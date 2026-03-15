import json
import random
from pathlib import Path
import dagshub
import mlflow

from src.utils.config import load_experiment_config
from src.rag.rag_service import RagService
from src.experiments.evaluator import Evaluator
from src.utils.logging import setup_logging


CONFIG_PATH = Path("configs/experiment.yaml")

logger = setup_logging("ExperimentRunner")


class ExperimentRunner:

    def __init__(self):

        # -----------------------------
        # Load Config
        # -----------------------------
        self.cfg = load_experiment_config(CONFIG_PATH)

        # -----------------------------
        # Initialize DagsHub + MLflow
        # -----------------------------
        dagshub.init(
            repo_owner="Nikhil-MLOPs",
            repo_name="Medical-Rag-Assistant-V4",
            mlflow=True,
        )

        mlflow.set_tracking_uri(self.cfg.tracking_uri)
        mlflow.set_experiment(self.cfg.experiment_name)

        # -----------------------------
        # Load Golden Dataset
        # -----------------------------
        dataset_path = Path(self.cfg.golden_dataset_path)

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Golden dataset not found at {dataset_path}"
            )

        with open(dataset_path, "r", encoding="utf-8") as f:
            self.dataset = [json.loads(line) for line in f]

        logger.info(
            f"Loaded {len(self.dataset)} golden samples."
        )

    # -------------------------------------------------
    # Main Experiment Loop
    # -------------------------------------------------
    def run(self):

        logger.info(
            f"Starting {self.cfg.num_trials} MLflow trials..."
        )

        for trial in range(1, self.cfg.num_trials + 1):

            logger.info(f"Running trial {trial}")

            # -----------------------------------------
            # Random parameter selection
            # -----------------------------------------
            params = {
                k: random.choice(v)
                for k, v in self.cfg.search_space.items()
            }

            with mlflow.start_run():

                # Log model name
                # mlflow.log_param("model", self.cfg.ollama_model)

                # Log parameters
                for p, v in params.items():
                    mlflow.log_param(p, v)

                # Initialize RAG with overrides
                rag = RagService(config_override=params)

                evaluator = Evaluator(rag, self.cfg)

                metrics_agg = {}

                # -------------------------------------
                # Evaluate full golden dataset
                # -------------------------------------
                for sample in self.dataset:

                    metrics = evaluator.evaluate_sample(sample)

                    for k, v in metrics.items():
                        metrics_agg.setdefault(k, []).append(v)

                # -------------------------------------
                # Aggregate Metrics
                # -------------------------------------
                final_metrics = {
                    k: sum(v) / len(v)
                    for k, v in metrics_agg.items()
                }

                # -------------------------------------
                # Composite Score
                # -------------------------------------
                composite_score = self.compute_weighted_score(
                    final_metrics
                )

                final_metrics["composite_score"] = composite_score

                # Log metrics to MLflow
                for k, v in final_metrics.items():
                    mlflow.log_metric(k, v)

                logger.info(
                    f"Trial {trial} composite_score={composite_score:.4f}"
                )

        logger.info("All trials completed.")

    # -------------------------------------------------
    # Weighted Composite Score
    # -------------------------------------------------
    def compute_weighted_score(self, metrics):

        score = 0.0

        for metric, weight in self.cfg.metrics_weights.items():

            if metric in metrics:
                score += weight * metrics[metric]

        return score


# -----------------------------------------------------
# Entrypoint
# -----------------------------------------------------
if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.run()