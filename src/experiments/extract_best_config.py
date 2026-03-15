import yaml
import mlflow
from pathlib import Path

CONFIG_PATH = "configs/experiment.yaml"
OUTPUT_PATH = "configs/best_config.yaml"


def main():

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    mlflow.set_tracking_uri(cfg["tracking_uri"])

    client = mlflow.tracking.MlflowClient()

    exp = client.get_experiment_by_name(cfg["experiment_name"])

    runs = client.search_runs(
        exp.experiment_id,
        order_by=["metrics.composite_score DESC"],
        max_results=1,
    )

    best_run = runs[0]

    best_params = best_run.data.params

    with open(OUTPUT_PATH, "w") as f:
        yaml.dump(best_params, f)

    print("Best config saved to configs/best_config.yaml")


if __name__ == "__main__":
    main()