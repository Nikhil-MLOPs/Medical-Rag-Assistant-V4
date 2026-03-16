import yaml
import mlflow
import json
from pathlib import Path

CONFIG_PATH = "configs/experiment.yaml"
OUTPUT = Path("reports/best_run.json")


def main():

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    mlflow.set_tracking_uri(cfg["tracking_uri"])

    client = mlflow.tracking.MlflowClient()

    exp = client.get_experiment_by_name(cfg["experiment_name"])

    runs = client.search_runs(
        exp.experiment_id,
        order_by=["metrics.composite_score DESC"],
        max_results=1
    )

    best_run = runs[0]

    best_info = {
        "run_id": best_run.info.run_id,
        "metrics": best_run.data.metrics,
        "params": best_run.data.params
    }

    # Create reports folder if it doesn't exist
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT, "w") as f:
        json.dump(best_info, f, indent=2)

    print("Best run saved:", OUTPUT)


if __name__ == "__main__":
    main()