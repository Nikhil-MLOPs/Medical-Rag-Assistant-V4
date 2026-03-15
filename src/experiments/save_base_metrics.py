import yaml
import json
import mlflow
from pathlib import Path

CONFIG_PATH = "configs/experiment.yaml"
OUTPUT = Path("reports/base_metrics.json")


def main():

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    mlflow.set_tracking_uri(cfg["tracking_uri"])

    client = mlflow.tracking.MlflowClient()

    exp = client.get_experiment_by_name(cfg["experiment_name"])

    runs = client.search_runs(exp.experiment_id)

    # assume first run was baseline
    base_run = runs[-1]

    metrics = base_run.data.metrics

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Base metrics saved to reports/base_metrics.json")


if __name__ == "__main__":
    main()