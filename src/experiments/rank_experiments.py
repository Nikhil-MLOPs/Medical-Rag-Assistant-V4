import yaml
import mlflow
import pandas as pd

CONFIG_PATH = "configs/experiment.yaml"
OUTPUT = "reports/experiment_leaderboard.csv"


def main():

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    mlflow.set_tracking_uri(cfg["tracking_uri"])

    client = mlflow.tracking.MlflowClient()

    exp = client.get_experiment_by_name(cfg["experiment_name"])

    runs = client.search_runs(
        exp.experiment_id,
        order_by=["metrics.composite_score DESC"]
    )

    rows = []

    for r in runs:

        row = {
            "run_id": r.info.run_id,
            "composite_score": r.data.metrics.get("composite_score")
        }

        row.update(r.data.params)

        rows.append(row)

    df = pd.DataFrame(rows)

    df["rank"] = range(1, len(df) + 1)

    df.to_csv(OUTPUT, index=False)

    print("Leaderboard saved:", OUTPUT)


if __name__ == "__main__":
    main()