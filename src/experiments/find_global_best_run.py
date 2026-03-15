import mlflow
import yaml
import json
from pathlib import Path

CONFIG_PATH = "configs/experiment.yaml"
OUTPUT_PATH = Path("reports/global_best_run.json")


def main():

    # -----------------------------
    # Load config
    # -----------------------------
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    mlflow.set_tracking_uri(cfg["tracking_uri"])

    client = mlflow.tracking.MlflowClient()

    experiments = client.search_experiments()

    all_runs = []

    # -----------------------------
    # Collect runs from all experiments
    # -----------------------------
    for exp in experiments:

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id]
        )

        for run in runs:

            if "composite_score" in run.data.metrics:

                all_runs.append({
                    "run": run,
                    "experiment_name": exp.name
                })

    if not all_runs:
        print("No runs with composite_score found.")
        return

    # -----------------------------
    # Find global best run
    # -----------------------------
    best_entry = max(
        all_runs,
        key=lambda x: x["run"].data.metrics["composite_score"]
    )

    best_run = best_entry["run"]
    best_experiment = best_entry["experiment_name"]

    result = {
        "experiment_name": best_experiment,
        "run_id": best_run.info.run_id,
        "composite_score": best_run.data.metrics["composite_score"],
        "params": best_run.data.params,
        "metrics": best_run.data.metrics
    }

    # -----------------------------
    # Print result
    # -----------------------------
    print("\n🏆 GLOBAL BEST RUN\n")

    print("Experiment:", result["experiment_name"])
    print("Run ID:", result["run_id"])
    print("Composite Score:", result["composite_score"])

    print("\nParameters:")
    for k, v in result["params"].items():
        print(f"{k}: {v}")

    # -----------------------------
    # Save result
    # -----------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=4)

    print(f"\n✅ Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()