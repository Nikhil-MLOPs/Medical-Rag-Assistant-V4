import yaml

BASE_FILES = {
    "retrieval": "configs/retrieval.yaml",
    "rag": "configs/rag.yaml",
}

BEST_FILE = "configs/best_config.yaml"
OUTPUT = "configs/comparison_report.yaml"


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():

    best = load_yaml(BEST_FILE)

    report = {}

    for stage, path in BASE_FILES.items():

        base = load_yaml(path)

        report[stage] = {}

        for param in best:

            if param in base:

                report[stage][param] = {
                    "base": base[param],
                    "best": best[param],
                }

    with open(OUTPUT, "w") as f:
        yaml.dump(report, f)

    print("Comparison saved to configs/comparison_report.yaml")


if __name__ == "__main__":
    main()