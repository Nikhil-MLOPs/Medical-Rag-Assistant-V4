import json
from pathlib import Path

BASE_METRICS_FILE = "reports/base_metrics.json"
BEST_RUN_FILE = "reports/best_run.json"
OUTPUT_FILE = Path("reports/metric_improvement_report.md")

BAR_LENGTH = 20


def percent_change(base, best):
    if base == 0:
        return None
    return ((best - base) / base) * 100


def bar(value, max_value=1.0):
    filled = int((value / max_value) * BAR_LENGTH)
    return "█" * filled + "░" * (BAR_LENGTH - filled)


def main():

    with open(BASE_METRICS_FILE) as f:
        base = json.load(f)

    with open(BEST_RUN_FILE) as f:
        best = json.load(f)["metrics"]

    lines = []

    lines.append("# 📊 RAG Experiment Improvement Report\n")
    lines.append("Comparison between **Base Configuration** and **Best Configuration**.\n")

    lines.append("## Metric Comparison\n")

    lines.append("| Metric | Base | Best | Change |")
    lines.append("|------|------|------|------|")

    for metric in base:

        base_val = base[metric]
        best_val = best.get(metric)

        if best_val is None:
            continue

        change = percent_change(base_val, best_val)

        if change is None:
            change_str = "Base=0"
        else:
            direction = "↑" if change > 0 else "↓"
            change_str = f"{direction} {abs(change):.2f}%"

        lines.append(
            f"| {metric} | {base_val:.3f} | {best_val:.3f} | {change_str} |"
        )

    lines.append("\n---\n")

    lines.append("## Visual Comparison\n")

    for metric in base:

        base_val = base[metric]
        best_val = best.get(metric)

        if best_val is None:
            continue

        lines.append(f"### {metric}\n")

        lines.append(
            f"Base : {bar(base_val)} {base_val:.3f}"
        )

        lines.append(
            f"Best : {bar(best_val)} {best_val:.3f}"
        )

        lines.append("")

    OUTPUT_FILE.parent.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n✅ Report generated: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()