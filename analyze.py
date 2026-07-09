from pathlib import Path

import matplotlib.pyplot as plt

from metrics_utils import compute_metrics, read_jsonl, write_curve_csv


PDIR = Path(__file__).resolve().parent
RDIR = PDIR / "results" / "real"


def main():
    raw_path = RDIR / "raw_results.jsonl"
    rows = read_jsonl(raw_path)
    _, curve_rows, lines = compute_metrics(rows)

    write_curve_csv(RDIR / "curve.csv", curve_rows)
    (RDIR / "analysis_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    xs = list(range(len(curve_rows)))
    ys = [r["mean_pgr"] for r in curve_rows]
    yerr_low = [r["mean_pgr"] - r["pgr_ci_low"] for r in curve_rows]
    yerr_high = [r["pgr_ci_high"] - r["mean_pgr"] for r in curve_rows]
    labels = [r["condition"] for r in curve_rows]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#6f6f6f", "#2f6f9f", "#268c64", "#c06c35", "#8b62a8", "#b04a4a"]
    ax.bar(xs, ys, color=colors[: len(xs)], alpha=0.82)
    ax.errorbar(xs, ys, yerr=[yerr_low, yerr_high], fmt="none", ecolor="black", capsize=4)
    ax.axhline(0.0, color="#555555", linewidth=0.8)
    ax.axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("PGR")
    ax.set_title("Weak-to-strong PGR by label channel")
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(RDIR / "figure_main.png", dpi=180)
    plt.close(fig)

    print("\n".join(lines))
    print("figure_main.png: PGR by label channel with seeded bootstrap 95% CIs; data in curve.csv")


if __name__ == "__main__":
    main()
