import csv
import json
import math
import random
from collections import defaultdict


PREFERRED_CONDITIONS = [
    "base-fewshot",
    "gold-ceiling",
    "real-weak",
    "shuffled-weak",
    "random-label",
    "constant-majority",
]


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean(vals):
    vals = [v for v in vals if v is not None and not math.isnan(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def percentile(sorted_vals, pct):
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * pct
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def bootstrap_ci(vals, n_boot=10000, seed=12345):
    vals = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], vals[0]
    rng = random.Random(seed)
    n = len(vals)
    boots = []
    for _ in range(n_boot):
        boots.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    boots.sort()
    return percentile(boots, 0.025), percentile(boots, 0.975)


def condition_order(conditions):
    ranked = {c: i for i, c in enumerate(PREFERRED_CONDITIONS)}
    return sorted(conditions, key=lambda c: (ranked.get(c, 999), c))


def compute_metrics(rows, n_boot=10000, seed=12345):
    by_cond = defaultdict(list)
    for row in rows:
        by_cond[row["condition"]].append(row)

    weak_acc = mean([r.get("weak_labeler_val_accuracy") for r in rows])
    ceiling_acc = mean(
        [r.get("student_val_accuracy") for r in by_cond.get("gold-ceiling", [])]
    )
    denom = ceiling_acc - weak_acc

    curve_rows = []
    pgr_by_cond = {}
    train_acc_by_cond = {}
    student_acc_by_cond = {}

    for cond in condition_order(by_cond):
        cond_rows = sorted(by_cond[cond], key=lambda r: str(r.get("seed", "")))
        student_vals = [r.get("student_val_accuracy") for r in cond_rows]
        train_vals = [r.get("train_label_gold_accuracy") for r in cond_rows]
        pgr_vals = []
        for r in cond_rows:
            acc = r.get("student_val_accuracy")
            if acc is None or denom == 0 or math.isnan(denom):
                pgr = float("nan")
            else:
                pgr = (float(acc) - weak_acc) / denom
            pgr_vals.append(pgr)

        lo, hi = bootstrap_ci(pgr_vals, n_boot=n_boot, seed=seed)
        pgr_by_cond[cond] = pgr_vals
        train_acc_by_cond[cond] = train_vals
        student_acc_by_cond[cond] = student_vals
        curve_rows.append(
            {
                "condition": cond,
                "n": len(cond_rows),
                "mean_pgr": mean(pgr_vals),
                "pgr_ci_low": lo,
                "pgr_ci_high": hi,
                "mean_student_acc": mean(student_vals),
                "mean_train_label_gold_accuracy": mean(train_vals),
            }
        )

    summary = {
        "weak_labeler_val_accuracy": weak_acc,
        "ceiling_acc": ceiling_acc,
    }
    for row in curve_rows:
        cond = row["condition"].replace("-", "_")
        summary[f"{cond}_student_acc_mean"] = row["mean_student_acc"]
        summary[f"{cond}_pgr_mean"] = row["mean_pgr"]
        summary[f"{cond}_pgr_ci_low"] = row["pgr_ci_low"]
        summary[f"{cond}_pgr_ci_high"] = row["pgr_ci_high"]
        summary[f"{cond}_train_label_gold_accuracy_mean"] = row[
            "mean_train_label_gold_accuracy"
        ]

    real = pgr_by_cond.get("real-weak", [])
    shuffled = pgr_by_cond.get("shuffled-weak", [])
    if real and shuffled:
        # Pair by row order after seed sort. The smoke and full designs share seeds.
        paired = [a - b for a, b in zip(real, shuffled)]
        lo, hi = bootstrap_ci(paired, n_boot=n_boot, seed=seed)
        summary["real_minus_shuffled_pgr_mean"] = mean(paired)
        summary["real_minus_shuffled_pgr_ci_low"] = lo
        summary["real_minus_shuffled_pgr_ci_high"] = hi

    lines = []
    for key in sorted(summary):
        val = summary[key]
        if val is None or (isinstance(val, float) and math.isnan(val)):
            text = "nan"
        else:
            text = f"{float(val):.6f}"
        lines.append(f"{key}: {text}")
    return summary, curve_rows, lines


def write_curve_csv(path, curve_rows):
    fieldnames = [
        "condition",
        "n",
        "mean_pgr",
        "pgr_ci_low",
        "pgr_ci_high",
        "mean_student_acc",
        "mean_train_label_gold_accuracy",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in curve_rows:
            writer.writerow(row)
