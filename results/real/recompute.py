#!/usr/bin/env python3
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path


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


def clean_numbers(vals):
    out = []
    for v in vals:
        if v is None:
            continue
        fv = float(v)
        if not math.isnan(fv):
            out.append(fv)
    return out


def mean(vals):
    vals = clean_numbers(vals)
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
    vals = clean_numbers(vals)
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
    summary = {
        "weak_labeler_val_accuracy": weak_acc,
        "ceiling_acc": ceiling_acc,
    }

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


def paired_bootstrap_stats(rows, left, right, n_boot=10000, seed=12345):
    by = defaultdict(dict)
    for row in rows:
        cond = row.get("condition")
        if cond in (left, right):
            by[int(row["seed"])][cond] = float(row["student_val_accuracy"])
    seeds = sorted(s for s, vals in by.items() if left in vals and right in vals)
    diffs = [by[s][left] - by[s][right] for s in seeds]
    observed = mean(diffs)
    rng = random.Random(seed)
    n = len(diffs)
    boots = []
    for _ in range(n_boot):
        boots.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    boots.sort()
    lo = percentile(boots, 0.025)
    hi = percentile(boots, 0.975)
    le_zero = sum(1 for v in boots if v <= 0.0) / n_boot
    ge_zero = sum(1 for v in boots if v >= 0.0) / n_boot
    p = min(1.0, 2.0 * min(le_zero, ge_zero))
    return {
        "claim": f"{left} vs {right}, student_val_accuracy",
        "test": "paired bootstrap (10000 resamples, seed=12345) of the per-seed accuracy difference",
        "statistic": observed,
        "p": p,
        "ci_lo": lo,
        "ci_hi": hi,
        "n": n,
        "n_seeds": n,
        "notes": (
            f"Statistic is mean({left} - {right}) across shared seeds {seeds}; "
            "p is two-sided as 2*min(Pr(bootstrap mean <= 0), "
            "Pr(bootstrap mean >= 0))."
        ),
    }


def compute_stats(rows):
    pairs = [
        ("real-weak", "shuffled-weak"),
        ("real-weak", "random-label"),
        ("real-weak", "constant-majority"),
        ("real-weak", "gold-ceiling"),
    ]
    return [paired_bootstrap_stats(rows, a, b) for a, b in pairs]


def main():
    rdir = Path(__file__).resolve().parent
    rows = read_jsonl(rdir / "raw_results.jsonl")
    if len(sys.argv) > 1 and sys.argv[1] == "--stats-json":
        print(json.dumps(compute_stats(rows), indent=2, sort_keys=True))
        return
    _, _, lines = compute_metrics(rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
