from pathlib import Path

from metrics_utils import compute_metrics, read_jsonl


def main():
    rdir = Path(__file__).resolve().parent / "results" / "real"
    rows = read_jsonl(rdir / "raw_results.jsonl")
    _, _, lines = compute_metrics(rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
