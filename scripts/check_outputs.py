#!/usr/bin/env python3
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/reproduce")
    parser.add_argument("--tables", default="paper_tables")
    args = parser.parse_args()

    required = [
        Path(args.results) / "policy_summary.csv",
        Path(args.results) / "policy_session_results.csv",
        Path(args.results) / "statistical_tests.csv",
        Path(args.tables) / "table_timebudget_negative.tex",
    ]
    optional = [Path(args.results) / "model_prediction_metrics.csv"]

    missing = [str(path) for path in required if not path.exists()]
    for path in optional:
        if not path.exists():
            print(f"WARNING: optional output missing: {path}")

    if missing:
        print("Missing required outputs:")
        for path in missing:
            print(f"  {path}")
        raise SystemExit(1)

    print("All required outputs found.")


if __name__ == "__main__":
    main()
