#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pandas as pd


POLICY_LABELS = {
    "bkt_timebudget": "BKT",
    "dkt_timebudget": "DKT",
    "gkt_timebudget": "GKT",
    "mars_timebudget": "MARS",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/reproduce")
    parser.add_argument("--out", default="paper_tables")
    args = parser.parse_args()

    results_dir = Path(args.results)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "policy_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")

    summary = pd.read_csv(summary_path)
    keep = ["bkt_timebudget", "dkt_timebudget", "gkt_timebudget", "mars_timebudget"]
    summary = summary[summary["policy"].isin(keep)].copy()
    summary["Policy"] = summary["policy"].map(POLICY_LABELS)
    order = {p: i for i, p in enumerate(keep)}
    summary["_order"] = summary["policy"].map(order)
    summary = summary.sort_values("_order")

    table = pd.DataFrame({
        "Policy": summary["Policy"],
        "Score": summary["mean_total_score"].map(lambda x: f"{x:.2f}"),
        "Score/min": summary["mean_score_per_minute"].map(lambda x: f"{x:.2f}"),
        "Marks": summary["mean_total_marks"].map(lambda x: f"{x:.2f}"),
        "Overtime": summary["overtime_rate"].map(lambda x: f"{x:.2f}"),
        "Time(s)": summary["mean_total_time_sec"].map(lambda x: f"{x:.2f}"),
    })
    table.to_latex(out_dir / "table_timebudget_negative.tex", index=False, escape=False)

    tests_path = results_dir / "statistical_tests.csv"
    if tests_path.exists():
        tests = pd.read_csv(tests_path)
        tests = tests[tests["comparison_policy"].isin(keep)]
        lines = ["Statistical comparison summary", ""]
        for row in tests.itertuples(index=False):
            label = POLICY_LABELS.get(row.comparison_policy, row.comparison_policy)
            lines.append(
                f"MARS vs {label}, {row.metric}: diff={row.mean_difference:.3f}, "
                f"95% CI [{row.ci95_low:.3f}, {row.ci95_high:.3f}], "
                f"Wilcoxon p={row.wilcoxon_p:.4g}"
            )
        (out_dir / "statistical_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {out_dir / 'table_timebudget_negative.tex'}")


if __name__ == "__main__":
    main()
