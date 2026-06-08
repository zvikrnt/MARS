#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot and report generator for time-budgeted routing experiments.

Inputs are read from a results directory, usually:
    results/time_budget_v1/

The script intentionally uses only pandas and matplotlib, with a headless
backend, so it can run on servers without a display.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PLOT_POLICIES_FOR_CUMULATIVE = [
    "fixed_elo_nearest",
    "greedy_marks_per_time",
    "mars_original",
    "mars_timebudget",
]


def df_to_markdown_like(df, max_rows=20):
    if len(df) == 0:
        return ""

    shown = df.head(max_rows).copy()
    return "```text\n" + shown.to_string(index=False) + "\n```"


def read_optional_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def save_fig(fig, out_dir, stem, generated):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ext in ["png", "pdf"]:
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        generated.append(path)

    plt.close(fig)


def policy_order(summary):
    if "policy" not in summary.columns:
        return []

    if "mean_total_marks" in summary.columns:
        ordered = summary.sort_values("mean_total_marks", ascending=False)["policy"].tolist()
    else:
        ordered = summary["policy"].tolist()

    return list(dict.fromkeys(ordered))


def maybe_policy_ci(session_results, metric):
    if len(session_results) == 0 or "policy" not in session_results.columns or metric not in session_results.columns:
        return {}

    rows = {}
    for policy, group in session_results.groupby("policy"):
        values = group[metric].dropna().astype(float)
        if len(values) > 1:
            rows[policy] = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
        else:
            rows[policy] = 0.0

    return rows


def bar_plot(summary, session_results, value_col, ylabel, title, stem, plot_dir, generated, test_duration=None):
    if len(summary) == 0 or value_col not in summary.columns:
        return

    order = policy_order(summary)
    df = summary.set_index("policy").loc[order].reset_index()
    x = np.arange(len(df))
    values = df[value_col].astype(float).to_numpy()
    ci_map = maybe_policy_ci(session_results, value_col.replace("mean_", ""))

    yerr = None
    if ci_map:
        yerr = np.asarray([ci_map.get(policy, 0.0) for policy in df["policy"]], dtype=float)

    fig, ax = plt.subplots(figsize=(max(9, len(df) * 1.1), 5.5))
    ax.bar(x, values, yerr=yerr, capsize=4 if yerr is not None else 0)
    ax.set_xticks(x)
    ax.set_xticklabels(df["policy"], rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if test_duration is not None:
        ax.axhline(test_duration, linestyle="--", linewidth=1.5, color="black", label="test duration")
        ax.legend()

    for idx, label in enumerate(df["policy"]):
        if label == "mars_timebudget":
            ax.get_xticklabels()[idx].set_fontweight("bold")

    fig.tight_layout()
    save_fig(fig, plot_dir, stem, generated)


def marks_vs_time_scatter(summary, plot_dir, generated):
    required = {"mean_total_time_sec", "mean_total_marks", "policy"}
    if len(summary) == 0 or not required.issubset(summary.columns):
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(summary["mean_total_time_sec"], summary["mean_total_marks"])

    for _, row in summary.iterrows():
        ax.annotate(
            row["policy"],
            (row["mean_total_time_sec"], row["mean_total_marks"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Mean total time used (sec)")
    ax.set_ylabel("Mean total marks")
    ax.set_title("Policy Marks vs Time Used")
    fig.tight_layout()
    save_fig(fig, plot_dir, "marks_vs_time_scatter", generated)


def heatmap_plot(df, value_col, title, stem, plot_dir, generated):
    if len(df) == 0 or "persona" not in df.columns or "policy" not in df.columns or value_col not in df.columns:
        return

    pivot = df.pivot_table(
        index="persona",
        columns="policy",
        values=value_col,
        aggfunc="mean",
    )

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(9, len(pivot.columns) * 1.1), max(5, len(pivot.index) * 0.45)))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    save_fig(fig, plot_dir, stem, generated)


def difficulty_distribution(session_results, plot_dir, generated):
    cols = ["easy_count", "medium_count", "hard_count"]
    if len(session_results) == 0 or "policy" not in session_results.columns or not set(cols).issubset(session_results.columns):
        return

    grouped = session_results.groupby("policy")[cols].mean()
    shares = grouped.div(grouped.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    order = shares.sum(axis=1).sort_values(ascending=False).index.tolist()
    shares = shares.loc[order]

    fig, ax = plt.subplots(figsize=(max(9, len(shares) * 1.1), 5.5))
    bottom = np.zeros(len(shares))

    for col in cols:
        values = shares[col].to_numpy(dtype=float)
        ax.bar(np.arange(len(shares)), values, bottom=bottom, label=col.replace("_count", ""))
        bottom += values

    ax.set_xticks(np.arange(len(shares)))
    ax.set_xticklabels(shares.index, rotation=35, ha="right")
    ax.set_ylabel("Share of selected questions")
    ax.set_title("Difficulty Distribution by Policy")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, plot_dir, "difficulty_distribution_by_policy", generated)


def cumulative_marks_example(selected, plot_dir, generated):
    if len(selected) == 0:
        return

    required = {"policy", "test_id", "step", "cumulative_marks"}
    if not required.issubset(selected.columns):
        return

    subset = selected[selected["policy"].isin(PLOT_POLICIES_FOR_CUMULATIVE)].copy()
    if len(subset) == 0:
        return

    test_counts = subset.groupby("test_id")["policy"].nunique().sort_values(ascending=False)
    example_tests = test_counts.head(3).index.tolist()
    subset = subset[subset["test_id"].isin(example_tests)].copy()

    fig, ax = plt.subplots(figsize=(9, 6))

    for (test_id, policy), group in subset.groupby(["test_id", "policy"]):
        group = group.sort_values("step")
        ax.plot(
            group["step"],
            group["cumulative_marks"],
            marker="o",
            linewidth=1.5,
            markersize=3,
            label=f"{policy} | {test_id}",
        )

    ax.set_xlabel("Selected question index")
    ax.set_ylabel("Cumulative marks")
    ax.set_title("Cumulative Marks for Representative Sessions")
    ax.legend(fontsize=7, ncol=1)
    fig.tight_layout()
    save_fig(fig, plot_dir, "cumulative_marks_example", generated)


def tuning_trials_plot(tuning, plot_dir, generated):
    if len(tuning) == 0 or "trial" not in tuning.columns or "objective" not in tuning.columns:
        return

    tuning = tuning.sort_values("trial").copy()
    tuning["best_so_far"] = tuning["objective"].cummax()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tuning["trial"], tuning["objective"], marker="o", label="objective")
    ax.plot(tuning["trial"], tuning["best_so_far"], linewidth=2, label="best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Validation objective")
    ax.set_title("MARS-TimeBudget Tuning Trials")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, plot_dir, "tuning_trials_plot", generated)


def latex_table(df, path, columns=None, float_format="%.3f"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(df) == 0:
        path.write_text("", encoding="utf-8")
        return

    out = df.copy()
    if columns:
        out = out[[col for col in columns if col in out.columns]]

    out.to_latex(path, index=False, float_format=float_format)


def create_tables(results_dir, session_results, summary, tests):
    tables_dir = Path(results_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    latex_table(
        summary,
        tables_dir / "table_policy_summary.tex",
        columns=[
            "policy",
            "mean_total_marks",
            "mean_marks_per_minute",
            "mean_total_time_sec",
            "overtime_rate",
            "mean_overtime_sec",
            "mean_accuracy",
            "mean_n_questions",
        ],
    )

    latex_table(
        tests,
        tables_dir / "table_statistical_tests.tex",
        columns=[
            "comparison_policy",
            "metric",
            "mean_difference_reference_minus_policy",
            "bootstrap_ci95_low",
            "bootstrap_ci95_high",
            "chosen_p_value",
            "cohens_d",
            "n_pairs",
        ],
    )

    if len(session_results) and "persona" in session_results.columns:
        persona = (
            session_results.groupby(["persona", "policy"], as_index=False)
            .agg(
                mean_total_marks=("total_marks", "mean"),
                overtime_rate=("overtime_violation", "mean"),
                mean_marks_per_minute=("marks_per_minute", "mean"),
            )
        )
    else:
        persona = pd.DataFrame()

    latex_table(
        persona,
        tables_dir / "table_persona_summary.tex",
        columns=["persona", "policy", "mean_total_marks", "mean_marks_per_minute", "overtime_rate"],
    )


def read_json_optional(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def top_policy_ranking(summary):
    if len(summary) == 0 or "mean_total_marks" not in summary.columns:
        return "No policy summary available."

    rows = []
    ordered = summary.sort_values("mean_total_marks", ascending=False)

    for _, row in ordered.iterrows():
        rows.append(
            f"- {row['policy']}: mean_total_marks={row['mean_total_marks']:.3f}, "
            f"marks_per_minute={row.get('mean_marks_per_minute', np.nan):.3f}, "
            f"overtime_rate={row.get('overtime_rate', np.nan):.3f}"
        )

    return "\n".join(rows)


def create_report(results_dir, dataset_summary, response_metrics, time_metrics, summary, tests, params):
    report_path = Path(results_dir) / "time_budget_report.md"

    lines = [
        "# Time-Budget Routing Report",
        "",
        "## Dataset Summary",
    ]

    if len(dataset_summary):
        for key, value in dataset_summary.iloc[0].to_dict().items():
            lines.append(f"- **{key}**: {value}")
    else:
        lines.append("Dataset summary file was not available.")

    lines.extend(["", "## Simulator Performance"])

    if len(response_metrics):
        lines.append("Response simulator:")
        lines.append(df_to_markdown_like(response_metrics))
    else:
        lines.append("Response simulator metrics were not available.")

    if len(time_metrics):
        lines.append("")
        lines.append("Time simulator / ensemble:")
        lines.append(df_to_markdown_like(time_metrics))
    else:
        lines.append("Time simulator metrics were not available.")

    lines.extend(["", "## Final Policy Ranking", top_policy_ranking(summary)])

    lines.extend(["", "## Best MARS-TimeBudget Parameters"])
    if params:
        for key, value in params.items():
            lines.append(f"- **{key}**: {value}")
    else:
        lines.append("Best parameter file was not available.")

    lines.extend(["", "## Key Statistical Comparisons"])
    if len(tests):
        cols = [
            "comparison_policy",
            "metric",
            "mean_difference_reference_minus_policy",
            "bootstrap_ci95_low",
            "bootstrap_ci95_high",
            "chosen_p_value",
            "cohens_d",
        ]
        lines.append(df_to_markdown_like(tests[[c for c in cols if c in tests.columns]]))
    else:
        lines.append("Statistical tests were not available.")

    lines.extend([
        "",
        "## Interpretation",
        (
            "Use these results in the paper only if MARS-TimeBudget improves total marks "
            "or marks-per-minute while reducing or controlling overtime compared with "
            "FixedElo and greedy baselines."
        ),
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/time_budget_v1")
    parser.add_argument("--test_duration", type=float, default=3600.0)
    args = parser.parse_args()

    results_dir = Path(args.results)
    plot_dir = results_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    session_results = read_optional_csv(results_dir / "policy_session_results.csv")
    summary = read_optional_csv(results_dir / "policy_summary.csv")
    tests = read_optional_csv(results_dir / "statistical_tests.csv")
    tuning = read_optional_csv(results_dir / "tuning_trials.csv")
    selected = read_optional_csv(results_dir / "policy_selected_questions.csv")
    dataset_summary = read_optional_csv(results_dir / "dataset_summary.csv")
    response_metrics = read_optional_csv(results_dir / "response_model_metrics.csv")
    time_metrics = read_optional_csv(results_dir / "time_model_ensemble_metrics.csv")
    if len(time_metrics) == 0:
        time_metrics = read_optional_csv(results_dir / "time_model_metrics.csv")

    params = read_json_optional(results_dir / "best_mars_timebudget_params.json")

    bar_plot(
        summary,
        session_results,
        "mean_total_marks",
        "Mean total marks",
        "Mean Total Marks by Policy",
        "policy_total_marks_bar",
        plot_dir,
        generated,
    )
    bar_plot(
        summary,
        session_results,
        "mean_marks_per_minute",
        "Mean marks per minute",
        "Marks per Minute by Policy",
        "policy_marks_per_minute_bar",
        plot_dir,
        generated,
    )
    bar_plot(
        summary,
        session_results,
        "overtime_rate",
        "Overtime rate",
        "Overtime Rate by Policy",
        "policy_overtime_rate_bar",
        plot_dir,
        generated,
    )
    bar_plot(
        summary,
        session_results,
        "mean_total_time_sec",
        "Mean total time used (sec)",
        "Mean Time Used by Policy",
        "policy_time_used_bar",
        plot_dir,
        generated,
        test_duration=args.test_duration,
    )

    marks_vs_time_scatter(summary, plot_dir, generated)
    heatmap_plot(session_results, "total_marks", "Persona Mean Total Marks", "persona_total_marks_heatmap", plot_dir, generated)
    heatmap_plot(session_results, "overtime_violation", "Persona Overtime Rate", "persona_overtime_heatmap", plot_dir, generated)
    difficulty_distribution(session_results, plot_dir, generated)
    cumulative_marks_example(selected, plot_dir, generated)
    tuning_trials_plot(tuning, plot_dir, generated)

    create_tables(results_dir, session_results, summary, tests)
    report_path = create_report(
        results_dir,
        dataset_summary,
        response_metrics,
        time_metrics,
        summary,
        tests,
        params,
    )

    print("Generated plot/table/report files:")
    for path in generated:
        print(path)
    print(results_dir / "tables" / "table_policy_summary.tex")
    print(results_dir / "tables" / "table_statistical_tests.tex")
    print(results_dir / "tables" / "table_persona_summary.tex")
    print(report_path)


if __name__ == "__main__":
    main()
