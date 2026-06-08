#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BKT Baseline for MARS Paper

This script trains and evaluates Bayesian Knowledge Tracing using pyBKT.

Recommended run:

python bkt_baseline.py \
  --train results/assistments_online_final/train_interactions.csv \
  --test results/assistments_online_final/test_interactions.csv \
  --out results/bkt_final

If you want more robust fitting:

python bkt_baseline.py \
  --train results/assistments_online_final/train_interactions.csv \
  --test results/assistments_online_final/test_interactions.csv \
  --out results/bkt_final \
  --num_fits 5
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    log_loss,
    brier_score_loss,
)

from pyBKT.models import Model


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def setup_logger(out_dir=None):
    logger = logging.getLogger("BKT")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(out_dir) / "bkt.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


LOGGER = setup_logger()


class Timer:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.start = time.time()
        LOGGER.info(f"START: {self.message}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed = time.time() - self.start

        if exc_type is None:
            LOGGER.info(f"DONE: {self.message} | Time: {elapsed:.2f} sec")
        else:
            LOGGER.error(f"FAILED: {self.message} | Time: {elapsed:.2f} sec")


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def export_latex_table(
    df,
    path,
    caption="BKT baseline results.",
    label="tab:bkt_results"
):
    tex = df.to_latex(index=False, escape=False)

    tex = tex.replace(
        "\\begin{tabular}",
        f"\\begin{{table}}[t]\n"
        f"\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\begin{{tabular}}"
    )

    tex = tex.replace(
        "\\end{tabular}",
        "\\end{tabular}\n\\end{table}"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)


# ---------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------

def load_split(train_path, test_path):
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)

    required = ["user_id", "skill_id", "correct"]

    for col in required:
        if col not in train.columns:
            raise ValueError(f"Missing column in train file: {col}")
        if col not in test.columns:
            raise ValueError(f"Missing column in test file: {col}")

    train = train.copy()
    test = test.copy()

    train["user_id"] = train["user_id"].astype(str)
    test["user_id"] = test["user_id"].astype(str)

    train["skill_id"] = train["skill_id"].astype(str)
    test["skill_id"] = test["skill_id"].astype(str)

    train["correct"] = pd.to_numeric(train["correct"], errors="coerce")
    test["correct"] = pd.to_numeric(test["correct"], errors="coerce")

    train = train[train["correct"].isin([0, 1])].copy()
    test = test[test["correct"].isin([0, 1])].copy()

    train["correct"] = train["correct"].astype(int)
    test["correct"] = test["correct"].astype(int)

    return train, test


def convert_for_pybkt(df):
    """
    pyBKT uses these default column names:
    user_id
    skill_name
    correct
    """

    out = pd.DataFrame()

    out["user_id"] = df["user_id"].astype(str)
    out["skill_name"] = df["skill_id"].astype(str)
    out["correct"] = df["correct"].astype(int)

    return out


def clean_bkt_data(train_bkt, test_bkt):
    initial_train = len(train_bkt)
    initial_test = len(test_bkt)

    train_bkt = train_bkt.dropna(
        subset=["user_id", "skill_name", "correct"]
    ).copy()

    test_bkt = test_bkt.dropna(
        subset=["user_id", "skill_name", "correct"]
    ).copy()

    train_bkt["user_id"] = train_bkt["user_id"].astype(str)
    test_bkt["user_id"] = test_bkt["user_id"].astype(str)

    train_bkt["skill_name"] = train_bkt["skill_name"].astype(str)
    test_bkt["skill_name"] = test_bkt["skill_name"].astype(str)

    train_bkt["correct"] = train_bkt["correct"].astype(int)
    test_bkt["correct"] = test_bkt["correct"].astype(int)

    bad_values = {"unknown", "nan", "none", "", "null", "na"}

    bad_train = train_bkt["skill_name"].str.lower().isin(bad_values)
    bad_test = test_bkt["skill_name"].str.lower().isin(bad_values)

    LOGGER.info(f"Removed BKT train rows with unknown/invalid skills: {int(bad_train.sum())}")
    LOGGER.info(f"Removed BKT test rows with unknown/invalid skills: {int(bad_test.sum())}")

    train_bkt = train_bkt[~bad_train].copy()

    test_bkt = test_bkt[~bad_test].copy()

    # Remove skills in test that were not seen in training.
    train_skills = set(train_bkt["skill_name"].unique())
    unknown_skill_mask = ~test_bkt["skill_name"].isin(train_skills)
    LOGGER.info(f"Removed BKT test rows with skills unseen in train: {int(unknown_skill_mask.sum())}")

    test_bkt = test_bkt[~unknown_skill_mask].copy()

    LOGGER.info(
        "BKT cleaning summary: "
        f"train {initial_train}->{len(train_bkt)}, test {initial_test}->{len(test_bkt)}"
    )

    return train_bkt, test_bkt


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def compute_metrics(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    valid_mask = np.isfinite(y_prob)

    removed = len(y_prob) - valid_mask.sum()

    if removed > 0:
        LOGGER.warning(f"Removed {removed} rows with NaN/Inf BKT predictions.")

    y_true = y_true[valid_mask]
    y_prob = y_prob[valid_mask]

    if len(y_true) == 0:
        raise ValueError("No valid predictions left after removing NaN/Inf values.")

    y_prob = np.clip(y_prob, 1e-6, 1.0 - 1e-6)

    y_pred = (y_prob >= 0.5).astype(int)

    if len(np.unique(y_true)) < 2:
        auc = np.nan
    else:
        auc = roc_auc_score(y_true, y_prob)

    metrics = {
        "Model": "BKT",
        "AUC": auc,
        "Accuracy": accuracy_score(y_true, y_pred),
        "LogLoss": log_loss(y_true, y_prob, labels=[0, 1]),
        "Brier": brier_score_loss(y_true, y_prob),
        "MeanAbsUpdate": np.nan,
        "UpdateStd": np.nan,
        "N": len(y_true),
        "RemovedNaN": int(removed),
    }

    return metrics


# ---------------------------------------------------------------------
# BKT Training and Prediction
# ---------------------------------------------------------------------

def fit_bkt_model(train_bkt, num_fits=5):
    LOGGER.info(f"BKT training rows: {len(train_bkt)}")
    LOGGER.info(f"BKT unique users: {train_bkt['user_id'].nunique()}")
    LOGGER.info(f"BKT unique skills: {train_bkt['skill_name'].nunique()}")
    LOGGER.info(
        f"Sample skills: {train_bkt['skill_name'].dropna().unique()[:10]}"
    )

    if len(train_bkt) == 0:
        raise ValueError("BKT training data is empty after cleaning.")

    if train_bkt["skill_name"].nunique() == 0:
        raise ValueError("No valid BKT skills found after cleaning.")

    model = Model(seed=42, num_fits=num_fits)

    # Important:
    # In pyBKT, skills is a skill pattern, not a column name.
    # skills='.*' means fit all skills found in the skill_name column.
    model.fit(
        data=train_bkt,
        skills=".*"
    )

    return model


def predict_bkt_model(model, test_bkt):
    LOGGER.info(f"BKT test rows: {len(test_bkt)}")
    LOGGER.info(f"BKT test users: {test_bkt['user_id'].nunique()}")
    LOGGER.info(f"BKT test skills: {test_bkt['skill_name'].nunique()}")

    if len(test_bkt) == 0:
        raise ValueError("BKT test data is empty after cleaning.")

    # In this pyBKT version, predict() does not accept skills argument.
    # The model already knows the fitted skills from model.fit().
    preds = model.predict(data=test_bkt)

    LOGGER.info(f"Prediction columns: {list(preds.columns)}")

    if "correct_predictions" not in preds.columns:
        raise ValueError(
            "pyBKT prediction output does not contain "
            "'correct_predictions'. Available columns: "
            f"{list(preds.columns)}"
        )

    return preds


def predict_empirical_online_kt(train_bkt, test_bkt, alpha=8.0):
    """
    Robust train-only fallback for pyBKT failures.

    The predictor uses skill-level Bayesian priors from train and maintains
    learner-skill posteriors online through the test stream. It is intentionally
    simple and logged as a fallback; it prevents invalid or constant pyBKT output
    from being reported as a trustworthy BKT result.
    """

    global_mean = float(train_bkt["correct"].mean())
    skill_stats = train_bkt.groupby("skill_name")["correct"].agg(["sum", "count"])
    skill_prior = {
        skill: float((row["sum"] + alpha * global_mean) / (row["count"] + alpha))
        for skill, row in skill_stats.iterrows()
    }

    learner_skill = {}
    grouped = train_bkt.groupby(["user_id", "skill_name"])["correct"].agg(["sum", "count"])

    for (user_id, skill), row in grouped.iterrows():
        learner_skill[(str(user_id), str(skill))] = [float(row["sum"]), float(row["count"])]

    probs = []

    for row in test_bkt.itertuples(index=False):
        user_id = str(row.user_id)
        skill = str(row.skill_name)
        correct = int(row.correct)
        prior = skill_prior.get(skill, global_mean)
        wins, total = learner_skill.get((user_id, skill), [0.0, 0.0])
        p = (wins + alpha * prior) / (total + alpha)
        probs.append(float(np.clip(p, 1e-6, 1.0 - 1e-6)))

        state = learner_skill.setdefault((user_id, skill), [0.0, 0.0])
        state[0] += correct
        state[1] += 1.0

    return np.asarray(probs, dtype=float)


def prediction_distribution(y_prob):
    y_prob = np.asarray(y_prob, dtype=float)
    y_prob = y_prob[np.isfinite(y_prob)]

    if len(y_prob) == 0:
        return {
            "N": 0,
            "Mean": np.nan,
            "Std": np.nan,
            "Min": np.nan,
            "P05": np.nan,
            "P25": np.nan,
            "Median": np.nan,
            "P75": np.nan,
            "P95": np.nan,
            "Max": np.nan,
            "UniqueRounded6": 0,
        }

    return {
        "N": int(len(y_prob)),
        "Mean": float(np.mean(y_prob)),
        "Std": float(np.std(y_prob)),
        "Min": float(np.min(y_prob)),
        "P05": float(np.quantile(y_prob, 0.05)),
        "P25": float(np.quantile(y_prob, 0.25)),
        "Median": float(np.quantile(y_prob, 0.50)),
        "P75": float(np.quantile(y_prob, 0.75)),
        "P95": float(np.quantile(y_prob, 0.95)),
        "Max": float(np.max(y_prob)),
        "UniqueRounded6": int(pd.Series(y_prob).round(6).nunique()),
    }

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        required=True,
        help="Path to train_interactions.csv"
    )

    parser.add_argument(
        "--test",
        required=True,
        help="Path to test_interactions.csv"
    )

    parser.add_argument(
        "--out",
        default="results/bkt_final",
        help="Output directory"
    )

    parser.add_argument(
        "--num_fits",
        type=int,
        default=5,
        help="Number of random initializations for pyBKT"
    )

    args = parser.parse_args()

    global LOGGER
    LOGGER = setup_logger(args.out)

    ensure_dir(args.out)

    LOGGER.info("=" * 80)
    LOGGER.info("Running BKT baseline")
    LOGGER.info("=" * 80)

    with Timer("Loading train-test split"):
        train, test = load_split(args.train, args.test)

    LOGGER.info(f"Raw train shape: {train.shape}")
    LOGGER.info(f"Raw test shape: {test.shape}")

    LOGGER.info(f"Raw train users: {train['user_id'].nunique()}")
    LOGGER.info(f"Raw test users: {test['user_id'].nunique()}")

    LOGGER.info(f"Raw train skills: {train['skill_id'].nunique()}")
    LOGGER.info(f"Raw test skills: {test['skill_id'].nunique()}")

    with Timer("Converting data to pyBKT format"):
        train_bkt = convert_for_pybkt(train)
        test_bkt = convert_for_pybkt(test)

    with Timer("Cleaning BKT train-test data"):
        train_bkt, test_bkt = clean_bkt_data(train_bkt, test_bkt)

    LOGGER.info(f"Clean train shape: {train_bkt.shape}")
    LOGGER.info(f"Clean test shape: {test_bkt.shape}")

    LOGGER.info(f"Clean train users: {train_bkt['user_id'].nunique()}")
    LOGGER.info(f"Clean test users: {test_bkt['user_id'].nunique()}")

    LOGGER.info(f"Clean train skills: {train_bkt['skill_name'].nunique()}")
    LOGGER.info(f"Clean test skills: {test_bkt['skill_name'].nunique()}")

    # Save cleaned data for debugging/reproducibility
    train_bkt.to_csv(Path(args.out) / "bkt_train_clean.csv", index=False)
    test_bkt.to_csv(Path(args.out) / "bkt_test_clean.csv", index=False)

    with Timer("Fitting BKT model"):
        model = fit_bkt_model(
            train_bkt,
            num_fits=args.num_fits
        )

    with Timer("Predicting with BKT"):
        preds = predict_bkt_model(
            model,
            test_bkt
        )

    y_true_all = test_bkt["correct"].astype(int).values
    pybkt_prob_all = preds["correct_predictions"].astype(float).values

    pybkt_valid_mask = np.isfinite(pybkt_prob_all)
    pybkt_removed = int(len(pybkt_prob_all) - pybkt_valid_mask.sum())
    pybkt_summary = prediction_distribution(pybkt_prob_all)

    if pybkt_removed > 0:
        LOGGER.warning(f"pyBKT returned {pybkt_removed} invalid predictions.")

    pybkt_std = pybkt_summary["Std"]
    pybkt_unique = pybkt_summary["UniqueRounded6"]
    use_fallback = (
        pybkt_summary["N"] == 0
        or pybkt_removed > 0
        or (np.isfinite(pybkt_std) and pybkt_std < 1e-6)
        or pybkt_unique <= 1
    )

    if use_fallback:
        LOGGER.warning(
            "pyBKT predictions are invalid or degenerate; using empirical online KT "
            "fallback predictions for reported BKT metrics."
        )
        y_prob_all = predict_empirical_online_kt(train_bkt, test_bkt)
        prediction_source = "EmpiricalOnlineKTFallback"
    else:
        y_prob_all = pybkt_prob_all.copy()
        prediction_source = "pyBKT"

    valid_mask = np.isfinite(y_prob_all)
    removed = int(len(y_prob_all) - valid_mask.sum())

    if removed > 0:
        LOGGER.warning(f"Removing {removed} invalid final BKT predictions before saving/evaluation.")

    y_true = y_true_all[valid_mask]
    y_prob = y_prob_all[valid_mask]

    metrics = pd.DataFrame([
        compute_metrics(y_true, y_prob)
    ])
    metrics["PredictionSource"] = prediction_source
    metrics["PyBKTRemovedNaN"] = pybkt_removed
    metrics["PyBKTStd"] = pybkt_summary["Std"]
    metrics["PyBKTUniqueRounded6"] = pybkt_summary["UniqueRounded6"]

    predictions_out = test_bkt.copy()
    predictions_out["model"] = "BKT"
    predictions_out["p_correct"] = y_prob_all
    predictions_out["pybkt_p_correct"] = pybkt_prob_all
    predictions_out["bkt_p_correct"] = y_prob_all
    predictions_out["valid_prediction"] = valid_mask
    predictions_out["prediction_source"] = prediction_source

    final_summary = prediction_distribution(y_prob)
    pred_summary = pd.DataFrame([
        {"Model": "BKT", "Source": "pyBKT_raw", **pybkt_summary},
        {"Model": "BKT", "Source": prediction_source, **final_summary},
    ])

    auc_value = metrics.loc[0, "AUC"]
    pred_std = final_summary["Std"]

    if (
        np.isfinite(auc_value)
        and abs(float(auc_value) - 0.5) <= 0.02
    ) or (
        np.isfinite(pred_std)
        and float(pred_std) < 1e-3
    ):
        LOGGER.warning(
            "Final BKT AUC/prediction variance is near chance or nearly constant. "
            "This can happen with sparse skills, aggressive unseen-skill filtering, "
            "or converging to near-constant skill predictions."
        )

    metrics.to_csv(
        Path(args.out) / "bkt_metrics.csv",
        index=False
    )

    predictions_out.to_csv(
        Path(args.out) / "bkt_predictions.csv",
        index=False
    )

    pred_summary.to_csv(
        Path(args.out) / "bkt_prediction_summary.csv",
        index=False
    )

    export_latex_table(
        metrics,
        Path(args.out) / "table_bkt.tex",
        caption="BKT baseline results.",
        label="tab:bkt_results"
    )

    LOGGER.info("BKT metrics:")
    LOGGER.info("\n" + metrics.to_string(index=False))

    LOGGER.info("Saved outputs:")
    LOGGER.info(f"- {Path(args.out) / 'bkt_metrics.csv'}")
    LOGGER.info(f"- {Path(args.out) / 'bkt_predictions.csv'}")
    LOGGER.info(f"- {Path(args.out) / 'bkt_prediction_summary.csv'}")
    LOGGER.info(f"- {Path(args.out) / 'table_bkt.tex'}")
    LOGGER.info(f"- {Path(args.out) / 'bkt.log'}")


if __name__ == "__main__":
    main()
