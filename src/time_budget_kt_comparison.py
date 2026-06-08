#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fast time-budgeted routing comparison across KT/rating probability models.

This script is intentionally separate from time_budget_experiment.py. It keeps
the same time-budget routing objective for every policy and changes only the
correctness probability estimator:

    mars_timebudget, dkt_timebudget, gkt_timebudget, bkt_timebudget.

The simulation samples correctness from the policy's current probability model;
logged correctness/response time is used for fitting and diagnostics, not as
counterfactual truth for arbitrary selections.
"""

import argparse
import json
import logging
import math
import os
import random
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, median_absolute_error, roc_auc_score
from tqdm import tqdm

try:
    import optuna
except Exception:
    optuna = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except Exception:
    torch = None
    nn = None
    DataLoader = None
    Dataset = object


LOGGER = logging.getLogger("time_budget_kt_comparison")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REQUIRED_COLUMNS = [
    "student_id", "test_id", "day_index", "timestamp", "question_id",
    "subject", "topic", "difficulty", "question_elo", "correct",
    "response_time_sec", "persona", "marks",
]

POLICIES = [
    "mars_timebudget",
    "bkt_timebudget",
    "dkt_timebudget",
    "gkt_timebudget",
]

def paper_policies(args):
    return POLICIES.copy()

DEFAULT_MARS_PARAMS = {
    "w_expected_marks": 1.0,
    "w_marks_per_time": 1.0,
    "w_difficulty_fit": 2.0,
    "w_topic_bonus": 1.0,
    "w_diversity": 1.0,
    "w_time_penalty": 0.01,
    "w_overtime": 5.0,
    "alpha_v": 1.5,
    "alpha_a": 120.0,
    "alpha_sigma": 50.0,
    "delta_max": 120.0,
    "beta_min": 0.02,
    "beta_max": 0.20,
    "v_max": 30.0,
    "k_max": 45.0,
    "k_min": 12.0,
    "n_prov": 30,
    "sigma2_0": 100.0,
    "target_acc": 0.70,
    "weak_topic_threshold": 0.60,
    "recent_topic_window": 5,
    "risk_aversion": 0.0,
}


def setup_logger(out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    for handler in list(LOGGER.handlers):
        LOGGER.removeHandler(handler)
        handler.close()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)
    file_handler = logging.FileHandler(Path(out_dir) / "time_budget_kt_comparison.log")
    file_handler.setFormatter(fmt)
    LOGGER.addHandler(file_handler)


class Timer:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.start = time.time()
        LOGGER.info("START: %s", self.message)
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.time() - self.start
        if exc_type is None:
            LOGGER.info("DONE: %s | Time: %.2f sec", self.message, elapsed)
        else:
            LOGGER.error("FAILED: %s | Time: %.2f sec", self.message, elapsed)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def elo_prob(rating, question_elo):
    return 1.0 / (1.0 + 10.0 ** ((float(question_elo) - float(rating)) / 400.0))


def compute_negative_penalty(marks, args):
    if not getattr(args, "negative_marking", False):
        return 0.0
    penalty = float(args.negative_penalty_ratio) * float(marks)
    penalty = max(penalty, float(args.min_negative_penalty))
    if args.max_negative_penalty is not None:
        penalty = min(penalty, float(args.max_negative_penalty))
    return float(penalty)


def expected_score(marks, p_correct, args):
    marks = np.asarray(marks, dtype=float)
    p_correct = np.asarray(p_correct, dtype=float)
    if not getattr(args, "negative_marking", False):
        return p_correct * marks
    penalty = np.vectorize(lambda m: compute_negative_penalty(m, args))(marks)
    return p_correct * marks - (1.0 - p_correct) * penalty


def realized_score(marks, correct, args):
    if int(correct):
        return float(marks)
    return -compute_negative_penalty(marks, args) if getattr(args, "negative_marking", False) else 0.0


def mode_or_first(series):
    series = series.dropna()
    if len(series) == 0:
        return np.nan
    modes = series.mode()
    return modes.iloc[0] if len(modes) else series.iloc[0]


def read_csv_robust(path):
    for enc in ["utf-8", "utf-8-sig", "latin1", "ISO-8859-1", "cp1252"]:
        try:
            LOGGER.info("Reading %s with encoding=%s", path, enc)
            return pd.read_csv(path, low_memory=False, encoding=enc, encoding_errors="replace", on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False, encoding="latin1", encoding_errors="replace", on_bad_lines="skip")


def clean_data(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.copy()
    for col in ["student_id", "day_index", "timestamp", "question_elo", "correct", "response_time_sec", "marks", "R_before"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS).copy()
    df = df[df["correct"].isin([0, 1])]
    df = df[(df["response_time_sec"] > 0) & (df["marks"] > 0)]
    LOGGER.info("Cleaned rows: %d -> %d", before, len(df))
    df["student_id"] = df["student_id"].astype(int)
    df["correct"] = df["correct"].astype(int)
    for col in ["test_id", "question_id", "subject", "topic", "difficulty", "persona"]:
        df[col] = df[col].astype(str)
    return df.sort_values(["student_id", "test_id", "timestamp"]).reset_index(drop=True)


def sample_for_speed(df, args):
    rng = np.random.default_rng(args.seed)
    out = df
    if args.max_students and out["student_id"].nunique() > args.max_students:
        keep = rng.choice(out["student_id"].unique(), size=args.max_students, replace=False)
        out = out[out["student_id"].isin(keep)].copy()
    if args.max_tests and out["test_id"].nunique() > args.max_tests:
        keep = rng.choice(out["test_id"].unique(), size=args.max_tests, replace=False)
        out = out[out["test_id"].isin(keep)].copy()
    LOGGER.info("Speed sample shape: %s | students=%d tests=%d", out.shape, out["student_id"].nunique(), out["test_id"].nunique())
    return out.reset_index(drop=True)


def split_by_test_id(df, seed):
    rng = np.random.default_rng(seed)
    test_ids = np.array(sorted(df["test_id"].unique()), dtype=object)
    rng.shuffle(test_ids)
    n = len(test_ids)
    n_train = max(1, int(round(0.70 * n)))
    n_val = max(1, int(round(0.15 * n))) if n >= 3 else 0
    train_ids = test_ids[:n_train]
    val_ids = test_ids[n_train:n_train + n_val]
    final_ids = test_ids[n_train + n_val:]
    if len(final_ids) == 0:
        final_ids = test_ids[-1:]
        train_ids = test_ids[:-1]
    return (
        df[df["test_id"].isin(train_ids)].copy(),
        df[df["test_id"].isin(val_ids)].copy(),
        df[df["test_id"].isin(final_ids)].copy(),
        train_ids, val_ids, final_ids,
    )


def save_dataset_summary(df, train, val, final, out_dir):
    row = {
        "rows": len(df),
        "students": df["student_id"].nunique(),
        "tests": df["test_id"].nunique(),
        "questions": df["question_id"].nunique(),
        "topics": df["topic"].nunique(),
        "mean_accuracy": df["correct"].mean(),
        "mean_response_time_sec": df["response_time_sec"].mean(),
        "mean_marks": df["marks"].mean(),
        "train_rows": len(train),
        "validation_rows": len(val),
        "final_rows": len(final),
        "train_tests": train["test_id"].nunique(),
        "validation_tests": val["test_id"].nunique(),
        "final_tests": final["test_id"].nunique(),
    }
    pd.DataFrame([row]).to_csv(Path(out_dir) / "dataset_summary.csv", index=False)


def build_question_bank(df):
    bank = (
        df.groupby("question_id")
        .agg(
            subject=("subject", mode_or_first),
            topic=("topic", mode_or_first),
            difficulty=("difficulty", mode_or_first),
            question_elo=("question_elo", "median"),
            marks=("marks", "median"),
            median_response_time_sec=("response_time_sec", "median"),
            empirical_accuracy=("correct", "mean"),
            n_attempts=("correct", "size"),
        )
        .reset_index()
    )
    return bank


def build_median_table(train, keys, min_count=5):
    table = train.groupby(keys)["response_time_sec"].agg(["median", "count"]).reset_index()
    table = table[table["count"] >= min_count]
    return {tuple(row[k] for k in keys): float(row["median"]) for _, row in table.iterrows()}


def build_time_model(train, min_count=5):
    return {
        "question_id": build_median_table(train, ["question_id"], min_count),
        "persona_subject_difficulty": build_median_table(train, ["persona", "subject", "difficulty"], min_count),
        "persona_difficulty": build_median_table(train, ["persona", "difficulty"], min_count),
        "topic_difficulty": build_median_table(train, ["topic", "difficulty"], min_count),
        "global": float(train["response_time_sec"].median()),
    }


def predict_time_rows(df, model):
    preds = []
    for row in df.itertuples(index=False):
        value = model["question_id"].get((str(row.question_id),))
        if value is None:
            value = model["persona_subject_difficulty"].get((str(row.persona), str(row.subject), str(row.difficulty)))
        if value is None:
            value = model["persona_difficulty"].get((str(row.persona), str(row.difficulty)))
        if value is None:
            value = model["topic_difficulty"].get((str(row.topic), str(row.difficulty)))
        if value is None:
            value = model["global"]
        preds.append(max(1.0, float(value)))
    return np.asarray(preds, dtype=float)


def save_time_metrics(final, time_model, out_dir):
    pred = predict_time_rows(final, time_model)
    y = final["response_time_sec"].to_numpy(dtype=float)
    row = {
        "Model": "GroupMedianTime",
        "MAE": mean_absolute_error(y, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MedianAbsError": median_absolute_error(y, pred),
        "N": len(final),
    }
    pd.DataFrame([row]).to_csv(Path(out_dir) / "time_predictor_metrics.csv", index=False)


def classification_metrics(name, y, p):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    auc = np.nan
    if len(np.unique(y)) > 1:
        try:
            auc = roc_auc_score(y, p)
        except Exception:
            auc = np.nan
    return {
        "Model": name,
        "AUC": auc,
        "Accuracy": accuracy_score(y, p >= 0.5),
        "LogLoss": log_loss(y, p, labels=[0, 1]),
        "Brier": brier_score_loss(y, p),
        "N": len(y),
    }


def concept_mapping(df, granularity):
    col = "topic" if granularity == "topic" else "question_id"
    values = sorted(df[col].astype(str).unique())
    return col, {v: i for i, v in enumerate(values)}


class SequenceDataset(Dataset):
    def __init__(self, df, concept_col, concept_to_id, max_seq_len):
        self.samples = []
        unk = 0
        for _, group in df.sort_values(["student_id", "timestamp"]).groupby("student_id"):
            concepts = [concept_to_id.get(str(v), unk) for v in group[concept_col].astype(str)]
            correct = group["correct"].astype(int).tolist()
            for start in range(0, max(0, len(concepts) - 1), max_seq_len - 1):
                c = concepts[start:start + max_seq_len]
                r = correct[start:start + max_seq_len]
                if len(c) >= 2:
                    x = [c[i] + len(concept_to_id) * r[i] for i in range(len(c) - 1)]
                    y_concept = c[1:]
                    y = r[1:]
                    self.samples.append((np.asarray(x), np.asarray(y_concept), np.asarray(y, dtype=np.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_sequences(batch):
    max_len = max(len(item[0]) for item in batch)
    x = np.zeros((len(batch), max_len), dtype=np.int64)
    c = np.zeros((len(batch), max_len), dtype=np.int64)
    y = np.zeros((len(batch), max_len), dtype=np.float32)
    mask = np.zeros((len(batch), max_len), dtype=np.float32)
    for i, (xi, ci, yi) in enumerate(batch):
        n = len(xi)
        x[i, :n] = xi
        c[i, :n] = ci
        y[i, :n] = yi
        mask[i, :n] = 1.0
    return torch.tensor(x), torch.tensor(c), torch.tensor(y), torch.tensor(mask)


class DKTNet(nn.Module):
    def __init__(self, num_concepts, emb_dim=64, hidden_dim=96):
        super().__init__()
        self.num_concepts = num_concepts
        self.embedding = nn.Embedding(num_concepts * 2, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, num_concepts)

    def forward(self, x):
        h, _ = self.lstm(self.embedding(x))
        return self.out(h)

    def predict_from_history(self, history_tokens, candidate_ids, priors, device):
        if len(history_tokens) == 0:
            return np.asarray([priors.get(int(cid), 0.5) for cid in candidate_ids], dtype=float)
        self.eval()
        with torch.no_grad():
            x = torch.tensor([history_tokens], dtype=torch.long, device=device)
            logits = self.forward(x)[0, -1]
            probs = torch.sigmoid(logits[torch.tensor(candidate_ids, dtype=torch.long, device=device)])
        return probs.detach().cpu().numpy().astype(float)


class GKTNet(nn.Module):
    def __init__(self, num_concepts, adjacency, emb_dim=48, hidden_dim=48):
        super().__init__()
        self.num_concepts = num_concepts
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(num_concepts * 2, emb_dim)
        self.gru = nn.GRUCell(emb_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.register_buffer("adjacency", torch.tensor(adjacency, dtype=torch.float32))

    def forward_sequence(self, x, concepts):
        batch, length = x.shape
        states = torch.zeros(batch, self.num_concepts, self.hidden_dim, device=x.device)
        logits = []
        for t in range(length):
            obs_c = concepts[:, t]
            current = states[torch.arange(batch, device=x.device), obs_c]
            updated = self.gru(self.embedding(x[:, t]), current)
            states = states.clone()
            states[torch.arange(batch, device=x.device), obs_c] = updated
            states = 0.90 * states + 0.10 * torch.einsum("ij,bjh->bih", self.adjacency, states)
            logits.append(self.out(states).squeeze(-1))
        return torch.stack(logits, dim=1)

    def predict_from_history(self, history, candidate_ids, priors, device):
        if len(history) == 0:
            return np.asarray([priors.get(int(cid), 0.5) for cid in candidate_ids], dtype=float)
        self.eval()
        states = torch.zeros(1, self.num_concepts, self.hidden_dim, device=device)
        with torch.no_grad():
            for concept, correct in history:
                token = int(concept) + self.num_concepts * int(correct)
                current = states[0, int(concept)]
                updated = self.gru(self.embedding(torch.tensor([token], device=device))[0], current)
                states[0, int(concept)] = updated
                states = 0.90 * states + 0.10 * torch.einsum("ij,bjh->bih", self.adjacency, states)
            logits = self.out(states).squeeze(0).squeeze(-1)
            probs = torch.sigmoid(logits[torch.tensor(candidate_ids, dtype=torch.long, device=device)])
        return probs.detach().cpu().numpy().astype(float)


def train_dkt(train, final, args, out_dir):
    if torch is None:
        LOGGER.warning("PyTorch is unavailable; DKT will use empirical concept priors.")
        return None, *concept_mapping(train, args.dkt_granularity), {}, "cpu"
    concept_col, concept_to_id = concept_mapping(train, args.dkt_granularity)
    num_concepts = len(concept_to_id)
    priors = train.groupby(concept_col)["correct"].mean().to_dict()
    priors = {concept_to_id[str(k)]: float(v) for k, v in priors.items() if str(k) in concept_to_id}
    dataset = SequenceDataset(train, concept_col, concept_to_id, args.max_seq_len)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DKTNet(num_concepts).to(device)
    if len(dataset) == 0:
        return model, concept_col, concept_to_id, priors, device
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_sequences)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    model.train()
    for epoch in range(args.dkt_epochs):
        losses = []
        for x, c, y, mask in tqdm(loader, desc=f"DKT epoch {epoch + 1}/{args.dkt_epochs}", leave=False):
            x, c, y, mask = x.to(device), c.to(device), y.to(device), mask.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(x).gather(2, c.unsqueeze(-1)).squeeze(-1)
                loss = (loss_fn(logits, y) * mask).sum() / mask.sum().clamp_min(1.0)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        LOGGER.info("DKT epoch %d loss %.4f", epoch + 1, float(np.mean(losses)) if losses else np.nan)
    preds, ys = logged_dkt_predictions(model, final, concept_col, concept_to_id, priors, device)
    pd.DataFrame([classification_metrics("DKT-TimeBudget", ys, preds)]).to_csv(Path(out_dir) / "dkt_metrics.csv", index=False)
    return model, concept_col, concept_to_id, priors, device


def logged_dkt_predictions(model, df, concept_col, concept_to_id, priors, device):
    preds, ys = [], []
    unk = 0
    for _, group in df.sort_values(["student_id", "timestamp"]).groupby("student_id"):
        history = []
        for row in group.itertuples(index=False):
            cid = concept_to_id.get(str(getattr(row, concept_col)), unk)
            p = model.predict_from_history(history, [cid], priors, device)[0] if model is not None else priors.get(cid, 0.5)
            preds.append(p)
            ys.append(int(row.correct))
            history.append(cid + len(concept_to_id) * int(row.correct))
    return np.asarray(preds), np.asarray(ys)


def topic_transition_adjacency(train, topic_to_id):
    n = len(topic_to_id)
    mat = np.eye(n, dtype=float) * 0.1
    for _, group in train.sort_values(["student_id", "timestamp"]).groupby("student_id"):
        topics = [topic_to_id.get(str(t)) for t in group["topic"].astype(str)]
        for a, b in zip(topics[:-1], topics[1:]):
            if a is not None and b is not None:
                mat[a, b] += 1.0
    row_sum = mat.sum(axis=1, keepdims=True)
    return mat / np.maximum(row_sum, 1.0)


def train_gkt(train, final, args, out_dir):
    concept_col, topic_to_id = concept_mapping(train, "topic")
    priors = train.groupby("topic")["correct"].mean().to_dict()
    priors = {topic_to_id[str(k)]: float(v) for k, v in priors.items() if str(k) in topic_to_id}
    if args.skip_gkt or torch is None:
        LOGGER.warning("Skipping neural GKT; using graph-smoothed topic priors for gkt_timebudget.")
        pd.DataFrame([classification_metrics("GKT-TimeBudget", final["correct"], [priors.get(topic_to_id.get(str(t), 0), 0.5) for t in final["topic"]])]).to_csv(Path(out_dir) / "gkt_metrics.csv", index=False)
        return None, concept_col, topic_to_id, priors, "cpu"
    adjacency = topic_transition_adjacency(train, topic_to_id)
    dataset = SequenceDataset(train, "topic", topic_to_id, args.max_seq_len)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GKTNet(len(topic_to_id), adjacency).to(device)
    if len(dataset):
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_sequences)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        model.train()
        for epoch in range(args.gkt_epochs):
            losses = []
            for x, c, y, mask in tqdm(loader, desc=f"GKT epoch {epoch + 1}/{args.gkt_epochs}", leave=False):
                x, c, y, mask = x.to(device), c.to(device), y.to(device), mask.to(device)
                opt.zero_grad()
                logits = model.forward_sequence(x, c).gather(2, c.unsqueeze(-1)).squeeze(-1)
                loss = (loss_fn(logits, y) * mask).sum() / mask.sum().clamp_min(1.0)
                loss.backward()
                opt.step()
                losses.append(float(loss.detach().cpu()))
            LOGGER.info("GKT epoch %d loss %.4f", epoch + 1, float(np.mean(losses)) if losses else np.nan)
    preds, ys = logged_gkt_predictions(model, final, topic_to_id, priors, device)
    pd.DataFrame([classification_metrics("GKT-TimeBudget", ys, preds)]).to_csv(Path(out_dir) / "gkt_metrics.csv", index=False)
    return model, concept_col, topic_to_id, priors, device


def logged_gkt_predictions(model, df, topic_to_id, priors, device):
    preds, ys = [], []
    for _, group in df.sort_values(["student_id", "timestamp"]).groupby("student_id"):
        history = []
        for row in group.itertuples(index=False):
            cid = topic_to_id.get(str(row.topic), 0)
            p = model.predict_from_history(history, [cid], priors, device)[0] if model is not None else priors.get(cid, 0.5)
            preds.append(p)
            ys.append(int(row.correct))
            history.append((cid, int(row.correct)))
    return np.asarray(preds), np.asarray(ys)


def fit_bkt(train, final, out_dir):
    params = {}
    global_acc = float(train["correct"].mean())
    for topic, group in train.groupby("topic"):
        acc = float(group["correct"].mean())
        params[str(topic)] = {
            "p_init": float(np.clip(acc, 0.05, 0.95)),
            "p_learn": 0.08,
            "p_guess": float(np.clip(acc * 0.45, 0.05, 0.35)),
            "p_slip": float(np.clip((1.0 - acc) * 0.45, 0.05, 0.35)),
        }
    default = {"p_init": global_acc, "p_learn": 0.08, "p_guess": 0.2, "p_slip": 0.1}
    preds, ys = [], []
    for _, group in final.sort_values(["student_id", "timestamp"]).groupby("student_id"):
        mastery = {}
        for row in group.itertuples(index=False):
            par = params.get(str(row.topic), default)
            m = mastery.get(str(row.topic), par["p_init"])
            p = m * (1 - par["p_slip"]) + (1 - m) * par["p_guess"]
            preds.append(p)
            ys.append(int(row.correct))
            mastery[str(row.topic)] = bkt_update(m, int(row.correct), par)
    pd.DataFrame([classification_metrics("BKT-TimeBudget", ys, preds)]).to_csv(Path(out_dir) / "bkt_metrics.csv", index=False)
    return params, default


def bkt_update(mastery, correct, par):
    p_correct = mastery * (1 - par["p_slip"]) + (1 - mastery) * par["p_guess"]
    if correct:
        posterior = mastery * (1 - par["p_slip"]) / max(p_correct, 1e-6)
    else:
        posterior = mastery * par["p_slip"] / max(1 - p_correct, 1e-6)
    return float(posterior + (1 - posterior) * par["p_learn"])


def load_mars_params(path):
    params = DEFAULT_MARS_PARAMS.copy()
    if path and Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            params.update(json.load(f))
    else:
        LOGGER.warning("MARS params file missing (%s); using defaults. Pass --mars_params to override.", path)
    return params


def default_params_path(args):
    if args.mars_params:
        return args.mars_params
    filename = "best_mars_timebudget_negative_params.json" if args.negative_marking else "best_mars_timebudget_params.json"
    local_path = Path(args.out) / filename
    if local_path.exists():
        return str(local_path)
    legacy = Path("results/time_budget_v1") / filename
    return str(legacy)


def new_mars_state(initial_rating=1500.0):
    return {
        "rating": float(initial_rating),
        "velocity": 0.0,
        "n_attempts": 0,
        "streak": 0,
        "rolling_updates": deque(maxlen=30),
        "correct_sum": 0.0,
        "topic_correct": defaultdict(float),
        "topic_total": defaultdict(float),
        "recent_topics": deque(maxlen=20),
        "answered": set(),
    }


def mars_sigma2(state):
    if len(state["rolling_updates"]) < 2:
        return 0.0
    return float(np.var(np.asarray(state["rolling_updates"], dtype=float)))


def mars_target_rating(state, params):
    sigma2 = mars_sigma2(state)
    sigma2_0 = float(params.get("sigma2_0", 100.0))
    acc = state["correct_sum"] / state["n_attempts"] if state["n_attempts"] else float(params.get("target_acc", 0.70))
    offset = (
        float(params.get("alpha_v", 1.5)) * np.clip(state["velocity"], -float(params.get("v_max", 30.0)), float(params.get("v_max", 30.0)))
        + float(params.get("alpha_a", 120.0)) * (acc - float(params.get("target_acc", 0.70)))
        - float(params.get("alpha_sigma", 50.0)) * (sigma2 / (sigma2 + sigma2_0))
    )
    return state["rating"] + float(np.clip(offset, -float(params.get("delta_max", 120.0)), float(params.get("delta_max", 120.0))))


def mars_update(state, row, correct, params):
    expected = elo_prob(state["rating"], row.question_elo)
    k = float(params.get("k_max", 45.0)) if state["n_attempts"] < int(params.get("n_prov", 30)) else float(params.get("k_min", 12.0))
    quality = 1.05 if state["streak"] >= 3 else 0.95 if state["streak"] <= -3 else 1.0
    delta_raw = k * quality * (int(correct) - expected)
    beta = float(params.get("beta_min", 0.02)) if state["n_attempts"] < int(params.get("n_prov", 30)) else float(params.get("beta_max", 0.20))
    state["velocity"] = beta * state["velocity"] + (1.0 - beta) * delta_raw
    update = float(np.clip(state["velocity"], -float(params.get("v_max", 30.0)), float(params.get("v_max", 30.0))))
    state["rating"] += update
    state["n_attempts"] += 1
    state["correct_sum"] += int(correct)
    state["streak"] = state["streak"] + 1 if correct else state["streak"] - 1
    if correct and state["streak"] < 0:
        state["streak"] = 1
    if not correct and state["streak"] > 0:
        state["streak"] = -1
    state["rolling_updates"].append(update)
    topic = str(row.topic)
    state["topic_total"][topic] += 1.0
    state["topic_correct"][topic] += int(correct)
    state["recent_topics"].append(topic)
    state["answered"].add(str(row.question_id))
    return update


def score_candidates(candidates, probs, pred_times, state, used_time, test_duration, params, args):
    scored = candidates.copy()
    scored["predicted_probability"] = np.clip(probs, 1e-4, 1 - 1e-4)
    scored["predicted_time_sec"] = np.clip(pred_times, 1.0, None)
    scored["expected_marks"] = scored["marks"] * scored["predicted_probability"]
    scored["negative_penalty"] = scored["marks"].map(lambda m: compute_negative_penalty(m, args))
    scored["expected_score"] = expected_score(scored["marks"], scored["predicted_probability"], args)
    target = mars_target_rating(state, params) if state is not None else 1500.0
    scored["difficulty_fit"] = 1.0 - np.minimum(np.abs(scored["question_elo"] - target) / 1200.0, 1.0)
    recent_topics = set(list(state["recent_topics"])[-int(params.get("recent_topic_window", 5)):]) if state is not None else set()
    topic_bonus, diversity = [], []
    for topic in scored["topic"].astype(str):
        if state is None:
            topic_bonus.append(0.0)
            diversity.append(1.0)
            continue
        total = state["topic_total"].get(topic, 0.0)
        acc = state["topic_correct"].get(topic, 0.0) / total if total else float(params.get("target_acc", 0.70))
        topic_bonus.append(max(0.0, float(params.get("weak_topic_threshold", 0.60)) - acc))
        diversity.append(0.0 if topic in recent_topics else 1.0)
    scored["topic_bonus"] = topic_bonus
    scored["diversity_bonus"] = diversity
    scored["overtime_risk"] = np.maximum(0.0, used_time + scored["predicted_time_sec"] - test_duration)
    scored["score"] = (
        float(params["w_expected_marks"]) * scored["expected_score"]
        + float(params["w_marks_per_time"]) * (scored["expected_score"] / scored["predicted_time_sec"].clip(lower=1.0))
        + float(params["w_difficulty_fit"]) * scored["difficulty_fit"]
        + float(params["w_topic_bonus"]) * scored["topic_bonus"]
        + float(params["w_diversity"]) * scored["diversity_bonus"]
        - float(params["w_time_penalty"]) * scored["predicted_time_sec"]
        - float(params["w_overtime"]) * scored["overtime_risk"]
        - float(params.get("risk_aversion", 0.0)) * (1.0 - scored["predicted_probability"]) * scored["negative_penalty"]
    )
    return scored


def make_candidate_pool(session_df, bank, candidate_pool, answered):
    if candidate_pool == "full_bank":
        candidates = bank.copy()
    else:
        candidates = build_question_bank(session_df)
    if answered:
        candidates = candidates[~candidates["question_id"].astype(str).isin(answered)].copy()
    return candidates


def final_sessions(final, args):
    sessions = list(final.groupby(["student_id", "test_id"], sort=False))
    if args.max_sessions and len(sessions) > args.max_sessions:
        rng = np.random.default_rng(args.seed)
        keep = set(rng.choice(np.arange(len(sessions)), size=args.max_sessions, replace=False).tolist())
        sessions = [s for i, s in enumerate(sessions) if i in keep]
    return sessions


def model_probs(policy, candidates, states, model_bundle):
    if policy == "mars_timebudget":
        st = states[policy]
        return np.asarray([elo_prob(st["rating"], elo) for elo in candidates["question_elo"]], dtype=float)
    if policy == "bkt_timebudget":
        mastery = states[policy]["mastery"]
        params, default = model_bundle["bkt"]
        out = []
        for topic in candidates["topic"].astype(str):
            par = params.get(topic, default)
            m = mastery.get(topic, par["p_init"])
            out.append(m * (1 - par["p_slip"]) + (1 - m) * par["p_guess"])
        return np.asarray(out, dtype=float)
    if policy == "dkt_timebudget":
        model, concept_col, concept_to_id, priors, device = model_bundle["dkt"]
        ids = [concept_to_id.get(str(v), 0) for v in candidates[concept_col].astype(str)]
        history = states[policy]["history"]
        if model is None:
            return np.asarray([priors.get(i, 0.5) for i in ids], dtype=float)
        return model.predict_from_history(history, ids, priors, device)
    if policy == "gkt_timebudget":
        model, _, topic_to_id, priors, device = model_bundle["gkt"]
        ids = [topic_to_id.get(str(v), 0) for v in candidates["topic"].astype(str)]
        history = states[policy]["history"]
        if model is None:
            return np.asarray([priors.get(i, 0.5) for i in ids], dtype=float)
        return model.predict_from_history(history, ids, priors, device)
    raise ValueError(policy)


def update_policy_state(policy, state, selected, correct, params, model_bundle):
    if policy == "mars_timebudget":
        before = state["rating"]
        mars_update(state, selected, correct, params)
        return before, state["rating"]
    if policy == "bkt_timebudget":
        before = np.nan
        bkt_params, default = model_bundle["bkt"]
        topic = str(selected.topic)
        par = bkt_params.get(topic, default)
        state["mastery"][topic] = bkt_update(state["mastery"].get(topic, par["p_init"]), correct, par)
        state["answered"].add(str(selected.question_id))
        return before, np.nan
    if policy == "dkt_timebudget":
        _, concept_col, concept_to_id, _, _ = model_bundle["dkt"]
        cid = concept_to_id.get(str(getattr(selected, concept_col)), 0)
        state["history"].append(cid + len(concept_to_id) * int(correct))
        state["answered"].add(str(selected.question_id))
        return np.nan, np.nan
    if policy == "gkt_timebudget":
        _, _, topic_to_id, _, _ = model_bundle["gkt"]
        cid = topic_to_id.get(str(selected.topic), 0)
        state["history"].append((cid, int(correct)))
        state["answered"].add(str(selected.question_id))
        return np.nan, np.nan
    return np.nan, np.nan


def init_states(initial_rating):
    return {
        "mars_timebudget": new_mars_state(initial_rating),
        "bkt_timebudget": {"mastery": {}, "answered": set()},
        "dkt_timebudget": {"history": [], "answered": set()},
        "gkt_timebudget": {"history": [], "answered": set()},
    }


def evaluate_routing(final, bank, time_model, model_bundle, params, args, out_dir, policies=None, write_outputs=True, desc_prefix="routing"):
    rng = np.random.default_rng(args.seed)
    sessions = final_sessions(final, args)
    result_rows, selected_rows = [], []
    policies = policies or POLICIES
    initial_rating = float(final["R_before"].median()) if "R_before" in final.columns and final["R_before"].notna().any() else 1500.0
    bank = bank.copy()
    bank["persona"] = final["persona"].mode().iloc[0] if len(final) else "unknown"
    for run_id in range(args.n_runs):
        for (student_id, test_id), session_df in tqdm(sessions, desc=f"{desc_prefix} run {run_id + 1}/{args.n_runs}"):
            session_df = session_df.sort_values("timestamp")
            persona = str(session_df["persona"].iloc[0])
            for policy in policies:
                states = init_states(initial_rating)
                used_time = 0.0
                total_marks = 0.0
                total_score = 0.0
                expected_marks = 0.0
                expected_score_total = 0.0
                negative_penalty_total = 0.0
                correct_sum = 0
                difficulty_counts = defaultdict(int)
                selected_topics = set()
                for step in range(1, args.max_questions + 1):
                    if used_time >= args.test_duration:
                        break
                    answered = states[policy].get("answered", set())
                    candidates = make_candidate_pool(session_df, bank, args.candidate_pool, answered)
                    if len(candidates) == 0:
                        break
                    candidates = candidates.copy()
                    candidates["persona"] = persona
                    pred_time = predict_time_rows(candidates, time_model)
                    probs = model_probs(policy, candidates, states, model_bundle)
                    mars_state = states["mars_timebudget"] if policy == "mars_timebudget" else new_mars_state(initial_rating)
                    scored = score_candidates(candidates, probs, pred_time, mars_state, used_time, args.test_duration, params, args)
                    selected = scored.loc[scored["score"].idxmax()]
                    if (
                        not args.allow_negative_expected_selection
                        and float(selected["expected_score"]) <= 0.0
                    ):
                        break
                    p = float(selected["predicted_probability"])
                    sampled_correct = int(rng.random() < p)
                    rating_before, rating_after = update_policy_state(policy, states[policy], selected, sampled_correct, params, model_bundle)
                    actual_time = float(selected["predicted_time_sec"])
                    used_time += actual_time
                    earned_marks = float(selected["marks"]) if sampled_correct else 0.0
                    earned_score = realized_score(selected["marks"], sampled_correct, args)
                    negative_penalty = float(selected["negative_penalty"])
                    total_marks += earned_marks
                    total_score += earned_score
                    expected_marks += float(selected["expected_marks"])
                    expected_score_total += float(selected["expected_score"])
                    negative_penalty_total += 0.0 if sampled_correct else negative_penalty
                    correct_sum += sampled_correct
                    difficulty_counts[str(selected["difficulty"]).lower()] += 1
                    selected_topics.add(str(selected["topic"]))
                    selected_rows.append({
                        "policy": policy,
                        "run_id": run_id,
                        "student_id": student_id,
                        "test_id": test_id,
                        "step": step,
                        "question_id": selected["question_id"],
                        "subject": selected["subject"],
                        "topic": selected["topic"],
                        "difficulty": selected["difficulty"],
                        "predicted_probability": p,
                        "predicted_time_sec": actual_time,
                        "sampled_correct": sampled_correct,
                        "marks": selected["marks"],
                        "negative_penalty": negative_penalty,
                        "earned_marks": earned_marks,
                        "earned_score": earned_score,
                        "expected_marks": float(selected["expected_marks"]),
                        "expected_score": float(selected["expected_score"]),
                        "cumulative_marks": total_marks,
                        "cumulative_score": total_score,
                        "cumulative_time_sec": used_time,
                        "rating_before": rating_before,
                        "rating_after": rating_after,
                    })
                result_rows.append({
                    "policy": policy,
                    "run_id": run_id,
                    "student_id": student_id,
                    "test_id": test_id,
                    "persona": persona,
                    "total_marks": total_marks,
                    "total_score": total_score,
                    "expected_marks": expected_marks,
                    "expected_score": expected_score_total,
                    "negative_penalty_total": negative_penalty_total,
                    "score_per_minute": total_score / max(used_time / 60.0, 1e-6),
                    "marks_per_minute": total_marks / max(used_time / 60.0, 1e-6),
                    "total_time_sec": used_time,
                    "overtime_sec": max(0.0, used_time - args.test_duration),
                    "overtime_violation": int(used_time > args.test_duration),
                    "accuracy": correct_sum / max(1, sum(difficulty_counts.values())),
                    "n_questions": sum(difficulty_counts.values()),
                    "unique_topics": len(selected_topics),
                    "easy_count": difficulty_counts.get("easy", 0),
                    "medium_count": difficulty_counts.get("medium", 0),
                    "hard_count": difficulty_counts.get("hard", 0),
                    "negative_marking_enabled": bool(args.negative_marking),
                    "negative_penalty_ratio": float(args.negative_penalty_ratio),
                })
    results = pd.DataFrame(result_rows)
    selected = pd.DataFrame(selected_rows)
    summary = summarize_policy_results(results)
    if write_outputs:
        results.to_csv(Path(out_dir) / "policy_session_results.csv", index=False)
        selected.to_csv(Path(out_dir) / "policy_selected_questions.csv", index=False)
        summary.to_csv(Path(out_dir) / "policy_summary.csv", index=False)
    return results, summary


def summarize_policy_results(results):
    rows = []
    for policy, group in results.groupby("policy"):
        row = {"policy": policy, "n_sessions": len(group)}
        for metric in [
            "total_marks", "total_score", "expected_marks", "expected_score",
            "negative_penalty_total", "marks_per_minute", "score_per_minute",
            "total_time_sec", "overtime_sec", "accuracy", "n_questions", "unique_topics",
        ]:
            vals = group[metric].astype(float)
            row[f"mean_{metric}"] = vals.mean()
            row[f"std_{metric}"] = vals.std(ddof=1)
            row[f"ci95_{metric}"] = 1.96 * vals.std(ddof=1) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
        row["overtime_rate"] = group["overtime_violation"].mean()
        row["negative_marking_enabled"] = bool(group["negative_marking_enabled"].iloc[0]) if "negative_marking_enabled" in group else False
        row["negative_penalty_ratio"] = float(group["negative_penalty_ratio"].iloc[0]) if "negative_penalty_ratio" in group else 0.0
        row["easy_share"] = group["easy_count"].sum() / max(group["n_questions"].sum(), 1)
        row["medium_share"] = group["medium_count"].sum() / max(group["n_questions"].sum(), 1)
        row["hard_share"] = group["hard_count"].sum() / max(group["n_questions"].sum(), 1)
        rows.append(row)
    sort_col = "mean_total_score" if "total_score" in results.columns else "mean_total_marks"
    return pd.DataFrame(rows).sort_values(sort_col, ascending=False)


def save_statistical_tests(results, out_dir, reference="mars_timebudget"):
    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None
    rows = []
    metrics = ["total_score", "score_per_minute", "total_marks", "marks_per_minute", "overtime_violation"]
    ref = results[results["policy"] == reference][["run_id", "student_id", "test_id"] + metrics]
    for policy in sorted(set(results["policy"]) - {reference}):
        other = results[results["policy"] == policy][["run_id", "student_id", "test_id"] + metrics]
        merged = ref.merge(other, on=["run_id", "student_id", "test_id"], suffixes=("_ref", "_other"))
        if len(merged) == 0:
            continue
        for metric in metrics:
            diff = merged[f"{metric}_ref"] - merged[f"{metric}_other"]
            boot = []
            rng = np.random.default_rng(123)
            for _ in range(500):
                boot.append(float(rng.choice(diff, size=len(diff), replace=True).mean()))
            p_value = np.nan
            if wilcoxon is not None and len(diff) > 1 and np.any(np.asarray(diff) != 0):
                try:
                    p_value = wilcoxon(diff).pvalue
                except Exception:
                    p_value = np.nan
            rows.append({
                "reference_policy": reference,
                "comparison_policy": policy,
                "metric": metric,
                "mean_difference": float(diff.mean()),
                "ci95_low": float(np.percentile(boot, 2.5)),
                "ci95_high": float(np.percentile(boot, 97.5)),
                "wilcoxon_p": p_value,
                "cohens_d": float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else np.nan,
                "n_pairs": len(diff),
            })
    pd.DataFrame(rows).to_csv(Path(out_dir) / "statistical_tests.csv", index=False)


def save_tables(summary, out_dir):
    table_dir = Path(out_dir)
    summary_cols = [
        "policy", "n_sessions", "mean_total_score", "ci95_total_score",
        "mean_score_per_minute", "ci95_score_per_minute", "mean_total_marks",
        "ci95_total_marks", "overtime_rate", "mean_total_time_sec",
    ]
    present = [c for c in summary_cols if c in summary.columns]
    summary[present].to_latex(table_dir / "table_policy_summary.tex", index=False, float_format="%.3f")
    tests_path = table_dir / "statistical_tests.csv"
    if tests_path.exists():
        tests = pd.read_csv(tests_path)
        tests.to_latex(table_dir / "table_statistical_tests.tex", index=False, float_format="%.4f")


def save_model_prediction_metrics(out_dir):
    rows = []
    filenames = ["mars_metrics.csv", "bkt_metrics.csv", "dkt_metrics.csv", "gkt_metrics.csv"]
    for filename in filenames:
        path = Path(out_dir) / filename
        if path.exists():
            rows.append(pd.read_csv(path))
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(Path(out_dir) / "model_prediction_metrics.csv", index=False)


def save_bar_plot(summary, value_col, ylabel, path_base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = summary.sort_values(value_col, ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(data["policy"], data[value_col])
    err_col = value_col.replace("mean_", "ci95_")
    if err_col in data:
        ax.errorbar(data["policy"], data[value_col], yerr=data[err_col], fmt="none", color="black", capsize=3)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    fig.savefig(f"{path_base}.png", dpi=200)
    fig.savefig(f"{path_base}.pdf")
    plt.close(fig)


def save_plots(summary, results, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        LOGGER.warning("Skipping plots because matplotlib is unavailable: %s", exc)
        return

    plot_dir = Path(out_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_bar_plot(summary, "mean_total_score", "Mean total score", plot_dir / "policy_total_score_bar")
    save_bar_plot(summary, "mean_score_per_minute", "Mean score per minute", plot_dir / "policy_score_per_minute_bar")
    save_bar_plot(summary, "mean_total_marks", "Mean total marks", plot_dir / "policy_total_marks_bar")

    fig, ax = plt.subplots(figsize=(10, 5))
    data = summary.sort_values("overtime_rate", ascending=False)
    ax.bar(data["policy"], data["overtime_rate"])
    ax.set_ylabel("Overtime rate")
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    fig.savefig(plot_dir / "policy_overtime_rate_bar.png", dpi=200)
    fig.savefig(plot_dir / "policy_overtime_rate_bar.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(summary["mean_total_time_sec"], summary["mean_total_score"])
    for row in summary.itertuples(index=False):
        ax.annotate(row.policy, (row.mean_total_time_sec, row.mean_total_score), fontsize=8)
    ax.set_xlabel("Mean total time sec")
    ax.set_ylabel("Mean total score")
    plt.tight_layout()
    fig.savefig(plot_dir / "score_vs_time_scatter.png", dpi=200)
    fig.savefig(plot_dir / "score_vs_time_scatter.pdf")
    plt.close(fig)

    diff = results.groupby("policy")[["easy_count", "medium_count", "hard_count"]].sum()
    shares = diff.div(diff.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(shares))
    for col in ["easy_count", "medium_count", "hard_count"]:
        ax.bar(shares.index, shares[col], bottom=bottom, label=col.replace("_count", ""))
        bottom += shares[col].to_numpy()
    ax.set_ylabel("Share")
    ax.legend()
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    fig.savefig(plot_dir / "difficulty_distribution_by_policy.png", dpi=200)
    fig.savefig(plot_dir / "difficulty_distribution_by_policy.pdf")
    plt.close(fig)


def sample_mars_params(rng):
    params = DEFAULT_MARS_PARAMS.copy()
    params.update({
        "w_expected_marks": float(rng.uniform(0.1, 8.0)),
        "w_marks_per_time": float(rng.uniform(0.1, 8.0)),
        "w_difficulty_fit": float(rng.uniform(0.0, 3.0)),
        "w_topic_bonus": float(rng.uniform(0.0, 2.0)),
        "w_diversity": float(rng.uniform(0.0, 2.0)),
        "w_time_penalty": float(rng.uniform(0.0, 3.0)),
        "w_overtime": float(rng.uniform(1.0, 30.0)),
        "alpha_v": float(rng.uniform(0.0, 3.0)),
        "alpha_a": float(rng.uniform(0.0, 200.0)),
        "alpha_sigma": float(rng.uniform(0.0, 100.0)),
        "delta_max": float(rng.uniform(50.0, 250.0)),
        "beta_min": float(rng.uniform(0.01, 0.10)),
        "beta_max": float(rng.uniform(0.10, 0.50)),
        "v_max": float(rng.uniform(10.0, 60.0)),
        "risk_aversion": float(rng.uniform(0.0, 5.0)),
    })
    return params


def optuna_mars_params(trial):
    params = DEFAULT_MARS_PARAMS.copy()
    params.update({
        "w_expected_marks": trial.suggest_float("w_expected_marks", 0.1, 8.0),
        "w_marks_per_time": trial.suggest_float("w_marks_per_time", 0.1, 8.0),
        "w_difficulty_fit": trial.suggest_float("w_difficulty_fit", 0.0, 3.0),
        "w_topic_bonus": trial.suggest_float("w_topic_bonus", 0.0, 2.0),
        "w_diversity": trial.suggest_float("w_diversity", 0.0, 2.0),
        "w_time_penalty": trial.suggest_float("w_time_penalty", 0.0, 3.0),
        "w_overtime": trial.suggest_float("w_overtime", 1.0, 30.0),
        "alpha_v": trial.suggest_float("alpha_v", 0.0, 3.0),
        "alpha_a": trial.suggest_float("alpha_a", 0.0, 200.0),
        "alpha_sigma": trial.suggest_float("alpha_sigma", 0.0, 100.0),
        "delta_max": trial.suggest_float("delta_max", 50.0, 250.0),
        "beta_min": trial.suggest_float("beta_min", 0.01, 0.10),
        "beta_max": trial.suggest_float("beta_max", 0.10, 0.50),
        "v_max": trial.suggest_float("v_max", 10.0, 60.0),
        "risk_aversion": trial.suggest_float("risk_aversion", 0.0, 5.0),
    })
    return params


def mars_validation_objective(summary):
    row = summary[summary["policy"] == "mars_timebudget"].iloc[0]
    return float(
        row["mean_total_score"]
        + 0.5 * row["mean_score_per_minute"]
        - 5.0 * row["overtime_rate"]
        - 0.01 * row["mean_overtime_sec"]
    )


def tune_mars_params(val, bank, time_model, args, out_dir):
    rng = np.random.default_rng(args.seed)
    trial_rows = []

    def evaluate_trial(params, trial_id):
        _, summary = evaluate_routing(
            val, bank, time_model, {}, params, args, out_dir,
            policies=["mars_timebudget"], write_outputs=False, desc_prefix=f"tune {trial_id}",
        )
        row = summary.iloc[0].to_dict()
        objective = mars_validation_objective(summary)
        trial_row = {"trial": trial_id, "objective": objective}
        trial_row.update(params)
        for key in ["mean_total_score", "mean_score_per_minute", "overtime_rate", "mean_overtime_sec", "mean_total_marks"]:
            trial_row[key] = row.get(key, np.nan)
        trial_rows.append(trial_row)
        return objective

    if optuna is not None:
        LOGGER.info("Using Optuna for MARS-TimeBudget tuning")

        def objective(trial):
            return evaluate_trial(optuna_mars_params(trial), trial.number)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
        best_params = DEFAULT_MARS_PARAMS.copy()
        best_params.update(study.best_params)
    else:
        LOGGER.info("Optuna not installed; using random search for MARS-TimeBudget tuning")
        best_params, best_obj = None, -np.inf
        for trial_id in range(args.n_trials):
            params = sample_mars_params(rng)
            obj = evaluate_trial(params, trial_id)
            if obj > best_obj:
                best_obj = obj
                best_params = params

    trials = pd.DataFrame(trial_rows).sort_values("objective", ascending=False)
    trials_name = "tuning_trials_negative.csv" if args.negative_marking else "tuning_trials.csv"
    params_name = "best_mars_timebudget_negative_params.json" if args.negative_marking else "best_mars_timebudget_params.json"
    trials.to_csv(Path(out_dir) / trials_name, index=False)
    with open(Path(out_dir) / params_name, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    LOGGER.info("Best tuning objective %.4f saved to %s", float(trials.iloc[0]["objective"]), params_name)
    return best_params, trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/fixedelo_bot_synthetic_dataset.csv")
    parser.add_argument("--out", default="results/time_budget_kt_comparison_v1")
    parser.add_argument("--mode", choices=["final", "tune_mars"], default="final")
    parser.add_argument("--mars_params", default=None)
    parser.add_argument("--candidate_pool", choices=["logged_session", "full_bank"], default="logged_session")
    parser.add_argument("--test_duration", type=float, default=3600.0)
    parser.add_argument("--max_students", type=int, default=50)
    parser.add_argument("--max_tests", type=int, default=200)
    parser.add_argument("--max_sessions", type=int, default=200)
    parser.add_argument("--max_questions", type=int, default=50)
    parser.add_argument("--n_runs", type=int, default=3)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--dkt_epochs", type=int, default=3)
    parser.add_argument("--gkt_epochs", type=int, default=3)
    parser.add_argument("--max_seq_len", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dkt_granularity", choices=["topic", "question"], default="topic")
    parser.add_argument("--skip_gkt", action="store_true")
    parser.add_argument("--negative_marking", action="store_true")
    parser.add_argument("--negative_penalty_ratio", type=float, default=0.25)
    parser.add_argument("--min_negative_penalty", type=float, default=0.0)
    parser.add_argument("--max_negative_penalty", type=float, default=None)
    parser.add_argument("--allow_negative_expected_selection", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logger(args.out)
    set_seed(args.seed)
    out_dir = Path(args.out)
    LOGGER.info(
        "Negative marking enabled=%s | ratio=%.4f | min=%.4f | max=%s",
        args.negative_marking,
        args.negative_penalty_ratio,
        args.min_negative_penalty,
        args.max_negative_penalty,
    )
    LOGGER.info("PyTorch available: %s", torch is not None)
    if torch is not None:
        LOGGER.info("Torch device: %s", "cuda" if torch.cuda.is_available() else "cpu")
    with open(out_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    with Timer("Loading and splitting data"):
        df = clean_data(read_csv_robust(args.data))
        df = sample_for_speed(df, args)
        train, val, final, train_ids, val_ids, final_ids = split_by_test_id(df, args.seed)
        save_dataset_summary(df, train, val, final, out_dir)
        pd.Series(train_ids).to_csv(out_dir / "train_test_ids.csv", index=False, header=["test_id"])
        pd.Series(val_ids).to_csv(out_dir / "validation_test_ids.csv", index=False, header=["test_id"])
        pd.Series(final_ids).to_csv(out_dir / "final_test_ids.csv", index=False, header=["test_id"])
        bank = build_question_bank(train)
        bank.to_csv(out_dir / "question_bank.csv", index=False)

    with Timer("Building group-median time predictor"):
        time_model = build_time_model(train)
        save_time_metrics(final, time_model, out_dir)

    if args.mode == "tune_mars":
        with Timer("Tuning MARS-TimeBudget on validation split"):
            tune_mars_params(val, bank, time_model, args, out_dir)
        LOGGER.info("Tuning complete. Run --mode final to evaluate frozen parameters on final split.")
        return

    params = load_mars_params(default_params_path(args))
    with open(out_dir / "used_mars_timebudget_params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    with Timer("Training/evaluating KT probability models"):
        dkt_bundle = train_dkt(train, final, args, out_dir)
        gkt_bundle = train_gkt(train, final, args, out_dir)
        bkt_bundle = fit_bkt(train, final, out_dir)
        mars_preds = [elo_prob(row.R_before if hasattr(row, "R_before") and not pd.isna(row.R_before) else 1500.0, row.question_elo) for row in final.itertuples(index=False)]
        pd.DataFrame([classification_metrics("MARS-TimeBudget", final["correct"], mars_preds)]).to_csv(out_dir / "mars_metrics.csv", index=False)
        save_model_prediction_metrics(out_dir)
        model_bundle = {"dkt": dkt_bundle, "gkt": gkt_bundle, "bkt": bkt_bundle}

    with Timer("Evaluating common time-budget router"):
        results, summary = evaluate_routing(final, bank, time_model, model_bundle, params, args, out_dir, policies=paper_policies(args))
        save_statistical_tests(results, out_dir)
        save_tables(summary, out_dir)
        save_plots(summary, results, out_dir)

    LOGGER.info("Saved outputs in %s", out_dir)
    LOGGER.info("Policy summary:\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
