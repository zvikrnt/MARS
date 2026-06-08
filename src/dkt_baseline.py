#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DKT Baseline for MARS Paper

Run:

python dkt_baseline.py \
  --data data/assistments2009.csv \
  --out results/dkt_big \
  --batch_size 1024 \
  --epochs 30 \
  --hidden_dim 512 \
  --emb_dim 256 \
  --amp \
  --compile
"""

import argparse
import logging
import random
import re
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def setup_logger(out_dir=None):
    logger = logging.getLogger("DKT")
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
        file_handler = logging.FileHandler(Path(out_dir) / "dkt_training.log")
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
# Utilities
# ---------------------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def export_latex_table(df, path, caption="DKT baseline results.", label="tab:dkt_results"):
    tex = df.to_latex(index=False, escape=False)
    tex = tex.replace(
        "\\begin{tabular}",
        f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\begin{{tabular}}"
    )
    tex = tex.replace("\\end{tabular}", "\\end{tabular}\n\\end{table}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)


# ---------------------------------------------------------------------
# Robust CSV Reader
# ---------------------------------------------------------------------

def read_csv_robust(path, max_rows=None):
    encodings = ["utf-8", "utf-8-sig", "latin1", "ISO-8859-1", "cp1252"]
    last_error = None

    file_size_mb = Path(path).stat().st_size / (1024 * 1024)
    LOGGER.info(f"Reading CSV file: {path}")
    LOGGER.info(f"File size: {file_size_mb:.2f} MB")

    for enc in encodings:
        try:
            LOGGER.info(f"Trying encoding: {enc}")
            df = pd.read_csv(
                path,
                nrows=max_rows,
                low_memory=False,
                encoding=enc,
                encoding_errors="replace",
                on_bad_lines="skip"
            )
            LOGGER.info(f"CSV loaded successfully using encoding: {enc}")
            LOGGER.info(f"Loaded shape: {df.shape}")
            return df
        except UnicodeDecodeError as e:
            last_error = e
            LOGGER.warning(f"Failed with encoding: {enc}")

    raise last_error


# ---------------------------------------------------------------------
# Dataset Normalization
# ---------------------------------------------------------------------

COLUMN_ALIASES = {
    "user_id": [
        "user_id", "user id", "student_id", "student id",
        "anon_student_id", "anon student id", "uid", "user"
    ],
    "item_id": [
        "problem_id", "problem id", "item_id", "item id",
        "question_id", "question id", "assistment_id",
        "assistment id", "assessment_item_id", "problem"
    ],
    "skill_id": [
        "skill_id", "skill id", "skill", "skill_name", "skill name",
        "kc", "kc_default", "kc(default)", "knowledge_component",
        "tags", "concept_id", "concept id"
    ],
    "correct": [
        "correct", "is_correct", "answer_correct",
        "answered_correctly", "correctness"
    ],
    "timestamp": [
        "timestamp", "time", "start_time", "start time",
        "created_at", "order_id", "order id", "order",
        "log_id", "problem_log_id"
    ],
}


def normalize_column_name(col):
    col = str(col).strip().lower()
    col = col.replace("-", "_").replace("/", "_")
    col = re.sub(r"\s+", "_", col)
    col = col.replace("__", "_")
    return col


def find_column(columns, canonical_name):
    normalized = {normalize_column_name(c): c for c in columns}

    for alias in COLUMN_ALIASES[canonical_name]:
        alias_norm = normalize_column_name(alias)
        if alias_norm in normalized:
            return normalized[alias_norm]

    for original in columns:
        n = normalize_column_name(original)

        if canonical_name == "user_id" and ("student" in n or "user" in n):
            return original

        if canonical_name == "item_id" and ("problem" in n or "question" in n or "item" in n):
            return original

        if canonical_name == "skill_id" and ("skill" in n or "kc" in n or "concept" in n):
            return original

    return None


def clean_skill_value(value):
    if pd.isna(value):
        return "unknown"

    value = str(value).strip()

    if not value:
        return "unknown"

    for sep in ["~~", ";", ",", "|"]:
        if sep in value:
            parts = [p.strip() for p in value.split(sep) if p.strip()]
            return parts[0] if parts else "unknown"

    return value


def load_public_dataset(path, max_rows=None):
    raw = read_csv_robust(path, max_rows=max_rows)

    LOGGER.info(f"Available columns: {list(raw.columns)}")

    df = pd.DataFrame()

    for canonical in COLUMN_ALIASES.keys():
        col = find_column(raw.columns, canonical)
        if col is not None:
            df[canonical] = raw[col]
            LOGGER.info(f"Mapped column: {canonical} -> {col}")
        else:
            df[canonical] = np.nan
            LOGGER.warning(f"Missing column: {canonical}")

    required = ["user_id", "skill_id", "correct"]
    missing = [c for c in required if df[c].isna().all()]

    if missing:
        raise ValueError(f"Missing required columns after mapping: {missing}")

    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["skill_id"] = df["skill_id"].apply(clean_skill_value).astype(str)

    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    df = df[df["correct"].isin([0, 1])].copy()
    df["correct"] = df["correct"].astype(int)

    if df["timestamp"].isna().all():
        df["timestamp"] = np.arange(len(df))
    else:
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        if ts.notna().mean() > 0.80:
            df["timestamp"] = ts
        else:
            df["timestamp"] = pd.factorize(df["timestamp"])[0]

    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    df = df[["user_id", "item_id", "skill_id", "correct", "timestamp"]]

    LOGGER.info(f"Normalized shape: {df.shape}")
    LOGGER.info(f"Users: {df['user_id'].nunique()}")
    LOGGER.info(f"Items: {df['item_id'].nunique()}")
    LOGGER.info(f"Skills: {df['skill_id'].nunique()}")

    return df


def filter_min_interactions(df, min_user_interactions=5):
    counts = df.groupby("user_id").size()
    keep_users = counts[counts >= min_user_interactions].index
    df = df[df["user_id"].isin(keep_users)].copy()
    df = df.reset_index(drop=True)
    return df


def chronological_split_by_user(df, test_ratio=0.2):
    train_parts = []
    test_parts = []

    for _, group in tqdm(
        df.groupby("user_id", sort=False),
        total=df["user_id"].nunique(),
        desc="Splitting users"
    ):
        group = group.sort_values("timestamp")
        n = len(group)

        if n < 3:
            train_parts.append(group)
            continue

        n_test = max(1, int(round(n * test_ratio)))
        split = max(1, n - n_test)

        train_parts.append(group.iloc[:split])
        test_parts.append(group.iloc[split:])

    train = pd.concat(train_parts, ignore_index=True)

    if test_parts:
        test = pd.concat(test_parts, ignore_index=True)
    else:
        test = pd.DataFrame(columns=df.columns)

    return train, test


def load_existing_split(train_path, test_path):
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)

    required = ["user_id", "skill_id", "correct", "timestamp"]

    for col in required:
        if col not in train.columns:
            raise ValueError(f"Missing column in train file: {col}")
        if col not in test.columns:
            raise ValueError(f"Missing column in test file: {col}")

    train = train[required].copy()
    test = test[required].copy()

    for df in [train, test]:
        df["user_id"] = df["user_id"].astype(str)
        df["skill_id"] = df["skill_id"].astype(str)
        df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        fallback_ts = pd.Series(np.arange(len(df)), index=df.index)
        df["timestamp"] = df["timestamp"].fillna(fallback_ts)

    train = train[train["correct"].isin([0, 1])].copy()
    test = test[test["correct"].isin([0, 1])].copy()

    train["correct"] = train["correct"].astype(int)
    test["correct"] = test["correct"].astype(int)

    train = train.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    test = test.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    return train, test


# ---------------------------------------------------------------------
# DKT Dataset
# ---------------------------------------------------------------------

class DKTDataset(Dataset):
    def __init__(self, df, skill_to_idx, max_seq_len=100):
        self.samples = []
        self.skill_to_idx = skill_to_idx
        self.max_seq_len = max_seq_len
        self.num_skills = len(skill_to_idx)

        user_groups = list(df.groupby("user_id", sort=False))

        for _, group in tqdm(
            user_groups,
            total=len(user_groups),
            desc="Building DKT sequences"
        ):
            group = group.sort_values("timestamp")

            skills = [
                self.skill_to_idx.get(str(s), -1)
                for s in group["skill_id"].astype(str).tolist()
            ]

            corrects = group["correct"].astype(int).tolist()

            pairs = [(s, c) for s, c in zip(skills, corrects) if s >= 0]

            if len(pairs) < 3:
                continue

            skills, corrects = zip(*pairs)
            skills = list(skills)
            corrects = list(corrects)

            for start in range(0, len(skills) - 1, max_seq_len):
                s_chunk = skills[start:start + max_seq_len + 1]
                c_chunk = corrects[start:start + max_seq_len + 1]

                if len(s_chunk) >= 3:
                    self.samples.append((s_chunk, c_chunk))

        LOGGER.info(f"Created DKT sequences: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        skills, corrects = self.samples[idx]

        x_skills = skills[:-1]
        x_corrects = corrects[:-1]
        y_skills = skills[1:]
        y_corrects = corrects[1:]

        L = min(len(x_skills), self.max_seq_len)

        x = np.zeros(self.max_seq_len, dtype=np.int64)
        q_next = np.zeros(self.max_seq_len, dtype=np.int64)
        y = np.zeros(self.max_seq_len, dtype=np.float32)
        mask = np.zeros(self.max_seq_len, dtype=np.float32)

        # Input token encodes skill + correctness
        # token = skill_id + num_skills * correctness
        x_token = [
            skill + self.num_skills * correct
            for skill, correct in zip(x_skills, x_corrects)
        ]

        x[:L] = x_token[:L]
        q_next[:L] = y_skills[:L]
        y[:L] = y_corrects[:L]
        mask[:L] = 1.0

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(q_next, dtype=torch.long),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
        )


# ---------------------------------------------------------------------
# DKT Model
# ---------------------------------------------------------------------

class DKTModel(nn.Module):
    def __init__(
        self,
        num_skills,
        emb_dim=256,
        hidden_dim=512,
        num_layers=1,
        dropout=0.2
    ):
        super().__init__()

        self.num_skills = num_skills

        self.embedding = nn.Embedding(
            num_embeddings=2 * num_skills,
            embedding_dim=emb_dim
        )

        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)

        self.output = nn.Linear(
            hidden_dim,
            num_skills
        )

    def forward(self, x, q_next):
        emb = self.embedding(x)
        h, _ = self.lstm(emb)
        h = self.dropout(h)

        logits_all = self.output(h)

        logits = logits_all.gather(
            dim=2,
            index=q_next.unsqueeze(-1)
        ).squeeze(-1)

        return logits


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def compute_metrics(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob = np.clip(y_prob, 1e-6, 1.0 - 1e-6)

    y_pred = (y_prob >= 0.5).astype(int)

    if len(np.unique(y_true)) < 2:
        auc = np.nan
    else:
        auc = roc_auc_score(y_true, y_prob)

    return {
        "Model": "DKT-LSTM",
        "AUC": auc,
        "Accuracy": accuracy_score(y_true, y_pred),
        "LogLoss": log_loss(y_true, y_prob, labels=[0, 1]),
        "Brier": brier_score_loss(y_true, y_prob),
        "N": len(y_true),
    }


# ---------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, loss_fn, device, use_amp):
    model.train()

    losses = []

    progress = tqdm(loader, desc="Training", leave=True, dynamic_ncols=True)

    for x, q_next, y, mask in progress:
        x = x.to(device, non_blocking=True)
        q_next = q_next.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
            logits = model(x, q_next)
            loss_raw = loss_fn(logits, y)
            loss = (loss_raw * mask).sum() / mask.sum().clamp_min(1.0)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        progress.set_postfix({"loss": f"{np.mean(losses):.4f}"})

    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, device, return_predictions=False):
    model.eval()

    y_true = []
    y_prob = []

    progress = tqdm(loader, desc="Evaluating", leave=True, dynamic_ncols=True)

    for x, q_next, y, mask in progress:
        x = x.to(device, non_blocking=True)
        q_next = q_next.to(device, non_blocking=True)

        logits = model(x, q_next)
        prob = torch.sigmoid(logits).detach().cpu().numpy()

        y_np = y.numpy()
        mask_np = mask.numpy().astype(bool)

        y_true.extend(y_np[mask_np].tolist())
        y_prob.extend(prob[mask_np].tolist())

    metrics = compute_metrics(y_true, y_prob)

    if not return_predictions:
        return metrics, pd.DataFrame()

    pred_df = pd.DataFrame({
        "model": "DKT-LSTM",
        "correct": np.asarray(y_true, dtype=int),
        "p_correct": np.asarray(y_prob, dtype=float),
    })

    return metrics, pred_df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default=None)
    parser.add_argument("--train", default=None)
    parser.add_argument("--test", default=None)
    parser.add_argument("--out", default="results/dkt")

    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--min_user_interactions", type=int, default=5)
    parser.add_argument("--test_ratio", type=float, default=0.2)

    parser.add_argument("--max_seq_len", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)

    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    global LOGGER
    LOGGER = setup_logger(args.out)

    ensure_dir(args.out)
    set_seed(args.seed)

    LOGGER.info("=" * 80)
    LOGGER.info("Running DKT baseline")
    LOGGER.info("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Using device: {device}")

    if device.type == "cuda":
        LOGGER.info(f"GPU: {torch.cuda.get_device_name(0)}")
        LOGGER.info(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        LOGGER.info(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    if args.train and args.test:
        with Timer("Loading existing train-test split"):
            train_df, test_df = load_existing_split(args.train, args.test)
        LOGGER.info("Using provided train/test files exactly for DKT evaluation split.")
    elif args.data:
        with Timer("Loading and normalizing dataset"):
            df = load_public_dataset(args.data, max_rows=args.max_rows)

        with Timer("Filtering low-interaction users"):
            df = filter_min_interactions(
                df,
                min_user_interactions=args.min_user_interactions
            )

        with Timer("Chronological train-test split"):
            train_df, test_df = chronological_split_by_user(
                df,
                test_ratio=args.test_ratio
            )
    else:
        raise ValueError("Provide either --data raw.csv or both --train and --test split files.")

    LOGGER.info(f"Train interactions: {len(train_df)}")
    LOGGER.info(f"Test interactions: {len(test_df)}")

    skill_values = sorted(train_df["skill_id"].astype(str).unique())
    skill_to_idx = {skill: idx for idx, skill in enumerate(skill_values)}

    train_df = train_df[train_df["skill_id"].astype(str).isin(skill_to_idx)].copy()
    unseen_test = ~test_df["skill_id"].astype(str).isin(skill_to_idx)
    LOGGER.info(f"Removed DKT test rows with skills unseen in train: {int(unseen_test.sum())}")
    test_df = test_df[~unseen_test].copy()

    LOGGER.info(f"Number of skills: {len(skill_to_idx)}")
    train_df.to_csv(Path(args.out) / "dkt_train_used.csv", index=False)
    test_df.to_csv(Path(args.out) / "dkt_test_used.csv", index=False)

    with Timer("Building DKT train dataset"):
        train_dataset = DKTDataset(
            train_df,
            skill_to_idx=skill_to_idx,
            max_seq_len=args.max_seq_len
        )

    with Timer("Building DKT test dataset"):
        test_dataset = DKTDataset(
            test_df,
            skill_to_idx=skill_to_idx,
            max_seq_len=args.max_seq_len
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True if 4 > 0 else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True if 4 > 0 else False
    )

    model = DKTModel(
        num_skills=len(skill_to_idx),
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    if args.compile and hasattr(torch, "compile"):
        LOGGER.info("Compiling model with torch.compile()")
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    best_auc = -1.0
    best_metrics = None
    last_pred_df = pd.DataFrame()

    for epoch in range(1, args.epochs + 1):
        LOGGER.info(f"Epoch {epoch}/{args.epochs}")

        with Timer(f"Training epoch {epoch}"):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                loss_fn,
                device,
                use_amp=args.amp
            )

        with Timer(f"Evaluating epoch {epoch}"):
            metrics, pred_df = evaluate(
                model,
                test_loader,
                device,
                return_predictions=True
            )
            last_pred_df = pred_df

        metrics["Epoch"] = epoch
        metrics["TrainLoss"] = train_loss

        LOGGER.info(f"Epoch {epoch} metrics: {metrics}")

        if not np.isnan(metrics["AUC"]) and metrics["AUC"] > best_auc:
            best_auc = metrics["AUC"]
            best_metrics = metrics.copy()

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "skill_to_idx": skill_to_idx,
                    "args": vars(args),
                    "metrics": best_metrics,
                },
                Path(args.out) / "best_dkt_model.pt"
            )

            LOGGER.info(f"Saved best model at epoch {epoch} with AUC={best_auc:.4f}")
            pred_df.to_csv(Path(args.out) / "dkt_predictions.csv", index=False)

    if best_metrics is None:
        best_metrics = metrics

    if not (Path(args.out) / "dkt_predictions.csv").exists() and len(last_pred_df):
        last_pred_df.to_csv(Path(args.out) / "dkt_predictions.csv", index=False)

    result_df = pd.DataFrame([best_metrics])

    result_df.to_csv(Path(args.out) / "dkt_metrics.csv", index=False)

    export_latex_table(
        result_df,
        Path(args.out) / "table_dkt.tex",
        caption="DKT baseline results.",
        label="tab:dkt_results"
    )

    LOGGER.info("Final best DKT metrics:")
    LOGGER.info("\n" + result_df.to_string(index=False))


if __name__ == "__main__":
    main()
