#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/fixedelo_bot_synthetic_dataset.csv")
    parser.add_argument("--out", default="data/sample/small_fixedelo_sample.csv")
    parser.add_argument("--max_sessions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            "Full dataset not found. Place fixedelo_bot_synthetic_dataset.csv in data/ first."
        )

    df = pd.read_csv(data_path, low_memory=False)
    rng = np.random.default_rng(args.seed)
    if "test_id" in df.columns:
        ids = np.array(sorted(df["test_id"].astype(str).unique()), dtype=object)
        keep = rng.choice(ids, size=min(args.max_sessions, len(ids)), replace=False)
        sample = df[df["test_id"].astype(str).isin(keep)].copy()
    elif "student_id" in df.columns:
        ids = np.array(sorted(df["student_id"].unique()), dtype=object)
        keep = rng.choice(ids, size=min(args.max_sessions, len(ids)), replace=False)
        sample = df[df["student_id"].isin(keep)].copy()
    else:
        sample = df.head(args.max_sessions).copy()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(sample)} rows")


if __name__ == "__main__":
    main()
