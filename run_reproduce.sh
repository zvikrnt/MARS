#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/reproduce

if [ ! -f data/fixedelo_bot_synthetic_dataset.csv ]; then
  echo "Full dataset not found. Place fixedelo_bot_synthetic_dataset.csv in data/ or generate it using src/generate_fixedelo_dataset.py after adding the source question bank."
  exit 1
fi

python src/time_budget_kt_comparison.py \
  --data data/fixedelo_bot_synthetic_dataset.csv \
  --out results/reproduce \
  --mode final \
  --negative_marking \
  --negative_penalty_ratio 0.25 \
  --candidate_pool logged_session \
  --max_students 100 \
  --max_sessions 300 \
  --max_questions 50 \
  --n_runs 3 \
  --dkt_epochs 10 \
  --gkt_epochs 10 \
  --mars_params configs/mars_timebudget_negative_params.json \
  --seed 42

python scripts/make_tables.py --results results/reproduce --out paper_tables

echo "Expected outputs:"
echo "  results/reproduce/policy_summary.csv"
echo "  results/reproduce/statistical_tests.csv"
echo "  results/reproduce/model_prediction_metrics.csv"
echo "  paper_tables/table_timebudget_negative.tex"
echo "  paper_tables/statistical_summary.txt"
