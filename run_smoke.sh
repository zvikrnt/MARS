#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/smoke

if [ ! -f data/sample/small_fixedelo_sample.csv ]; then
  echo "Sample dataset not found. Run python scripts/make_sample_dataset.py after placing the full dataset in data/."
  exit 1
fi

python src/time_budget_kt_comparison.py \
  --data data/sample/small_fixedelo_sample.csv \
  --out results/smoke \
  --mode final \
  --negative_marking \
  --negative_penalty_ratio 0.25 \
  --candidate_pool logged_session \
  --max_students 20 \
  --max_sessions 50 \
  --max_questions 50 \
  --n_runs 1 \
  --dkt_epochs 1 \
  --gkt_epochs 1 \
  --mars_params configs/mars_timebudget_negative_params.json \
  --seed 42

echo "Smoke outputs:"
echo "  results/smoke/policy_summary.csv"
echo "  results/smoke/statistical_tests.csv"
echo "  results/smoke/model_prediction_metrics.csv"
