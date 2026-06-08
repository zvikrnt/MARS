# MARS Reproducibility Repository

This repository contains anonymized code and configuration files for reproducing the experiments in the paper:

**"MARS: A Momentum-Aware Adaptive Rating System for Personalized Assessment"**

## Repository Contents

- `src/`: experiment code and baselines.
- `configs/`: tuned parameters, experiment configuration, and seeds.
- `scripts/`: sample generation, output checks, and paper table generation.
- `data/`: dataset instructions and optional small sample data.


## Installation

```bash
python -m venv mars_env
source mars_env/bin/activate
pip install -r requirements.txt
```

On Windows:

```bat
mars_env\Scripts\activate
pip install -r requirements.txt
```

## Data

The full reproduction expects:

```text
data/fixedelo_bot_synthetic_dataset.csv
```

The synthetic dataset can be generated with `python src/generate_fixedelo_dataset.py` after adding the source question bank described in `data/README.md`.

For smoke tests, create a small sample after placing the full dataset in `data/`:

```bash
python scripts/make_sample_dataset.py
```

## Quick Smoke Test

```bash
bash run_smoke.sh
```

## Full Reproduction

```bash
bash run_reproduce.sh
```

## Main Paper Configuration

- `N=50`
- `T=3600`
- `negative_penalty_ratio=0.25`
- `candidate_pool=logged_session`
- `seed=42`
- `max_students=100`
- `max_sessions=300`
- `n_runs=3`
- `dkt_epochs=10`
- `gkt_epochs=10`

## Models Compared

Default paper comparison:

- MARS-TimeBudget
- BKT-TimeBudget
- DKT-TimeBudget
- GKT-TimeBudget


## Outputs

Main generated outputs:

- `results/reproduce/policy_summary.csv`
- `results/reproduce/statistical_tests.csv`
- `results/reproduce/policy_selected_questions.csv`
- `results/reproduce/model_prediction_metrics.csv`
- `paper_tables/table_timebudget_negative.tex`
- `results/reproduce/plots/`

## Reproducibility Notes

- Random seeds are fixed in `configs/seeds.json`.
- Splits are by `test_id`.
- All policies use the same candidate pool and routing objective.
- Only the probability estimator differs across KT baselines.

## Double-Blind Anonymity

This repository is anonymized for peer review. Author-identifying metadata, local paths, and institution names have been removed.



## License

MIT License.
