# Reproducibility

The synthetic assessment dataset is generated with Fixed Elo routing using `K=32` and seed `20260602`.

The time-budget comparison uses seed `42`, splits by `test_id`, and evaluates all policies with the same candidate pool and routing objective. Only the probability estimator differs across KT baselines.

Negative marking uses:

```text
expected score = p * marks - (1 - p) * 0.25 * marks
```

The realized score is:

```text
correct => marks
wrong   => -0.25 * marks
```

The main paper configuration uses:

- `max_questions=50`
- `test_duration=3600`
- `candidate_pool=logged_session`
- `max_students=100`
- `max_sessions=300`
- `n_runs=3`
- `dkt_epochs=10`
- `gkt_epochs=10`

To reproduce tables:

```bash
bash run_reproduce.sh
python scripts/make_tables.py --results results/reproduce --out paper_tables
```
