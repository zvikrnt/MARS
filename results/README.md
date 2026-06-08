# Results

`results/reported/` contains compact reported outputs if available:

- `policy_summary.csv`
- `statistical_tests.csv`
- `model_prediction_metrics.csv`
- `experiment_config.json`

Large generated files such as `policy_session_results.csv` and `policy_selected_questions.csv` are produced by the reproduction scripts and may be excluded from version control.

Run:

```bash
bash run_reproduce.sh
```

to generate `results/reproduce/`.
