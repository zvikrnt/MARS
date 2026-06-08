# Data

The full experiment expects:

```text
data/fixedelo_bot_synthetic_dataset.csv
```

The dataset can be generated with:

```bash
python src/generate_fixedelo_dataset.py
```

Generation requires the source question bank:

```text
data/tmp_master_merged.json
```

or:

```text
tmp_master_merged.json
```

The paper experiments use 300 learners, 6000 test sessions, 1091 questions, 277500 interactions, and 10 personas.

If the full dataset is not included due to size, place it in `data/` and run:

```bash
python scripts/make_sample_dataset.py
```

This creates `data/sample/small_fixedelo_sample.csv` for smoke testing.
