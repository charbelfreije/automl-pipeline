# Goal-Aware AutoML Pipeline

A modular AutoML system built from scratch. You describe your goal in plain English (e.g. *"I want to detect heart disease"*); the pipeline infers the task type, auto-detects the target column, searches across preprocessing strategies and model families via cross-validation, evaluates the winner on a held-out test set, and produces plots and a report — then lets you make live predictions from the saved pipeline.

**On the heart-disease benchmark:** evaluated **66 pipelines** (7 model families × 3 preprocessing strategies × hyperparameters) via 5-fold CV. Best model — Logistic Regression — reached **test accuracy 0.82, ROC-AUC 0.86, F1 0.82**.

---

## What makes it "goal-aware"

Instead of hardcoding the target column and task, the pipeline:
1. Takes a natural-language goal string from the user.
2. Scores every column for how well it matches that goal and **auto-selects the target**.
3. **Infers the task type** (classification vs regression) from the target.
4. Chooses and searches the appropriate models and metrics for that task.

---

## Results (heart-disease dataset)

| Metric | Test score |
|---|---|
| Accuracy | 0.82 |
| ROC-AUC | 0.86 |
| Precision | 0.78 |
| Recall | 0.86 |
| F1 | 0.82 |

Best pipeline: `LogisticRegression` + median-impute & scale, `C=1.0` (CV 0.876).
Plots (ROC, feature importance, confusion matrix, precision-recall) are in [`automl_outputs/`](automl_outputs/).

---

## Architecture

The system is split into focused modules:

| Module | Role |
|---|---|
| `main.py` | CLI entry point — training-mode wizard and prediction mode |
| `config.py` | Run configuration, metrics definitions, CLI argument handling |
| `data_io.py` | Data loading, dataset bundling, train/test splitting |
| `analysis.py` | Column-type inference, schema building, column statistics |
| `target_detection.py` | Scores columns against the goal, selects the target |
| `preprocessing.py` | Feature grouping and preprocessing strategy definitions |
| `models.py` | Model family definitions and hyperparameter grids |
| `search.py` | Cross-validated pipeline search across models × strategies |
| `evaluation.py` | Held-out test evaluation |
| `metrics.py` | Metric computation for classification and regression |
| `reporting.py` | Plots and the text run summary |

---

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run — interactive training wizard
python main.py
```

The wizard loads the dataset, asks for your goal, runs the search, and writes the
best pipeline plus plots and a summary to `automl_outputs/`. Choose prediction
mode to load that saved pipeline and classify a single example interactively.

The included `HeartDiseaseTrain-Test.csv` (public UCI Heart Disease dataset) lets
you run it out of the box.

---

## Project structure

```
.
├── main.py, config.py, data_io.py, analysis.py,
│   target_detection.py, preprocessing.py, models.py,
│   search.py, evaluation.py, metrics.py, reporting.py
├── HeartDiseaseTrain-Test.csv     # sample dataset (public)
├── requirements.txt
├── README.md
├── .gitignore
└── automl_outputs/                # plots + run summary (saved models gitignored)
```

## License

MIT — see `LICENSE`.
