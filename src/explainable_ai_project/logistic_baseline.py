"""
Training pathway, block 1B, step 13: logistic regression baseline.

A simple, well-understood model to compare the Random Forests against.
If the Random Forest is not clearly better than this, its extra complexity
is not worth it.

For each dataset:
    1. train on the training part
       (features are scaled first: logistic regression needs that, Random Forests do not)
    2. score on the VALIDATION part (the test part stays untouched until step 21)
    3. save the model and its coefficients

Metrics (at the usual 0.5 cut-off for "approved"):
    AUC         how well it ranks approved above rejected (0.5 = guessing, 1 = perfect)
    accuracy    share of correct predictions
    precision   of those predicted "approved", the share that really were
    recall      of those really approved, the share it found
    F1          balance of precision and recall
    Brier       how close the probabilities are to what happened (lower is better)

Put this file in:  src/explainable_ai_project/logistic_baseline.py

Writes:
    models/logistic_dataset_1_v1.joblib, models/logistic_dataset_2_v1.joblib
    results/tables/model_results.csv                     one row per model, dataset and part
    results/tables/logistic_coefficients_dataset_X.csv   how each feature pushes approval
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from explainable_ai_project.load_split import load_split


# ------------------------------------------------------------------
# Where the files go
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PACKAGE_FOLDER = THIS_FILE.parent
SRC_FOLDER = PACKAGE_FOLDER.parent
PROJECT_ROOT = SRC_FOLDER.parent

MODELS_FOLDER = PROJECT_ROOT / "models"
TABLES_FOLDER = PROJECT_ROOT / "results" / "tables"
MODEL_RESULTS_FILE = TABLES_FOLDER / "model_results.csv"

SEED = 42
VERSION = "v1"
DATASET_NAMES = ["dataset_1", "dataset_2"]
MODEL_NAME = "logistic_regression"

# A probability at or above this counts as "approved"
CUT_OFF = 0.5


# ------------------------------------------------------------------
# Metrics (later model steps import these, so every model is scored the same way)
# ------------------------------------------------------------------

def score_predictions(labels, probabilities):
    """All metrics for one set of predictions, as a dictionary."""

    predicted_approved = probabilities >= CUT_OFF
    predictions = predicted_approved.astype(int)

    auc = roc_auc_score(labels, probabilities)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    brier = brier_score_loss(labels, probabilities)

    scores = {
        "auc": round(auc, 4),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "brier": round(brier, 4),
    }
    return scores


def save_model_result(model_name, dataset_name, split_name, scores):
    """Write one row into results/tables/model_results.csv.

    If a row for the same model, dataset and part already exists, it is replaced,
    so running a step again never creates duplicates.
    """

    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)

    new_row = {
        "model": model_name,
        "dataset": dataset_name,
        "split": split_name,
    }
    for metric_name, value in scores.items():
        new_row[metric_name] = value
    new_table = pd.DataFrame([new_row])

    file_exists = MODEL_RESULTS_FILE.exists()
    if file_exists:
        old_table = pd.read_csv(MODEL_RESULTS_FILE)

        same_model = old_table["model"] == model_name
        same_dataset = old_table["dataset"] == dataset_name
        same_split = old_table["split"] == split_name
        is_old_version = same_model & same_dataset & same_split

        kept_rows = old_table[~is_old_version]
        results_table = pd.concat([kept_rows, new_table], ignore_index=True)
    else:
        results_table = new_table

    results_table.to_csv(MODEL_RESULTS_FILE, index=False)


# ------------------------------------------------------------------
# The baseline
# ------------------------------------------------------------------

def build_logistic_model():
    """Scale the features, then logistic regression."""

    model = Pipeline(steps=[
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(max_iter=2000, random_state=SEED)),
    ])
    return model


def coefficient_table(model, feature_names):
    """How strongly each (scaled) feature pushes the approval probability up or down."""

    logistic_part = model.named_steps["logistic"]
    coefficients = logistic_part.coef_[0]

    rows = []
    for position in range(len(feature_names)):
        name = feature_names[position]
        value = coefficients[position]

        if value >= 0:
            direction = "raises approval"
        else:
            direction = "lowers approval"

        rows.append({
            "feature": name,
            "coefficient": round(value, 4),
            "size": round(abs(value), 4),
            "direction": direction,
        })

    table = pd.DataFrame(rows)
    table = table.sort_values("size", ascending=False)
    table = table.reset_index(drop=True)
    return table


def train_one_baseline(dataset_name):
    print(dataset_name)

    train = load_split(dataset_name, "train")
    validation = load_split(dataset_name, "validation")

    model = build_logistic_model()
    model.fit(train["features"], train["labels"])

    probabilities = model.predict_proba(validation["features"])
    approval_probabilities = probabilities[:, 1]

    scores = score_predictions(validation["labels"], approval_probabilities)
    save_model_result(MODEL_NAME, dataset_name, "validation", scores)

    print("  validation:", scores)

    feature_names = list(train["features"].columns)
    coefficients = coefficient_table(model, feature_names)
    print("  strongest features:")
    top_five = coefficients.head(5)
    print(top_five.to_string(index=False))

    MODELS_FOLDER.mkdir(parents=True, exist_ok=True)
    model_file = MODELS_FOLDER / ("logistic_" + dataset_name + "_" + VERSION + ".joblib")
    joblib.dump(model, model_file)

    coefficients_file = TABLES_FOLDER / ("logistic_coefficients_" + dataset_name + ".csv")
    coefficients.to_csv(coefficients_file, index=False)

    print("  saved", model_file.relative_to(PROJECT_ROOT))
    print("  saved", coefficients_file.relative_to(PROJECT_ROOT))
    print()

    return model


def train_logistic_baselines():
    """Run the step on both datasets.

    Returns a dictionary: {"dataset_1": model, "dataset_2": model}
    """

    models = {}
    for dataset_name in DATASET_NAMES:
        model = train_one_baseline(dataset_name)
        models[dataset_name] = model

    print("Saved", MODEL_RESULTS_FILE.relative_to(PROJECT_ROOT))
    return models