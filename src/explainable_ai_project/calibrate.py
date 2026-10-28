"""
Training pathway, block 1B, step 19: calibrate (Topic 3: calibration).

A model is CALIBRATED if its probabilities can be taken literally:
of all applicants it gives 0.8, about 80 % should really be approved.
Random Forests are often a little off (too careful near 0 and 1).
This matters here because the amount sweep stops at "P(approve) >= 0.8",
so 0.8 has to mean 0.8.

Calibration adds a small correction on top of the finished model. It does not
retrain the forest; it learns how to turn its raw probability into a better one.
Two common corrections are tried:
    sigmoid    a smooth S-shaped curve (few parameters, stable)
    isotonic   a step curve that can take any rising shape (more flexible)
Both only ever map a higher raw probability to a higher (or equal) calibrated one,
so the monotonic rule from step 15 still holds. Step 20 checks this.

How the method is chosen fairly:
    The correction is learned on the validation part. To score it on rows it has
    NOT seen, the validation part is cut into 5 pieces: the correction is learned
    on 4 pieces and scored on the 5th, five times over ("out-of-fold").
    The method with the lower out-of-fold Brier score wins, and is then learned
    on the whole validation part for the final model.

Put this file in:  src/explainable_ai_project/calibrate.py

Writes:
    models/rf_calibrated_dataset_X_v1.joblib        calibrated models A and B
    results/tables/calibration_choice.csv           Brier score per method
    results/figures/calibration_dataset_X.png       reliability diagram before and after
    results/tables/model_results.csv                (adds the calibrated rows)
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save figures to files without opening a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold

from explainable_ai_project.load_split import load_split
from explainable_ai_project.logistic_baseline import save_model_result
from explainable_ai_project.logistic_baseline import score_predictions
from explainable_ai_project.monotonic_forest_a import monotonic_model_file


# ------------------------------------------------------------------
# Where the files go
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PACKAGE_FOLDER = THIS_FILE.parent
SRC_FOLDER = PACKAGE_FOLDER.parent
PROJECT_ROOT = SRC_FOLDER.parent

MODELS_FOLDER = PROJECT_ROOT / "models"
TABLES_FOLDER = PROJECT_ROOT / "results" / "tables"
FIGURES_FOLDER = PROJECT_ROOT / "results" / "figures"

SEED = 42
VERSION = "v1"
DATASET_NAMES = ["dataset_1", "dataset_2"]
MODEL_NAME = "random_forest_monotonic_calibrated"

METHODS = ["sigmoid", "isotonic"]
FOLD_COUNT = 5
DIAGRAM_BINS = 10   # number of probability groups in the reliability diagram


def calibrated_model_file(dataset_name):
    file_name = "rf_calibrated_" + dataset_name + "_" + VERSION + ".joblib"
    return MODELS_FOLDER / file_name


def load_calibrated_model(dataset_name):
    """Load a calibrated model (used by the later steps and at inference)."""
    model_file = calibrated_model_file(dataset_name)
    model = joblib.load(model_file)
    return model


# ------------------------------------------------------------------
# Building a calibrated model
# ------------------------------------------------------------------

def calibrate_on(forest, features, labels, method):
    """Learn a correction of the given method on these rows. The forest itself is not changed."""

    # FrozenEstimator: "this model is already trained, do not train it again"
    frozen_forest = FrozenEstimator(forest)

    calibrated = CalibratedClassifierCV(frozen_forest, method=method)
    calibrated.fit(features, labels)
    return calibrated


def out_of_fold_probabilities(forest, features, labels, method):
    """Calibrated probabilities for every validation row, each from a correction
    that was learned WITHOUT that row (5 folds)."""

    folds = StratifiedKFold(n_splits=FOLD_COUNT, shuffle=True, random_state=SEED)
    probabilities = np.zeros(len(labels))

    for learn_positions, score_positions in folds.split(features, labels):
        learn_features = features.iloc[learn_positions]
        learn_labels = labels.iloc[learn_positions]
        score_features = features.iloc[score_positions]

        calibrated = calibrate_on(forest, learn_features, learn_labels, method)
        fold_probabilities = calibrated.predict_proba(score_features)
        probabilities[score_positions] = fold_probabilities[:, 1]

    return probabilities


# ------------------------------------------------------------------
# Reliability diagram
# ------------------------------------------------------------------

def draw_reliability_diagram(dataset_name, labels, raw_probabilities, calibrated_probabilities, method):
    """Predicted probability against how often applicants were really approved.
    A perfectly calibrated model lies on the diagonal."""

    raw_observed, raw_predicted = calibration_curve(labels, raw_probabilities, n_bins=DIAGRAM_BINS)
    new_observed, new_predicted = calibration_curve(labels, calibrated_probabilities, n_bins=DIAGRAM_BINS)

    raw_brier = brier_score_loss(labels, raw_probabilities)
    new_brier = brier_score_loss(labels, calibrated_probabilities)

    raw_label = "before (Brier " + str(round(raw_brier, 4)) + ")"
    new_label = "after, " + method + " (Brier " + str(round(new_brier, 4)) + ")"

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], linestyle="--", color="#9AA5B1", label="perfectly calibrated")
    axis.plot(raw_predicted, raw_observed, marker="o", color="#C0392B", label=raw_label)
    axis.plot(new_predicted, new_observed, marker="o", color="#2F8F5B", label=new_label)

    axis.set_xlabel("Predicted approval probability")
    axis.set_ylabel("Share really approved")
    axis.set_title(dataset_name + ": reliability diagram (validation, out-of-fold)")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend(loc="upper left")
    figure.tight_layout()

    FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)
    figure_file = FIGURES_FOLDER / ("calibration_" + dataset_name + ".png")
    figure.savefig(figure_file, dpi=150)
    plt.close(figure)
    print("  saved", figure_file.relative_to(PROJECT_ROOT))


# ------------------------------------------------------------------
# One dataset
# ------------------------------------------------------------------

def calibrate_one_model(dataset_name):
    print(dataset_name)

    forest_file = monotonic_model_file(dataset_name)
    forest = joblib.load(forest_file)

    validation = load_split(dataset_name, "validation")
    features = validation["features"]
    labels = validation["labels"]

    # Before: the raw forest probabilities
    raw_probability_table = forest.predict_proba(features)
    raw_probabilities = raw_probability_table[:, 1]
    raw_brier = brier_score_loss(labels, raw_probabilities)
    print("  raw Brier score:", round(raw_brier, 4))

    # Try both methods fairly (out-of-fold)
    choice_rows = []
    out_of_fold_results = {}

    for method in METHODS:
        probabilities = out_of_fold_probabilities(forest, features, labels, method)
        method_brier = brier_score_loss(labels, probabilities)
        out_of_fold_results[method] = probabilities

        print("  " + method + " Brier score (out-of-fold):", round(method_brier, 4))
        choice_rows.append({
            "dataset": dataset_name,
            "method": method,
            "brier_raw": round(raw_brier, 4),
            "brier_calibrated_out_of_fold": round(method_brier, 4),
        })

    # Pick the lower Brier score
    best_method = METHODS[0]
    best_brier = None
    for row in choice_rows:
        if best_brier is None or row["brier_calibrated_out_of_fold"] < best_brier:
            best_brier = row["brier_calibrated_out_of_fold"]
            best_method = row["method"]

    for row in choice_rows:
        row["chosen"] = row["method"] == best_method
    print("  chosen:", best_method)

    # Honest validation scores: the out-of-fold probabilities of the chosen method
    best_probabilities = out_of_fold_results[best_method]
    scores = score_predictions(labels, best_probabilities)
    save_model_result(MODEL_NAME, dataset_name, "validation", scores)
    print("  validation (out-of-fold):", scores)

    draw_reliability_diagram(dataset_name, labels, raw_probabilities, best_probabilities, best_method)

    # Final model: the chosen correction learned on the whole validation part
    final_model = calibrate_on(forest, features, labels, best_method)

    MODELS_FOLDER.mkdir(parents=True, exist_ok=True)
    model_file = calibrated_model_file(dataset_name)
    joblib.dump(final_model, model_file)
    print("  saved", model_file.relative_to(PROJECT_ROOT))
    print()

    return final_model, choice_rows


def calibrate_models():
    """Step 19: calibrate model A (dataset 1) and model B (dataset 2)."""

    calibrated_models = {}
    all_choice_rows = []

    for dataset_name in DATASET_NAMES:
        final_model, choice_rows = calibrate_one_model(dataset_name)
        calibrated_models[dataset_name] = final_model
        all_choice_rows.extend(choice_rows)

    choice_table = pd.DataFrame(all_choice_rows)
    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)
    choice_file = TABLES_FOLDER / "calibration_choice.csv"
    choice_table.to_csv(choice_file, index=False)
    print("Saved", choice_file.relative_to(PROJECT_ROOT))

    return calibrated_models