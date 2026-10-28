"""
Training pathway, block 1B, step 21: test once (Topic 2: confusion matrices).

Until now every number came from the VALIDATION part, which was also used to
tune and calibrate. Those numbers are slightly optimistic. The TEST part has
never been looked at, so it gives the honest final results.

The rule: look at the test results, report them, and do NOT go back and change
the models because of them. (Re-running this step on the same saved models
gives the same numbers, which is fine.)

Every model is scored on the test part of its own dataset:
    logistic_regression                    baseline (step 13)
    random_forest_unconstrained            step 14
    random_forest_monotonic                step 15 to 17, before calibration
    random_forest_monotonic_calibrated     the FINAL models A and B (step 19)
    ebm_monotonic                          Topic 4 comparison (step 18)

For the final models A and B it also draws:
    a confusion matrix     how many approved / rejected applicants were predicted right
    a reliability diagram  whether the probabilities can be taken literally

Put this file in:  src/explainable_ai_project/test_once.py

Writes:
    results/tables/model_results.csv          (adds one "test" row per model and dataset)
    results/tables/test_results.csv           the test rows only, side by side
    results/tables/confusion_matrices.csv
    results/figures/confusion_matrices.png
    results/figures/reliability_test.png
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save figures to files without opening a window

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.metrics import confusion_matrix

from explainable_ai_project.ebm_comparison import ebm_inputs
from explainable_ai_project.load_split import load_split
from explainable_ai_project.logistic_baseline import CUT_OFF
from explainable_ai_project.logistic_baseline import save_model_result
from explainable_ai_project.logistic_baseline import score_predictions


# ------------------------------------------------------------------
# Where the files are
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PACKAGE_FOLDER = THIS_FILE.parent
SRC_FOLDER = PACKAGE_FOLDER.parent
PROJECT_ROOT = SRC_FOLDER.parent

MODELS_FOLDER = PROJECT_ROOT / "models"
TABLES_FOLDER = PROJECT_ROOT / "results" / "tables"
FIGURES_FOLDER = PROJECT_ROOT / "results" / "figures"

VERSION = "v1"
DATASET_NAMES = ["dataset_1", "dataset_2"]
FINAL_MODEL = "random_forest_monotonic_calibrated"
FINAL_MODEL_LABELS = {
    "dataset_1": "Model A (dataset 1)",
    "dataset_2": "Model B (dataset 2)",
}

# Model name in the results table -> (file name start, which inputs it reads)
# "features" = one-hot model inputs from the preprocessor, "rows" = original columns (EBM)
MODELS_TO_TEST = {
    "logistic_regression": ("logistic_", "features"),
    "random_forest_unconstrained": ("rf_unconstrained_", "features"),
    "random_forest_monotonic": ("rf_monotonic_", "features"),
    "random_forest_monotonic_calibrated": ("rf_calibrated_", "features"),
    "ebm_monotonic": ("ebm_", "rows"),
}

DIAGRAM_BINS = 10


# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------

def load_saved_model(file_start, dataset_name):
    model_file = MODELS_FOLDER / (file_start + dataset_name + "_" + VERSION + ".joblib")
    file_exists = model_file.exists()
    if not file_exists:
        print("  missing", model_file.relative_to(PROJECT_ROOT), "- skipped")
        return None
    model = joblib.load(model_file)
    return model


def test_probabilities(model, test, input_kind):
    if input_kind == "rows":
        inputs = ebm_inputs(test["rows"])
    else:
        inputs = test["features"]

    probability_table = model.predict_proba(inputs)
    approval_probabilities = probability_table[:, 1]
    return approval_probabilities


def test_one_dataset(dataset_name):
    """Score every saved model on this dataset's test part. Returns the final model's probabilities."""

    print(dataset_name)
    test = load_split(dataset_name, "test")
    labels = test["labels"]

    final_probabilities = None

    for model_name, model_info in MODELS_TO_TEST.items():
        file_start = model_info[0]
        input_kind = model_info[1]

        model = load_saved_model(file_start, dataset_name)
        if model is None:
            continue

        probabilities = test_probabilities(model, test, input_kind)
        scores = score_predictions(labels, probabilities)
        save_model_result(model_name, dataset_name, "test", scores)

        print("  " + model_name + ":", "AUC", scores["auc"], "| F1", scores["f1"], "| Brier", scores["brier"])

        if model_name == FINAL_MODEL:
            final_probabilities = probabilities

    print()
    return labels, final_probabilities


# ------------------------------------------------------------------
# Confusion matrices for the final models
# ------------------------------------------------------------------

def confusion_counts(labels, probabilities):
    """[[true rejected, false approved], [false rejected, true approved]]"""

    predicted_approved = probabilities >= CUT_OFF
    predictions = predicted_approved.astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return matrix


def confusion_rows(matrix, dataset_name):
    row = {
        "dataset": dataset_name,
        "model": FINAL_MODEL,
        "correctly_rejected": int(matrix[0][0]),
        "wrongly_approved": int(matrix[0][1]),
        "wrongly_rejected": int(matrix[1][0]),
        "correctly_approved": int(matrix[1][1]),
    }
    return row


def draw_confusion_matrices(matrices):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    class_names = ["rejected", "approved"]

    for position in range(len(DATASET_NAMES)):
        dataset_name = DATASET_NAMES[position]
        axis = axes[position]
        matrix = matrices[dataset_name]

        axis.imshow(matrix, cmap="Blues")

        for true_class in range(2):
            for predicted_class in range(2):
                count = matrix[true_class][predicted_class]
                axis.text(predicted_class, true_class, str(count), ha="center", va="center", fontsize=14)

        axis.set_xticks([0, 1])
        axis.set_xticklabels(class_names)
        axis.set_yticks([0, 1])
        axis.set_yticklabels(class_names)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Really")
        axis.set_title(FINAL_MODEL_LABELS[dataset_name] + ", test part")

    figure.suptitle("Confusion matrices of the final calibrated models (cut-off 0.5)")
    figure.tight_layout()

    FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)
    figure_file = FIGURES_FOLDER / "confusion_matrices.png"
    figure.savefig(figure_file, dpi=150)
    plt.close(figure)
    print("Saved", figure_file.relative_to(PROJECT_ROOT))


# ------------------------------------------------------------------
# Reliability diagram on the test part
# ------------------------------------------------------------------

def draw_reliability_on_test(final_results):
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], linestyle="--", color="#9AA5B1", label="perfectly calibrated")

    colours = {"dataset_1": "#3B6FD8", "dataset_2": "#2F8F5B"}

    for dataset_name in DATASET_NAMES:
        labels = final_results[dataset_name]["labels"]
        probabilities = final_results[dataset_name]["probabilities"]

        observed, predicted = calibration_curve(labels, probabilities, n_bins=DIAGRAM_BINS)
        brier = brier_score_loss(labels, probabilities)

        line_label = FINAL_MODEL_LABELS[dataset_name] + " (Brier " + str(round(brier, 4)) + ")"
        axis.plot(predicted, observed, marker="o", color=colours[dataset_name], label=line_label)

    axis.set_xlabel("Predicted approval probability")
    axis.set_ylabel("Share really approved")
    axis.set_title("Reliability on the test part")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend(loc="upper left")
    figure.tight_layout()

    figure_file = FIGURES_FOLDER / "reliability_test.png"
    figure.savefig(figure_file, dpi=150)
    plt.close(figure)
    print("Saved", figure_file.relative_to(PROJECT_ROOT))


# ------------------------------------------------------------------
# The step
# ------------------------------------------------------------------

def test_results_table():
    results = pd.read_csv(TABLES_FOLDER / "model_results.csv")
    is_test = results["split"] == "test"
    test_rows = results[is_test]
    test_rows = test_rows.sort_values(["dataset", "auc"], ascending=[True, False])
    test_rows = test_rows.reset_index(drop=True)
    return test_rows


def test_once():
    """Step 21: score every model on the untouched test part."""

    print("Scoring every model on the TEST part (do not change the models after this)")
    print()

    final_results = {}
    matrices = {}
    confusion_table_rows = []

    for dataset_name in DATASET_NAMES:
        labels, final_probabilities = test_one_dataset(dataset_name)
        if final_probabilities is None:
            print("Final calibrated model missing for", dataset_name, "- run step 19 first")
            continue

        final_results[dataset_name] = {"labels": labels, "probabilities": final_probabilities}

        matrix = confusion_counts(labels, final_probabilities)
        matrices[dataset_name] = matrix
        row = confusion_rows(matrix, dataset_name)
        confusion_table_rows.append(row)

    # Tables
    test_table = test_results_table()
    test_file = TABLES_FOLDER / "test_results.csv"
    test_table.to_csv(test_file, index=False)

    confusion_table = pd.DataFrame(confusion_table_rows)
    confusion_file = TABLES_FOLDER / "confusion_matrices.csv"
    confusion_table.to_csv(confusion_file, index=False)

    print("Test results:")
    print(test_table.to_string(index=False))
    print()
    print("Confusion matrices of the final models:")
    print(confusion_table.to_string(index=False))
    print()
    print("Saved", test_file.relative_to(PROJECT_ROOT))
    print("Saved", confusion_file.relative_to(PROJECT_ROOT))

    # Figures (only if both final models were found)
    both_found = len(final_results) == len(DATASET_NAMES)
    if both_found:
        draw_confusion_matrices(matrices)
        draw_reliability_on_test(final_results)

    return test_table