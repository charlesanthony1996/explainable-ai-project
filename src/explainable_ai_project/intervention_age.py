"""
Training pathway, block 1C, step 28: intervention, remove age (Topic 3: intervention).

Steps 24 to 27 found that model A gives every applicant aged 61+ about the same
low probability, and that income and credit score do not explain the gap.
This step tests the obvious fix: take age away from the model.

For each dataset:
    1. build the same model WITHOUT the age column
       (same tuned settings, same monotonic rule, same calibration method)
    2. score both versions on the TEST part:
           accuracy     AUC, F1, Brier score
           fairness     approval rates and true positive rates per age group,
                        demographic parity ratio, equal opportunity difference
           ranking      within-group AUC (can the model tell older applicants apart?)
           sweep        share of each age group that gets a maximum amount at 0.8
    3. compare "with age" and "without age" side by side

How to read the result:
    - gap shrinks, accuracy about the same   -> age was used directly; removing it helps
    - gap stays                              -> the gap comes through other features
                                                that are tied to age (here: credit score),
                                                so removing age alone is not enough
    - accuracy drops a lot                   -> age carried real information

The models without age are saved separately; the production models are not replaced.

Put this file in:  src/explainable_ai_project/intervention_remove_age.py

Writes:
    models/rf_calibrated_no_age_dataset_X_v1.joblib
    results/tables/intervention_remove_age.csv          one row per dataset and version
    results/tables/intervention_by_age_group.csv        per age group, with and without age
    results/figures/intervention_remove_age.png
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save figures to files without opening a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from explainable_ai_project.test_set_sweep import load_sweep_limits
from explainable_ai_project.test_set_sweep import sweep_applicants
from explainable_ai_project.calibrate import calibrate_on
from explainable_ai_project.calibrate import load_calibrated_model
from explainable_ai_project.load_split import load_split
from explainable_ai_project.load_split import read_feature_names
from explainable_ai_project.load_split import read_preprocessor
from explainable_ai_project.logistic_baseline import score_predictions
from explainable_ai_project.monotonic_forest_a import build_monotonic_forest
from explainable_ai_project.tune_forest import load_best_settings


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

VERSION = "v1"
DATASET_NAMES = ["dataset_1", "dataset_2"]
AGE_GROUP_ORDER = ["18-30", "31-45", "46-60", "61+"]
REMOVED_FEATURE = "age"
SWEEP_THRESHOLD = 0.8
CUT_OFF = 0.5

MODEL_LABELS = {
    "dataset_1": "Model A (dataset 1)",
    "dataset_2": "Model B (dataset 2)",
}


# ------------------------------------------------------------------
# A model that ignores the age column
# ------------------------------------------------------------------

class WithoutAge:
    """Wraps a model trained without age so it accepts the usual feature table.

    The amount sweep and the other helpers build features WITH age;
    this wrapper simply drops the age column before asking the model.
    """

    def __init__(self, model):
        self.model = model

    def predict_proba(self, features):
        features_without_age = features.drop(columns=[REMOVED_FEATURE])
        probabilities = self.model.predict_proba(features_without_age)
        return probabilities


def calibration_method_for(dataset_name):
    """The calibration method chosen in step 19 (isotonic if the table is missing)."""

    choice_file = TABLES_FOLDER / "calibration_choice.csv"
    if not choice_file.exists():
        return "isotonic"

    choices = pd.read_csv(choice_file)
    is_dataset = choices["dataset"] == dataset_name
    is_chosen = choices["chosen"] == True
    chosen_rows = choices[is_dataset & is_chosen]

    if len(chosen_rows) == 0:
        return "isotonic"

    method = chosen_rows["method"].iloc[0]
    return method


def build_model_without_age(dataset_name):
    """Same settings, same constraint, same calibration, but no age column."""

    train = load_split(dataset_name, "train")
    validation = load_split(dataset_name, "validation")

    train_features = train["features"].drop(columns=[REMOVED_FEATURE])
    validation_features = validation["features"].drop(columns=[REMOVED_FEATURE])
    feature_names = list(train_features.columns)

    settings = load_best_settings(dataset_name)
    forest = build_monotonic_forest(feature_names, settings)
    forest.fit(train_features, train["labels"])

    method = calibration_method_for(dataset_name)
    calibrated = calibrate_on(forest, validation_features, validation["labels"], method)

    MODELS_FOLDER.mkdir(parents=True, exist_ok=True)
    model_file = MODELS_FOLDER / ("rf_calibrated_no_age_" + dataset_name + "_" + VERSION + ".joblib")
    joblib.dump(calibrated, model_file)
    print("  trained without age (" + method + " calibration), saved", model_file.relative_to(PROJECT_ROOT))

    wrapped = WithoutAge(calibrated)
    return wrapped


# ------------------------------------------------------------------
# Measuring one version
# ------------------------------------------------------------------

def within_group_auc(labels, probabilities):
    both_outcomes_present = len(np.unique(labels)) == 2
    if not both_outcomes_present:
        return np.nan
    auc = roc_auc_score(labels, probabilities)
    return auc


def measure_version(model, version_name, dataset_name, test, sweep_inputs):
    """All measures for one version of the model, on the test part."""

    labels = test["labels"].to_numpy()
    groups = test["rows"]["age_group"].to_numpy()

    probability_table = model.predict_proba(test["features"])
    probabilities = probability_table[:, 1]
    scores = score_predictions(labels, probabilities)

    # Sweep: who gets a maximum amount at the threshold
    preprocessor = sweep_inputs["preprocessor"]
    feature_names = sweep_inputs["feature_names"]
    limits = sweep_inputs["limits"]
    sweep_result = sweep_applicants(test["rows"], model, preprocessor, feature_names, limits)
    max_column = "max_amount_" + str(SWEEP_THRESHOLD)
    maximum_amounts = sweep_result[max_column]
    has_amount = maximum_amounts.notna()
    gets_an_amount = has_amount.to_numpy()

    predicted_approved = probabilities >= CUT_OFF

    group_rows = []
    approval_rates = []
    true_positive_rates = []

    for group in AGE_GROUP_ORDER:
        in_group = groups == group
        group_labels = labels[in_group]
        group_probabilities = probabilities[in_group]
        group_predicted = predicted_approved[in_group]

        approval_rate = group_predicted.mean()
        really_approved = group_labels == 1
        if really_approved.sum() > 0:
            true_positive_rate = group_predicted[really_approved].mean()
        else:
            true_positive_rate = np.nan

        auc = within_group_auc(group_labels, group_probabilities)
        amount_share = gets_an_amount[in_group].mean()

        approval_rates.append(approval_rate)
        true_positive_rates.append(true_positive_rate)

        group_rows.append({
            "dataset": dataset_name,
            "version": version_name,
            "age_group": group,
            "applicants": int(in_group.sum()),
            "really_approved": round(group_labels.mean(), 3),
            "model_approves_at_0.5": round(approval_rate, 3),
            "true_positive_rate_at_0.5": round(true_positive_rate, 3),
            "within_group_auc": round(auc, 3),
            "share_getting_an_amount_at_0.8": round(amount_share, 3),
        })

    highest_rate = max(approval_rates)
    lowest_rate = min(approval_rates)
    if highest_rate > 0:
        parity_ratio = lowest_rate / highest_rate
    else:
        parity_ratio = np.nan

    valid_rates = []
    for rate in true_positive_rates:
        if not np.isnan(rate):
            valid_rates.append(rate)
    opportunity_difference = max(valid_rates) - min(valid_rates)

    summary_row = {
        "dataset": dataset_name,
        "version": version_name,
        "auc": scores["auc"],
        "f1": scores["f1"],
        "brier": scores["brier"],
        "demographic_parity_ratio_at_0.5": round(parity_ratio, 3),
        "equal_opportunity_difference_at_0.5": round(opportunity_difference, 3),
    }

    return summary_row, group_rows


# ------------------------------------------------------------------
# One dataset
# ------------------------------------------------------------------

def compare_one_dataset(dataset_name):
    print(dataset_name + ":", MODEL_LABELS[dataset_name])

    test = load_split(dataset_name, "test")
    sweep_inputs = {
        "preprocessor": read_preprocessor(dataset_name),
        "feature_names": read_feature_names(dataset_name),
        "limits": load_sweep_limits(dataset_name),
    }

    with_age_model = load_calibrated_model(dataset_name)
    without_age_model = build_model_without_age(dataset_name)

    summary_with, groups_with = measure_version(with_age_model, "with age", dataset_name, test, sweep_inputs)
    summary_without, groups_without = measure_version(without_age_model, "without age", dataset_name, test, sweep_inputs)

    summary_rows = [summary_with, summary_without]
    group_rows = groups_with + groups_without

    summary_table = pd.DataFrame(summary_rows)
    summary_to_show = summary_table.drop(columns=["dataset"])
    print(summary_to_show.to_string(index=False))

    group_table = pd.DataFrame(group_rows)
    columns_to_show = [
        "version",
        "age_group",
        "model_approves_at_0.5",
        "within_group_auc",
        "share_getting_an_amount_at_0.8",
    ]
    print(group_table[columns_to_show].to_string(index=False))
    print()

    return summary_rows, group_rows


# ------------------------------------------------------------------
# Figure
# ------------------------------------------------------------------

def draw_figure(group_table):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    bar_width = 0.38
    positions = np.arange(len(AGE_GROUP_ORDER))
    column = "share_getting_an_amount_at_0.8"

    for position in range(len(DATASET_NAMES)):
        dataset_name = DATASET_NAMES[position]
        axis = axes[position]
        is_dataset = group_table["dataset"] == dataset_name

        is_with = group_table["version"] == "with age"
        is_without = group_table["version"] == "without age"
        with_rows = group_table[is_dataset & is_with]
        without_rows = group_table[is_dataset & is_without]
        with_rates = with_rows[column].to_numpy()
        without_rates = without_rows[column].to_numpy()

        axis.bar(positions - bar_width / 2, with_rates, bar_width, color="#C0392B", label="with age")
        axis.bar(positions + bar_width / 2, without_rates, bar_width, color="#2F8F5B", label="without age")

        axis.set_xticks(positions)
        axis.set_xticklabels(AGE_GROUP_ORDER)
        axis.set_xlabel("Age group")
        axis.set_title(MODEL_LABELS[dataset_name] + ", test part")
        axis.set_ylim(0, 1)
        axis.legend(loc="upper right")

    axes[0].set_ylabel("Share getting a maximum amount (P >= " + str(SWEEP_THRESHOLD) + ")")
    figure.suptitle("Intervention: does removing age from the model change who gets an amount?")
    figure.tight_layout()

    FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)
    figure_file = FIGURES_FOLDER / "intervention_remove_age.png"
    figure.savefig(figure_file, dpi=150)
    plt.close(figure)
    print("Saved", figure_file.relative_to(PROJECT_ROOT))


# ------------------------------------------------------------------
# The step
# ------------------------------------------------------------------

def intervention_remove_age():
    """Step 28: retrain both models without age and compare with the originals."""

    all_summary_rows = []
    all_group_rows = []

    for dataset_name in DATASET_NAMES:
        summary_rows, group_rows = compare_one_dataset(dataset_name)
        all_summary_rows.extend(summary_rows)
        all_group_rows.extend(group_rows)

    summary_table = pd.DataFrame(all_summary_rows)
    group_table = pd.DataFrame(all_group_rows)

    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)
    summary_file = TABLES_FOLDER / "intervention_remove_age.csv"
    group_file = TABLES_FOLDER / "intervention_by_age_group.csv"
    summary_table.to_csv(summary_file, index=False)
    group_table.to_csv(group_file, index=False)
    print("Saved", summary_file.relative_to(PROJECT_ROOT))
    print("Saved", group_file.relative_to(PROJECT_ROOT))

    draw_figure(group_table)
    return summary_table, group_table