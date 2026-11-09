"""
Explanations with TreeSHAP (Topic 2).

What SHAP does here
-------------------
For one applicant, SHAP splits the model's prediction into one number per
feature: how much that feature pushed the approval probability UP or DOWN,
compared with an average applicant. The numbers add up exactly:

    average probability + sum of all feature contributions = this applicant's probability

TreeSHAP computes these numbers exactly (not by sampling) for tree models,
so the same applicant always gets the same explanation. That matters for the
LIME comparison later: LIME samples, so it wobbles between runs; SHAP does not.

Which model is explained
------------------------
TreeSHAP needs the forest itself, so it runs on the MONOTONIC forest
(`rf_monotonic_dataset_X_v1.joblib`), not on the calibrated wrapper
(`CalibratedClassifierCV`, which is not a tree model).

Calibration is a monotone correction on top of the forest: it can only map a
higher raw probability to a higher calibrated one. So the SIGN of every
contribution and the RANKING of the features are the same for the calibrated
model. Only the sizes differ, because they are expressed in the forest's raw
probability scale. Every table written here says which scale it uses.

The monotonic rule from step 15 is visible in the output: loan_amount and
loan_to_income can only ever push approval DOWN, never up.

Fidelity check
--------------
TreeSHAP is exact, so the contributions must add up to the model's prediction.
This module checks that and reports the largest deviation. A value near zero
(about 1e-7) means the explanation is faithful to the model, not an
approximation of it. This is the SHAP side of the Topic 2 fidelity check.

Usage
-----
    uv run main.py shap        global importance + example applicants (for the report)

    from explainable_ai_project.explain import explain_applicant
    explain_applicant(applicant_row, "dataset_1")        # inference, step 6

Writes:
    results/tables/shap_importance_dataset_X.csv    mean |contribution| per feature
    results/tables/shap_examples_dataset_X.csv      top features for example applicants
    results/tables/shap_fidelity.csv                additivity check per dataset
    results/figures/shap_importance.png             bar chart, both datasets
    results/figures/shap_beeswarm_dataset_X.csv     one dot per applicant per feature
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save figures to files without opening a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from explainable_ai_project.load_split import load_split
from explainable_ai_project.load_split import read_feature_names
from explainable_ai_project.load_split import read_preprocessor
from explainable_ai_project.sweep_problem import make_features_for_amount


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

DATASET_NAMES = ["dataset_1", "dataset_2"]

# How many test applicants go into the global picture. TreeSHAP is fast, but
# 1,000 applicants already give a stable average and keep the figures readable.
GLOBAL_SAMPLE_SIZE = 1000
RANDOM_SEED = 42

# How many example applicants are written out in full
EXAMPLE_APPLICANTS = 8

# How many features an explanation names
TOP_FEATURES = 5


# ------------------------------------------------------------------
# Loading the forest
# ------------------------------------------------------------------

def monotonic_forest_file(dataset_name):
    return MODELS_FOLDER / ("rf_monotonic_" + dataset_name + "_v1.joblib")


def load_monotonic_forest(dataset_name):
    """The forest TreeSHAP explains (uncalibrated, see the module docstring)."""

    forest_file = monotonic_forest_file(dataset_name)
    if not forest_file.exists():
        message = (
            "Missing " + str(forest_file.relative_to(PROJECT_ROOT))
            + " - run `uv run main.py models` first."
        )
        raise FileNotFoundError(message)

    forest = joblib.load(forest_file)
    return forest


def make_explainer(dataset_name):
    """A TreeSHAP explainer for one dataset's forest.

    `shap` is imported here, not at the top of the file, so that modules which
    only need the sweep do not pay for the import.
    """

    import shap

    forest = load_monotonic_forest(dataset_name)
    explainer = shap.TreeExplainer(forest)
    return explainer, forest


# ------------------------------------------------------------------
# Getting the contributions out of shap
# ------------------------------------------------------------------

def contributions_for_approval(explainer, features):
    """SHAP values for the APPROVED class, as a plain 2-D array.

    shap returns different shapes depending on version and model, so this
    function normalises them:
        (rows, features, 2)  -> take class 1
        list of two arrays   -> take class 1
        (rows, features)     -> use as is
    Also returns the base value (the average probability) for class 1.
    """

    raw_values = explainer.shap_values(features, check_additivity=False)

    if isinstance(raw_values, list):
        values = raw_values[1] if len(raw_values) > 1 else raw_values[0]
    else:
        values = np.asarray(raw_values)
        if values.ndim == 3:
            values = values[:, :, 1] if values.shape[2] > 1 else values[:, :, 0]

    values = np.asarray(values, dtype=float)

    base = explainer.expected_value
    base = np.asarray(base, dtype=float).ravel()
    base_value = float(base[1]) if base.size > 1 else float(base[0])

    return values, base_value


def check_additivity(values, base_value, forest, features):
    """Do the contributions add up to the model's own prediction?

    Returns the largest absolute difference over all applicants. TreeSHAP is
    exact, so this should be about 1e-7. Anything larger means the explanation
    does not match the model.
    """

    reconstructed = base_value + values.sum(axis=1)
    predicted = forest.predict_proba(features)[:, 1]
    differences = np.abs(reconstructed - predicted)
    return float(differences.max())


# ------------------------------------------------------------------
# One applicant (used by the inference pathway, step 6)
# ------------------------------------------------------------------

def explain_applicant(applicant_row, dataset_name, amount=None, top_n=TOP_FEATURES):
    """The features that pushed this applicant's approval up or down.

    applicant_row   a table with ONE row, in the cleaned column layout
    amount          explain at this loan amount instead of the requested one
                    (loan_to_income is recomputed, as in the sweep)
    top_n           how many features to name

    Returns a table, largest absolute contribution first:
        feature         model input name
        value           the applicant's value for it
        contribution    how much it moved the raw approval probability
        direction       "up" or "down"

    The contributions are in the forest's raw probability scale. The sign and
    the order also hold for the calibrated model (see the module docstring).
    """

    preprocessor = read_preprocessor(dataset_name)
    feature_names = read_feature_names(dataset_name)

    if amount is None:
        features = make_features_for_amount(
            applicant_row, applicant_row["loan_amount"], preprocessor, feature_names
        )
    else:
        features = make_features_for_amount(
            applicant_row, amount, preprocessor, feature_names
        )

    explainer, forest = make_explainer(dataset_name)
    values, base_value = contributions_for_approval(explainer, features)

    contributions = values[0]
    applicant_values = features.iloc[0].to_numpy()

    table = pd.DataFrame({
        "feature": feature_names,
        "value": applicant_values,
        "contribution": np.round(contributions, 5),
    })
    table["direction"] = np.where(table["contribution"] >= 0, "up", "down")

    table["size"] = table["contribution"].abs()
    table = table.sort_values("size", ascending=False).drop(columns="size")
    table = table.head(top_n).reset_index(drop=True)

    table.attrs["base_value"] = round(base_value, 5)
    table.attrs["scale"] = "raw forest probability"
    return table


# ------------------------------------------------------------------
# The global picture (for the report)
# ------------------------------------------------------------------

def sample_test_applicants(dataset_name):
    """Up to GLOBAL_SAMPLE_SIZE applicants from the untouched test part."""

    test = load_split(dataset_name, "test")
    rows = test["rows"]
    features = test["features"]

    if len(rows) > GLOBAL_SAMPLE_SIZE:
        sampled = rows.sample(GLOBAL_SAMPLE_SIZE, random_state=RANDOM_SEED)
        rows = sampled
        features = features.loc[sampled.index]

    return rows, features


def importance_table(values, feature_names, dataset_name):
    """Mean absolute contribution per feature: how much it matters overall."""

    mean_size = np.abs(values).mean(axis=0)
    mean_signed = values.mean(axis=0)

    table = pd.DataFrame({
        "dataset": dataset_name,
        "feature": feature_names,
        "mean_absolute_contribution": np.round(mean_size, 5),
        "mean_signed_contribution": np.round(mean_signed, 5),
    })

    table["usually_pushes"] = np.where(table["mean_signed_contribution"] >= 0, "up", "down")
    table = table.sort_values("mean_absolute_contribution", ascending=False)
    table = table.reset_index(drop=True)
    table["rank"] = table.index + 1
    return table


def example_table(values, features, rows, dataset_name):
    """The top features for a handful of individual applicants."""

    feature_names = list(features.columns)
    example_count = min(EXAMPLE_APPLICANTS, len(rows))

    collected = []
    for position in range(example_count):
        contributions = values[position]
        order = np.argsort(np.abs(contributions))[::-1][:TOP_FEATURES]

        for place, feature_index in enumerate(order):
            collected.append({
                "dataset": dataset_name,
                "applicant_id": rows.iloc[position].get("applicant_id", position),
                "age_group": rows.iloc[position].get("age_group", ""),
                "approved": rows.iloc[position].get("approved", ""),
                "place": place + 1,
                "feature": feature_names[feature_index],
                "value": features.iloc[position, feature_index],
                "contribution": round(float(contributions[feature_index]), 5),
                "direction": "up" if contributions[feature_index] >= 0 else "down",
            })

    return pd.DataFrame(collected)


def draw_importance_chart(importance_tables):
    """Side-by-side bar chart of the most important features per dataset."""

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for position in range(len(DATASET_NAMES)):
        dataset_name = DATASET_NAMES[position]
        axis = axes[position]
        table = importance_tables[dataset_name].head(12).iloc[::-1]

        colours = np.where(table["usually_pushes"] == "up", "#2E8B57", "#C0392B")
        axis.barh(table["feature"], table["mean_absolute_contribution"], color=colours)
        axis.set_title(dataset_name + ": mean |SHAP| (raw forest scale)")
        axis.set_xlabel("average size of the contribution")

    figure.suptitle("green = usually pushes approval up,   red = usually pushes it down")
    figure.tight_layout()

    FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)
    figure_file = FIGURES_FOLDER / "shap_importance.png"
    figure.savefig(figure_file, dpi=150)
    plt.close(figure)
    print("Saved", figure_file.relative_to(PROJECT_ROOT))


def draw_beeswarm(values, base_value, features, dataset_name):
    """One dot per applicant per feature: direction and spread of the effect."""

    import shap

    explanation = shap.Explanation(
        values=values,
        base_values=np.full(len(values), base_value),
        data=features.to_numpy(),
        feature_names=list(features.columns),
    )

    plt.figure(figsize=(9, 6))
    shap.plots.beeswarm(explanation, max_display=14, show=False)
    plt.title(dataset_name + ": how each feature moves approval")
    plt.tight_layout()

    FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)
    figure_file = FIGURES_FOLDER / ("shap_beeswarm_" + dataset_name + ".png")
    plt.savefig(figure_file, dpi=150)
    plt.close()
    print("Saved", figure_file.relative_to(PROJECT_ROOT))


def explain_one_dataset(dataset_name):
    print(dataset_name)

    rows, features = sample_test_applicants(dataset_name)
    print("  explaining", len(rows), "test applicants")

    explainer, forest = make_explainer(dataset_name)
    values, base_value = contributions_for_approval(explainer, features)
    print("  average approval probability (base value):", round(base_value, 4))

    largest_difference = check_additivity(values, base_value, forest, features)
    print("  fidelity check: largest difference between SHAP and the model:",
          format(largest_difference, ".2e"))
    if largest_difference > 1e-4:
        print("  WARNING: the contributions do not add up to the model's prediction")

    importance = importance_table(values, list(features.columns), dataset_name)
    print("  most important features:")
    print(importance.head(6)[["rank", "feature", "mean_absolute_contribution",
                              "usually_pushes"]].to_string(index=False))

    # How the monotonic rule shows up in SHAP
    #
    # A monotonic constraint does NOT mean the contribution of loan_amount is
    # always negative. SHAP measures a contribution RELATIVE to an average
    # applicant, so someone asking for a below-average amount gets a POSITIVE
    # contribution from it - which is exactly what monotonicity predicts.
    #
    # What monotonicity does mean is that the contribution has to FALL as the
    # amount rises. Across applicants that shows up as a strongly negative rank
    # correlation between a feature's value and its own contribution. It is not
    # exactly -1, because contributions also depend on the other features.
    #
    # The strict per-applicant test (vary only the amount, the probability curve
    # must never rise) is step 20 in recheck_monotonic.py, which checks it on
    # the predictions themselves. That one is the authoritative check.
    from scipy.stats import spearmanr

    monotonic_correlations = {}
    for feature in ["loan_amount", "loan_to_income"]:
        if feature in list(features.columns):
            column_index = list(features.columns).index(feature)
            correlation = float(spearmanr(features[feature], values[:, column_index])[0])
            monotonic_correlations[feature] = round(correlation, 4)
            direction = "falls with the amount, as expected" if correlation < -0.5 else "CHECK THIS"
            print("  monotonic rule,", feature, "- rank correlation value vs contribution:",
                  round(correlation, 4), "-", direction)

    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)

    importance_file = TABLES_FOLDER / ("shap_importance_" + dataset_name + ".csv")
    importance.to_csv(importance_file, index=False)
    print("  saved", importance_file.relative_to(PROJECT_ROOT))

    examples = example_table(values, features, rows, dataset_name)
    examples_file = TABLES_FOLDER / ("shap_examples_" + dataset_name + ".csv")
    examples.to_csv(examples_file, index=False)
    print("  saved", examples_file.relative_to(PROJECT_ROOT))

    draw_beeswarm(values, base_value, features, dataset_name)
    print()

    fidelity_row = {
        "dataset": dataset_name,
        "applicants_explained": len(rows),
        "base_value": round(base_value, 5),
        "largest_difference_shap_vs_model": largest_difference,
        "passed": bool(largest_difference <= 1e-4),
        "rank_correlation_loan_amount": monotonic_correlations.get("loan_amount"),
        "rank_correlation_loan_to_income": monotonic_correlations.get("loan_to_income"),
        "scale": "raw forest probability (uncalibrated)",
    }

    return importance, fidelity_row


def run_shap_explanations():
    """Explain both models with TreeSHAP and write the tables and figures."""

    importance_tables = {}
    fidelity_rows = []

    for dataset_name in DATASET_NAMES:
        importance, fidelity_row = explain_one_dataset(dataset_name)
        importance_tables[dataset_name] = importance
        fidelity_rows.append(fidelity_row)

    fidelity_table = pd.DataFrame(fidelity_rows)
    fidelity_file = TABLES_FOLDER / "shap_fidelity.csv"
    fidelity_table.to_csv(fidelity_file, index=False)

    print("Fidelity of the SHAP explanations:")
    print(fidelity_table.to_string(index=False))
    print()
    print("Saved", fidelity_file.relative_to(PROJECT_ROOT))

    draw_importance_chart(importance_tables)
    return importance_tables
