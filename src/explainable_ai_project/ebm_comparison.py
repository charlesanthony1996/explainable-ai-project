"""
Training pathway, block 1B, step 18: EBM comparison (Topic 4: performance versus explainability).

An Explainable Boosting Machine (EBM) is a GAM, a "glass-box" model:
its prediction is just a SUM of one curve per feature,

    approval score = f1(age) + f2(income) + f3(credit score) + ... + f11(loan purpose)

so you can draw every curve and see exactly how each feature pushes the decision.
A Random Forest of hundreds of trees cannot be read like that; it needs SHAP.

This step asks: do we lose accuracy by using the readable model?
    1. train an EBM on each dataset, on the same training rows as the Random Forests
       - with the same rule: loan_amount and loan_to_income may only LOWER approval
       - with no interactions between features, so it stays a pure GAM
       - on the original columns (it handles categories itself, no one-hot needed)
    2. score it on validation with the same metrics as every other model
    3. check the amount sweep, like steps 15 and 17
    4. draw the shape curves for the four most important number features
    5. put all models side by side: accuracy and how explainable each one is

Put this file in:  src/explainable_ai_project/ebm_comparison.py

Writes:
    models/ebm_dataset_X_v1.joblib
    results/figures/ebm_shapes_dataset_X.png      the curves the EBM learned
    results/tables/ebm_importance_dataset_X.csv   how much each feature matters
    results/tables/ebm_comparison.csv             all models side by side
    results/tables/model_results.csv              (adds the EBM rows)
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save figures to files without opening a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier

from explainable_ai_project.load_split import load_split
from explainable_ai_project.logistic_baseline import save_model_result
from explainable_ai_project.logistic_baseline import score_predictions
from explainable_ai_project.sweep_problem import amount_grid


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
MODEL_RESULTS_FILE = TABLES_FOLDER / "model_results.csv"

SEED = 42
VERSION = "v1"
DATASET_NAMES = ["dataset_1", "dataset_2"]
MODEL_NAME = "ebm_monotonic"

# The EBM uses the original columns (before one-hot encoding)
INPUT_COLUMNS = [
    "age",
    "dependents",
    "annual_income",
    "credit_score",
    "loan_amount",
    "loan_term_months",
    "loan_to_income",
    "marital_status",
    "education",
    "employment_status",
    "loan_purpose",
]

DECREASING_FEATURES = ["loan_amount", "loan_to_income"]

# The four number features whose curves are drawn
FEATURES_TO_DRAW = ["loan_amount", "loan_to_income", "credit_score", "annual_income"]

APPLICANT_COUNT = 300

# How each model can be explained (for the comparison table)
EXPLAINABILITY = {
    "logistic_regression": "glass box: one weight per feature, straight-line effects only",
    "random_forest_unconstrained": "black box: needs SHAP; can break the amount rule",
    "random_forest_monotonic": "black box: needs SHAP; follows the amount rule",
    "ebm_monotonic": "glass box: one curve per feature, follows the amount rule",
}


# ------------------------------------------------------------------
# The EBM
# ------------------------------------------------------------------

def build_ebm():
    constraints = []
    for column in INPUT_COLUMNS:
        if column in DECREASING_FEATURES:
            constraints.append(-1)
        else:
            constraints.append(0)

    model = ExplainableBoostingClassifier(
        interactions=0,                  # no feature pairs: a pure GAM, every effect is one curve
        monotone_constraints=constraints,
        random_state=SEED,
        n_jobs=-1,
    )
    return model


def ebm_inputs(rows):
    """The original columns the EBM reads."""
    inputs = rows[INPUT_COLUMNS]
    return inputs


def check_ebm_sweep(model, train_rows, validation_rows):
    """Change only the amount for 300 applicants; count how many curves rise."""

    sample_rows = validation_rows.sample(n=APPLICANT_COUNT, random_state=SEED)
    grid = amount_grid(train_rows)

    curves = {}
    for amount in grid:
        changed_rows = sample_rows.copy()
        changed_rows["loan_amount"] = amount
        incomes = changed_rows["annual_income"]
        changed_rows["loan_to_income"] = amount / incomes

        probabilities = model.predict_proba(ebm_inputs(changed_rows))
        approval_probabilities = probabilities[:, 1]
        curves[amount] = approval_probabilities

    curve_table = pd.DataFrame(curves).T

    step_changes = curve_table.diff()
    went_up = step_changes > 0.01
    rises_anywhere = went_up.any(axis=0)
    share_rising = rises_anywhere.sum() / APPLICANT_COUNT
    return round(share_rising, 3)


# ------------------------------------------------------------------
# What the EBM learned
# ------------------------------------------------------------------

def importance_table(model):
    """Average absolute contribution of each feature: how much it moves predictions."""

    names = model.term_names_
    importances = model.term_importances()

    rows = []
    for position in range(len(names)):
        rows.append({
            "feature": names[position],
            "importance": round(importances[position], 4),
        })

    table = pd.DataFrame(rows)
    table = table.sort_values("importance", ascending=False)
    table = table.reset_index(drop=True)
    return table


def draw_shapes(model, dataset_name, train_rows):
    """One panel per feature: the curve the EBM adds to the approval score."""

    explanation = model.explain_global()
    term_names = list(model.term_names_)

    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    axes_list = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]

    for position in range(len(FEATURES_TO_DRAW)):
        feature = FEATURES_TO_DRAW[position]
        axis = axes_list[position]

        term_number = term_names.index(feature)
        curve_data = explanation.data(term_number)

        bin_edges = np.array(curve_data["names"], dtype=float)
        scores = np.array(curve_data["scores"], dtype=float)

        # Draw only the range where 1 % to 99 % of training applicants are
        lowest_shown = train_rows[feature].quantile(0.01)
        highest_shown = train_rows[feature].quantile(0.99)

        axis.stairs(scores, bin_edges, color="#3B6FD8", linewidth=2)
        axis.axhline(0, color="#9AA5B1", linewidth=0.8)
        axis.set_xlim(lowest_shown, highest_shown)
        axis.set_title(feature)
        axis.set_xlabel(feature)
        axis.set_ylabel("effect on approval score")

    figure.suptitle(dataset_name + ": what the EBM learned (above 0 raises approval, below 0 lowers it)")
    figure.tight_layout()

    FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)
    figure_file = FIGURES_FOLDER / ("ebm_shapes_" + dataset_name + ".png")
    figure.savefig(figure_file, dpi=150)
    plt.close(figure)
    print("  saved", figure_file.relative_to(PROJECT_ROOT))


# ------------------------------------------------------------------
# One dataset
# ------------------------------------------------------------------

def train_one_ebm(dataset_name):
    print(dataset_name)

    train = load_split(dataset_name, "train")
    validation = load_split(dataset_name, "validation")

    # 1. Train
    model = build_ebm()
    model.fit(ebm_inputs(train["rows"]), train["labels"])

    # 2. Score on validation
    probabilities = model.predict_proba(ebm_inputs(validation["rows"]))
    approval_probabilities = probabilities[:, 1]
    scores = score_predictions(validation["labels"], approval_probabilities)
    save_model_result(MODEL_NAME, dataset_name, "validation", scores)
    print("  validation:", scores)

    # 3. Amount sweep check
    share_rising = check_ebm_sweep(model, train["rows"], validation["rows"])
    print("  applicants whose approval rises at least once as the amount grows:", share_rising)

    # 4. What it learned
    importances = importance_table(model)
    print("  most important features:")
    top_five = importances.head(5)
    print(top_five.to_string(index=False))

    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)
    importance_file = TABLES_FOLDER / ("ebm_importance_" + dataset_name + ".csv")
    importances.to_csv(importance_file, index=False)
    print("  saved", importance_file.relative_to(PROJECT_ROOT))

    draw_shapes(model, dataset_name, train["rows"])

    MODELS_FOLDER.mkdir(parents=True, exist_ok=True)
    model_file = MODELS_FOLDER / ("ebm_" + dataset_name + "_" + VERSION + ".joblib")
    joblib.dump(model, model_file)
    print("  saved", model_file.relative_to(PROJECT_ROOT))
    print()

    return model


# ------------------------------------------------------------------
# All models side by side
# ------------------------------------------------------------------

def comparison_table():
    """Validation scores of every model so far, with how each one can be explained."""

    results = pd.read_csv(MODEL_RESULTS_FILE)

    is_validation = results["split"] == "validation"
    validation_results = results[is_validation].copy()

    explainability_texts = []
    for model_name in validation_results["model"]:
        text = EXPLAINABILITY.get(model_name, "")
        explainability_texts.append(text)
    validation_results["explainability"] = explainability_texts

    columns_to_keep = ["dataset", "model", "auc", "f1", "brier", "explainability"]
    table = validation_results[columns_to_keep]
    table = table.sort_values(["dataset", "auc"], ascending=[True, False])
    table = table.reset_index(drop=True)
    return table


def run_ebm_comparison():
    """Step 18: train an EBM on both datasets and compare it with the other models."""

    models = {}
    for dataset_name in DATASET_NAMES:
        model = train_one_ebm(dataset_name)
        models[dataset_name] = model

    table = comparison_table()
    print("All models on the validation part:")
    print(table.to_string(index=False))

    comparison_file = TABLES_FOLDER / "ebm_comparison.csv"
    table.to_csv(comparison_file, index=False)
    print("Saved", comparison_file.relative_to(PROJECT_ROOT))

    return models