"""
Training pathway, block 1B, step 14: show the sweep problem.

The project estimates the maximum loan amount by "sweeping": take one applicant,
try many loan amounts, and find the highest amount the model still approves.
That only works if asking for MORE money never makes approval MORE likely.

This step shows that an ordinary (unconstrained) Random Forest breaks that rule:
    1. train a normal Random Forest on each dataset
    2. take 300 validation applicants
    3. change ONLY their loan amount (and loan-to-income, which depends on it)
       across the training range, and ask the model each time
    4. plot the average approval probability against the amount

In dataset 1 the curve goes UP with the amount, so a sweep would always answer
"the largest amount we tried". Step 15 fixes this with a monotonic constraint.

The helper make_features_for_amount() is reused by step 15 and the sweep in step 22.

Put this file in:  src/explainable_ai_project/sweep_problem.py

Writes:
    models/rf_unconstrained_dataset_X_v1.joblib
    results/figures/sweep_problem.png
    results/tables/sweep_problem.csv     average probability per amount
    results/tables/model_results.csv     (adds the unconstrained Random Forest rows)
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save figures to files without opening a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from explainable_ai_project.load_split import load_split
from explainable_ai_project.load_split import read_feature_names
from explainable_ai_project.load_split import read_preprocessor
from explainable_ai_project.logistic_baseline import save_model_result
from explainable_ai_project.logistic_baseline import score_predictions


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
MODEL_NAME = "random_forest_unconstrained"

APPLICANT_COUNT = 300     # validation applicants used for the curves
AMOUNT_STEPS = 30         # how many amounts to try between the training minimum and maximum
CURVES_TO_DRAW = 30       # thin single-applicant curves drawn behind the average


# ------------------------------------------------------------------
# Changing only the loan amount (reused by steps 15 and 22)
# ------------------------------------------------------------------

def make_features_for_amount(rows, amount, preprocessor, feature_names):
    """Model inputs for the same applicants, but with a different loan amount.

    loan_to_income depends on the amount, so it is recomputed too.
    Every other value stays exactly as it was.
    """

    changed_rows = rows.copy()
    changed_rows["loan_amount"] = amount

    incomes = changed_rows["annual_income"]
    changed_rows["loan_to_income"] = amount / incomes

    feature_values = preprocessor.transform(changed_rows)
    features = pd.DataFrame(feature_values, columns=feature_names, index=rows.index)
    return features


def amount_grid(train_rows):
    """Evenly spaced amounts from the smallest to the largest amount in the training part."""

    lowest_amount = train_rows["loan_amount"].min()
    highest_amount = train_rows["loan_amount"].max()
    grid = np.linspace(lowest_amount, highest_amount, AMOUNT_STEPS)
    return grid


def probability_curves(model, rows, grid, preprocessor, feature_names):
    """Approval probability for every applicant at every amount.

    Returns a table: one row per amount, one column per applicant.
    """

    curves = {}
    for amount in grid:
        features = make_features_for_amount(rows, amount, preprocessor, feature_names)
        probabilities = model.predict_proba(features)
        approval_probabilities = probabilities[:, 1]
        curves[amount] = approval_probabilities

    curve_table = pd.DataFrame(curves).T
    return curve_table


def count_rising_curves(curve_table):
    """How many applicants get a HIGHER approval probability for a larger amount.

    ends_higher:    probability at the largest amount > probability at the smallest
    rises_anywhere: probability goes up at least once between two neighbouring amounts
    """

    first_row = curve_table.iloc[0]
    last_row = curve_table.iloc[-1]
    ends_higher = last_row > first_row + 0.01

    step_changes = curve_table.diff()
    went_up = step_changes > 0.01
    rises_anywhere = went_up.any(axis=0)

    applicant_count = curve_table.shape[1]
    share_ends_higher = ends_higher.sum() / applicant_count
    share_rises_anywhere = rises_anywhere.sum() / applicant_count

    return round(share_ends_higher, 3), round(share_rises_anywhere, 3)


# ------------------------------------------------------------------
# The unconstrained Random Forest
# ------------------------------------------------------------------

def build_unconstrained_forest():
    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=SEED,
    )
    return model


def show_problem_for_one_dataset(dataset_name):
    print(dataset_name)

    train = load_split(dataset_name, "train")
    validation = load_split(dataset_name, "validation")
    preprocessor = read_preprocessor(dataset_name)
    feature_names = read_feature_names(dataset_name)

    # 1. Train an ordinary Random Forest
    model = build_unconstrained_forest()
    model.fit(train["features"], train["labels"])

    probabilities = model.predict_proba(validation["features"])
    approval_probabilities = probabilities[:, 1]
    scores = score_predictions(validation["labels"], approval_probabilities)
    save_model_result(MODEL_NAME, dataset_name, "validation", scores)
    print("  validation:", scores)

    # 2. Pick validation applicants
    sample_rows = validation["rows"].sample(n=APPLICANT_COUNT, random_state=SEED)

    # 3. Change only the amount and ask the model each time
    grid = amount_grid(train["rows"])
    curve_table = probability_curves(model, sample_rows, grid, preprocessor, feature_names)

    share_ends_higher, share_rises_anywhere = count_rising_curves(curve_table)
    print("  applicants whose approval is higher at the largest amount than at the smallest:",
          share_ends_higher)
    print("  applicants whose approval rises at least once as the amount grows:",
          share_rises_anywhere)

    # 4. Save the model
    MODELS_FOLDER.mkdir(parents=True, exist_ok=True)
    model_file = MODELS_FOLDER / ("rf_unconstrained_" + dataset_name + "_" + VERSION + ".joblib")
    joblib.dump(model, model_file)
    print("  saved", model_file.relative_to(PROJECT_ROOT))
    print()

    result = {
        "grid": grid,
        "curve_table": curve_table,
        "share_ends_higher": share_ends_higher,
        "share_rises_anywhere": share_rises_anywhere,
    }
    return result


# ------------------------------------------------------------------
# Figure and table
# ------------------------------------------------------------------

def draw_figure(results):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for position in range(len(DATASET_NAMES)):
        dataset_name = DATASET_NAMES[position]
        axis = axes[position]
        result = results[dataset_name]

        grid = result["grid"]
        curve_table = result["curve_table"]

        # Thin grey lines: single applicants
        for column_number in range(CURVES_TO_DRAW):
            single_curve = curve_table.iloc[:, column_number]
            axis.plot(grid, single_curve, color="#B8C2CC", linewidth=0.8)

        # Thick line: the average
        average_curve = curve_table.mean(axis=1)
        axis.plot(grid, average_curve, color="#C0392B", linewidth=2.5, label="average of 300 applicants")

        share_text = str(round(result["share_ends_higher"] * 100)) + " % end higher"
        axis.set_title(dataset_name + ": unconstrained Random Forest (" + share_text + ")")
        axis.set_xlabel("Requested loan amount (everything else unchanged)")
        axis.set_ylabel("Predicted approval probability")
        axis.set_ylim(0, 1)
        axis.legend(loc="best")

    figure.suptitle("The sweep problem: asking for more should never raise approval")
    figure.tight_layout()

    FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)
    figure_file = FIGURES_FOLDER / "sweep_problem.png"
    figure.savefig(figure_file, dpi=150)
    plt.close(figure)

    print("Saved", figure_file.relative_to(PROJECT_ROOT))


def save_table(results):
    rows = []
    for dataset_name in DATASET_NAMES:
        result = results[dataset_name]
        grid = result["grid"]
        average_curve = result["curve_table"].mean(axis=1)

        for position in range(len(grid)):
            amount = grid[position]
            average_probability = average_curve.iloc[position]
            rows.append({
                "dataset": dataset_name,
                "loan_amount": round(amount, 0),
                "average_approval_probability": round(average_probability, 4),
            })

    table = pd.DataFrame(rows)

    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)
    table_file = TABLES_FOLDER / "sweep_problem.csv"
    table.to_csv(table_file, index=False)
    print("Saved", table_file.relative_to(PROJECT_ROOT))


def show_sweep_problem():
    """Run the step on both datasets."""

    results = {}
    for dataset_name in DATASET_NAMES:
        result = show_problem_for_one_dataset(dataset_name)
        results[dataset_name] = result

    draw_figure(results)
    save_table(results)
    return results