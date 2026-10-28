"""
Training pathway, block 1B, step 20: re-check monotonicity.

Step 15 made the forests monotonic: a larger loan can never raise approval.
Step 19 then put a calibration correction on top. That correction should keep
the rule, but "should" is not proof, and the whole amount sweep depends on it.

This step proves it on the FINAL calibrated models A and B:
    1. take 500 validation applicants (the test part stays untouched)
    2. change only their loan amount, in 50 steps across the training range
       (loan-to-income is recomputed, everything else stays the same)
    3. check that every applicant's approval probability never goes up
       from one step to the next

If even one curve rises, the step stops with an error, because the
maximum-amount estimate would then not be trustworthy.

Put this file in:  src/explainable_ai_project/recheck_monotonicity.py

Writes:
    results/tables/monotonicity_check.csv    one row per model: applicants checked,
                                             curves that rose, largest rise found
"""

from pathlib import Path

import numpy as np
import pandas as pd

from explainable_ai_project.calibrate import load_calibrated_model
from explainable_ai_project.load_split import load_split
from explainable_ai_project.load_split import read_feature_names
from explainable_ai_project.load_split import read_preprocessor
from explainable_ai_project.sweep_problem import probability_curves


# ------------------------------------------------------------------
# Where the files go
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PACKAGE_FOLDER = THIS_FILE.parent
SRC_FOLDER = PACKAGE_FOLDER.parent
PROJECT_ROOT = SRC_FOLDER.parent

TABLES_FOLDER = PROJECT_ROOT / "results" / "tables"

SEED = 42
DATASET_NAMES = ["dataset_1", "dataset_2"]
MODEL_LABELS = {
    "dataset_1": "calibrated model A",
    "dataset_2": "calibrated model B",
}

APPLICANT_COUNT = 500
AMOUNT_STEPS = 50

# Differences smaller than this are rounding noise in the computer, not a real rise
ROUNDING_TOLERANCE = 1e-9


def fine_amount_grid(train_rows):
    """50 evenly spaced amounts from the smallest to the largest training amount."""

    lowest_amount = train_rows["loan_amount"].min()
    highest_amount = train_rows["loan_amount"].max()
    grid = np.linspace(lowest_amount, highest_amount, AMOUNT_STEPS)
    return grid


def find_rises(curve_table):
    """Look at every step of every applicant's curve.

    curve_table has one row per amount and one column per applicant.
    Returns: how many applicants rise at least once, and the largest single rise.
    """

    step_changes = curve_table.diff()
    step_changes = step_changes.iloc[1:]   # the first row has no previous amount

    is_rise = step_changes > ROUNDING_TOLERANCE
    applicant_rises = is_rise.any(axis=0)
    rising_applicant_count = int(applicant_rises.sum())

    largest_rise = step_changes.max().max()
    if largest_rise < 0:
        largest_rise = 0.0

    return rising_applicant_count, largest_rise


def check_one_model(dataset_name):
    model_label = MODEL_LABELS[dataset_name]
    print(dataset_name + ":", model_label)

    model = load_calibrated_model(dataset_name)

    train = load_split(dataset_name, "train")
    validation = load_split(dataset_name, "validation")
    preprocessor = read_preprocessor(dataset_name)
    feature_names = read_feature_names(dataset_name)

    sample_rows = validation["rows"].sample(n=APPLICANT_COUNT, random_state=SEED)
    grid = fine_amount_grid(train["rows"])

    curve_table = probability_curves(model, sample_rows, grid, preprocessor, feature_names)
    rising_applicant_count, largest_rise = find_rises(curve_table)

    lowest_amount = round(grid[0])
    highest_amount = round(grid[-1])
    print("  checked", APPLICANT_COUNT, "applicants at", AMOUNT_STEPS, "amounts from",
          lowest_amount, "to", highest_amount)
    print("  curves that rise at least once:", rising_applicant_count)
    print("  largest single rise:", round(largest_rise, 6))

    passed = rising_applicant_count == 0
    if passed:
        print("  PASS: approval never rises as the loan amount grows")
    else:
        print("  FAIL: some approval probabilities rise with the loan amount")
    print()

    result_row = {
        "dataset": dataset_name,
        "model": model_label,
        "applicants_checked": APPLICANT_COUNT,
        "amounts_per_applicant": AMOUNT_STEPS,
        "lowest_amount": lowest_amount,
        "highest_amount": highest_amount,
        "curves_that_rise": rising_applicant_count,
        "largest_rise": round(largest_rise, 6),
        "passed": passed,
    }
    return result_row


def recheck_monotonicity():
    """Step 20: prove both calibrated models still follow the amount rule.

    Stops with an error if either model fails, because the amount sweep
    (step 22) would then give untrustworthy maximum amounts.
    """

    result_rows = []
    for dataset_name in DATASET_NAMES:
        result_row = check_one_model(dataset_name)
        result_rows.append(result_row)

    result_table = pd.DataFrame(result_rows)

    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)
    result_file = TABLES_FOLDER / "monotonicity_check.csv"
    result_table.to_csv(result_file, index=False)
    print("Saved", result_file.relative_to(PROJECT_ROOT))

    all_passed = result_table["passed"].all()
    if not all_passed:
        message = "Monotonicity check failed. See results/tables/monotonicity_check.csv. "
        message = message + "Check the constraint (step 15) and the calibration (step 19) before sweeping."
        raise AssertionError(message)

    print("Both calibrated models passed: the amount sweep is safe to use.")
    return result_table