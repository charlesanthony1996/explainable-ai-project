"""
Training pathway, block 1B, step 22: sweep on the test set.

This is the step that answers the project's question:
"what is the largest loan this applicant would likely be approved for?"

For each applicant, the calibrated model is asked about many loan amounts.
Because the model is monotonic (checked in step 20), approval only goes down
as the amount goes up, so the answer is simply:

    the highest amount where P(approve) is still at or above the threshold

Thresholds 0.7, 0.8 and 0.9 are all reported (0.8 is the default).

Which amounts are tried (from step 4 and step 8):
    In dataset 1, nobody asked for more than about 40 % of their income, and
    income and amount are strongly tied. Asking the model about a loan of
    3 x income would be guessing. So each applicant's amounts stay inside:
        - the training range of loan amounts, AND
        - the training range of loan-to-income (1st to 99th percentile)
    These limits come from the TRAINING part only and are saved, so inference
    uses exactly the same ones.

Upper limit:
    If an applicant still passes the threshold at the highest amount we may try,
    the maximum is reported as that limit and flagged (max_is_upper_limit),
    because the real maximum may be higher than the data can show.

Consistency check on the test part:
    approved applicants  -> their requested amount should be at or below their maximum
    rejected applicants  -> no maximum found, or their requested amount is above it
Because the model is monotonic, this is the same as asking whether
P(approve) at the requested amount is above the threshold, so it is reported
as a check that the sweep and the model agree, not as a new accuracy measure.

Put this file in:  src/explainable_ai_project/amount_sweep.py

Writes:
    models/sweep_limits_dataset_X.json            amount and loan-to-income limits (used at inference)
    results/tables/sweep_test_dataset_X.csv       one row per test applicant
    results/tables/sweep_summary.csv              per dataset and threshold
    results/figures/sweep_maximum_amounts.png     histogram of maximum amounts at 0.8
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to files without opening a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from explainable_ai_project.calibrate import load_calibrated_model
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
THRESHOLDS = [0.7, 0.8, 0.9]
DEFAULT_THRESHOLD = 0.8

AMOUNT_STEPS = 100            # amounts tried per applicant
LOW_PERCENTILE = 0.01         # loan-to-income limits: 1st ...
HIGH_PERCENTILE = 0.99        # ... to 99th percentile of the training part


# ------------------------------------------------------------------
# Limits: which amounts may be tried (from the training part only)
# ------------------------------------------------------------------

def sweep_limits_file(dataset_name):
    return MODELS_FOLDER / ("sweep_limits_" + dataset_name + ".json")


def compute_sweep_limits(train_rows):
    amounts = train_rows["loan_amount"]
    ratios = train_rows["loan_to_income"]

    limits = {
        "lowest_amount": float(amounts.min()),
        "highest_amount": float(amounts.max()),
        "lowest_loan_to_income": float(ratios.quantile(LOW_PERCENTILE)),
        "highest_loan_to_income": float(ratios.quantile(HIGH_PERCENTILE)),
    }
    return limits


def save_sweep_limits(limits, dataset_name):
    MODELS_FOLDER.mkdir(parents=True, exist_ok=True)
    limits_file = sweep_limits_file(dataset_name)
    with open(limits_file, "w", encoding="utf-8") as file:
        json.dump(limits, file, indent=2)
    print("  saved", limits_file.relative_to(PROJECT_ROOT))


def load_sweep_limits(dataset_name):
    """Read the saved limits (used at inference)."""
    limits_file = sweep_limits_file(dataset_name)
    with open(limits_file, "r", encoding="utf-8") as file:
        limits = json.load(file)
    return limits


def amount_range_per_applicant(rows, limits):
    """Lowest and highest amount to try for each applicant.

    lowest  = the larger of: the smallest training amount, and income x lowest loan-to-income
    highest = the smaller of: the largest training amount, and income x highest loan-to-income
    """

    incomes = rows["annual_income"].to_numpy()

    low_by_ratio = incomes * limits["lowest_loan_to_income"]
    high_by_ratio = incomes * limits["highest_loan_to_income"]

    lows = np.maximum(low_by_ratio, limits["lowest_amount"])
    highs = np.minimum(high_by_ratio, limits["highest_amount"])

    # Very low or very high incomes can make the range empty: then try one amount only
    range_is_empty = highs < lows
    highs = np.where(range_is_empty, lows, highs)

    return lows, highs


# ------------------------------------------------------------------
# The sweep (also used at inference for a single applicant)
# ------------------------------------------------------------------

def sweep_applicants(rows, model, preprocessor, feature_names, limits):
    """Run the sweep for many applicants at once.

    Returns a table with, per applicant:
        sweep_lowest_amount, sweep_highest_amount   the range that was tried
        max_amount_0.7, max_amount_0.8, max_amount_0.9
            the highest tried amount with P(approve) >= threshold
            (empty if not even the lowest amount reaches the threshold)
    """

    lows, highs = amount_range_per_applicant(rows, limits)
    applicant_count = len(rows)

    # Start with "no amount found" for every applicant and threshold
    max_amounts = {}
    for threshold in THRESHOLDS:
        max_amounts[threshold] = np.full(applicant_count, np.nan)

    for step in range(AMOUNT_STEPS):
        share_of_range = step / (AMOUNT_STEPS - 1)
        amounts = lows + (highs - lows) * share_of_range

        features = make_features_for_amount(rows, amounts, preprocessor, feature_names)
        probability_table = model.predict_proba(features)
        probabilities = probability_table[:, 1]

        # Amounts go up step by step, so the last amount that still passes is the maximum
        for threshold in THRESHOLDS:
            passes = probabilities >= threshold
            max_amounts[threshold] = np.where(passes, amounts, max_amounts[threshold])

    result = pd.DataFrame(index=rows.index)
    result["sweep_lowest_amount"] = np.round(lows, 0)
    result["sweep_highest_amount"] = np.round(highs, 0)
    for threshold in THRESHOLDS:
        column = "max_amount_" + str(threshold)
        result[column] = np.round(max_amounts[threshold], 0)

    # True if the maximum is the highest amount we were allowed to try:
    # the real maximum may be higher, but the data cannot tell us
    default_maximum = max_amounts[DEFAULT_THRESHOLD]
    reached_upper_limit = default_maximum >= highs
    result["max_is_upper_limit"] = reached_upper_limit

    return result


def estimate_max_amount(applicant_row, dataset_name):
    """Inference helper: the sweep for ONE applicant (a table with one row).

    Uses the saved calibrated model, preprocessor and limits of that dataset.
    """

    model = load_calibrated_model(dataset_name)
    preprocessor = read_preprocessor(dataset_name)
    feature_names = read_feature_names(dataset_name)
    limits = load_sweep_limits(dataset_name)

    result = sweep_applicants(applicant_row, model, preprocessor, feature_names, limits)
    return result


# ------------------------------------------------------------------
# The test set
# ------------------------------------------------------------------

def consistency_for(table, threshold):
    """Share of test applicants whose real decision agrees with the sweep at this threshold."""

    column = "max_amount_" + str(threshold)
    max_amounts = table[column]
    requested = table["loan_amount"]
    really_approved = table["approved"] == 1

    no_amount_found = max_amounts.isna()
    within_maximum = requested <= max_amounts   # False when no amount was found

    approved_and_within = really_approved & within_maximum
    rejected_and_not_within = (~really_approved) & (no_amount_found | ~within_maximum)

    agrees = approved_and_within | rejected_and_not_within
    share = agrees.mean()
    return round(share, 4)


def summary_rows_for(table, dataset_name):
    rows = []
    for threshold in THRESHOLDS:
        column = "max_amount_" + str(threshold)
        max_amounts = table[column]

        no_amount_found = max_amounts.isna()
        share_without_amount = no_amount_found.mean()

        found_amounts = max_amounts.dropna()
        if len(found_amounts) > 0:
            median_amount = found_amounts.median()
        else:
            median_amount = np.nan

        # Among applicants who got a maximum: how many hit the upper limit of the sweep
        if threshold == DEFAULT_THRESHOLD and len(found_amounts) > 0:
            found_rows = table[~no_amount_found]
            share_at_limit = found_rows["max_is_upper_limit"].mean()
            share_at_limit = round(share_at_limit, 4)
        else:
            share_at_limit = np.nan

        rows.append({
            "dataset": dataset_name,
            "threshold": threshold,
            "share_with_no_amount": round(share_without_amount, 4),
            "share_of_maximums_at_upper_limit": share_at_limit,
            "median_max_amount": round(median_amount, 0),
            "median_requested_amount": round(table["loan_amount"].median(), 0),
            "consistency": consistency_for(table, threshold),
        })
    return rows


def sweep_one_test_set(dataset_name):
    print(dataset_name)

    model = load_calibrated_model(dataset_name)
    preprocessor = read_preprocessor(dataset_name)
    feature_names = read_feature_names(dataset_name)

    train = load_split(dataset_name, "train")
    test = load_split(dataset_name, "test")

    # Limits from the training part only
    limits = compute_sweep_limits(train["rows"])
    print("  loan-to-income tried:", round(limits["lowest_loan_to_income"], 3), "to",
          round(limits["highest_loan_to_income"], 3))
    save_sweep_limits(limits, dataset_name)

    # Probability at the amount each applicant really requested
    probability_table = model.predict_proba(test["features"])
    probabilities_at_requested = probability_table[:, 1]

    # The sweep
    sweep_result = sweep_applicants(test["rows"], model, preprocessor, feature_names, limits)

    table = pd.DataFrame()
    table["applicant_id"] = test["rows"]["applicant_id"]
    table["age_group"] = test["rows"]["age_group"]
    table["annual_income"] = test["rows"]["annual_income"]
    table["loan_amount"] = test["rows"]["loan_amount"]
    table["approved"] = test["rows"]["approved"]
    table["p_approve_at_requested"] = np.round(probabilities_at_requested, 4)
    table = pd.concat([table, sweep_result], axis=1)

    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)
    table_file = TABLES_FOLDER / ("sweep_test_" + dataset_name + ".csv")
    table.to_csv(table_file, index=False)
    print("  saved", table_file.relative_to(PROJECT_ROOT))
    print()

    return table


def draw_histograms(tables):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    column = "max_amount_" + str(DEFAULT_THRESHOLD)

    for position in range(len(DATASET_NAMES)):
        dataset_name = DATASET_NAMES[position]
        axis = axes[position]
        table = tables[dataset_name]

        found_amounts = table[column].dropna()
        no_amount_share = table[column].isna().mean()
        no_amount_percent = round(no_amount_share * 100)

        axis.hist(found_amounts, bins=40, color="#3B6FD8", alpha=0.85, label="estimated maximum")
        axis.hist(table["loan_amount"], bins=40, histtype="step", color="#C0392B", linewidth=1.5,
                  label="amount really requested")

        title = dataset_name + ": maximum at P >= " + str(DEFAULT_THRESHOLD)
        title = title + " (" + str(no_amount_percent) + " % get no amount)"
        axis.set_title(title)
        axis.set_xlabel("Loan amount")
        axis.set_ylabel("Test applicants")
        axis.legend(loc="best")

    figure.tight_layout()
    FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)
    figure_file = FIGURES_FOLDER / "sweep_maximum_amounts.png"
    figure.savefig(figure_file, dpi=150)
    plt.close(figure)
    print("Saved", figure_file.relative_to(PROJECT_ROOT))


def sweep_on_test_set():
    """Step 22: estimate the maximum amount for every test applicant."""

    tables = {}
    summary_rows = []

    for dataset_name in DATASET_NAMES:
        table = sweep_one_test_set(dataset_name)
        tables[dataset_name] = table

        rows = summary_rows_for(table, dataset_name)
        summary_rows.extend(rows)

    summary_table = pd.DataFrame(summary_rows)
    summary_file = TABLES_FOLDER / "sweep_summary.csv"
    summary_table.to_csv(summary_file, index=False)

    print("Sweep summary (test part):")
    print(summary_table.to_string(index=False))
    print()
    print("Saved", summary_file.relative_to(PROJECT_ROOT))

    draw_histograms(tables)
    return tables