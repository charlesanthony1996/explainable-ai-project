"""
Training pathway, block 1C, step 25: maximum amount by age group (Topic 3: fairness).

Step 24 showed that the models treat age groups differently. This step asks
two questions about the MAXIMUM AMOUNT (from the step 22 sweep, threshold 0.8):

1. How different is it per age group?
       share who get any amount, median and mean maximum,
       share whose maximum is the upper limit of the sweep

2. Is the difference explained by income or credit score?
   Older applicants may simply earn less or have lower credit scores.
   To check, applicants are put into fifths ("quintiles") of income, of credit
   score, and of both. Then each age group's rate is recalculated as if every
   age group had the SAME mix of incomes / credit scores as the whole test part
   ("adjusted rate"). If the gap shrinks a lot after adjusting, income or
   credit score explains it. If it stays, something else does (possibly age).

   The adjusted rate of one age group =
       sum over the fifths of: (share of ALL applicants in that fifth)
                               x (that age group's rate inside that fifth)
   Fifths where an age group has no applicants are left out, and the shares
   of the remaining fifths are scaled up so they still add up to 1.

Put this file in:  src/explainable_ai_project/fairness_max_amount.py

Reads:
    results/tables/sweep_test_dataset_X.csv   (from step 22)

Writes:
    results/tables/max_amount_by_age_group.csv
    results/tables/max_amount_adjusted.csv
    results/figures/max_amount_by_age_group.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to files without opening a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from explainable_ai_project.load_split import load_split


# ------------------------------------------------------------------
# Where the files are
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PACKAGE_FOLDER = THIS_FILE.parent
SRC_FOLDER = PACKAGE_FOLDER.parent
PROJECT_ROOT = SRC_FOLDER.parent

TABLES_FOLDER = PROJECT_ROOT / "results" / "tables"
FIGURES_FOLDER = PROJECT_ROOT / "results" / "figures"

DATASET_NAMES = ["dataset_1", "dataset_2"]
AGE_GROUP_ORDER = ["18-30", "31-45", "46-60", "61+"]
THRESHOLD = 0.8
MAX_COLUMN = "max_amount_" + str(THRESHOLD)
FIFTHS = 5

MODEL_LABELS = {
    "dataset_1": "Model A (dataset 1)",
    "dataset_2": "Model B (dataset 2)",
}

# How the rate is adjusted: name -> which fifths are used
ADJUSTMENTS = {
    "not adjusted": [],
    "same income mix": ["income_fifth"],
    "same credit score mix": ["credit_score_fifth"],
    "same income and credit score mix": ["income_fifth", "credit_score_fifth"],
}


# ------------------------------------------------------------------
# Reading the sweep results
# ------------------------------------------------------------------

def read_sweep_table(dataset_name):
    """Sweep results of step 22, plus credit score from the test rows."""

    sweep_file = TABLES_FOLDER / ("sweep_test_" + dataset_name + ".csv")
    file_exists = sweep_file.exists()
    if not file_exists:
        raise FileNotFoundError("Missing " + sweep_file.name + ". Run step 22 (sweep on the test set) first.")

    sweep_table = pd.read_csv(sweep_file)

    test = load_split(dataset_name, "test")
    credit_scores = test["rows"][["applicant_id", "credit_score"]]
    table = sweep_table.merge(credit_scores, on="applicant_id", how="left")

    has_amount = table[MAX_COLUMN].notna()
    table["gets_an_amount"] = has_amount.astype(int)

    # Fifths of income and credit score, from the lowest (1) to the highest (5)
    income_fifths = pd.qcut(table["annual_income"], FIFTHS, labels=False, duplicates="drop")
    credit_fifths = pd.qcut(table["credit_score"], FIFTHS, labels=False, duplicates="drop")
    table["income_fifth"] = income_fifths + 1
    table["credit_score_fifth"] = credit_fifths + 1

    return table


# ------------------------------------------------------------------
# 1. Plain comparison per age group
# ------------------------------------------------------------------

def summary_per_age_group(table, dataset_name):
    rows = []
    for group in AGE_GROUP_ORDER:
        in_group = table["age_group"] == group
        group_rows = table[in_group]

        found_rows = group_rows[group_rows["gets_an_amount"] == 1]
        found_amounts = found_rows[MAX_COLUMN]

        if len(found_rows) > 0:
            median_maximum = round(found_amounts.median(), 0)
            mean_maximum = round(found_amounts.mean(), 0)
            share_at_limit = round(found_rows["max_is_upper_limit"].mean(), 3)
        else:
            median_maximum = np.nan
            mean_maximum = np.nan
            share_at_limit = np.nan

        share_with_amount = group_rows["gets_an_amount"].mean()
        median_income = group_rows["annual_income"].median()
        median_credit_score = group_rows["credit_score"].median()

        rows.append({
            "dataset": dataset_name,
            "age_group": group,
            "applicants": len(group_rows),
            "share_getting_an_amount": round(share_with_amount, 3),
            "median_maximum": median_maximum,
            "mean_maximum": mean_maximum,
            "share_of_maximums_at_upper_limit": share_at_limit,
            "median_income": round(median_income, 0),
            "median_credit_score": round(median_credit_score, 0),
        })

    summary = pd.DataFrame(rows)
    return summary


# ------------------------------------------------------------------
# 2. Adjusted rates (same income / credit score mix for every age group)
# ------------------------------------------------------------------

def make_cell_labels(table, fifth_columns):
    """One label per applicant naming their cell, e.g. 'income 2 | credit 5'."""

    if len(fifth_columns) == 0:
        labels = pd.Series("everyone", index=table.index)
        return labels

    labels = pd.Series("", index=table.index)
    for column in fifth_columns:
        column_text = column + " " + table[column].astype(str)
        labels = labels + column_text + " | "
    return labels


def adjusted_rate(table, group, fifth_columns):
    """Share of the age group getting an amount, as if it had the overall mix of cells."""

    cell_labels = make_cell_labels(table, fifth_columns)

    overall_shares = cell_labels.value_counts(normalize=True)

    in_group = table["age_group"] == group
    group_cells = cell_labels[in_group]
    group_outcomes = table.loc[in_group, "gets_an_amount"]

    rate_per_cell = group_outcomes.groupby(group_cells).mean()

    weighted_sum = 0.0
    weight_total = 0.0
    for cell, rate in rate_per_cell.items():
        weight = overall_shares[cell]
        weighted_sum = weighted_sum + weight * rate
        weight_total = weight_total + weight

    if weight_total == 0:
        return np.nan

    rate = weighted_sum / weight_total
    return rate


def adjusted_table_for(table, dataset_name):
    rows = []
    for adjustment_name, fifth_columns in ADJUSTMENTS.items():
        rates = {}
        for group in AGE_GROUP_ORDER:
            rate = adjusted_rate(table, group, fifth_columns)
            rates[group] = rate

        highest_rate = max(rates.values())
        lowest_rate = min(rates.values())
        gap = highest_rate - lowest_rate

        for group in AGE_GROUP_ORDER:
            rows.append({
                "dataset": dataset_name,
                "adjustment": adjustment_name,
                "age_group": group,
                "share_getting_an_amount": round(rates[group], 3),
                "gap_between_groups": round(gap, 3),
            })

    adjusted = pd.DataFrame(rows)
    return adjusted


# ------------------------------------------------------------------
# Figure
# ------------------------------------------------------------------

def draw_figure(adjusted):
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    adjustment_names = list(ADJUSTMENTS.keys())
    colours = ["#9AA5B1", "#3B6FD8", "#D9822B", "#2F8F5B"]

    bar_count = len(adjustment_names)
    bar_width = 0.8 / bar_count
    positions = np.arange(len(AGE_GROUP_ORDER))

    for dataset_position in range(len(DATASET_NAMES)):
        dataset_name = DATASET_NAMES[dataset_position]
        axis = axes[dataset_position]
        is_dataset = adjusted["dataset"] == dataset_name

        for bar_number in range(bar_count):
            adjustment_name = adjustment_names[bar_number]
            is_adjustment = adjusted["adjustment"] == adjustment_name
            rows = adjusted[is_dataset & is_adjustment]
            rates = rows["share_getting_an_amount"].to_numpy()

            offset = (bar_number - (bar_count - 1) / 2) * bar_width
            axis.bar(positions + offset, rates, bar_width, color=colours[bar_number], label=adjustment_name)

        axis.set_xticks(positions)
        axis.set_xticklabels(AGE_GROUP_ORDER)
        axis.set_xlabel("Age group")
        axis.set_title(MODEL_LABELS[dataset_name] + ", test part")
        axis.set_ylim(0, 1)
        axis.legend(loc="upper right", fontsize=8)

    axes[0].set_ylabel("Share getting a maximum amount (P >= " + str(THRESHOLD) + ")")
    figure.suptitle("Does income or credit score explain the age gap?")
    figure.tight_layout()

    FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)
    figure_file = FIGURES_FOLDER / "max_amount_by_age_group.png"
    figure.savefig(figure_file, dpi=150)
    plt.close(figure)
    print("Saved", figure_file.relative_to(PROJECT_ROOT))


# ------------------------------------------------------------------
# The step
# ------------------------------------------------------------------

def maximum_amount_by_age_group():
    """Step 25: maximum amount per age group, and whether income or credit score explain the gap."""

    summaries = []
    adjusted_tables = []

    for dataset_name in DATASET_NAMES:
        print(dataset_name + ":", MODEL_LABELS[dataset_name])
        table = read_sweep_table(dataset_name)

        summary = summary_per_age_group(table, dataset_name)
        adjusted = adjusted_table_for(table, dataset_name)
        summaries.append(summary)
        adjusted_tables.append(adjusted)

        columns_to_show = [
            "age_group",
            "applicants",
            "share_getting_an_amount",
            "median_maximum",
            "median_income",
            "median_credit_score",
        ]
        print(summary[columns_to_show].to_string(index=False))

        print("  gap in 'share getting an amount' between the highest and lowest age group:")
        for adjustment_name in ADJUSTMENTS:
            is_adjustment = adjusted["adjustment"] == adjustment_name
            adjustment_rows = adjusted[is_adjustment]
            gap = adjustment_rows["gap_between_groups"].iloc[0]
            print("    " + adjustment_name + ":", gap)
        print()

    summary_table = pd.concat(summaries, ignore_index=True)
    adjusted_table = pd.concat(adjusted_tables, ignore_index=True)

    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)
    summary_file = TABLES_FOLDER / "max_amount_by_age_group.csv"
    adjusted_file = TABLES_FOLDER / "max_amount_adjusted.csv"
    summary_table.to_csv(summary_file, index=False)
    adjusted_table.to_csv(adjusted_file, index=False)
    print("Saved", summary_file.relative_to(PROJECT_ROOT))
    print("Saved", adjusted_file.relative_to(PROJECT_ROOT))

    draw_figure(adjusted_table)
    return summary_table, adjusted_table