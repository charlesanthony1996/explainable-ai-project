"""
Training pathway, block 1A, step 8: add loan-to-income and age groups.

Adds two new columns to each cleaned dataset:

    loan_to_income   loan amount divided by annual income.
                     A MODEL INPUT. It tells the model how big the loan is
                     for this applicant, not just in absolute money. The amount
                     sweep recomputes it for every amount it tries.

    age_group        18-30, 31-45, 46-60 or 61+.
                     NOT a model input. Used only in the Topic 3 fairness checks
                     to compare groups.

Then saves the result as the cleaned dataset that every later step reads.

Put this file in:  src/explainable_ai_project/add_features.py
Run it with:       uv run python -m explainable_ai_project.add_features

Writes:
    datasets/processed/dataset_1_clean.csv
    datasets/processed/dataset_2_clean.csv
    results/tables/loan_to_income_summary.csv
    results/tables/age_group_summary.csv
"""

from pathlib import Path

import pandas as pd

from explainable_ai_project.cleaning import check_ranges_and_clean
from explainable_ai_project.common_columns import LABEL_COLUMN
from explainable_ai_project.common_columns import keep_common_features
from explainable_ai_project.harmonize import harmonise_labels_and_units
from explainable_ai_project.load import load_both_raw_datasets


# ------------------------------------------------------------------
# Where the files go
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PACKAGE_FOLDER = THIS_FILE.parent
SRC_FOLDER = PACKAGE_FOLDER.parent
PROJECT_ROOT = SRC_FOLDER.parent

PROCESSED_FOLDER = PROJECT_ROOT / "datasets" / "processed"
TABLES_FOLDER = PROJECT_ROOT / "results" / "tables"


# ------------------------------------------------------------------
# New column names and the age groups
# ------------------------------------------------------------------
LOAN_TO_INCOME_COLUMN = "loan_to_income"
AGE_GROUP_COLUMN = "age_group"

# Each group includes its upper edge: 17 < age <= 30 is "18-30", and so on.
AGE_GROUP_EDGES = [17, 30, 45, 60, 200]
AGE_GROUP_LABELS = ["18-30", "31-45", "46-60", "61+"]


# ------------------------------------------------------------------
# The two new columns
# ------------------------------------------------------------------

def add_loan_to_income(data):
    """loan_to_income = loan_amount / annual_income"""

    result = data.copy()

    loan_amounts = result["loan_amount"]
    incomes = result["annual_income"]
    ratios = loan_amounts / incomes

    result[LOAN_TO_INCOME_COLUMN] = ratios
    return result


def add_age_group(data):
    """Put every applicant into one of the four age groups."""

    result = data.copy()

    ages = result["age"]
    groups = pd.cut(ages, bins=AGE_GROUP_EDGES, labels=AGE_GROUP_LABELS)

    # Store as plain text, so the CSV file reads back the same way
    result[AGE_GROUP_COLUMN] = groups.astype(str)
    return result


# ------------------------------------------------------------------
# Summaries for the report
# ------------------------------------------------------------------

def loan_to_income_summary(data, dataset_name):
    ratios = data[LOAN_TO_INCOME_COLUMN]

    lowest = ratios.min()
    percentile_1 = ratios.quantile(0.01)
    median = ratios.median()
    percentile_99 = ratios.quantile(0.99)
    highest = ratios.max()

    summary_row = {
        "dataset": dataset_name,
        "min": round(lowest, 3),
        "percentile_1": round(percentile_1, 3),
        "median": round(median, 3),
        "percentile_99": round(percentile_99, 3),
        "max": round(highest, 3),
    }
    return summary_row


def age_group_summary(data, dataset_name):
    """Rows and approval rate per age group."""

    rows = []
    for group in AGE_GROUP_LABELS:
        in_group = data[AGE_GROUP_COLUMN] == group
        group_data = data[in_group]

        row_count = len(group_data)
        if row_count > 0:
            approval_rate = group_data[LABEL_COLUMN].mean()
            approval_rate = round(approval_rate, 3)
        else:
            approval_rate = None

        rows.append({
            "dataset": dataset_name,
            "age_group": group,
            "rows": row_count,
            "approval_rate": approval_rate,
        })

    return rows


# ------------------------------------------------------------------
# Both datasets
# ------------------------------------------------------------------

def add_loan_to_income_and_age_groups(clean_datasets):
    """Run the step on both datasets and save the cleaned files.

    clean_datasets is the dictionary returned by check_ranges_and_clean().
    Returns a dictionary: {"dataset_1": table, "dataset_2": table}
    """

    PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)
    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)

    final_datasets = {}
    ratio_rows = []
    age_rows = []

    for dataset_name, clean_data in clean_datasets.items():
        data = add_loan_to_income(clean_data)
        data = add_age_group(data)
        final_datasets[dataset_name] = data

        ratio_row = loan_to_income_summary(data, dataset_name)
        ratio_rows.append(ratio_row)

        group_rows = age_group_summary(data, dataset_name)
        age_rows.extend(group_rows)

        output_file = PROCESSED_FOLDER / (dataset_name + "_clean.csv")
        data.to_csv(output_file, index=False)
        relative_path = output_file.relative_to(PROJECT_ROOT)
        row_count = data.shape[0]
        column_count = data.shape[1]
        print(dataset_name + ": saved", relative_path, "-", row_count, "rows x", column_count, "columns")

    print()
    ratio_table = pd.DataFrame(ratio_rows)
    print("Loan-to-income:")
    print(ratio_table.to_string(index=False))

    print()
    age_table = pd.DataFrame(age_rows)
    print("Age groups:")
    print(age_table.to_string(index=False))

    ratio_file = TABLES_FOLDER / "loan_to_income_summary.csv"
    ratio_table.to_csv(ratio_file, index=False)
    age_file = TABLES_FOLDER / "age_group_summary.csv"
    age_table.to_csv(age_file, index=False)

    print()
    print("Saved", ratio_file.relative_to(PROJECT_ROOT))
    print("Saved", age_file.relative_to(PROJECT_ROOT))

    return final_datasets


if __name__ == "__main__":
    loaded_datasets = load_both_raw_datasets()
    print()
    common = keep_common_features(loaded_datasets)
    print()
    harmonised = harmonise_labels_and_units(common)
    print()
    cleaned = check_ranges_and_clean(harmonised)
    print()
    add_loan_to_income_and_age_groups(cleaned)