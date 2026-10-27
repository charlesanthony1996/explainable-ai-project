"""
Training pathway, block 1A, step 7: range checks and cleaning.

This is the only step that removes rows. A row is removed if:
    1. one of the 10 features or the label is missing
    2. it is a duplicate of an earlier row (ignoring the applicant ID)
    3. a value is outside its allowed range (RANGE_RULES below)
    4. the label is not exactly 0 or 1

Every removed row is written down with its ID and the reason, so the
report can say exactly what was removed and why (Topic 3: data validation).

Put this file in:  src/explainable_ai_project/cleaning.py
Run it with:       uv run python -m explainable_ai_project.cleaning

Writes:
    results/tables/dataset_1_range_checks.csv
    results/tables/dataset_2_range_checks.csv
    results/tables/removed_rows.csv         ID and reason of every removed row
    results/tables/cleaning_summary.csv     rows before, removed per reason, rows after
"""

from pathlib import Path

import pandas as pd

from explainable_ai_project.common_columns import ID_COLUMN
from explainable_ai_project.common_columns import LABEL_COLUMN
from explainable_ai_project.common_columns import keep_common_features
from explainable_ai_project.harmonize import harmonise_labels_and_units
from explainable_ai_project.load import load_both_raw_datasets


# ------------------------------------------------------------------
# Where the result tables go
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PACKAGE_FOLDER = THIS_FILE.parent
SRC_FOLDER = PACKAGE_FOLDER.parent
PROJECT_ROOT = SRC_FOLDER.parent

TABLES_FOLDER = PROJECT_ROOT / "results" / "tables"


# ------------------------------------------------------------------
# Allowed range per number column: (lowest allowed, highest allowed)
# None means "no limit on this side".
# The same rules are used for both datasets.
# ------------------------------------------------------------------
RANGE_RULES = {
    "age": (18, 100),                  # adults only
    "dependents": (0, 20),
    "annual_income": (1, None),        # must be positive
    "credit_score": (300, 850),        # usual credit score scale
    "loan_amount": (1, None),          # must be positive
    "loan_term_months": (1, 480),      # up to 40 years
}


def limit_as_text(limit):
    if limit is None:
        return "no limit"
    return str(limit)


# ------------------------------------------------------------------
# The three checks. Each returns True/False per row: True = remove.
# ------------------------------------------------------------------

def rows_with_missing_values(data):
    columns_to_check = []
    for column in data.columns:
        if column != ID_COLUMN:
            columns_to_check.append(column)

    has_missing = data[columns_to_check].isna().any(axis=1)
    return has_missing


def rows_that_are_duplicates(data):
    """Same values in every column except the ID as an earlier row."""

    columns_to_compare = []
    for column in data.columns:
        if column != ID_COLUMN:
            columns_to_compare.append(column)

    is_duplicate = data.duplicated(subset=columns_to_compare)
    return is_duplicate


def rows_outside_ranges(data):
    """True/False per row, plus a table with one row per rule for the report."""

    outside_any_range = pd.Series(False, index=data.index)
    rule_rows = []

    for column, limits in RANGE_RULES.items():
        lowest = limits[0]
        highest = limits[1]
        values = data[column]

        too_low = pd.Series(False, index=data.index)
        if lowest is not None:
            too_low = values < lowest

        too_high = pd.Series(False, index=data.index)
        if highest is not None:
            too_high = values > highest

        outside_this_range = too_low | too_high
        violation_count = int(outside_this_range.sum())

        rule_rows.append({
            "column": column,
            "allowed_min": limit_as_text(lowest),
            "allowed_max": limit_as_text(highest),
            "actual_min": values.min(),
            "actual_max": values.max(),
            "rows_outside": violation_count,
        })

        outside_any_range = outside_any_range | outside_this_range

    # The label must be exactly 0 or 1
    labels = data[LABEL_COLUMN]
    label_is_valid = labels.isin([0, 1])
    label_is_invalid = ~label_is_valid
    invalid_label_count = int(label_is_invalid.sum())

    rule_rows.append({
        "column": LABEL_COLUMN,
        "allowed_min": "0",
        "allowed_max": "1",
        "actual_min": labels.min(),
        "actual_max": labels.max(),
        "rows_outside": invalid_label_count,
    })

    outside_any_range = outside_any_range | label_is_invalid

    rules_table = pd.DataFrame(rule_rows)
    return outside_any_range, rules_table


# ------------------------------------------------------------------
# Cleaning one dataset
# ------------------------------------------------------------------

def removed_rows_record(data, remove_flags, dataset_name, reason):
    """ID and reason for every row that is about to be removed."""

    removed = data[remove_flags]

    record = pd.DataFrame()
    record[ID_COLUMN] = removed[ID_COLUMN]
    record["dataset"] = dataset_name
    record["reason"] = reason
    return record


def clean_one_dataset(harmonised_data, dataset_name):
    """Run the checks in order and remove the rows that fail.

    Returns the cleaned table, the range rules table, the removed-rows record
    and one summary row.
    """

    data = harmonised_data.copy()
    rows_before = len(data)
    removed_records = []

    # 1. Missing values
    remove_flags = rows_with_missing_values(data)
    missing_count = int(remove_flags.sum())
    record = removed_rows_record(data, remove_flags, dataset_name, "missing value")
    removed_records.append(record)
    data = data[~remove_flags]

    # 2. Duplicates
    remove_flags = rows_that_are_duplicates(data)
    duplicate_count = int(remove_flags.sum())
    record = removed_rows_record(data, remove_flags, dataset_name, "duplicate")
    removed_records.append(record)
    data = data[~remove_flags]

    # 3. Range checks (including the label)
    remove_flags, rules_table = rows_outside_ranges(data)
    range_count = int(remove_flags.sum())
    record = removed_rows_record(data, remove_flags, dataset_name, "outside allowed range")
    removed_records.append(record)
    data = data[~remove_flags]

    data = data.reset_index(drop=True)
    rows_after = len(data)

    removed_table = pd.concat(removed_records, ignore_index=True)

    summary_row = {
        "dataset": dataset_name,
        "rows_before": rows_before,
        "removed_missing": missing_count,
        "removed_duplicates": duplicate_count,
        "removed_out_of_range": range_count,
        "rows_after": rows_after,
    }

    return data, rules_table, removed_table, summary_row


# ------------------------------------------------------------------
# Both datasets
# ------------------------------------------------------------------

def check_ranges_and_clean(harmonised_datasets):
    """Run the step on both datasets.

    harmonised_datasets is the dictionary returned by harmonise_labels_and_units().
    Returns a dictionary: {"dataset_1": clean table, "dataset_2": clean table}
    """

    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)

    clean_datasets = {}
    all_removed_tables = []
    summary_rows = []

    for dataset_name, harmonised_data in harmonised_datasets.items():
        result = clean_one_dataset(harmonised_data, dataset_name)
        clean_data = result[0]
        rules_table = result[1]
        removed_table = result[2]
        summary_row = result[3]

        clean_datasets[dataset_name] = clean_data
        all_removed_tables.append(removed_table)
        summary_rows.append(summary_row)

        print(dataset_name)
        print(rules_table.to_string(index=False))
        print("  removed: missing", summary_row["removed_missing"],
              "| duplicates", summary_row["removed_duplicates"],
              "| out of range", summary_row["removed_out_of_range"])
        print("  rows:", summary_row["rows_before"], "->", summary_row["rows_after"])

        rules_file = TABLES_FOLDER / (dataset_name + "_range_checks.csv")
        rules_table.to_csv(rules_file, index=False)
        print("  saved", rules_file.relative_to(PROJECT_ROOT))
        print()

    removed_rows = pd.concat(all_removed_tables, ignore_index=True)
    removed_file = TABLES_FOLDER / "removed_rows.csv"
    removed_rows.to_csv(removed_file, index=False)
    print("Saved", removed_file.relative_to(PROJECT_ROOT), "(" + str(len(removed_rows)) + " rows removed in total)")

    summary_table = pd.DataFrame(summary_rows)
    summary_file = TABLES_FOLDER / "cleaning_summary.csv"
    summary_table.to_csv(summary_file, index=False)
    print("Saved", summary_file.relative_to(PROJECT_ROOT))

    return clean_datasets


# if __name__ == "__main__":
#     loaded_datasets = load_both_raw_datasets()
#     print()
#     common = keep_common_features(loaded_datasets)
#     print()
#     harmonised = harmonise_labels_and_units(common)
#     print()
#     check_ranges_and_clean(harmonised)