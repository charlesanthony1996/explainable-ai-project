"""
Training pathway, block 1A, step 3: leakage scan.

For every raw column, measure how well that column ON ITS OWN predicts approval.
A column that predicts approval almost perfectly is usually "leakage":
information that only exists because the decision was already made
(for example a risk score the bank used to decide). A model trained on it
looks great in testing but is useless for new applicants.

The measure is the single-feature AUC:
    0.5  = the column says nothing about approval
    1.0  = the column separates approved from rejected perfectly

This step only reports. The models later use only the 10 common columns,
so a leaky column found here is documented, not used.

Put this file in:  src/explainable_ai_project/leakage_scan.py
Run it with:       uv run python -m explainable_ai_project.leakage_scan

Writes:
    results/tables/dataset_1_leakage_scan.csv
    results/tables/dataset_2_leakage_scan.csv
"""

from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import roc_auc_score

from explainable_ai_project.load import datasets
from explainable_ai_project.load import load_both_raw_datasets
from explainable_ai_project.missing_and_duplicates import id_columns


# ------------------------------------------------------------------
# Where the result tables go
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PACKAGE_FOLDER = THIS_FILE.parent
SRC_FOLDER = PACKAGE_FOLDER.parent
PROJECT_ROOT = SRC_FOLDER.parent

TABLES_FOLDER = PROJECT_ROOT / "results" / "tables"


# ------------------------------------------------------------------
# Limits
# ------------------------------------------------------------------
LEAKAGE_LIMIT = 0.95   # above this: almost certainly leakage
CHECK_LIMIT = 0.85     # above this: look at the column before trusting it

# Text columns with more distinct values than this (dates, free text) are skipped.
# Scoring each row by its own category would make them look like perfect leakage.
MAX_TEXT_VALUES = 50


def scores_for_numeric_column(values):
    """Numbers can be used as scores directly. Missing values get the median."""

    median_value = values.median()
    scores = values.fillna(median_value)
    return scores


def scores_for_text_column(values, labels):
    """Give every row the approval rate of its category, e.g. 'Married' -> 0.71.

    Returns None if the column has too many distinct values to be a category.
    """

    text_values = values.astype(str)

    distinct_count = text_values.nunique()
    if distinct_count > MAX_TEXT_VALUES:
        return None

    approval_rate_per_category = labels.groupby(text_values).mean()
    scores = text_values.map(approval_rate_per_category)
    return scores


def single_feature_auc(values, labels):
    """AUC of one column against the approval label, or None if it cannot be scored."""

    column_is_numeric = is_numeric_dtype(values)

    if column_is_numeric:
        scores = scores_for_numeric_column(values)
    else:
        scores = scores_for_text_column(values, labels)
        if scores is None:
            return None

    # A column with only one value cannot separate anything
    distinct_scores = scores.nunique()
    if distinct_scores < 2:
        return 0.5

    auc = roc_auc_score(labels, scores)

    # A column that predicts approval "backwards" is just as informative,
    # so 0.2 counts the same as 0.8
    if auc < 0.5:
        auc = 1 - auc

    return auc


def verdict_for(auc):
    if auc is None:
        return "skipped: date or free text"
    if auc > LEAKAGE_LIMIT:
        return "likely leakage"
    if auc > CHECK_LIMIT:
        return "check before use"
    return "ok"


def scan_one_dataset(raw_data, dataset_name):
    """Single-feature AUC for every column of one dataset, most predictive first."""

    label_column = datasets[dataset_name]["label_column"]
    id_column = id_columns[dataset_name]
    labels = raw_data[label_column]

    rows = []

    for column in raw_data.columns:
        if column == label_column:
            continue
        if column == id_column:
            continue

        auc = single_feature_auc(raw_data[column], labels)
        verdict = verdict_for(auc)

        if auc is not None:
            auc = round(auc, 4)

        rows.append({
            "column": column,
            "single_feature_auc": auc,
            "verdict": verdict,
        })

    table = pd.DataFrame(rows)
    table = table.sort_values("single_feature_auc", ascending=False, na_position="last")
    table = table.reset_index(drop=True)
    return table


def list_or_none(items):
    if len(items) == 0:
        return "none"
    return items


def print_scan_result(table, dataset_name):
    print(dataset_name)

    print("  five most predictive columns:")
    top_five = table.head(5)
    print(top_five.to_string(index=False))

    leaky_rows = table[table["verdict"] == "likely leakage"]
    check_rows = table[table["verdict"] == "check before use"]
    skipped_rows = table[table["verdict"] == "skipped: date or free text"]

    leaky_columns = list(leaky_rows["column"])
    check_columns = list(check_rows["column"])
    skipped_columns = list(skipped_rows["column"])

    print("  likely leakage:", list_or_none(leaky_columns))
    print("  check before use:", list_or_none(check_columns))
    if len(skipped_columns) > 0:
        print("  skipped:", skipped_columns)


def run_leakage_scan(raw_datasets):
    """Scan both datasets.

    raw_datasets is the dictionary returned by load_both_raw_datasets().
    Returns a dictionary: {"dataset_1": table, "dataset_2": table}
    """

    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)

    scan_tables = {}

    for dataset_name, raw_data in raw_datasets.items():
        table = scan_one_dataset(raw_data, dataset_name)
        scan_tables[dataset_name] = table

        output_file = TABLES_FOLDER / (dataset_name + "_leakage_scan.csv")
        table.to_csv(output_file, index=False)

        print_scan_result(table, dataset_name)
        print("  saved", output_file.relative_to(PROJECT_ROOT))
        print()

    return scan_tables


# if __name__ == "__main__":
#     loaded_datasets = load_both_raw_datasets()
#     print()
#     run_leakage_scan(loaded_datasets)