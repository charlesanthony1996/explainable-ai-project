"""
Training pathway, block 1A, step 6: harmonise labels and units.

After step 5 both datasets have the same column names, but not the same values:
dataset 1 says "Graduate" where dataset 2 says "Bachelor", "Vehicle" where it
says "Auto", and so on. This step:
    1. maps every category value to one shared label
       (any value not in the map becomes "Other" and is reported)
    2. compares the numeric columns of both datasets side by side and warns
       if a unit looks different (monthly instead of annual income,
       years instead of months)
    3. applies unit conversions, if any are needed (none are, at the moment)

Put this file in:  src/explainable_ai_project/harmonise.py
Run it with:       uv run python -m explainable_ai_project.harmonise

Writes:
    results/tables/category_counts.csv    every shared label and how often it appears
    results/tables/unmapped_values.csv    values that were not in the map
    results/tables/unit_comparison.csv    min, median, max of each number column
"""

from pathlib import Path

import pandas as pd

from explainable_ai_project.common_columns import keep_common_features
from explainable_ai_project.load import load_both_raw_datasets


# ------------------------------------------------------------------
# Where the result tables go
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PACKAGE_FOLDER = THIS_FILE.parent
SRC_FOLDER = PACKAGE_FOLDER.parent
PROJECT_ROOT = SRC_FOLDER.parent

TABLES_FOLDER = PROJECT_ROOT / "results" / "tables"


CATEGORY_COLUMNS = ["marital_status", "education", "employment_status", "loan_purpose"]

NUMBER_COLUMNS = ["age", "dependents", "annual_income", "credit_score", "loan_amount", "loan_term_months"]


# ------------------------------------------------------------------
# Category maps: original value -> shared label
# Checked against the real values of both datasets.
#
# TEAM DECISIONS TO CONFIRM:
#   - Education: dataset 1 "Graduate" is treated as a bachelor-level degree, so
#     "Graduate", "Associate" and "Bachelor" all become "Undergraduate".
#   - Marital status: dataset 2 "Widowed" has no match in dataset 1 -> "Other".
#   - Loan purpose: "Personal", "Debt Consolidation" and "Other" share one group.
# ------------------------------------------------------------------
CATEGORY_MAPS = {
    "dataset_1": {
        "marital_status": {
            "Single": "Single",
            "Married": "Married",
            "Divorced": "Divorced",
        },
        "education": {
            "High School": "High School",
            "Graduate": "Undergraduate",
            "Postgraduate": "Postgraduate",
        },
        "employment_status": {
            "Employed": "Employed",
            "Self-Employed": "Self-Employed",
            "Unemployed": "Unemployed",
        },
        "loan_purpose": {
            "Home": "Home",
            "Education": "Education",
            "Vehicle": "Vehicle",
            "Personal": "Personal or other",
        },
    },
    "dataset_2": {
        "marital_status": {
            "Single": "Single",
            "Married": "Married",
            "Divorced": "Divorced",
            "Widowed": "Other",
        },
        "education": {
            "High School": "High School",
            "Associate": "Undergraduate",
            "Bachelor": "Undergraduate",
            "Master": "Postgraduate",
            "Doctorate": "Postgraduate",
        },
        "employment_status": {
            "Employed": "Employed",
            "Self-Employed": "Self-Employed",
            "Unemployed": "Unemployed",
        },
        "loan_purpose": {
            "Home": "Home",
            "Education": "Education",
            "Auto": "Vehicle",
            "Debt Consolidation": "Personal or other",
            "Other": "Personal or other",
        },
    },
}


# ------------------------------------------------------------------
# Unit sanity rules
# If a median falls outside these ranges, the unit is probably different.
# ------------------------------------------------------------------
UNIT_RULES = {
    "annual_income": {
        "lowest_median": 12000,
        "warning": "median income is very low: is it monthly instead of annual?",
    },
    "loan_term_months": {
        "lowest_median": 6,
        "warning": "median term is very short: is it in years instead of months?",
    },
}


# ------------------------------------------------------------------
# 1. Category labels
# ------------------------------------------------------------------

def map_one_column(values, value_map):
    """Replace every value with its shared label. Unknown values become 'Other'."""

    clean_values = values.astype(str)
    clean_values = clean_values.str.strip()

    new_values = []
    for value in clean_values:
        if value in value_map:
            new_value = value_map[value]
        else:
            new_value = "Other"
        new_values.append(new_value)

    return clean_values, new_values


def find_unmapped_values(clean_values, value_map, dataset_name, column):
    """Rows for the report: every value that was not in the map, and how often it occurs."""

    rows = []
    unique_values = clean_values.unique()

    for value in unique_values:
        if value not in value_map:
            is_this_value = clean_values == value
            count = int(is_this_value.sum())
            rows.append({
                "dataset": dataset_name,
                "column": column,
                "value": value,
                "rows": count,
            })

    return rows


def harmonise_categories(common_data, dataset_name):
    """Return a copy with shared category labels, plus the list of unmapped values."""

    data = common_data.copy()
    dataset_maps = CATEGORY_MAPS[dataset_name]
    unmapped_rows = []

    for column in CATEGORY_COLUMNS:
        value_map = dataset_maps[column]
        clean_values, new_values = map_one_column(data[column], value_map)

        column_unmapped = find_unmapped_values(clean_values, value_map, dataset_name, column)
        unmapped_rows.extend(column_unmapped)

        data[column] = new_values

    return data, unmapped_rows


def category_counts(data, dataset_name):
    """How often each shared label appears, so both datasets can be compared."""

    rows = []
    for column in CATEGORY_COLUMNS:
        counts = data[column].value_counts()
        for label, count in counts.items():
            rows.append({
                "dataset": dataset_name,
                "column": column,
                "label": label,
                "rows": int(count),
            })
    return rows


# ------------------------------------------------------------------
# 2. Units
# ------------------------------------------------------------------

def unit_comparison(harmonised_datasets):
    """Min, median and max of every number column, for both datasets side by side."""

    rows = []
    for column in NUMBER_COLUMNS:
        row = {"column": column}
        for dataset_name, data in harmonised_datasets.items():
            values = data[column]
            row[dataset_name + "_min"] = values.min()
            row[dataset_name + "_median"] = values.median()
            row[dataset_name + "_max"] = values.max()
        rows.append(row)

    table = pd.DataFrame(rows)
    return table


def check_units(harmonised_datasets):
    """Print a warning for every median that suggests a different unit."""

    warning_count = 0

    for dataset_name, data in harmonised_datasets.items():
        for column, rule in UNIT_RULES.items():
            median_value = data[column].median()
            if median_value < rule["lowest_median"]:
                print("WARNING", dataset_name, column + ":", rule["warning"])
                warning_count += 1

    if warning_count == 0:
        print("Units look the same in both datasets (annual income, term in months).")


def convert_units(data, dataset_name):
    """Put any unit conversion here, one per line, with a comment saying why.

    Example, if a dataset had monthly income:
        data["annual_income"] = data["annual_income"] * 12

    At the moment both datasets use annual income and terms in months,
    so nothing is converted.
    """

    converted = data.copy()
    return converted


# ------------------------------------------------------------------
# Both steps on both datasets
# ------------------------------------------------------------------

def save_table(rows_or_table, file_name, columns=None):
    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)

    if isinstance(rows_or_table, pd.DataFrame):
        table = rows_or_table
    else:
        table = pd.DataFrame(rows_or_table, columns=columns)

    output_file = TABLES_FOLDER / file_name
    table.to_csv(output_file, index=False)
    print("Saved", output_file.relative_to(PROJECT_ROOT))
    return table


def harmonise_labels_and_units(common_datasets):
    """Run the step on both datasets.

    common_datasets is the dictionary returned by keep_common_features().
    Returns a dictionary: {"dataset_1": harmonised table, "dataset_2": harmonised table}
    """

    harmonised_datasets = {}
    all_unmapped_rows = []
    all_count_rows = []

    # 1. Category labels
    for dataset_name, common_data in common_datasets.items():
        data, unmapped_rows = harmonise_categories(common_data, dataset_name)
        data = convert_units(data, dataset_name)

        harmonised_datasets[dataset_name] = data
        all_unmapped_rows.extend(unmapped_rows)

        count_rows = category_counts(data, dataset_name)
        all_count_rows.extend(count_rows)

    if len(all_unmapped_rows) == 0:
        print("Every category value was mapped to a shared label.")
    else:
        print("WARNING: these values were not in the map and became 'Other':")
        unmapped_table = pd.DataFrame(all_unmapped_rows)
        print(unmapped_table.to_string(index=False))
        print("Add them to CATEGORY_MAPS if they deserve their own group.")

    # Shared labels side by side: one row per label, one column per dataset
    counts_table = pd.DataFrame(all_count_rows)
    side_by_side = counts_table.pivot_table(
        index=["column", "label"],
        columns="dataset",
        values="rows",
        aggfunc="sum",
        fill_value=0,
    )
    side_by_side = side_by_side.astype(int)
    print()
    print(side_by_side.to_string())

    # 2. Units
    print()
    units_table = unit_comparison(harmonised_datasets)
    print(units_table.to_string(index=False))
    check_units(harmonised_datasets)

    print()
    save_table(all_count_rows, "category_counts.csv")
    save_table(all_unmapped_rows, "unmapped_values.csv", columns=["dataset", "column", "value", "rows"])
    save_table(units_table, "unit_comparison.csv")

    return harmonised_datasets


# if __name__ == "__main__":
#     loaded_datasets = load_both_raw_datasets()
#     print()
#     common = keep_common_features(loaded_datasets)
#     print()
#     harmonise_labels_and_units(common)