"""
Training pathway, block 1A, step 5: keep the 10 common features.

Both datasets describe the same things with different column names
(for example "Annual_Income" and "AnnualIncome"). This step:
    1. keeps only the 10 features both datasets have, plus the approval label
    2. renames them to one shared set of names
    3. adds an applicant ID that is unique across both datasets
    4. checks that each feature has the same kind of values in both
       (numbers in both, or text in both)

Category labels and units are made the same in the next step (step 6).

Put this file in:  src/explainable_ai_project/common_features.py
Run it with:       uv run python -m explainable_ai_project.common_features

Writes:
    results/tables/common_features.csv   which raw column became which shared name
"""

from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from explainable_ai_project.load import load_both_raw_datasets
from explainable_ai_project.missing_and_duplicates import id_columns


# ------------------------------------------------------------------
# Where the result table goes
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PACKAGE_FOLDER = THIS_FILE.parent
SRC_FOLDER = PACKAGE_FOLDER.parent
PROJECT_ROOT = SRC_FOLDER.parent

TABLES_FOLDER = PROJECT_ROOT / "results" / "tables"


# ------------------------------------------------------------------
# The shared names, and which raw column each one comes from
# ------------------------------------------------------------------
ID_COLUMN = "applicant_id"
LABEL_COLUMN = "approved"

# shared name: (column in dataset 1, column in dataset 2)
COMMON_COLUMNS = {
    "age": ("Age", "Age"),
    "marital_status": ("Marital_Status", "MaritalStatus"),
    "dependents": ("Dependents", "NumberOfDependents"),
    "education": ("Education", "EducationLevel"),
    "employment_status": ("Employment_Status", "EmploymentStatus"),
    "annual_income": ("Annual_Income", "AnnualIncome"),
    "credit_score": ("Credit_Score", "CreditScore"),
    "loan_amount": ("Loan_Amount_Requested", "LoanAmount"),
    "loan_term_months": ("Loan_Term", "LoanDuration"),
    "loan_purpose": ("Loan_Purpose", "LoanPurpose"),
    LABEL_COLUMN: ("Loan_Approval_Status", "LoanApproved"),
}

# Prefix for the new IDs, so "D1-17" and "D2-17" can never be mixed up
ID_PREFIXES = {
    "dataset_1": "D1-",
    "dataset_2": "D2-",
}


def raw_name_for(shared_name, dataset_name):
    """The raw column name of a shared feature in one dataset."""

    names = COMMON_COLUMNS[shared_name]
    if dataset_name == "dataset_1":
        return names[0]
    return names[1]


def make_ids(raw_data, dataset_name):
    """Build IDs like 'D1-17'. Dataset 2 has no ID column, so it uses the row number."""

    prefix = ID_PREFIXES[dataset_name]
    id_column = id_columns[dataset_name]

    ids = []

    if id_column is not None:
        original_ids = raw_data[id_column].tolist()
        for original_id in original_ids:
            new_id = prefix + str(original_id)
            ids.append(new_id)
    else:
        row_count = len(raw_data)
        for row_number in range(1, row_count + 1):
            new_id = prefix + str(row_number)
            ids.append(new_id)

    return ids


def keep_common_columns(raw_data, dataset_name):
    """Return a new table with the ID, the 10 common features and the label, under shared names."""

    # 1. Find the raw names for this dataset, and stop if any is missing
    raw_names = []
    missing_columns = []

    for shared_name in COMMON_COLUMNS:
        raw_name = raw_name_for(shared_name, dataset_name)
        raw_names.append(raw_name)
        if raw_name not in raw_data.columns:
            missing_columns.append(raw_name)

    if len(missing_columns) > 0:
        message = dataset_name + " is missing these columns: " + str(missing_columns)
        raise ValueError(message)

    # 2. Keep only those columns, in the same order for both datasets
    common_data = raw_data[raw_names].copy()

    # 3. Rename them to the shared names
    new_names = {}
    for shared_name in COMMON_COLUMNS:
        raw_name = raw_name_for(shared_name, dataset_name)
        new_names[raw_name] = shared_name
    common_data = common_data.rename(columns=new_names)

    # 4. Add the ID as the first column
    ids = make_ids(raw_data, dataset_name)
    common_data.insert(0, ID_COLUMN, ids)

    return common_data


def kind_of_values(values):
    if is_numeric_dtype(values):
        return "number"
    return "text"


def column_overview(common_datasets):
    """One row per shared feature: its raw names and the kind of values in each dataset."""

    data_1 = common_datasets["dataset_1"]
    data_2 = common_datasets["dataset_2"]

    rows = []
    for shared_name in COMMON_COLUMNS:
        kind_1 = kind_of_values(data_1[shared_name])
        kind_2 = kind_of_values(data_2[shared_name])
        same_kind = kind_1 == kind_2

        rows.append({
            "shared_name": shared_name,
            "dataset_1_column": raw_name_for(shared_name, "dataset_1"),
            "dataset_2_column": raw_name_for(shared_name, "dataset_2"),
            "dataset_1_kind": kind_1,
            "dataset_2_kind": kind_2,
            "same_kind": same_kind,
        })

    table = pd.DataFrame(rows)
    return table


def keep_common_features(raw_datasets):
    """Run the step on both datasets.

    raw_datasets is the dictionary returned by load_both_raw_datasets().
    Returns a dictionary: {"dataset_1": common table, "dataset_2": common table}
    """

    common_datasets = {}

    for dataset_name, raw_data in raw_datasets.items():
        common_data = keep_common_columns(raw_data, dataset_name)
        common_datasets[dataset_name] = common_data

        row_count = common_data.shape[0]
        column_count = common_data.shape[1]
        print(dataset_name + ":", row_count, "rows x", column_count, "columns (ID, 10 features, label)")

    print()

    overview = column_overview(common_datasets)
    print(overview.to_string(index=False))

    different_kinds = overview[~overview["same_kind"]]
    if len(different_kinds) > 0:
        print()
        print("WARNING: these features have numbers in one dataset and text in the other:")
        print(list(different_kinds["shared_name"]))
    else:
        print()
        print("Every feature has the same kind of values in both datasets.")

    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)
    output_file = TABLES_FOLDER / "common_features.csv"
    overview.to_csv(output_file, index=False)
    print("Saved", output_file.relative_to(PROJECT_ROOT))

    return common_datasets


# if __name__ == "__main__":
#     loaded_datasets = load_both_raw_datasets()
#     print()
#     keep_common_features(loaded_datasets)