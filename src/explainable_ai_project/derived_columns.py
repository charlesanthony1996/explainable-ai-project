"""
Training pathway, block 1A, step 4: columns derived from the loan amount.

The amount sweep (training step 22, inference step 5) changes ONLY the loan
amount and asks the model again. Any other input that depends on the amount
would then be out of date, and the model would see an impossible applicant.

This step finds every numeric column that moves together with the loan amount,
using two measures:
    correlation       straight-line link, from -1 to 1
    R² from amount    how much of the column a small model can predict from
                      the amount alone (0 = nothing, 1 = everything). This also
                      catches curved links, such as a monthly payment that
                      depends on amount, interest rate and term together.

It then says what each finding means for the project:
    - a column that is NOT a model input: no action, mention it in the report
    - a column that IS a model input: the sweep must stay within realistic
      amounts for that applicant (handled in training step 22)

Both datasets are checked: dataset 2 has columns calculated from the amount,
and dataset 1 has model inputs that are tightly linked to it.

Put this file in:  src/explainable_ai_project/derived_columns.py
Run it with:       uv run python -m explainable_ai_project.derived_columns

Writes:
    results/tables/dataset_1_derived_columns.csv
    results/tables/dataset_2_derived_columns.csv
"""

from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score

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
# Settings
# ------------------------------------------------------------------
SEED = 42

# The loan amount column in each raw dataset
AMOUNT_COLUMNS = {
    "dataset_1": "Loan_Amount_Requested",
    "dataset_2": "LoanAmount",
}

# The raw names of the model inputs (the common features, without the amount itself)
MODEL_INPUT_COLUMNS = {
    "dataset_1": [
        "Age", "Marital_Status", "Dependents", "Education", "Employment_Status",
        "Annual_Income", "Credit_Score", "Loan_Term", "Loan_Purpose",
    ],
    "dataset_2": [
        "Age", "MaritalStatus", "NumberOfDependents", "EducationLevel", "EmploymentStatus",
        "AnnualIncome", "CreditScore", "LoanDuration", "LoanPurpose",
    ],
}

STRONG_CORRELATION = 0.5   # size of correlation that counts as strongly tied
STRONG_R2 = 0.3            # R² from the amount that counts as strongly tied
PARTLY_R2 = 0.1            # R² from the amount that counts as partly tied


def r2_from_amount(amounts, values):
    """How much of 'values' a small model can predict from the amount alone.

    Uses 3-fold cross-validation, so the model is always scored on rows it did not see.
    """

    amount_table = amounts.to_frame()

    model = HistGradientBoostingRegressor(max_iter=100, random_state=SEED)
    fold_scores = cross_val_score(model, amount_table, values, cv=3, scoring="r2")

    average_score = fold_scores.mean()

    # Below 0 means "worse than guessing the average", which we report as 0
    if average_score < 0:
        average_score = 0.0

    return average_score


def link_strength(correlation, r2):
    correlation_size = abs(correlation)

    if correlation_size >= STRONG_CORRELATION:
        return "strongly tied"
    if r2 >= STRONG_R2:
        return "strongly tied"
    if r2 >= PARTLY_R2:
        return "partly tied"
    return "not tied"


def action_for(strength, is_model_input):
    if strength == "not tied":
        return "none"
    if is_model_input:
        return "model input: keep the sweep within realistic amounts for the applicant"
    return "not a model input: mention in the report"


def scan_one_dataset(raw_data, dataset_name):
    """Check every numeric column of one dataset against its loan amount."""

    amount_column = AMOUNT_COLUMNS[dataset_name]
    label_column = datasets[dataset_name]["label_column"]
    id_column = id_columns[dataset_name]
    model_inputs = MODEL_INPUT_COLUMNS[dataset_name]

    amounts = raw_data[amount_column]
    rows = []

    for column in raw_data.columns:
        # Skip the amount itself, the label and the ID
        if column == amount_column:
            continue
        if column == label_column:
            continue
        if column == id_column:
            continue

        # Only numbers can be calculated from the amount
        column_is_numeric = is_numeric_dtype(raw_data[column])
        if not column_is_numeric:
            continue

        values = raw_data[column]

        correlation = values.corr(amounts)
        r2 = r2_from_amount(amounts, values)
        strength = link_strength(correlation, r2)

        is_model_input = column in model_inputs
        action = action_for(strength, is_model_input)

        rows.append({
            "column": column,
            "correlation_with_amount": round(correlation, 3),
            "r2_from_amount": round(r2, 3),
            "link": strength,
            "model_input": is_model_input,
            "action": action,
        })

    table = pd.DataFrame(rows)
    table = table.sort_values("r2_from_amount", ascending=False)
    table = table.reset_index(drop=True)
    return table


def print_scan_result(table, dataset_name):
    print(dataset_name)

    tied_rows = table[table["link"] != "not tied"]

    if len(tied_rows) == 0:
        print("  no column is tied to the loan amount")
        return

    columns_to_show = ["column", "correlation_with_amount", "r2_from_amount", "link", "model_input"]
    print(tied_rows[columns_to_show].to_string(index=False))

    model_input_rows = tied_rows[tied_rows["model_input"]]
    model_input_count = len(model_input_rows)
    if model_input_count > 0:
        model_input_names = list(model_input_rows["column"])
        print("  model inputs tied to the amount:", model_input_names)
        print("  -> the sweep must stay within realistic amounts for these (training step 22)")


def find_derived_columns(raw_datasets):
    """Run the check on both datasets.

    raw_datasets is the dictionary returned by load_both_raw_datasets().
    Returns a dictionary: {"dataset_1": table, "dataset_2": table}
    """

    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)

    result_tables = {}

    for dataset_name, raw_data in raw_datasets.items():
        table = scan_one_dataset(raw_data, dataset_name)
        result_tables[dataset_name] = table

        output_file = TABLES_FOLDER / (dataset_name + "_derived_columns.csv")
        table.to_csv(output_file, index=False)

        print_scan_result(table, dataset_name)
        print("  saved", output_file.relative_to(PROJECT_ROOT))
        print()

    return result_tables


# if __name__ == "__main__":
#     loaded_datasets = load_both_raw_datasets()
#     print()
#     find_derived_columns(loaded_datasets)