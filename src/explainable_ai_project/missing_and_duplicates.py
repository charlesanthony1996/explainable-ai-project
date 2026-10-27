from pathlib import Path

import pandas as pd

from explainable_ai_project.load import load_both_raw_datasets

this_file = Path(__file__).resolve()
package_folder = this_file.parent
src_folder = package_folder.parent
project_root = src_folder.parent
 
tables_folder = project_root / "results" / "tables"

id_columns = {
    "dataset_1": "Applicant_ID",
    "dataset_2": None,
}

def missing_values_table(raw_data):
    """One row per column: how many values are missing, and what share that is."""
 
    row_count = len(raw_data)
    rows = []
 
    for column in raw_data.columns:
        missing_flags = raw_data[column].isna()
        missing_count = int(missing_flags.sum())
 
        missing_percent = 100 * missing_count / row_count
        missing_percent = round(missing_percent, 2)
 
        rows.append({
            "column": column,
            "missing_count": missing_count,
            "missing_percent": missing_percent,
        })
 
    table = pd.DataFrame(rows)
 
    # Columns with the most missing values first
    table = table.sort_values("missing_count", ascending=False)
    table = table.reset_index(drop=True)
    return table


def count_rows_with_missing_values(raw_data):
    """How many rows have at least one missing value."""
 
    row_has_missing = raw_data.isna().any(axis=1)
    count = int(row_has_missing.sum())
    return count
 
 
def count_exact_duplicates(raw_data):
    """Rows that are identical to an earlier row in every column."""
 
    duplicate_flags = raw_data.duplicated()
    count = int(duplicate_flags.sum())
    return count
 
 
def count_duplicates_ignoring_id(raw_data, id_column):
    """Rows identical to an earlier row in every column except the ID."""
 
    if id_column is None:
        return None
 
    columns_without_id = []
    for column in raw_data.columns:
        if column != id_column:
            columns_without_id.append(column)
 
    duplicate_flags = raw_data.duplicated(subset=columns_without_id)
    count = int(duplicate_flags.sum())
    return count
 
 
def check_one_dataset(raw_data, dataset_name):
    """Run every check on one dataset, save its table, return a summary row."""
 
    print(dataset_name)
 
    # Missing values per column
    missing_table = missing_values_table(raw_data)
    output_file = tables_folder / (dataset_name + "_missing_values.csv")
    missing_table.to_csv(output_file, index=False)
 
    total_missing = int(missing_table["missing_count"].sum())
    rows_with_missing = count_rows_with_missing_values(raw_data)
    print("  missing values in total:", total_missing)
    print("  rows with at least one missing value:", rows_with_missing)
 
    # Show only the columns that actually have missing values
    columns_with_missing = missing_table[missing_table["missing_count"] > 0]
    if len(columns_with_missing) > 0:
        print("  columns with missing values:")
        print(columns_with_missing.to_string(index=False))
 
    # Duplicates
    exact_duplicates = count_exact_duplicates(raw_data)
    print("  exact duplicate rows:", exact_duplicates)
 
    id_column = id_columns[dataset_name]
    duplicates_ignoring_id = count_duplicates_ignoring_id(raw_data, id_column)
    if duplicates_ignoring_id is not None:
        print("  duplicate rows when ignoring", id_column + ":", duplicates_ignoring_id)
 
    print("  saved", output_file.relative_to(project_root))
    print()
 
    # Dataset 2 has no ID column, so this check does not apply to it
    if duplicates_ignoring_id is None:
        duplicates_ignoring_id = "no ID column"
 
    summary_row = {
        "dataset": dataset_name,
        "rows": len(raw_data),
        "columns": raw_data.shape[1],
        "missing_values_total": total_missing,
        "rows_with_missing_values": rows_with_missing,
        "columns_with_missing_values": len(columns_with_missing),
        "exact_duplicate_rows": exact_duplicates,
        "duplicate_rows_ignoring_id": duplicates_ignoring_id,
    }
    return summary_row
 
 
def check_missing_and_duplicates(raw_datasets):
    """Run the checks on both datasets.
 
    raw_datasets is the dictionary returned by load_both_raw_datasets().
    Returns the summary table (one row per dataset).
    """
 
    tables_folder.mkdir(parents=True, exist_ok=True)
 
    summary_rows = []
    for dataset_name, raw_data in raw_datasets.items():
        summary_row = check_one_dataset(raw_data, dataset_name)
        summary_rows.append(summary_row)
 
    summary_table = pd.DataFrame(summary_rows)
    summary_file = tables_folder / "missing_and_duplicates_summary.csv"
    summary_table.to_csv(summary_file, index=False)
    print("Saved", summary_file.relative_to(project_root))
 
    return summary_table
 
 
# if __name__ == "__main__":
#     loaded_datasets = load_both_raw_datasets()
#     print()
#     check_missing_and_duplicates(loaded_datasets)
 