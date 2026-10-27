from pathlib import Path

import pandas as pd
import requests

# path handling
this_file = Path(__file__).resolve()
package_folder = this_file.parent
src_folder = package_folder.parent
project_root = src_folder.parent

raw_datasets_folder = project_root / "datasets" / "root"

datasets = {
    "dataset_1": {
        "url": "https://raw.githubusercontent.com/charlesanthony1996/human-centered-and-explainable-ai-datasets/refs/heads/main/Loan%20Dataset.csv",
        "file": raw_datasets_folder / "loan_dataset_1.csv",
        "expected_rows": 52000,
        "expected_columns": 27,
        "label_column": "Loan_Approval_Status"

    },

    "dataset_2": {
        "url": "https://raw.githubusercontent.com/charlesanthony1996/human-centered-and-explainable-ai-datasets/refs/heads/main/Loan.csv",
        "file": raw_datasets_folder / "loan_dataset_2.csv",
        "expected_rows": 20000,
        "expected_columns": 36,
        "label_column": "LoanApproved"
    }
}

#  Download one dataset into datasets/raw/, unless it is already there.
def download_dataset(dataset_name):

    settings = datasets[dataset_name]
    url = settings["url"]
    target_file = settings["file"]
 
    file_exists = target_file.exists()
    if file_exists:
        print(dataset_name + ": already downloaded")
        return
 
    print(dataset_name + ": downloading ...")
    response = requests.get(url, timeout=120)
 
    # Stop with a clear error if GitHub returns an error page
    response.raise_for_status()
 
    raw_datasets_folder.mkdir(parents=True, exist_ok=True)
    file_bytes = response.content
    target_file.write_bytes(file_bytes)
 
    print(dataset_name + ": saved to", target_file.relative_to(project_root))

#  Read one dataset and check its size and its approval column
def load_dataset(dataset_name):
 
    settings = datasets[dataset_name]
    file_path = settings["file"]
 
    raw_data = pd.read_csv(file_path)
 
    row_count = raw_data.shape[0]
    column_count = raw_data.shape[1]
    print(dataset_name + ":", row_count, "rows x", column_count, "columns")
 
    # Check 1: the size is what we expect
    expected_rows = settings["expected_rows"]
    expected_columns = settings["expected_columns"]
    rows_match = row_count == expected_rows
    columns_match = column_count == expected_columns
 
    if rows_match and columns_match:
        print("  size OK")
    else:
        print("  WARNING: expected", expected_rows, "x", expected_columns, "- the file may have changed")
 
    # Check 2: the approval column exists
    label_column = settings["label_column"]
    label_exists = label_column in raw_data.columns
 
    if not label_exists:
        message = dataset_name + " has no column called " + label_column
        raise ValueError(message)
 
    approval_rate = raw_data[label_column].mean()
    approval_percent = round(approval_rate * 100, 1)
    print("  approval rate:", approval_percent, "%")
 
    return raw_data

def load_both_raw_datasets():

    print("loaded the datasets")
    # pass

# Download (first time only) and load both datasets.
# Returns a dictionary: {"dataset_1": table, "dataset_2": table}
def load_both_raw_datasets():
 
    raw_datasets = {}
 
    for dataset_name in datasets:
        download_dataset(dataset_name)
        raw_data = load_dataset(dataset_name)
        raw_datasets[dataset_name] = raw_data
 
    return raw_datasets

# if __name__ == "__main__":

#     load_both_raw_datasets()
#     print("loading from load.py")