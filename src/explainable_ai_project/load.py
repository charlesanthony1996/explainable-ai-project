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


def load_dataset():

    pass

def load_both_raw_datasets():

    print("loaded the datasets")
    # pass

if __name__ == "__main__":

    load_both_raw_datasets()
    print("loading from load.py")