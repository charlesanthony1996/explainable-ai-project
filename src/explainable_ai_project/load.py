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

def download_dataset():

    pass


def load_dataset():

    pass

def load_both_raw_datasets():

    print("loaded the datasets")
    # pass

if __name__ == "__main__":

    load_both_raw_datasets()
    print("loading from load.py")