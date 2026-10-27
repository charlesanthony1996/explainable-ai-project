import os
import random

from explainable_ai_project.load import load_both_raw_datasets
from explainable_ai_project.missing_and_duplicates import check_missing_and_duplicates
from explainable_ai_project.leakage_scan import run_leakage_scan
from explainable_ai_project.derived_columns import find_derived_columns
from explainable_ai_project.common_columns import keep_common_features


def main():

    loaded_datasets = load_both_raw_datasets()
    check_missing_and_duplicates(loaded_datasets)
    run_leakage_scan(loaded_datasets)
    find_derived_columns(loaded_datasets)
    keep_common_features(loaded_datasets)


if __name__ == "__main__":
    main()