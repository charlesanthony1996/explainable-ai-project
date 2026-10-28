import os
import random

from explainable_ai_project.load import load_both_raw_datasets
from explainable_ai_project.missing_and_duplicates import check_missing_and_duplicates
from explainable_ai_project.leakage_scan import run_leakage_scan

from explainable_ai_project.derived_columns import find_derived_columns
from explainable_ai_project.common_columns import keep_common_features
from explainable_ai_project.harmonize import harmonise_labels_and_units

from explainable_ai_project.cleaning import check_ranges_and_clean
from explainable_ai_project.add_features import add_loan_to_income_and_age_groups
from explainable_ai_project.split import split_70_15_15

from explainable_ai_project.preprocessor import fit_preprocessors
from explainable_ai_project.governance_report import write_reports_and_governance_log

# from the train folder
from explainable_ai_project.load_split import check_all_splits_load
from explainable_ai_project.logistic_baseline import train_logistic_baselines
from explainable_ai_project.sweep_problem import show_sweep_problem

def main():

    loaded_datasets = load_both_raw_datasets()

    check_missing_and_duplicates(loaded_datasets)

    run_leakage_scan(loaded_datasets)

    find_derived_columns(loaded_datasets)

    common = keep_common_features(loaded_datasets)

    harmonized = harmonise_labels_and_units(common)

    cleaned = check_ranges_and_clean(harmonized)

    final = add_loan_to_income_and_age_groups(cleaned)

    splits_per_dataset = split_70_15_15(final)

    fit_preprocessors(splits_per_dataset)

    write_reports_and_governance_log()

    # train and test the models
    check_all_splits_load()

    train_logistic_baselines()

    # from sweep_problem.py
    show_sweep_problem()



if __name__ == "__main__":
    main()