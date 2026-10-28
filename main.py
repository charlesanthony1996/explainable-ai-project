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

# random forest A, tuning and running the second random forest b
from explainable_ai_project.monotonic_forest_a import train_monotonic_forest_a
from explainable_ai_project.tune_forest import tune_and_retrain_a
from explainable_ai_project.monotonic_forest_b import train_monotonic_forest_b

# ebm comparison
from explainable_ai_project.ebm_comparison import run_ebm_comparison
from explainable_ai_project.calibrate import calibrate_models
from explainable_ai_project.recheck_monotonic import recheck_monotonicity

# test once and metrics
from explainable_ai_project.testing_once import test_once
from explainable_ai_project.test_set_sweep import sweep_on_test_set

# save, log and generate model card
# from explainable_ai_project.model_card import save_log_and_write_model_card
from explainable_ai_project.model_card_2 import save_log_and_write_model_card

# approval rates by age group
from explainable_ai_project.age_approval import approval_rates_by_age_group

# maximum amount by age group
from explainable_ai_project.maximum_age_fairness import maximum_amount_by_age_group

# calibration by age group
from explainable_ai_project.age_group_calibration import calibration_by_age_group

# choosing deferral rules
from explainable_ai_project.deferral import choose_deferral_rules

from explainable_ai_project.intervention_age import intervention_remove_age

# generate fairness report
# from explainable_ai_project.fairness_report import write_fairness_report
from explainable_ai_project.fairness_report_2 import write_fairness_report

# collecting documents for rag
from explainable_ai_project.documents_for_rag import collect_documents

# extract and clean text
from explainable_ai_project.extract_clean import extract_and_clean_text

# chunking
from explainable_ai_project.chunk import chunk_documents

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

    train_monotonic_forest_a()

    tune_and_retrain_a()

    train_monotonic_forest_b()

    run_ebm_comparison()

    calibrate_models()

    recheck_monotonicity()

    test_once()

    sweep_on_test_set()

    save_log_and_write_model_card()

    approval_rates_by_age_group()

    maximum_amount_by_age_group()

    calibration_by_age_group()

    choose_deferral_rules()

    intervention_remove_age()

    write_fairness_report()

    collect_documents()

    extract_and_clean_text()

    chunk_documents()



if __name__ == "__main__":
    main()