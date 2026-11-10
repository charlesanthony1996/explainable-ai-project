"""
Entry point for the trustworthy loan assessment assistant.

Usage
-----
    uv run main.py data        block 1A, steps 1-11:  load, check, clean, split, preprocessor
    uv run main.py models      block 1B, steps 12-23: baseline, forests, EBM, calibrate, test
    uv run main.py fairness    block 1C, steps 24-29: Topic 3 checks before the LLM
    uv run main.py rag         block 1D, steps 30-35: documents, chunks, embeddings, FAISS
    uv run main.py shap        Topic 2: TreeSHAP importance, examples, fidelity check
    uv run main.py train       all training blocks in order (the old behaviour, plus shap)

    uv run main.py retrieval_check   step 35 on its own: does retrieval find the
                                     right document? (needs the index)

    uv run main.py ask         inference pathway: one applicant, one question,
                               one checked answer (needs Ollama running)
    uv run main.py evaluate    evaluation pathway: 24 questions through the
                               three configurations, with the paired tests

Each training block reads what it needs from disk and writes its own outputs, so
the blocks can be run one at a time. Running `data` once is enough; after that
`models`, `fairness` and `rag` can be repeated on their own.

Imports live inside the stage functions on purpose: `interpret` (EBM) and
`sentence_transformers` (torch) take seconds to import, and the inference
pathway must not pay for them.
"""

import argparse
import sys
import time


# ------------------------------------------------------------------
# Block 1A - data (steps 1 to 11)
# ------------------------------------------------------------------
def stage_data():
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

    loaded_datasets = load_both_raw_datasets()          # step 1
    check_missing_and_duplicates(loaded_datasets)       # step 2
    run_leakage_scan(loaded_datasets)                   # step 3
    find_derived_columns(loaded_datasets)               # step 4

    common = keep_common_features(loaded_datasets)      # step 5
    harmonized = harmonise_labels_and_units(common)     # step 6
    cleaned = check_ranges_and_clean(harmonized)        # step 7
    final = add_loan_to_income_and_age_groups(cleaned)  # step 8

    splits_per_dataset = split_70_15_15(final)          # step 9
    fit_preprocessors(splits_per_dataset)               # step 10
    write_reports_and_governance_log()                  # step 11


# ------------------------------------------------------------------
# Block 1B - models (steps 12 to 23)
# ------------------------------------------------------------------
def stage_models():
    from explainable_ai_project.load_split import check_all_splits_load
    from explainable_ai_project.logistic_baseline import train_logistic_baselines
    from explainable_ai_project.sweep_problem import show_sweep_problem
    from explainable_ai_project.monotonic_forest_a import train_monotonic_forest_a
    from explainable_ai_project.tune_forest import tune_and_retrain_a
    from explainable_ai_project.monotonic_forest_b import train_monotonic_forest_b
    from explainable_ai_project.ebm_comparison import run_ebm_comparison
    from explainable_ai_project.calibrate import calibrate_models
    from explainable_ai_project.recheck_monotonic import recheck_monotonicity
    from explainable_ai_project.testing_once import test_once
    from explainable_ai_project.test_set_sweep import sweep_on_test_set
    from explainable_ai_project.model_card_2 import save_log_and_write_model_card

    check_all_splits_load()             # step 12
    train_logistic_baselines()          # step 13
    show_sweep_problem()                # step 14
    train_monotonic_forest_a()          # step 15
    tune_and_retrain_a()                # step 16
    train_monotonic_forest_b()          # step 17
    run_ebm_comparison()                # step 18
    calibrate_models()                  # step 19
    recheck_monotonicity()              # step 20
    test_once()                         # step 21
    sweep_on_test_set()                 # step 22
    save_log_and_write_model_card()     # step 23


# ------------------------------------------------------------------
# Block 1C - Topic 3 checks before the LLM (steps 24 to 29)
# ------------------------------------------------------------------
def stage_fairness():
    from explainable_ai_project.age_approval import approval_rates_by_age_group
    from explainable_ai_project.maximum_age_fairness import maximum_amount_by_age_group
    from explainable_ai_project.age_group_calibration import calibration_by_age_group
    from explainable_ai_project.deferral import choose_deferral_rules
    from explainable_ai_project.intervention_age import intervention_remove_age
    from explainable_ai_project.fairness_report_2 import write_fairness_report

    approval_rates_by_age_group()       # step 24
    maximum_amount_by_age_group()       # step 25
    calibration_by_age_group()          # step 26
    choose_deferral_rules()             # step 27
    intervention_remove_age()           # step 28
    write_fairness_report()             # step 29


# ------------------------------------------------------------------
# Block 1D - RAG index (steps 30 to 35)
# ------------------------------------------------------------------
def stage_rag():
    from explainable_ai_project.documents_for_rag import collect_documents
    from explainable_ai_project.extract_clean import extract_and_clean_text
    from explainable_ai_project.chunk import chunk_documents
    from explainable_ai_project.embed import embed_chunks
    from explainable_ai_project.faiss_indexing import build_faiss_index

    collect_documents()                 # step 30
    extract_and_clean_text()            # step 31
    chunk_documents()                   # step 32
    embed_chunks()                      # step 33
    build_faiss_index()                 # step 34
    stage_retrieval_check()             # step 35


# ------------------------------------------------------------------
# Block 1D, step 35 - does retrieval find the right documents?
# ------------------------------------------------------------------
def stage_retrieval_check():
    from explainable_ai_project.retrieve import retrieval_check

    retrieval_check()


# ------------------------------------------------------------------
# Topic 2 - TreeSHAP explanations
# ------------------------------------------------------------------
def stage_shap():
    from explainable_ai_project.explain import run_shap_explanations

    run_shap_explanations()


# ------------------------------------------------------------------
# Whole training pathway
# ------------------------------------------------------------------
def stage_train():
    stage_data()
    stage_models()
    stage_fairness()
    stage_shap()
    stage_rag()


# ------------------------------------------------------------------
# Inference and evaluation (both need Ollama)
# ------------------------------------------------------------------
def stage_ask():
    from explainable_ai_project.assistant import run_ask

    run_ask()


def stage_evaluate(reuse_answers=False):
    from explainable_ai_project.evaluation import run_evaluation

    return run_evaluation(reuse_answers=reuse_answers)


STAGES = {
    "data": stage_data,
    "models": stage_models,
    "fairness": stage_fairness,
    "shap": stage_shap,
    "rag": stage_rag,
    "retrieval_check": stage_retrieval_check,
    "train": stage_train,
    "ask": stage_ask,
    "evaluate": stage_evaluate,
}


def main():
    parser = argparse.ArgumentParser(
        description="Trustworthy loan assessment assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "stage",
        choices=sorted(STAGES),
        help="which block to run",
    )
    parser.add_argument(
        "--reuse-answers",
        action="store_true",
        help="evaluate only: redo the analysis on the answers already "
             "generated, without asking the model anything again",
    )
    arguments = parser.parse_args()

    stage_name = arguments.stage
    stage_function = STAGES[stage_name]

    if arguments.reuse_answers and stage_name != "evaluate":
        parser.error("--reuse-answers only applies to the evaluate stage")

    print("=" * 70)
    print("running:", stage_name)
    print("=" * 70)

    started_at = time.time()
    if stage_name == "evaluate":
        exit_code = stage_function(reuse_answers=arguments.reuse_answers)
    else:
        exit_code = stage_function()
    seconds = round(time.time() - started_at, 1)

    print()
    print("=" * 70)
    print("finished:", stage_name, "in", seconds, "seconds")
    print("=" * 70)

    return exit_code or 0


if __name__ == "__main__":
    sys.exit(main())
