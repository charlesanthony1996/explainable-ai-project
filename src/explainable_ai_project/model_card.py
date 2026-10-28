"""
Training pathway, block 1B, step 23: save, log, model card (Topic 3: revocation, Topic 5: governance).

Three jobs:

1. LOG every saved model in results/governance_log.csv with its md5 hash
   (same log as step 11), so anyone can check later that a model file
   has not changed since the results were produced.

2. REVOCATION DEMO: an applicant withdraws consent and asks for their data
   to be removed. This step shows what that means in practice:
       - remove one applicant from the training data
       - retrain model A without them (same settings)
       - compare the old and new model: hashes, validation AUC,
         and how much the predictions changed
       - log the event
   The new model is saved as a separate demo file; the production model A
   is NOT replaced. (A real revocation would also redo calibration and the
   steps after it.)

3. MODEL CARD: docs/model_card.md, a short standard document describing
   what the models are for, how they were built, how well they work and
   where they should not be trusted. Numbers are filled in from the results
   tables; the team's own text comes from docs/model_card_notes.md
   (created once, never overwritten).

Put this file in:  src/explainable_ai_project/model_card.py

Writes:
    results/governance_log.csv                   (adds model rows and the revocation event)
    models/rf_monotonic_dataset_1_revocation_demo.joblib
    results/tables/revocation_demo.csv
    docs/model_card.md
    docs/model_card_notes.md                     (only the first time)
"""

import datetime
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from explainable_ai_project.load_split import load_split
from explainable_ai_project.monotonic_forest_a import build_monotonic_forest
from explainable_ai_project.monotonic_forest_a import monotonic_model_file
from explainable_ai_project.tune_forest import load_best_settings


# ------------------------------------------------------------------
# Where everything is
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PACKAGE_FOLDER = THIS_FILE.parent
SRC_FOLDER = PACKAGE_FOLDER.parent
PROJECT_ROOT = SRC_FOLDER.parent

MODELS_FOLDER = PROJECT_ROOT / "models"
TABLES_FOLDER = PROJECT_ROOT / "results" / "tables"
DOCS_FOLDER = PROJECT_ROOT / "docs"
GOVERNANCE_LOG_FILE = PROJECT_ROOT / "results" / "governance_log.csv"

MODEL_CARD_FILE = DOCS_FOLDER / "model_card.md"
MODEL_CARD_NOTES_FILE = DOCS_FOLDER / "model_card_notes.md"

VERSION = "v1"
DATASET_NAMES = ["dataset_1", "dataset_2"]
LOG_COLUMNS = ["date", "step", "file", "md5", "size_kb", "note"]

# The applicant removed in the revocation demo: the first one in dataset 1's training part
REVOCATION_DATASET = "dataset_1"


# ==================================================================
# Small helpers
# ==================================================================

def now_text():
    now = datetime.datetime.now()
    text = now.strftime("%Y-%m-%d %H:%M")
    return text


def compute_md5(file_path):
    hasher = hashlib.md5()
    with open(file_path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    md5_text = hasher.hexdigest()
    return md5_text


def read_log():
    log_exists = GOVERNANCE_LOG_FILE.exists()
    if not log_exists:
        empty_log = pd.DataFrame(columns=LOG_COLUMNS)
        return empty_log
    log_table = pd.read_csv(GOVERNANCE_LOG_FILE)
    return log_table


def add_log_rows(new_rows):
    """Append rows to the governance log. Rows for a file+hash already logged are skipped."""

    log_table = read_log()

    already_logged = set()
    for row in log_table.itertuples(index=False):
        pair = (row.file, row.md5)
        already_logged.add(pair)

    rows_to_add = []
    for row in new_rows:
        pair = (row["file"], row["md5"])
        if pair in already_logged:
            continue
        rows_to_add.append(row)

    if len(rows_to_add) > 0:
        new_table = pd.DataFrame(rows_to_add)
        log_table = pd.concat([log_table, new_table], ignore_index=True)

    GOVERNANCE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log_table.to_csv(GOVERNANCE_LOG_FILE, index=False)
    return len(rows_to_add)


def log_row_for(file_path, step, note):
    relative_text = str(file_path.relative_to(PROJECT_ROOT))
    size_in_kb = file_path.stat().st_size / 1024

    row = {
        "date": now_text(),
        "step": step,
        "file": relative_text,
        "md5": compute_md5(file_path),
        "size_kb": round(size_in_kb, 1),
        "note": note,
    }
    return row


def table_to_markdown(table):
    columns = list(table.columns)

    header_cells = []
    separator_cells = []
    for column in columns:
        header_cells.append(str(column))
        separator_cells.append("---")

    lines = []
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("| " + " | ".join(separator_cells) + " |")

    for row in table.itertuples(index=False):
        cells = []
        for value in row:
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")

    text = "\n".join(lines)
    return text


def read_table(file_name):
    file_path = TABLES_FOLDER / file_name
    file_exists = file_path.exists()
    if not file_exists:
        return None
    table = pd.read_csv(file_path)
    return table


# ==================================================================
# 1. Log every saved model
# ==================================================================

# (file name start, step, note) for each dataset
MODEL_FILES = [
    ("logistic_", ".joblib", "13 logistic baseline", "logistic regression baseline"),
    ("rf_unconstrained_", ".joblib", "14 sweep problem", "unconstrained Random Forest (comparison only)"),
    ("rf_monotonic_", ".joblib", "15-17 monotonic forest", "monotonic Random Forest, tuned, before calibration"),
    ("ebm_", ".joblib", "18 EBM", "monotonic EBM (Topic 4 comparison)"),
    ("rf_calibrated_", ".joblib", "19 calibrate", "FINAL model: calibrated monotonic Random Forest"),
]

SETTINGS_FILES = [
    ("best_settings_", ".json", "16-17 tuning", "chosen Random Forest settings"),
    ("sweep_limits_", ".json", "22 sweep", "amount and loan-to-income limits for the sweep"),
]


def log_saved_models():
    new_rows = []
    missing_files = []

    for dataset_name in DATASET_NAMES:
        for file_start, file_end, step, note in MODEL_FILES:
            file_path = MODELS_FOLDER / (file_start + dataset_name + "_" + VERSION + file_end)
            if file_path.exists():
                row = log_row_for(file_path, step, note + ", " + dataset_name)
                new_rows.append(row)
            else:
                missing_files.append(file_path.name)

        for file_start, file_end, step, note in SETTINGS_FILES:
            file_path = MODELS_FOLDER / (file_start + dataset_name + file_end)
            if file_path.exists():
                row = log_row_for(file_path, step, note + ", " + dataset_name)
                new_rows.append(row)
            else:
                missing_files.append(file_path.name)

    added_count = add_log_rows(new_rows)
    print("Governance log:", added_count, "new model rows")
    if len(missing_files) > 0:
        print("  not found:", missing_files)


# ==================================================================
# 2. Revocation demo
# ==================================================================

def revocation_demo():
    dataset_name = REVOCATION_DATASET
    print("Revocation demo on", dataset_name)

    train = load_split(dataset_name, "train")
    validation = load_split(dataset_name, "validation")

    # The applicant who withdraws consent
    removed_position = 0
    removed_id = train["rows"]["applicant_id"].iloc[removed_position]
    print("  applicant withdrawing consent:", removed_id)

    keep_flags = train["rows"]["applicant_id"] != removed_id
    kept_features = train["features"][keep_flags]
    kept_labels = train["labels"][keep_flags]
    removed_features = train["features"].iloc[[removed_position]]

    # Old model (production model A, before calibration)
    old_file = monotonic_model_file(dataset_name)
    old_model = joblib.load(old_file)
    old_hash = compute_md5(old_file)

    # New model: same settings, without the applicant
    settings = load_best_settings(dataset_name)
    feature_names = list(train["features"].columns)
    new_model = build_monotonic_forest(feature_names, settings)
    new_model.fit(kept_features, kept_labels)

    new_file = MODELS_FOLDER / ("rf_monotonic_" + dataset_name + "_revocation_demo.joblib")
    joblib.dump(new_model, new_file)
    new_hash = compute_md5(new_file)

    # Compare old and new
    old_validation_table = old_model.predict_proba(validation["features"])
    new_validation_table = new_model.predict_proba(validation["features"])
    old_validation = old_validation_table[:, 1]
    new_validation = new_validation_table[:, 1]

    old_auc = roc_auc_score(validation["labels"], old_validation)
    new_auc = roc_auc_score(validation["labels"], new_validation)

    differences = np.abs(new_validation - old_validation)
    mean_difference = differences.mean()
    largest_difference = differences.max()

    old_on_removed_table = old_model.predict_proba(removed_features)
    new_on_removed_table = new_model.predict_proba(removed_features)
    old_on_removed = old_on_removed_table[0, 1]
    new_on_removed = new_on_removed_table[0, 1]

    result = {
        "date": now_text(),
        "dataset": dataset_name,
        "removed_applicant": removed_id,
        "training_rows_before": len(train["labels"]),
        "training_rows_after": len(kept_labels),
        "old_model_md5": old_hash,
        "new_model_md5": new_hash,
        "validation_auc_before": round(old_auc, 4),
        "validation_auc_after": round(new_auc, 4),
        "mean_prediction_change": round(mean_difference, 5),
        "largest_prediction_change": round(largest_difference, 5),
        "removed_applicant_p_before": round(old_on_removed, 4),
        "removed_applicant_p_after": round(new_on_removed, 4),
    }

    for name, value in result.items():
        print("  " + name + ":", value)

    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)
    result_table = pd.DataFrame([result])
    result_file = TABLES_FOLDER / "revocation_demo.csv"
    result_table.to_csv(result_file, index=False)
    print("  saved", result_file.relative_to(PROJECT_ROOT))

    note = "applicant " + removed_id + " removed; model retrained; old md5 " + old_hash
    log_row = log_row_for(new_file, "23 revocation", note)
    add_log_rows([log_row])
    print("  logged the revocation in", GOVERNANCE_LOG_FILE.relative_to(PROJECT_ROOT))
    print()

    return result


# ==================================================================
# 3. Model card
# ==================================================================

NOTES_TEMPLATE = """# Model card notes (written by the team)

This file is created once and never overwritten. Its text is copied into
docs/model_card.md every time step 23 runs.

## Intended use
TODO (academic proof of concept: estimate the largest loan amount an applicant
would likely be approved for, and explain it; supports a human, never decides alone)

## Out of scope
TODO (real lending decisions; applicants outside the training ranges;
any use without human review - credit scoring is high-risk under the EU AI Act, Annex III)

## Ethical considerations
TODO (age differences in approval; what the fairness checks in 1C found)

## Caveats
TODO (synthetic data; dataset 1's maximum often equals the upper sweep limit;
the EBM was as accurate as the Random Forest on the test part)
"""


def make_notes_file_once():
    notes_exist = MODEL_CARD_NOTES_FILE.exists()
    if notes_exist:
        return
    DOCS_FOLDER.mkdir(parents=True, exist_ok=True)
    MODEL_CARD_NOTES_FILE.write_text(NOTES_TEMPLATE, encoding="utf-8")
    print("Created", MODEL_CARD_NOTES_FILE.relative_to(PROJECT_ROOT), "- fill in the TODO parts")


def add_section(lines, title, table, note=None):
    lines.append("## " + title)
    lines.append("")
    if note is not None:
        lines.append(note)
        lines.append("")
    if table is None:
        lines.append("Not available yet: run the step that makes this table.")
    else:
        lines.append(table_to_markdown(table))
    lines.append("")


def settings_table():
    rows = []
    for dataset_name in DATASET_NAMES:
        settings_file = MODELS_FOLDER / ("best_settings_" + dataset_name + ".json")
        if not settings_file.exists():
            continue
        with open(settings_file, "r", encoding="utf-8") as file:
            settings = json.load(file)
        row = {"dataset": dataset_name}
        for name, value in settings.items():
            row[name] = value
        rows.append(row)

    if len(rows) == 0:
        return None
    table = pd.DataFrame(rows)
    return table


def write_model_card():
    make_notes_file_once()

    lines = []
    lines.append("# Model card: loan maximum-amount assistant")
    lines.append("")
    lines.append("Numbers filled in automatically on " + now_text()
                 + ". The team's text comes from `docs/model_card_notes.md`.")
    lines.append("")

    lines.append("## Model details")
    lines.append("")
    lines.append("- **Final models:** model A (dataset 1) and model B (dataset 2), each a Random Forest classifier.")
    lines.append("- **Monotonic constraint:** loan_amount and loan_to_income may only lower the approval "
                 "probability, so the maximum-amount sweep is valid (checked in step 20).")
    lines.append("- **Calibration:** a correction learned on the validation part, so probabilities can be "
                 "taken literally (step 19).")
    lines.append("- **Output:** P(approve) at the requested amount, and the highest amount with "
                 "P(approve) >= 0.8 (also 0.7 and 0.9).")
    lines.append("- **Inputs:** age, dependents, annual income, credit score, loan amount, loan term, "
                 "loan-to-income, marital status, education, employment status, loan purpose.")
    lines.append("")

    add_section(lines, "Tuned settings", settings_table())
    add_section(lines, "Calibration", read_table("calibration_choice.csv"))

    test_results = read_table("test_results.csv")
    if test_results is not None:
        columns_to_show = ["dataset", "model", "auc", "accuracy", "precision", "recall", "f1", "brier"]
        test_results = test_results[columns_to_show]
    add_section(lines, "Results on the untouched test part", test_results,
                note="Every model scored once on the test part (step 21). The final model is "
                     "random_forest_monotonic_calibrated.")

    add_section(lines, "Confusion matrices (cut-off 0.5)", read_table("confusion_matrices.csv"))
    add_section(lines, "Monotonicity check", read_table("monotonicity_check.csv"))
    add_section(lines, "Maximum-amount sweep on the test part", read_table("sweep_summary.csv"))
    add_section(lines, "Revocation demo", read_table("revocation_demo.csv"))

    notes_text = MODEL_CARD_NOTES_FILE.read_text(encoding="utf-8")
    lines.append("---")
    lines.append("")
    lines.append(notes_text)

    text = "\n".join(lines)
    MODEL_CARD_FILE.write_text(text, encoding="utf-8")
    print("Saved", MODEL_CARD_FILE.relative_to(PROJECT_ROOT))


# ==================================================================
# The step
# ==================================================================

def save_log_and_write_model_card():
    """Step 23: log every model, run the revocation demo, write the model card."""

    log_saved_models()
    print()
    revocation_demo()
    write_model_card()