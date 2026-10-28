"""
Training pathway, block 1A, step 11: reports and governance log.

Collects what steps 1 to 10 saved and writes three things:

    docs/data_quality.md          a readable report of every 1A check, built from the
                                  tables in results/tables/. Rewritten on every run.

    docs/datasheet.md             facts about both datasets (sources, size, features,
                                  age groups) plus the team's own text from
                                  docs/datasheet_notes.md. Rewritten on every run.

    results/governance_log.csv    one row per data file and model file with its md5 hash,
                                  so anyone can check that a file has not changed.
                                  Only new rows are added; old rows are never deleted.

docs/datasheet_notes.md is created ONCE with headings for the team to fill in.
It is never overwritten, so your text is safe.

This step reads everything from disk, so it can run on its own after steps 1 to 10.

Put this file in:  src/explainable_ai_project/reports_governance.py
Run it with:       uv run python -m explainable_ai_project.reports_governance
"""

import datetime
import hashlib
import json
from pathlib import Path

import pandas as pd

from explainable_ai_project.load import datasets


# ------------------------------------------------------------------
# Where everything is
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PACKAGE_FOLDER = THIS_FILE.parent
SRC_FOLDER = PACKAGE_FOLDER.parent
PROJECT_ROOT = SRC_FOLDER.parent

RAW_FOLDER = PROJECT_ROOT / "datasets" / "raw"
PROCESSED_FOLDER = PROJECT_ROOT / "datasets" / "processed"
MODELS_FOLDER = PROJECT_ROOT / "models"
TABLES_FOLDER = PROJECT_ROOT / "results" / "tables"
DOCS_FOLDER = PROJECT_ROOT / "docs"

DATA_QUALITY_FILE = DOCS_FOLDER / "data_quality.md"
DATASHEET_FILE = DOCS_FOLDER / "datasheet.md"
DATASHEET_NOTES_FILE = DOCS_FOLDER / "datasheet_notes.md"
GOVERNANCE_LOG_FILE = PROJECT_ROOT / "results" / "governance_log.csv"

DATASET_NAMES = ["dataset_1", "dataset_2"]
VERSION = "v1"


# ==================================================================
# Small helpers
# ==================================================================

def read_table(file_name):
    """Read a table from results/tables/, or return None if that step has not run yet."""

    file_path = TABLES_FOLDER / file_name
    file_exists = file_path.exists()
    if not file_exists:
        print("  missing", file_name, "- run the step that makes it first")
        return None

    table = pd.read_csv(file_path)
    return table


def table_to_markdown(table):
    """Turn a table into a Markdown table, one line per row."""

    columns = list(table.columns)

    header_cells = []
    separator_cells = []
    for column in columns:
        header_cells.append(str(column))
        separator_cells.append("---")

    header = "| " + " | ".join(header_cells) + " |"
    separator = "| " + " | ".join(separator_cells) + " |"
    lines = [header, separator]

    for row in table.itertuples(index=False):
        cells = []
        for value in row:
            cells.append(str(value))
        line = "| " + " | ".join(cells) + " |"
        lines.append(line)

    text = "\n".join(lines)
    return text


def add_table_section(lines, title, table, note=None):
    """Add a heading, an optional sentence and a table (or a 'not available' line)."""

    lines.append("### " + title)
    lines.append("")
    if note is not None:
        lines.append(note)
        lines.append("")
    if table is None:
        lines.append("Not available yet: run the step that makes this table.")
    elif len(table) == 0:
        lines.append("None.")
    else:
        lines.append(table_to_markdown(table))
    lines.append("")


def compute_md5(file_path):
    """md5 hash of a file, read in 1 MB pieces."""

    hasher = hashlib.md5()
    with open(file_path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    md5_text = hasher.hexdigest()
    return md5_text


def now_text():
    now = datetime.datetime.now()
    text = now.strftime("%Y-%m-%d %H:%M")
    return text


# ==================================================================
# 1. Governance log
# ==================================================================

def files_to_log():
    """Every file 1A produced: (step, file path, note)."""

    entries = []

    for dataset_name in DATASET_NAMES:
        raw_file = datasets[dataset_name]["file"]
        entries.append(("1 load", raw_file, "raw data, " + dataset_name))

    for dataset_name in DATASET_NAMES:
        clean_file = PROCESSED_FOLDER / (dataset_name + "_clean.csv")
        entries.append(("8 add features", clean_file, "cleaned data with loan_to_income and age_group"))

    for dataset_name in DATASET_NAMES:
        split_file = PROCESSED_FOLDER / ("splits_" + dataset_name + ".csv")
        entries.append(("9 split", split_file, "train / validation / test IDs, seed 42"))

    for dataset_name in DATASET_NAMES:
        preprocessor_file = MODELS_FOLDER / ("preprocessor_" + dataset_name + "_" + VERSION + ".joblib")
        entries.append(("10 preprocessor", preprocessor_file, "fitted on the training part only"))

        names_file = MODELS_FOLDER / ("feature_names_" + dataset_name + "_" + VERSION + ".json")
        entries.append(("10 preprocessor", names_file, "model input columns in order"))

    return entries


def read_existing_log():
    log_exists = GOVERNANCE_LOG_FILE.exists()
    if not log_exists:
        columns = ["date", "step", "file", "md5", "size_kb", "note"]
        empty_log = pd.DataFrame(columns=columns)
        return empty_log

    existing_log = pd.read_csv(GOVERNANCE_LOG_FILE)
    return existing_log


def update_governance_log():
    """Add a row for every file that is new or has changed since it was last logged."""

    existing_log = read_existing_log()

    # Every (file, md5) pair already in the log
    already_logged = set()
    for row in existing_log.itertuples(index=False):
        pair = (row.file, row.md5)
        already_logged.add(pair)

    new_rows = []
    missing_files = []

    for step, file_path, note in files_to_log():
        file_exists = file_path.exists()
        if not file_exists:
            missing_files.append(file_path.name)
            continue

        relative_text = str(file_path.relative_to(PROJECT_ROOT))
        md5_text = compute_md5(file_path)

        pair = (relative_text, md5_text)
        if pair in already_logged:
            continue

        size_in_kb = file_path.stat().st_size / 1024
        size_in_kb = round(size_in_kb, 1)

        new_rows.append({
            "date": now_text(),
            "step": step,
            "file": relative_text,
            "md5": md5_text,
            "size_kb": size_in_kb,
            "note": note,
        })

    if len(new_rows) > 0:
        new_table = pd.DataFrame(new_rows)
        updated_log = pd.concat([existing_log, new_table], ignore_index=True)
    else:
        updated_log = existing_log

    GOVERNANCE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    updated_log.to_csv(GOVERNANCE_LOG_FILE, index=False)

    print("Governance log:", len(new_rows), "new rows,", len(updated_log), "rows in total")
    if len(missing_files) > 0:
        print("  not found (run the step that makes them):", missing_files)

    return updated_log


# ==================================================================
# 2. Data quality report
# ==================================================================

def leakage_section(lines):
    lines.append("## 3. Leakage scan (step 3)")
    lines.append("")
    lines.append("Single-feature AUC: how well each raw column alone predicts approval "
                 "(0.5 = not at all, 1.0 = perfectly). Above 0.95 counts as leakage.")
    lines.append("")
    for dataset_name in DATASET_NAMES:
        table = read_table(dataset_name + "_leakage_scan.csv")
        flagged = None
        if table is not None:
            is_flagged = table["verdict"] != "ok"
            flagged = table[is_flagged]
        add_table_section(lines, dataset_name + ": columns that are not 'ok'", flagged)


def derived_section(lines):
    lines.append("## 4. Columns tied to the loan amount (step 4)")
    lines.append("")
    lines.append("Columns that move with the loan amount. The amount sweep changes only the amount, "
                 "so model inputs among these limit how far the sweep may go.")
    lines.append("")
    for dataset_name in DATASET_NAMES:
        table = read_table(dataset_name + "_derived_columns.csv")
        tied = None
        if table is not None:
            is_tied = table["link"] != "not tied"
            tied = table[is_tied]
        add_table_section(lines, dataset_name, tied)


def labels_section(lines):
    lines.append("## 6. Labels and units (step 6)")
    lines.append("")

    unmapped = read_table("unmapped_values.csv")
    add_table_section(lines, "Category values not in the map (became 'Other')", unmapped)

    counts = read_table("category_counts.csv")
    side_by_side = None
    if counts is not None:
        side_by_side = counts.pivot_table(
            index=["column", "label"],
            columns="dataset",
            values="rows",
            aggfunc="sum",
            fill_value=0,
        )
        side_by_side = side_by_side.astype(int)
        side_by_side = side_by_side.reset_index()
    add_table_section(lines, "Shared labels and their counts", side_by_side)

    units = read_table("unit_comparison.csv")
    add_table_section(lines, "Number columns side by side", units)


def cleaning_section(lines):
    lines.append("## 7. Range checks and cleaning (step 7)")
    lines.append("")

    summary = read_table("cleaning_summary.csv")
    add_table_section(lines, "Rows removed per reason", summary)

    for dataset_name in DATASET_NAMES:
        table = read_table(dataset_name + "_range_checks.csv")
        add_table_section(lines, dataset_name + ": range rules", table)


def preprocessor_section(lines):
    lines.append("## 10. Preprocessor (step 10)")
    lines.append("")
    lines.append("Fitted on the training part only. Numbers pass through; categories are one-hot encoded.")
    lines.append("")

    for dataset_name in DATASET_NAMES:
        names_file = MODELS_FOLDER / ("feature_names_" + dataset_name + "_" + VERSION + ".json")
        names_exist = names_file.exists()

        lines.append("### " + dataset_name)
        lines.append("")
        if names_exist:
            with open(names_file, "r", encoding="utf-8") as file:
                feature_names = json.load(file)
            lines.append(str(len(feature_names)) + " model features: " + ", ".join(feature_names))
        else:
            lines.append("Not available yet: run step 10.")
        lines.append("")


def write_data_quality_report():
    lines = []
    lines.append("# Data quality report")
    lines.append("")
    lines.append("Written by `reports_governance.py` on " + now_text() + ". Do not edit by hand; rerun instead.")
    lines.append("")

    # 1 and 2
    lines.append("## 1–2. Raw datasets, missing values and duplicates (steps 1 and 2)")
    lines.append("")
    summary = read_table("missing_and_duplicates_summary.csv")
    add_table_section(lines, "Summary", summary)

    # 3 and 4
    leakage_section(lines)
    derived_section(lines)

    # 5
    lines.append("## 5. Common features (step 5)")
    lines.append("")
    common = read_table("common_features.csv")
    add_table_section(lines, "Which raw column became which shared name", common)

    # 6 and 7
    labels_section(lines)
    cleaning_section(lines)

    # 8
    lines.append("## 8. Loan-to-income and age groups (step 8)")
    lines.append("")
    ratios = read_table("loan_to_income_summary.csv")
    add_table_section(lines, "Loan-to-income", ratios)
    ages = read_table("age_group_summary.csv")
    add_table_section(lines, "Rows and approval rate per age group", ages)

    # 9
    lines.append("## 9. Split (step 9)")
    lines.append("")
    split_summary = read_table("split_summary.csv")
    add_table_section(lines, "Rows and approval rate per part", split_summary)
    split_groups = read_table("split_age_groups.csv")
    test_groups = None
    if split_groups is not None:
        is_test = split_groups["split"] == "test"
        test_groups = split_groups[is_test]
    add_table_section(lines, "Age groups in the test part", test_groups)

    # 10
    preprocessor_section(lines)

    lines.append("## Governance log")
    lines.append("")
    lines.append("Every data and model file with its md5 hash: `results/governance_log.csv`.")
    lines.append("")

    text = "\n".join(lines)
    DOCS_FOLDER.mkdir(parents=True, exist_ok=True)
    DATA_QUALITY_FILE.write_text(text, encoding="utf-8")
    print("Saved", DATA_QUALITY_FILE.relative_to(PROJECT_ROOT))


# ==================================================================
# 3. Datasheet
# ==================================================================

NOTES_TEMPLATE = """# Datasheet notes (written by the team)

This file is created once and never overwritten. Its text is copied into
docs/datasheet.md every time the reports step runs.

## How the approval labels were made
TODO

## Evidence that the data is synthetic
TODO (see docs/data_quality.md: for example, approval rises with the requested amount
in dataset 1, and nobody in dataset 1 requested more than 40 % of their income)

## Leakage and derived columns
TODO (RiskScore in dataset 2 almost perfectly decides approval; MonthlyLoanPayment is
calculated from the amount; neither is a model input)

## Limitations
TODO (currency unknown, different credit score ranges, no true maximum amount,
small 61+ group in dataset 2)

## Privacy and consent
TODO (synthetic records, no real people; no individual records go into the RAG documents)

## Licence
TODO (licence of dataset 2 on Kaggle)
"""


def make_notes_file_once():
    notes_exist = DATASHEET_NOTES_FILE.exists()
    if notes_exist:
        return

    DOCS_FOLDER.mkdir(parents=True, exist_ok=True)
    DATASHEET_NOTES_FILE.write_text(NOTES_TEMPLATE, encoding="utf-8")
    print("Created", DATASHEET_NOTES_FILE.relative_to(PROJECT_ROOT), "- fill in the TODO parts")


def write_datasheet():
    make_notes_file_once()

    lines = []
    lines.append("# Datasheet")
    lines.append("")
    lines.append("Facts are filled in automatically on " + now_text()
                 + ". The team's text comes from `docs/datasheet_notes.md`.")
    lines.append("")

    lines.append("## Sources")
    lines.append("")
    for dataset_name in DATASET_NAMES:
        url = datasets[dataset_name]["url"]
        lines.append("- " + dataset_name + ": " + url)
    lines.append("")

    summary = read_table("missing_and_duplicates_summary.csv")
    add_table_section(lines, "Size", summary)

    common = read_table("common_features.csv")
    add_table_section(lines, "Model features (plus loan_to_income = loan amount / income)", common)

    ages = read_table("age_group_summary.csv")
    add_table_section(lines, "Age groups (used for fairness checks only, not a model input)", ages)

    notes_text = DATASHEET_NOTES_FILE.read_text(encoding="utf-8")
    lines.append("---")
    lines.append("")
    lines.append(notes_text)

    text = "\n".join(lines)
    DATASHEET_FILE.write_text(text, encoding="utf-8")
    print("Saved", DATASHEET_FILE.relative_to(PROJECT_ROOT))


# ==================================================================
# All three
# ==================================================================

def write_reports_and_governance_log():
    """Run the step. Reads everything from disk, so it needs no arguments."""

    update_governance_log()
    print()
    write_data_quality_report()
    write_datasheet()


# if __name__ == "__main__":
#     write_reports_and_governance_log()