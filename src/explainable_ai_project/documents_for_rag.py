"""
Training pathway, block 1D, step 30: collect documents (Topic 1: trustworthy AI framework).

The RAG index (steps 31 to 35) is built from the files in rag_docs/.
Only what is in this folder can ever be cited by the assistant, so this step
decides what the assistant is allowed to know.

    rag_docs/
        external/   official documents, downloaded BY HAND (see below)
        project/    the project's own documents, copied here automatically
        sources.csv one row per document: title, source, status, pages, md5

External documents (download once, save under exactly these names):

    1. rag_docs/external/eba_gl_2020_06_loan_origination.pdf
       EBA Guidelines on loan origination and monitoring (EBA/GL/2020/06).
       Page: https://www.eba.europa.eu/regulation-and-policy/credit-risk/guidelines-on-loan-origination-and-monitoring
       Download the PDF called "Guidelines on loan origination and monitoring".

    2. rag_docs/external/eu_ai_act_2024_1689.pdf
       EU Artificial Intelligence Act, Regulation (EU) 2024/1689.
       Page: https://eur-lex.europa.eu/eli/reg/2024/1689/oj
       Download the English PDF. The Act was amended in 2026; if EUR-Lex offers a
       consolidated version (the current text including amendments), use that one.

Project documents (copied from docs/, made by earlier steps):
    datasheet.md (step 11), data_quality.md (step 11),
    model_card.md (step 23), fairness_report.md (step 29)
plus a feature glossary written by this step, which explains every input and
output in plain words so the assistant can explain terms like "loan-to-income".

No individual applicant records go into rag_docs/. The step checks the
project documents for applicant IDs and stops if it finds any.

Put this file in:  src/explainable_ai_project/collect_documents.py
Needs:             uv add "pypdf==6.20.0"

Writes:
    rag_docs/project/*.md
    rag_docs/sources.csv
    results/governance_log.csv     (adds one row per document)
"""

import datetime
import hashlib
import re
import shutil
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


# ------------------------------------------------------------------
# Where everything is
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PACKAGE_FOLDER = THIS_FILE.parent
SRC_FOLDER = PACKAGE_FOLDER.parent
PROJECT_ROOT = SRC_FOLDER.parent

DOCS_FOLDER = PROJECT_ROOT / "docs"
RAG_DOCS_FOLDER = PROJECT_ROOT / "rag_docs"
EXTERNAL_FOLDER = RAG_DOCS_FOLDER / "external"
PROJECT_FOLDER = RAG_DOCS_FOLDER / "project"
SOURCES_FILE = RAG_DOCS_FOLDER / "sources.csv"
GOVERNANCE_LOG_FILE = PROJECT_ROOT / "results" / "governance_log.csv"

LOG_COLUMNS = ["date", "step", "file", "md5", "size_kb", "note"]
APPLICANT_ID_PATTERN = r"\bD[12]-\d+\b"


# ------------------------------------------------------------------
# The documents
# ------------------------------------------------------------------
EXTERNAL_DOCUMENTS = [
    {
        "id": "eba_guidelines",
        "title": "EBA Guidelines on loan origination and monitoring (EBA/GL/2020/06)",
        "file": "eba_gl_2020_06_loan_origination.pdf",
        "source": "https://www.eba.europa.eu/regulation-and-policy/credit-risk/guidelines-on-loan-origination-and-monitoring",
        "why": "how lenders must assess creditworthiness, including when using automated models",
        # Words that must appear, to check it is the right document
        "check_words": ["creditworthiness", "loan origination"],
    },
    {
        "id": "eu_ai_act",
        "title": "EU Artificial Intelligence Act, Regulation (EU) 2024/1689",
        "file": "eu_ai_act_2024_1689.pdf",
        "source": "https://eur-lex.europa.eu/eli/reg/2024/1689/oj",
        "why": "credit scoring is high-risk (Annex III); data governance (Art. 10), "
               "transparency (Art. 13), human oversight (Art. 14), right to explanation (Art. 86)",
        "check_words": ["Annex III", "creditworthiness", "human oversight"],
    },
]

PROJECT_DOCUMENTS = [
    {
        "id": "datasheet",
        "title": "Datasheet of the two loan datasets",
        "file": "datasheet.md",
        "made_by": "step 11",
    },
    {
        "id": "data_quality",
        "title": "Data quality report",
        "file": "data_quality.md",
        "made_by": "step 11",
    },
    {
        "id": "model_card",
        "title": "Model card of models A and B",
        "file": "model_card.md",
        "made_by": "step 23",
    },
    {
        "id": "fairness_report",
        "title": "Fairness report",
        "file": "fairness_report.md",
        "made_by": "step 29",
    },
]

GLOSSARY_FILE_NAME = "feature_glossary.md"

GLOSSARY_TEXT = """# Feature glossary: what every input and output means

This glossary explains, in plain words, every value the loan assessment assistant
uses or reports.

## Inputs (what the applicant provides)

- **Age:** the applicant's age in years. Used by the models; also used to form age
  groups (18-30, 31-45, 46-60, 61+) for the fairness checks.
- **Dependents:** the number of people who depend financially on the applicant.
- **Annual income:** yearly income before tax, in the dataset's currency (not stated in the data).
- **Credit score:** a number summarising the applicant's credit history; higher is better.
  The two datasets use different ranges: about 300-850 in dataset 1 and about 340-710 in dataset 2.
- **Loan amount:** the amount of money requested.
- **Loan term:** how long the loan runs, in months.
- **Marital status, education, employment status, loan purpose:** categories describing
  the applicant and the loan.

## Values the system calculates

- **Loan-to-income:** the loan amount divided by the annual income. 0.3 means the loan
  equals 30 % of one year's income. Higher values make approval less likely.
- **P(approve):** the model's calibrated probability that a loan of this amount would be
  approved for this applicant. 0.8 means: among applicants given 0.8, about 80 % were approved.
- **Estimated maximum amount:** the highest loan amount for which P(approve) is still at
  least 0.8. It is an estimate based on past decisions in the data, not a promise.
- **Upper limit flag:** when the estimated maximum equals the highest amount the system was
  allowed to check, the real maximum may be higher; the data does not show it.
- **No amount found:** even the smallest amount checked does not reach P(approve) 0.8.

## Deferral (sending a case to a human)

- **Borderline:** P(approve) is very close to the 0.8 threshold.
- **Unusual input:** a value lies outside the range seen in the training data.
- **Missing data:** a required field is empty.
- **Disagreement:** the two models (trained on different datasets) give very different answers.
- **Weak group:** the applicant belongs to an age group the model cannot assess reliably.

## Models

- **Model A:** trained on dataset 1 (the course dataset).
- **Model B:** trained on dataset 2 (Financial Risk for Loan Approval).
- Both are Random Forests with a monotonic rule: asking for a larger loan can never raise
  P(approve). Their probabilities are calibrated.
"""


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

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


def log_documents(file_paths):
    """Add each document to the governance log, unless the same file and hash is already there."""

    if GOVERNANCE_LOG_FILE.exists():
        log_table = pd.read_csv(GOVERNANCE_LOG_FILE)
    else:
        log_table = pd.DataFrame(columns=LOG_COLUMNS)

    already_logged = set()
    for row in log_table.itertuples(index=False):
        pair = (row.file, row.md5)
        already_logged.add(pair)

    new_rows = []
    for file_path in file_paths:
        relative_text = str(file_path.relative_to(PROJECT_ROOT))
        md5_text = compute_md5(file_path)
        pair = (relative_text, md5_text)
        if pair in already_logged:
            continue

        size_in_kb = file_path.stat().st_size / 1024
        new_rows.append({
            "date": now_text(),
            "step": "30 collect documents",
            "file": relative_text,
            "md5": md5_text,
            "size_kb": round(size_in_kb, 1),
            "note": "RAG document",
        })

    if len(new_rows) > 0:
        new_table = pd.DataFrame(new_rows)
        log_table = pd.concat([log_table, new_table], ignore_index=True)
        GOVERNANCE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        log_table.to_csv(GOVERNANCE_LOG_FILE, index=False)

    print("Governance log:", len(new_rows), "new document rows")


# ------------------------------------------------------------------
# External documents: check what was downloaded by hand
# ------------------------------------------------------------------

def check_external_document(document):
    """Is the PDF there, can it be read, and is it the right document?"""

    file_path = EXTERNAL_FOLDER / document["file"]
    row = {
        "id": document["id"],
        "kind": "external",
        "title": document["title"],
        "file": "external/" + document["file"],
        "source": document["source"],
        "why_included": document["why"],
        "status": "",
        "pages": None,
        "md5": None,
    }

    file_exists = file_path.exists()
    if not file_exists:
        row["status"] = "MISSING - download by hand"
        print("  MISSING:", document["title"])
        print("    download from:", document["source"])
        print("    save as:", file_path.relative_to(PROJECT_ROOT))
        return row, None

    try:
        reader = PdfReader(file_path)
        page_count = len(reader.pages)
    except Exception as error:
        row["status"] = "UNREADABLE - download again"
        print("  UNREADABLE:", file_path.name, "-", error)
        return row, None

    # Read every page and look for the check words
    # (the AI Act mentions creditworthiness only once, in Annex III near the end)
    text_parts = []
    for page_number in range(page_count):
        page = reader.pages[page_number]
        page_text = page.extract_text()
        if page_text is not None:
            text_parts.append(page_text)
    sample_text = " ".join(text_parts)
    sample_text_lower = sample_text.lower()

    # PDF text extraction sometimes breaks words with a space ("creditworthi ness"),
    # so compare with all spaces and line breaks removed
    text_pieces = sample_text_lower.split()
    compact_text = "".join(text_pieces)

    missing_words = []
    for word in document["check_words"]:
        word_lower = word.lower()
        word_pieces = word_lower.split()
        compact_word = "".join(word_pieces)
        if compact_word not in compact_text:
            missing_words.append(word)

    if len(missing_words) > 0:
        row["status"] = "CHECK - expected words not found: " + ", ".join(missing_words)
        print("  CHECK:", file_path.name, "- expected words not found:", missing_words)
    else:
        row["status"] = "ok"
        print("  ok:", file_path.name, "-", page_count, "pages")

    row["pages"] = page_count
    row["md5"] = compute_md5(file_path)
    return row, file_path


# ------------------------------------------------------------------
# Project documents: copy them in
# ------------------------------------------------------------------

def check_no_applicant_ids(file_path):
    text = file_path.read_text(encoding="utf-8")
    found_ids = re.findall(APPLICANT_ID_PATTERN, text)
    if len(found_ids) > 0:
        message = file_path.name + " contains applicant IDs (" + str(found_ids[:3]) + "). "
        message = message + "Individual records must not go into rag_docs/."
        raise ValueError(message)


def copy_project_document(document):
    source_file = DOCS_FOLDER / document["file"]
    target_file = PROJECT_FOLDER / document["file"]

    row = {
        "id": document["id"],
        "kind": "project",
        "title": document["title"],
        "file": "project/" + document["file"],
        "source": "docs/" + document["file"] + " (" + document["made_by"] + ")",
        "why_included": "project documentation the assistant may cite",
        "status": "",
        "pages": None,
        "md5": None,
    }

    source_exists = source_file.exists()
    if not source_exists:
        row["status"] = "MISSING - run " + document["made_by"] + " first"
        print("  MISSING:", source_file.relative_to(PROJECT_ROOT), "- run", document["made_by"], "first")
        return row, None

    check_no_applicant_ids(source_file)
    shutil.copyfile(source_file, target_file)

    row["status"] = "ok"
    row["md5"] = compute_md5(target_file)
    print("  ok:", target_file.relative_to(PROJECT_ROOT))
    return row, target_file


def write_glossary():
    target_file = PROJECT_FOLDER / GLOSSARY_FILE_NAME
    target_file.write_text(GLOSSARY_TEXT, encoding="utf-8")

    row = {
        "id": "feature_glossary",
        "kind": "project",
        "title": "Feature glossary",
        "file": "project/" + GLOSSARY_FILE_NAME,
        "source": "written by step 30",
        "why_included": "plain-word definitions of every input and output",
        "status": "ok",
        "pages": None,
        "md5": compute_md5(target_file),
    }
    print("  ok:", target_file.relative_to(PROJECT_ROOT))
    return row, target_file


def remove_old_loose_copy():
    """Step 29 copied the fairness report straight into rag_docs/. It now lives in rag_docs/project/."""

    old_copy = RAG_DOCS_FOLDER / "fairness_report.md"
    if old_copy.exists():
        old_copy.unlink()
        print("  removed the old copy rag_docs/fairness_report.md (now in rag_docs/project/)")


# ------------------------------------------------------------------
# The step
# ------------------------------------------------------------------

def collect_documents():
    """Step 30: check the external PDFs, copy the project documents, write sources.csv."""

    EXTERNAL_FOLDER.mkdir(parents=True, exist_ok=True)
    PROJECT_FOLDER.mkdir(parents=True, exist_ok=True)

    source_rows = []
    collected_files = []

    print("External documents (downloaded by hand):")
    for document in EXTERNAL_DOCUMENTS:
        row, file_path = check_external_document(document)
        source_rows.append(row)
        if file_path is not None:
            collected_files.append(file_path)
    print()

    print("Project documents (copied from docs/):")
    remove_old_loose_copy()
    for document in PROJECT_DOCUMENTS:
        row, file_path = copy_project_document(document)
        source_rows.append(row)
        if file_path is not None:
            collected_files.append(file_path)

    row, file_path = write_glossary()
    source_rows.append(row)
    collected_files.append(file_path)
    print()

    sources_table = pd.DataFrame(source_rows)
    sources_table["checked_on"] = now_text()
    sources_table.to_csv(SOURCES_FILE, index=False)
    print("Saved", SOURCES_FILE.relative_to(PROJECT_ROOT))

    log_documents(collected_files)

    is_ok = sources_table["status"] == "ok"
    ok_count = int(is_ok.sum())
    total_count = len(sources_table)
    print(ok_count, "of", total_count, "documents ready for the RAG index")

    if ok_count < total_count:
        print("Fix the MISSING or CHECK documents above, then run this step again.")

    return sources_table