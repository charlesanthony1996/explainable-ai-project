"""
Training pathway, block 1D, step 31: extract and clean text.

Turns every document collected in step 30 into clean plain text, page by page,
ready to be cut into chunks (step 32).

PDFs (rag_docs/external/):
    1. Extract the text of each page with pypdf in "layout" mode.
       (The normal mode breaks justified text into pieces like
        "creditworthi ness"; layout mode keeps the words whole.)
    2. Skip contents pages (most lines end with a page number).
    3. Remove lines repeated at the top or bottom of most pages
       (running headers and footers such as
        "FINAL REPORT – GUIDELINES ON LOAN ORIGINATION AND MONITORING"
        or "02024R1689 — EN — 27.07.2026 — 001.001 — 11") and bare page numbers.
    4. Remove the EUR-Lex amendment markers (►B, ▼M1, ◄).
    5. Join words split across lines with a soft hyphen ("appli- cations").
    6. Join the lines of each paragraph into one line, keep blank lines
       between paragraphs, squeeze repeated spaces, and remove spaces
       left in front of punctuation ("outputs ;" -> "outputs;").

Project documents (rag_docs/project/*.md):
    Kept almost as they are (headings and tables matter for chunking);
    only whitespace is tidied.

Each document becomes one JSON Lines file: one line per page, with the
document id, title, file, page number and clean text. The page number is
kept so the assistant can cite "EU AI Act, page 132".

Put this file in:  src/explainable_ai_project/extract_clean_text.py

Reads:
    rag_docs/sources.csv          (from step 30: which documents are ready)
    rag_docs/external/*.pdf, rag_docs/project/*.md

Writes:
    rag_index/clean_text/<document id>.jsonl
    results/tables/rag_text_summary.csv
"""

import json
import re
from collections import Counter
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

RAG_DOCS_FOLDER = PROJECT_ROOT / "rag_docs"
SOURCES_FILE = RAG_DOCS_FOLDER / "sources.csv"
CLEAN_TEXT_FOLDER = PROJECT_ROOT / "rag_index" / "clean_text"
TABLES_FOLDER = PROJECT_ROOT / "results" / "tables"

# ------------------------------------------------------------------
# Cleaning settings
# ------------------------------------------------------------------
EDGE_LINES = 3                # lines at the top and bottom of a page checked for headers/footers
REPEATED_SHARE = 0.5          # a line on at least half of all pages counts as header/footer
CONTENTS_SHARE = 0.6          # a page where 60 % of lines end in a page number is a contents page
CONTENTS_MIN_LINES = 8

# EUR-Lex marks amendments with symbols like ►B, ▼M1, ◄
AMENDMENT_MARKER_PATTERN = r"[►▼◄]\s*(?:[BMACR]\d*)?"
SOFT_HYPHEN = "\u00ad"

# Words that must survive cleaning: used to check nothing important was broken
CHECK_WORDS = ["creditworthiness", "human oversight", "transparency"]


# ==================================================================
# PDF: extract
# ==================================================================

def extract_pdf_pages(file_path):
    """Raw text of every page, using layout mode (keeps words whole)."""

    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        page_text = page.extract_text(extraction_mode="layout")
        if page_text is None:
            page_text = ""
        pages.append(page_text)
    return pages


def page_lines(page_text):
    """Lines of a page with the extra layout spaces squeezed, empty lines kept as ''."""

    lines = []
    for raw_line in page_text.split("\n"):
        squeezed = re.sub(r"[ \t]+", " ", raw_line)
        stripped = squeezed.strip()
        lines.append(stripped)
    return lines


# ==================================================================
# PDF: find contents pages and repeated headers/footers
# ==================================================================

def is_contents_page(lines):
    """Most non-empty lines end with a page number, like 'Executive Summary 4'."""

    non_empty = []
    for line in lines:
        if line != "":
            non_empty.append(line)

    if len(non_empty) < CONTENTS_MIN_LINES:
        return False

    ending_in_number = 0
    for line in non_empty:
        if re.search(r"\s\d{1,3}$", line):
            ending_in_number += 1

    share = ending_in_number / len(non_empty)
    return share >= CONTENTS_SHARE


def line_pattern(line):
    """The same line with every number replaced by '#', so 'page 11' and 'page 12' match."""
    pattern = re.sub(r"\d+", "#", line)
    return pattern


def edge_lines_of(lines):
    """The first and last few non-empty lines of a page."""

    non_empty = []
    for line in lines:
        if line != "":
            non_empty.append(line)

    top = non_empty[:EDGE_LINES]
    bottom = non_empty[-EDGE_LINES:]
    edges = top + bottom
    return edges


def find_repeated_patterns(all_page_lines):
    """Line patterns that appear at the top or bottom of at least half of the pages."""

    page_count = len(all_page_lines)
    counts = Counter()

    for lines in all_page_lines:
        patterns_on_page = set()
        for line in edge_lines_of(lines):
            patterns_on_page.add(line_pattern(line))
        for pattern in patterns_on_page:
            counts[pattern] += 1

    repeated = set()
    for pattern, count in counts.items():
        share = count / page_count
        if share >= REPEATED_SHARE:
            repeated.add(pattern)
    return repeated


# ==================================================================
# PDF: clean one page
# ==================================================================

def remove_header_and_footer_lines(lines, repeated_patterns):
    kept = []
    removed_count = 0
    for line in lines:
        is_repeated = line != "" and line_pattern(line) in repeated_patterns
        is_page_number = re.fullmatch(r"\d{1,3}", line) is not None
        if is_repeated or is_page_number:
            removed_count += 1
            continue
        kept.append(line)
    return kept, removed_count


def join_paragraphs(lines):
    """Join the lines of each paragraph; blank lines separate paragraphs."""

    paragraphs = []
    current = []

    for line in lines:
        if line == "":
            if len(current) > 0:
                paragraphs.append(" ".join(current))
                current = []
        else:
            current.append(line)

    if len(current) > 0:
        paragraphs.append(" ".join(current))

    text = "\n\n".join(paragraphs)
    return text


def clean_page_text(lines, repeated_patterns):
    kept_lines, removed_count = remove_header_and_footer_lines(lines, repeated_patterns)

    # Amendment markers can stand alone on a line or start one
    without_markers = []
    for line in kept_lines:
        cleaned_line = re.sub(AMENDMENT_MARKER_PATTERN, "", line)
        cleaned_line = cleaned_line.strip()
        is_only_marker = cleaned_line == "" and line != ""
        if is_only_marker:
            continue
        without_markers.append(cleaned_line)

    text = join_paragraphs(without_markers)

    # "appli\u00ad cations" -> "applications" (soft hyphen at a line end)
    soft_hyphen_break = SOFT_HYPHEN + " "
    text = text.replace(soft_hyphen_break, "")
    text = text.replace(SOFT_HYPHEN, "")

    text = re.sub(r"[ ]{2,}", " ", text)

    # "outputs ;" -> "outputs;" (layout mode sometimes leaves a space before punctuation)
    text = re.sub(r" ([,;:.)])", r"\1", text)

    text = text.strip()
    return text, removed_count


def clean_pdf(file_path):
    """Clean pages of one PDF. Returns a list of (page number, text) and some counts."""

    raw_pages = extract_pdf_pages(file_path)

    all_page_lines = []
    for raw_page in raw_pages:
        all_page_lines.append(page_lines(raw_page))

    repeated_patterns = find_repeated_patterns(all_page_lines)

    clean_pages = []
    contents_pages = 0
    removed_lines = 0

    for position in range(len(all_page_lines)):
        lines = all_page_lines[position]
        page_number = position + 1

        if is_contents_page(lines):
            contents_pages += 1
            continue

        text, removed_count = clean_page_text(lines, repeated_patterns)
        removed_lines += removed_count

        if text == "":
            continue
        clean_pages.append((page_number, text))

    counts = {
        "pages_in_file": len(raw_pages),
        "contents_pages_skipped": contents_pages,
        "header_footer_lines_removed": removed_lines,
        "repeated_patterns": sorted(repeated_patterns),
    }
    return clean_pages, counts


# ==================================================================
# Markdown project documents
# ==================================================================

def clean_markdown(file_path):
    """Markdown keeps its headings and tables; only whitespace is tidied."""

    text = file_path.read_text(encoding="utf-8")
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    # A markdown file has no pages: it is stored as "page" 0
    clean_pages = [(0, text)]
    counts = {
        "pages_in_file": 0,
        "contents_pages_skipped": 0,
        "header_footer_lines_removed": 0,
        "repeated_patterns": [],
    }
    return clean_pages, counts


# ==================================================================
# The step
# ==================================================================

def read_ready_documents():
    if not SOURCES_FILE.exists():
        raise FileNotFoundError("Missing rag_docs/sources.csv. Run step 30 (collect documents) first.")

    sources = pd.read_csv(SOURCES_FILE)
    is_ready = sources["status"] == "ok"
    ready = sources[is_ready]

    not_ready = sources[~is_ready]
    for row in not_ready.itertuples(index=False):
        print("  skipped (not ready in step 30):", row.file)

    return ready


def count_check_words(clean_pages):
    all_text_parts = []
    for page_number, text in clean_pages:
        all_text_parts.append(text)
    all_text = " ".join(all_text_parts)
    all_text_lower = all_text.lower()

    counts = {}
    for word in CHECK_WORDS:
        counts[word] = all_text_lower.count(word)
    return counts, len(all_text.split())


def save_jsonl(document, clean_pages):
    CLEAN_TEXT_FOLDER.mkdir(parents=True, exist_ok=True)
    output_file = CLEAN_TEXT_FOLDER / (document.id + ".jsonl")

    with open(output_file, "w", encoding="utf-8") as file:
        for page_number, text in clean_pages:
            record = {
                "doc_id": document.id,
                "title": document.title,
                "file": document.file,
                "page": page_number,
                "text": text,
            }
            line = json.dumps(record, ensure_ascii=False)
            file.write(line + "\n")

    return output_file


def extract_and_clean_text():
    """Step 31: clean text of every ready document, saved page by page."""

    documents = read_ready_documents()
    summary_rows = []

    for document in documents.itertuples(index=False):
        file_path = RAG_DOCS_FOLDER / document.file
        print(document.id)

        if file_path.suffix.lower() == ".pdf":
            clean_pages, counts = clean_pdf(file_path)
        else:
            clean_pages, counts = clean_markdown(file_path)

        word_counts, total_words = count_check_words(clean_pages)
        output_file = save_jsonl(document, clean_pages)

        if counts["pages_in_file"] > 0:
            print("  pages:", counts["pages_in_file"], "| contents pages skipped:", counts["contents_pages_skipped"],
                  "| header/footer lines removed:", counts["header_footer_lines_removed"])
            for pattern in counts["repeated_patterns"]:
                print("  removed repeated line:", pattern)
        print("  words:", total_words, "| check words:", word_counts)
        print("  saved", output_file.relative_to(PROJECT_ROOT))

        summary_row = {
            "doc_id": document.id,
            "kind": document.kind,
            "pages_in_file": counts["pages_in_file"],
            "pages_kept": len(clean_pages),
            "contents_pages_skipped": counts["contents_pages_skipped"],
            "header_footer_lines_removed": counts["header_footer_lines_removed"],
            "words": total_words,
        }
        for word, count in word_counts.items():
            summary_row["count_" + word.replace(" ", "_")] = count
        summary_rows.append(summary_row)

    summary_table = pd.DataFrame(summary_rows)
    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)
    summary_file = TABLES_FOLDER / "rag_text_summary.csv"
    summary_table.to_csv(summary_file, index=False)
    print()
    print("Saved", summary_file.relative_to(PROJECT_ROOT))

    return summary_table