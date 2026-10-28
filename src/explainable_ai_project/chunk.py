"""
Training pathway, block 1D, step 32: chunk.

Cuts the clean text from step 31 into small pieces ("chunks"). The assistant
later searches these chunks and gives the best few to Qwen, so each chunk must be:
    - small enough to be embedded whole (see CHUNK SIZE below)
    - about ONE thing: a chunk never mixes two sections
    - traceable: it keeps its document, section heading and page numbers,
      so an answer can cite "EU AI Act, Article 86, page 108"

How it works, per document:
    1. Walk through the paragraphs in page order.
    2. Recognise section headings and remember the current one:
         EU AI Act       "Article 86 Right to explanation ...", "ANNEX III ...", "CHAPTER III ..."
         EBA Guidelines  "4.3.4 Models for creditworthiness assessment ..."
         markdown files  lines starting with "#"
       A new section always starts a new chunk.
    3. Add paragraphs to the current chunk until it reaches the size limit.
       A paragraph that is too long on its own is split at sentence ends.
    4. The next chunk starts with the last few sentences' worth of words of the
       previous one (the OVERLAP), so a sentence cut at the border is still
       found in one piece.

CHUNK SIZE:
    The plan said about 500 tokens. The embedding model planned for step 33
    (all-MiniLM-L6-v2) only reads the first 256 tokens of a text and silently
    ignores the rest. One word is roughly 1.3 tokens in this kind of text, so
    chunks are limited to 180 words (about 235 tokens) with a 40-word overlap.
    Step 33 counts the tokens and reports any chunk that would still be cut off.

Each chunk also gets an "embedding text": the same text with its document title
and section in front, e.g. "EU AI Act | Article 86 Right to explanation ... | <text>".
That helps the search find the chunk when the question names the law or the article.

Put this file in:  src/explainable_ai_project/chunk_documents.py

Reads:
    rag_index/clean_text/*.jsonl         (from step 31)

Writes:
    rag_index/chunks.jsonl               one line per chunk
    results/tables/rag_chunk_summary.csv
"""

import json
import re
from pathlib import Path

import pandas as pd


# ------------------------------------------------------------------
# Where everything is
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PACKAGE_FOLDER = THIS_FILE.parent
SRC_FOLDER = PACKAGE_FOLDER.parent
PROJECT_ROOT = SRC_FOLDER.parent

CLEAN_TEXT_FOLDER = PROJECT_ROOT / "rag_index" / "clean_text"
CHUNKS_FILE = PROJECT_ROOT / "rag_index" / "chunks.jsonl"
TABLES_FOLDER = PROJECT_ROOT / "results" / "tables"

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
CHUNK_WORDS = 180          # largest chunk, in words
OVERLAP_WORDS = 40         # words repeated from the end of the previous chunk
MIN_CHUNK_WORDS = 25       # a smaller leftover is merged into the chunk before it

# A chunk can start with OVERLAP_WORDS carried over, so a new piece may add at most
# this many words. That keeps every chunk at or below CHUNK_WORDS.
PIECE_WORDS = CHUNK_WORDS - OVERLAP_WORDS
HEADING_MAX_WORDS = 20     # a heading is short
DROP_BELOW_WORDS = 5       # chunks shorter than this (e.g. a lone "EBA/GL/2020/06") are dropped

# Short titles used in front of the embedding text
SHORT_TITLES = {
    "eba_guidelines": "EBA Guidelines on loan origination and monitoring",
    "eu_ai_act": "EU AI Act",
    "datasheet": "Project datasheet",
    "data_quality": "Project data quality report",
    "model_card": "Project model card",
    "fairness_report": "Project fairness report",
    "feature_glossary": "Project feature glossary",
}


# ==================================================================
# Reading the clean text
# ==================================================================

def read_clean_documents():
    """Every document from step 31, as a list of its page records."""

    if not CLEAN_TEXT_FOLDER.exists():
        raise FileNotFoundError("Missing rag_index/clean_text/. Run step 31 (extract and clean text) first.")

    documents = {}
    for file_path in sorted(CLEAN_TEXT_FOLDER.glob("*.jsonl")):
        records = []
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                records.append(record)
        doc_id = file_path.stem
        documents[doc_id] = records
    return documents


def paragraphs_with_pages(records):
    """All paragraphs of a document in order, each with its page number."""

    paragraphs = []
    for record in records:
        page = record["page"]
        pieces = record["text"].split("\n\n")
        for piece in pieces:
            paragraph = piece.strip()
            if paragraph != "":
                paragraphs.append((page, paragraph))
    return paragraphs


# ==================================================================
# Recognising section headings
# ==================================================================

def heading_of(paragraph, doc_id):
    """The heading text if this paragraph starts a new section, otherwise None.

    Also returns the rest of the paragraph (text after a markdown heading line).
    """

    words = paragraph.split()
    is_short = len(words) <= HEADING_MAX_WORDS

    # Markdown: a first line starting with '#'
    first_line = paragraph.split("\n")[0]
    if first_line.startswith("#"):
        heading = first_line.lstrip("#").strip()
        rest_lines = paragraph.split("\n")[1:]
        rest = "\n".join(rest_lines).strip()
        return heading, rest

    if doc_id == "eu_ai_act":
        starts_like_heading = re.match(r"^(Article \d+[a-z]?|ANNEX [IVXLC]+|CHAPTER [IVXLC]+|SECTION \d+)\b", paragraph)
        if starts_like_heading and is_short:
            return paragraph, ""

    if doc_id == "eba_guidelines":
        numbered_heading = re.match(r"^\d+(\.\d+)*\.?\s+[A-Z]", paragraph)
        ends_like_sentence = paragraph.endswith(".") or paragraph.endswith(";")
        if numbered_heading and is_short and not ends_like_sentence:
            return paragraph, ""

    return None, paragraph


# ==================================================================
# Splitting text that is too long
# ==================================================================

def split_long_paragraph(paragraph):
    """Split a paragraph longer than PIECE_WORDS at sentence ends (or, if needed, by words)."""

    sentences = re.split(r"(?<=[.;:!?])\s+", paragraph)
    pieces = []
    current_words = []

    for sentence in sentences:
        sentence_words = sentence.split()

        # A single sentence longer than a piece is cut by words
        if len(sentence_words) > PIECE_WORDS:
            if len(current_words) > 0:
                pieces.append(" ".join(current_words))
                current_words = []
            for start in range(0, len(sentence_words), PIECE_WORDS):
                part = sentence_words[start:start + PIECE_WORDS]
                pieces.append(" ".join(part))
            continue

        too_big = len(current_words) + len(sentence_words) > PIECE_WORDS
        if too_big and len(current_words) > 0:
            pieces.append(" ".join(current_words))
            current_words = []

        current_words.extend(sentence_words)

    if len(current_words) > 0:
        pieces.append(" ".join(current_words))

    return pieces


# ==================================================================
# Building the chunks of one document
# ==================================================================

class ChunkBuilder:
    """Collects paragraphs into chunks for one document."""

    def __init__(self, doc_id, title):
        self.doc_id = doc_id
        self.title = title
        self.chunks = []
        self.section = ""
        self.current_texts = []
        self.current_word_count = 0
        self.current_pages = []
        self.has_content = False

    def start_section(self, heading):
        # Two headings in a row (e.g. "CHAPTER III ..." then "SECTION 1 ..."):
        # combine them instead of making a chunk that holds only a heading
        only_heading_so_far = self.current_word_count > 0 and not self.has_content
        if only_heading_so_far:
            self.section = self.section + " > " + heading
            return True

        self.finish_chunk(keep_overlap=False)
        self.section = heading
        return False

    def add_paragraph(self, page, paragraph, is_heading=False):
        if not is_heading:
            self.has_content = True

        paragraph_word_count = len(paragraph.split())

        if paragraph_word_count > PIECE_WORDS:
            pieces = split_long_paragraph(paragraph)
        else:
            pieces = [paragraph]

        for piece in pieces:
            piece_word_count = len(piece.split())
            too_big = self.current_word_count + piece_word_count > CHUNK_WORDS
            if too_big and self.current_word_count > 0:
                self.finish_chunk(keep_overlap=True)

            self.current_texts.append(piece)
            self.current_word_count = self.current_word_count + piece_word_count
            self.current_pages.append(page)

    def finish_chunk(self, keep_overlap):
        if self.current_word_count == 0:
            return

        text = "\n\n".join(self.current_texts)
        is_tiny = self.current_word_count < MIN_CHUNK_WORDS
        previous_exists = len(self.chunks) > 0

        # A tiny leftover in the same section is added to the previous chunk instead
        if is_tiny and previous_exists and self.chunks[-1]["section"] == self.section:
            previous = self.chunks[-1]
            previous["text"] = previous["text"] + "\n\n" + text
            previous["words"] = len(previous["text"].split())
            previous["page_end"] = max(previous["page_end"], max(self.current_pages))
        else:
            chunk = {
                "doc_id": self.doc_id,
                "title": self.title,
                "section": self.section,
                "page_start": min(self.current_pages),
                "page_end": max(self.current_pages),
                "text": text,
                "words": self.current_word_count,
            }
            self.chunks.append(chunk)

        # Start the next chunk with the end of this one (overlap)
        if keep_overlap:
            all_words = text.split()
            overlap_words = all_words[-OVERLAP_WORDS:]
            overlap_text = " ".join(overlap_words)
            last_page = max(self.current_pages)
            self.current_texts = [overlap_text]
            self.current_word_count = len(overlap_words)
            self.current_pages = [last_page]
        else:
            self.current_texts = []
            self.current_word_count = 0
            self.current_pages = []
            self.has_content = False

    def finish_document(self):
        self.finish_chunk(keep_overlap=False)
        return self.chunks


def drop_tiny_chunks(chunks):
    kept = []
    for chunk in chunks:
        if chunk["words"] >= DROP_BELOW_WORDS:
            kept.append(chunk)
    return kept


def chunk_one_document(doc_id, records):
    title = records[0]["title"]
    builder = ChunkBuilder(doc_id, title)

    for page, paragraph in paragraphs_with_pages(records):
        heading, rest = heading_of(paragraph, doc_id)

        if heading is not None:
            builder.start_section(heading)
            # The heading text is also part of the chunk, so it can be found by search
            builder.add_paragraph(page, heading, is_heading=True)

        if rest != "":
            builder.add_paragraph(page, rest)

    chunks = builder.finish_document()
    chunks = drop_tiny_chunks(chunks)
    return chunks


def add_ids_and_embedding_text(chunks, doc_id):
    short_title = SHORT_TITLES.get(doc_id, doc_id)

    for number in range(len(chunks)):
        chunk = chunks[number]
        chunk_id = doc_id + "-" + str(number + 1).zfill(4)
        chunk["chunk_id"] = chunk_id

        header_parts = [short_title]
        if chunk["section"] != "":
            header_parts.append(chunk["section"])
        header = " | ".join(header_parts)
        chunk["embedding_text"] = header + " | " + chunk["text"]


# ==================================================================
# The step
# ==================================================================

def chunk_documents():
    """Step 32: cut every clean document into chunks and save them."""

    documents = read_clean_documents()
    all_chunks = []
    summary_rows = []

    for doc_id, records in documents.items():
        chunks = chunk_one_document(doc_id, records)
        add_ids_and_embedding_text(chunks, doc_id)
        all_chunks.extend(chunks)

        word_counts = []
        sections = set()
        for chunk in chunks:
            word_counts.append(chunk["words"])
            sections.add(chunk["section"])

        summary_row = {
            "doc_id": doc_id,
            "chunks": len(chunks),
            "sections": len(sections),
            "mean_words": round(sum(word_counts) / len(word_counts), 1),
            "largest_chunk_words": max(word_counts),
            "smallest_chunk_words": min(word_counts),
        }
        summary_rows.append(summary_row)
        print(doc_id + ":", len(chunks), "chunks in", len(sections), "sections")

    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as file:
        for chunk in all_chunks:
            line = json.dumps(chunk, ensure_ascii=False)
            file.write(line + "\n")

    summary_table = pd.DataFrame(summary_rows)
    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)
    summary_file = TABLES_FOLDER / "rag_chunk_summary.csv"
    summary_table.to_csv(summary_file, index=False)

    print()
    print(summary_table.to_string(index=False))
    print()
    print("Total:", len(all_chunks), "chunks")
    print("Saved", CHUNKS_FILE.relative_to(PROJECT_ROOT))
    print("Saved", summary_file.relative_to(PROJECT_ROOT))

    return all_chunks