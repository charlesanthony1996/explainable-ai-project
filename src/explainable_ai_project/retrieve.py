"""
Inference pathway, step 10: find the documents that support an answer.

Searches the FAISS index built in training step 34. The question is turned into
a vector with the SAME embedding model that made the index (read from
rag_index/embedding_info.json, so the two can never drift apart), then the
closest chunks are returned with their source, section and page.

Because every vector was normalised to length 1 in step 33, the inner product
that FAISS reports IS the cosine similarity:

    sim(u, v) = (u . v) / (|u| |v|)   and with |u| = |v| = 1:   sim = u . v

so a score of 1.0 means "same meaning" and 0.0 means "unrelated".

The index, the chunk texts and the embedding model are loaded once and kept in
module-level caches. The first question therefore costs a few seconds (the model
is read from disk), every later question is fast.

This module also contains training step 35, the retrieval check, which was
missing: 15 realistic questions whose answer is known to live in a particular
document, and the share of them where a chunk from that document really appears
in the top 4. Without this number there is no evidence that retrieval works at
all, and the B-versus-C comparison in the evaluation cannot be interpreted.

Usage
-----
    from explainable_ai_project.retrieve import retrieve
    chunks = retrieve("Is credit scoring high-risk under the EU AI Act?", top_k=4)

    uv run main.py retrieval_check        # step 35

Reads:
    rag_index/faiss.index          the searchable index      (step 34)
    rag_index/index_chunks.json    chunk details in index order
    rag_index/embedding_info.json  which model made the index (step 33)

Writes (retrieval check only):
    results/tables/rag_retrieval_check.csv      one row per question
    results/tables/rag_retrieval_summary.csv    the hit rate
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Where everything is
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PACKAGE_FOLDER = THIS_FILE.parent
SRC_FOLDER = PACKAGE_FOLDER.parent
PROJECT_ROOT = SRC_FOLDER.parent

RAG_INDEX_FOLDER = PROJECT_ROOT / "rag_index"
INDEX_FILE = RAG_INDEX_FOLDER / "faiss.index"
INDEX_CHUNKS_FILE = RAG_INDEX_FOLDER / "index_chunks.json"
EMBEDDING_INFO_FILE = RAG_INDEX_FOLDER / "embedding_info.json"

TABLES_FOLDER = PROJECT_ROOT / "results" / "tables"

DEFAULT_TOP_K = 4

# Loaded once, then kept
_index = None
_chunks = None
_model = None


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

def _require(file_path):
    if not file_path.exists():
        message = (
            "Missing " + str(file_path.relative_to(PROJECT_ROOT))
            + " - run `uv run main.py rag` first."
        )
        raise FileNotFoundError(message)
    return file_path


def load_index():
    """The FAISS index (cached)."""

    global _index
    if _index is None:
        import faiss

        _require(INDEX_FILE)
        _index = faiss.read_index(str(INDEX_FILE))
    return _index


def load_chunks():
    """Chunk details in index order: position -> chunk (cached)."""

    global _chunks
    if _chunks is None:
        _require(INDEX_CHUNKS_FILE)
        with open(INDEX_CHUNKS_FILE, "r", encoding="utf-8") as file:
            _chunks = json.load(file)
    return _chunks


def embedding_model_name():
    """Which model made the index. Reading it from the index means the question
    can never be embedded with a different model than the chunks were."""

    _require(EMBEDDING_INFO_FILE)
    with open(EMBEDDING_INFO_FILE, "r", encoding="utf-8") as file:
        info = json.load(file)
    return info["model"]


def load_model():
    """The sentence-transformers model (cached, imported lazily)."""

    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        name = embedding_model_name()
        print("Loading the embedding model", name, "...")
        _model = SentenceTransformer(name)
    return _model


# ------------------------------------------------------------------
# Searching
# ------------------------------------------------------------------

def source_label(chunk):
    """A short citation label, e.g. 'eba_guidelines p. 12' or 'fairness_report'."""

    label = chunk["doc_id"]
    page_start = chunk.get("page_start", 0)

    if page_start and page_start > 0:
        page_end = chunk.get("page_end", page_start)
        if page_end and page_end != page_start:
            label = label + " p. " + str(page_start) + "-" + str(page_end)
        else:
            label = label + " p. " + str(page_start)

    return label


def retrieve(question, top_k=DEFAULT_TOP_K):
    """The chunks closest in meaning to the question.

    Returns a list of dictionaries, best match first:
        chunk_id, doc_id, title, section, page_start, page_end, text
        score   cosine similarity, 1.0 = same meaning
        source  short citation label for the prompt
    """

    index = load_index()
    chunks = load_chunks()
    model = load_model()

    question_vector = model.encode(
        [question],
        normalize_embeddings=True,     # must match step 33
        show_progress_bar=False,
    )
    question_vector = np.asarray(question_vector, dtype="float32")

    scores, positions = index.search(question_vector, top_k)

    found = []
    for place in range(len(positions[0])):
        position = int(positions[0][place])
        if position < 0:               # FAISS uses -1 when it has no more hits
            continue

        chunk = dict(chunks[position])
        chunk["score"] = round(float(scores[0][place]), 4)
        chunk["source"] = source_label(chunk)
        found.append(chunk)

    return found


def format_chunks_for_prompt(chunks):
    """The retrieved chunks as text to put in the prompt.

    Every chunk is labelled with its source so the model can cite it, and the
    answer check can verify that a citation points at something real.
    """

    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        header = (
            "[" + str(number) + "] source: " + chunk["source"]
            + " | section: " + str(chunk.get("section", ""))
        )
        blocks.append(header + "\n" + chunk["text"])

    return "\n\n".join(blocks)


# ------------------------------------------------------------------
# Training step 35: does retrieval actually work?
# ------------------------------------------------------------------

# 15 questions a user of this system could really ask. For each one, the
# documents that must supply the answer. A question counts as a hit if at least
# one chunk from an expected document is in the top 4.
CHECK_QUESTIONS = [
    ("Is credit scoring a high-risk AI system under the EU AI Act?",
     ["eu_ai_act"]),
    ("What does the EU AI Act require about the quality of training data?",
     ["eu_ai_act"]),
    ("What information must be given to the people who use a high-risk AI system?",
     ["eu_ai_act"]),
    ("What does human oversight of a high-risk AI system have to make possible?",
     ["eu_ai_act"]),
    ("Do I have a right to an explanation of a decision made about me by an AI system?",
     ["eu_ai_act"]),
    ("How should a lender assess whether a borrower can repay a loan?",
     ["eba_guidelines"]),
    ("What do the guidelines say about using automated models in lending decisions?",
     ["eba_guidelines"]),
    ("What should be checked about a borrower's income and financial situation?",
     ["eba_guidelines"]),
    ("How often does the model approve applicants aged 61 and over?",
     ["fairness_report"]),
    ("Is the model equally well calibrated for every age group?",
     ["fairness_report"]),
    ("Did removing age from the model reduce the differences between age groups?",
     ["fairness_report"]),
    ("Which settings were chosen for the random forest, and how was it calibrated?",
     ["model_card"]),
    ("What does loan-to-income mean, and what counts as a high value?",
     ["feature_glossary"]),
    ("Was there any leakage found in the datasets?",
     ["data_quality"]),
    ("Where do the two loan datasets come from and how large are they?",
     ["datasheet", "data_quality"]),
]


def retrieval_check(top_k=DEFAULT_TOP_K):
    """Step 35: the share of questions where the right document is in the top k."""

    rows = []

    for question, expected_documents in CHECK_QUESTIONS:
        chunks = retrieve(question, top_k=top_k)

        found_documents = [chunk["doc_id"] for chunk in chunks]
        hit = any(document in expected_documents for document in found_documents)

        best_place = ""
        for place, document in enumerate(found_documents, start=1):
            if document in expected_documents:
                best_place = place
                break

        rows.append({
            "question": question,
            "expected": ", ".join(expected_documents),
            "retrieved": ", ".join(found_documents),
            "best_score": chunks[0]["score"] if chunks else np.nan,
            "place_of_first_hit": best_place,
            "hit": hit,
        })

    table = pd.DataFrame(rows)
    hit_rate = table["hit"].mean()

    places = pd.to_numeric(table["place_of_first_hit"], errors="coerce")
    mean_place = places.mean()

    print()
    print("Retrieval check: is a chunk from the right document in the top", top_k, "?")
    print(table[["expected", "retrieved", "place_of_first_hit", "hit"]].to_string(index=False))
    print()
    print("hit rate:", round(hit_rate, 4), "(" + str(int(table["hit"].sum())),
          "of", str(len(table)) + ")")
    print("average place of the first correct chunk:", round(mean_place, 2))

    misses = table[~table["hit"]]
    if len(misses) > 0:
        print()
        print("Questions that found nothing from the expected document:")
        for _, row in misses.iterrows():
            print("  -", row["question"])
            print("    expected:", row["expected"], "| got:", row["retrieved"])

    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)

    table_file = TABLES_FOLDER / "rag_retrieval_check.csv"
    table.to_csv(table_file, index=False)
    print()
    print("Saved", table_file.relative_to(PROJECT_ROOT))

    summary = pd.DataFrame([{
        "questions": len(table),
        "top_k": top_k,
        "hits": int(table["hit"].sum()),
        "hit_rate": round(hit_rate, 4),
        "average_place_of_first_hit": round(mean_place, 2),
        "embedding_model": embedding_model_name(),
    }])
    summary_file = TABLES_FOLDER / "rag_retrieval_summary.csv"
    summary.to_csv(summary_file, index=False)
    print("Saved", summary_file.relative_to(PROJECT_ROOT))

    return table
