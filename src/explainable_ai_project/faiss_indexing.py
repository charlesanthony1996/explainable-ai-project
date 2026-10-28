"""
Training pathway, block 1D, step 34: FAISS index.

Puts the chunk vectors from step 33 into a FAISS index: a structure that,
given the vector of a question, finds the most similar chunk vectors quickly.

Index type: IndexFlatIP ("flat inner product").
    - "flat" = it compares the question with EVERY chunk, so results are exact.
      With about 900 chunks that takes well under a millisecond, so no
      approximate (faster but less exact) index is needed.
    - "inner product" = the score is the dot product of two vectors. Because
      step 33 normalised every vector to length 1, this score is exactly the
      cosine similarity: 1 = same meaning, 0 = unrelated.

The index only stores vectors, numbered 0, 1, 2, ... in the order of
rag_index/embedding_chunk_ids.json. To go from a search hit back to the chunk
text, page and section, this step also saves a lookup file with the chunks
in exactly that order.

Checks before saving:
    - the number of vectors, chunk ids and chunks all match
    - every chunk finds ITSELF as its best match (proves the order is right)
    - near-duplicate chunks are counted (they waste search results)

The index is kept separate from the loan models: it holds documents,
the models hold applicants. They only meet inside the prompt (inference).

Put this file in:  src/explainable_ai_project/faiss_index.py
Needs:             uv add "faiss-cpu==1.15.1"

Reads:
    rag_index/embeddings.npy, rag_index/embedding_chunk_ids.json   (step 33)
    rag_index/chunks.jsonl                                          (step 32)

Writes:
    rag_index/faiss.index          the searchable index
    rag_index/index_chunks.json    chunk details, in index order (position -> chunk)
    rag_index/index_info.json      size, type, date, hash of the vectors
"""

import datetime
import hashlib
import json
from pathlib import Path

import faiss
import numpy as np


# ------------------------------------------------------------------
# Where everything is
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PACKAGE_FOLDER = THIS_FILE.parent
SRC_FOLDER = PACKAGE_FOLDER.parent
PROJECT_ROOT = SRC_FOLDER.parent

RAG_INDEX_FOLDER = PROJECT_ROOT / "rag_index"
CHUNKS_FILE = RAG_INDEX_FOLDER / "chunks.jsonl"
EMBEDDINGS_FILE = RAG_INDEX_FOLDER / "embeddings.npy"
CHUNK_IDS_FILE = RAG_INDEX_FOLDER / "embedding_chunk_ids.json"

INDEX_FILE = RAG_INDEX_FOLDER / "faiss.index"
INDEX_CHUNKS_FILE = RAG_INDEX_FOLDER / "index_chunks.json"
INDEX_INFO_FILE = RAG_INDEX_FOLDER / "index_info.json"

# Two chunks this similar are treated as near-duplicates
DUPLICATE_SIMILARITY = 0.98

# What each chunk keeps in the lookup file (the search results need these)
FIELDS_TO_KEEP = ["chunk_id", "doc_id", "title", "section", "page_start", "page_end", "text"]


# ==================================================================
# Helpers
# ==================================================================

def compute_md5(file_path):
    hasher = hashlib.md5()
    with open(file_path, "rb") as file:
        while True:
            piece = file.read(1024 * 1024)
            if not piece:
                break
            hasher.update(piece)
    md5_text = hasher.hexdigest()
    return md5_text


def check_file_exists(file_path, step_name):
    if not file_path.exists():
        relative_path = file_path.relative_to(PROJECT_ROOT)
        message = "Missing " + str(relative_path) + ". Run " + step_name + " first."
        raise FileNotFoundError(message)


def read_inputs():
    check_file_exists(EMBEDDINGS_FILE, "step 33 (embed)")
    check_file_exists(CHUNK_IDS_FILE, "step 33 (embed)")
    check_file_exists(CHUNKS_FILE, "step 32 (chunk)")

    vectors = np.load(EMBEDDINGS_FILE)
    vectors = vectors.astype(np.float32)

    with open(CHUNK_IDS_FILE, "r", encoding="utf-8") as file:
        chunk_ids = json.load(file)

    chunks_by_id = {}
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        for line in file:
            chunk = json.loads(line)
            chunks_by_id[chunk["chunk_id"]] = chunk

    return vectors, chunk_ids, chunks_by_id


def is_up_to_date(vectors_md5):
    files_exist = INDEX_FILE.exists() and INDEX_CHUNKS_FILE.exists() and INDEX_INFO_FILE.exists()
    if not files_exist:
        return False

    with open(INDEX_INFO_FILE, "r", encoding="utf-8") as file:
        info = json.load(file)

    same_vectors = info.get("embeddings_md5") == vectors_md5
    return same_vectors


# ==================================================================
# Checks
# ==================================================================

def check_alignment(vectors, chunk_ids, chunks_by_id):
    """The vectors, the chunk ids and the chunks must line up exactly."""

    vector_count = vectors.shape[0]
    id_count = len(chunk_ids)

    if vector_count != id_count:
        message = f"{vector_count} vectors but {id_count} chunk ids. Run step 33 (embed) again."
        raise ValueError(message)

    missing_ids = []
    for chunk_id in chunk_ids:
        if chunk_id not in chunks_by_id:
            missing_ids.append(chunk_id)

    if len(missing_ids) > 0:
        message = f"{len(missing_ids)} chunk ids have no chunk (e.g. {missing_ids[:3]}). "
        message = message + "The chunks changed after embedding: run step 33 (embed) again."
        raise ValueError(message)

    lengths = np.linalg.norm(vectors, axis=1)
    largest_error = np.abs(lengths - 1.0).max()
    if largest_error > 0.001:
        raise ValueError("The vectors are not normalised. Step 33 must use normalize_embeddings=True.")

    print("  alignment OK:", vector_count, "vectors, chunk ids and chunks match; all vectors have length 1")


def check_self_search(index, vectors):
    """Every chunk's vector, searched for, should return that chunk first."""

    scores, positions = index.search(vectors, 2)

    found_itself = 0
    near_duplicates = 0

    for row in range(vectors.shape[0]):
        best_position = positions[row][0]
        best_score = scores[row][0]
        second_score = scores[row][1]

        # Ties with an identical chunk can put the twin first; both count as "itself"
        if best_position == row or best_score - second_score < 1e-6:
            found_itself += 1

        if second_score >= DUPLICATE_SIMILARITY:
            near_duplicates += 1

    share = found_itself / vectors.shape[0]
    print("  self-search:", found_itself, "of", vectors.shape[0], "chunks find themselves first",
          "(" + str(round(share * 100, 1)) + " %)")
    print("  near-duplicate chunks (similarity >= " + str(DUPLICATE_SIMILARITY) + " with another chunk):",
          near_duplicates)

    if share < 0.99:
        raise ValueError("The index order does not match the chunks. Run steps 33 and 34 again.")

    return near_duplicates


# ==================================================================
# Building, saving and loading
# ==================================================================

def build_index(vectors):
    dimensions = vectors.shape[1]
    index = faiss.IndexFlatIP(dimensions)
    index.add(vectors)
    return index


def save_index(index, chunk_ids, chunks_by_id, vectors_md5, near_duplicates):
    RAG_INDEX_FOLDER.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_FILE))

    # Position i in the index -> details of chunk i
    index_chunks = []
    for chunk_id in chunk_ids:
        chunk = chunks_by_id[chunk_id]
        kept = {}
        for field in FIELDS_TO_KEEP:
            kept[field] = chunk[field]
        index_chunks.append(kept)

    with open(INDEX_CHUNKS_FILE, "w", encoding="utf-8") as file:
        json.dump(index_chunks, file, ensure_ascii=False)

    now = datetime.datetime.now()
    info = {
        "index_type": "IndexFlatIP (exact search, cosine similarity on normalised vectors)",
        "vectors": int(index.ntotal),
        "dimensions": int(index.d),
        "near_duplicate_chunks": int(near_duplicates),
        "embeddings_md5": vectors_md5,
        "index_md5": compute_md5(INDEX_FILE),
        "made_on": now.strftime("%Y-%m-%d %H:%M"),
    }
    with open(INDEX_INFO_FILE, "w", encoding="utf-8") as file:
        json.dump(info, file, indent=2)

    print("Saved", INDEX_FILE.relative_to(PROJECT_ROOT), "-", index.ntotal, "vectors")
    print("Saved", INDEX_CHUNKS_FILE.relative_to(PROJECT_ROOT))
    print("Saved", INDEX_INFO_FILE.relative_to(PROJECT_ROOT))


def load_index():
    """Load the saved index and its chunk lookup (used by retrieval and inference)."""

    check_file_exists(INDEX_FILE, "step 34 (FAISS index)")
    check_file_exists(INDEX_CHUNKS_FILE, "step 34 (FAISS index)")

    index = faiss.read_index(str(INDEX_FILE))
    with open(INDEX_CHUNKS_FILE, "r", encoding="utf-8") as file:
        index_chunks = json.load(file)

    if index.ntotal != len(index_chunks):
        raise ValueError("Index and chunk lookup do not match. Run step 34 again.")

    return index, index_chunks


# ==================================================================
# The step
# ==================================================================

def build_faiss_index():
    """Step 34: build, check and save the FAISS index."""

    vectors, chunk_ids, chunks_by_id = read_inputs()
    vectors_md5 = compute_md5(EMBEDDINGS_FILE)

    if is_up_to_date(vectors_md5):
        print("FAISS index is up to date (same vectors): nothing to do.")
        return

    print("Building the FAISS index from", vectors.shape[0], "vectors of", vectors.shape[1], "numbers")
    check_alignment(vectors, chunk_ids, chunks_by_id)

    index = build_index(vectors)
    near_duplicates = check_self_search(index, vectors)

    save_index(index, chunk_ids, chunks_by_id, vectors_md5, near_duplicates)