"""
Training pathway, block 1D, step 33: embed.

Turns every chunk from step 32 into a vector (a list of 384 numbers) with the
Sentence Transformers model all-MiniLM-L6-v2. Chunks with similar meaning get
similar vectors, so step 34 can find the chunks closest to a user's question
even when they use different words.

What this step does:
    1. Load the chunks and the model.
       (The first run downloads the model from Hugging Face, about 90 MB.)
    2. Count the tokens of every chunk. The model reads at most 256 tokens and
       silently ignores the rest, so every chunk that is too long is reported.
    3. Embed the "embedding_text" of every chunk (title | section | text).
       The vectors are normalised to length 1, so the dot product of two
       vectors is their cosine similarity (1 = same meaning, 0 = unrelated).
    4. Save the vectors, the chunk order, and a small info file.

If the chunks and the model have not changed since the last run, the step
skips the work and reuses the saved vectors (so main.py stays fast).

Put this file in:  src/explainable_ai_project/embed_chunks.py
Needs:             uv add "sentence-transformers==6.1.0"

Reads:
    rag_index/chunks.jsonl                (from step 32)

Writes:
    rag_index/embeddings.npy              one row per chunk, 384 numbers each
    rag_index/embedding_chunk_ids.json    which chunk each row belongs to
    rag_index/embedding_info.json         model, size, date, chunks file hash
    results/tables/rag_token_check.csv    chunks that are too long for the model (if any)
"""

import datetime
import hashlib
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
CHUNKS_FILE = RAG_INDEX_FOLDER / "chunks.jsonl"
EMBEDDINGS_FILE = RAG_INDEX_FOLDER / "embeddings.npy"
CHUNK_IDS_FILE = RAG_INDEX_FOLDER / "embedding_chunk_ids.json"
INFO_FILE = RAG_INDEX_FOLDER / "embedding_info.json"
TABLES_FOLDER = PROJECT_ROOT / "results" / "tables"
TOKEN_CHECK_FILE = TABLES_FOLDER / "rag_token_check.csv"

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64


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


def read_chunks():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError("Missing rag_index/chunks.jsonl. Run step 32 (chunk) first.")

    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        for line in file:
            chunk = json.loads(line)
            chunks.append(chunk)
    return chunks


def is_up_to_date(chunks_md5):
    """True if the saved vectors were made from the same chunks with the same model."""

    files_exist = INFO_FILE.exists() and EMBEDDINGS_FILE.exists() and CHUNK_IDS_FILE.exists()
    if not files_exist:
        return False

    with open(INFO_FILE, "r", encoding="utf-8") as file:
        info = json.load(file)

    same_chunks = info.get("chunks_md5") == chunks_md5
    same_model = info.get("model") == MODEL_NAME
    return same_chunks and same_model


def load_model():
    """Load the embedding model (downloads it the first time)."""

    # Imported here so the other steps do not need to load the large library
    from sentence_transformers import SentenceTransformer

    print("Loading", MODEL_NAME, "(the first time this downloads about 90 MB)")
    model = SentenceTransformer(MODEL_NAME)
    print("  runs on:", model.device, "| reads at most", model.max_seq_length, "tokens per text")
    return model


# ==================================================================
# Token check
# ==================================================================

def check_token_counts(model, chunks):
    """Count tokens per chunk; report those longer than the model reads."""

    texts = []
    for chunk in chunks:
        texts.append(chunk["embedding_text"])

    tokenized = model.tokenizer(texts, add_special_tokens=True)
    token_lists = tokenized["input_ids"]

    limit = model.max_seq_length
    rows = []
    token_counts = []

    for position in range(len(chunks)):
        chunk = chunks[position]
        token_count = len(token_lists[position])
        token_counts.append(token_count)

        if token_count > limit:
            rows.append({
                "chunk_id": chunk["chunk_id"],
                "words": chunk["words"],
                "tokens": token_count,
                "tokens_ignored": token_count - limit,
            })

    too_long = pd.DataFrame(rows, columns=["chunk_id", "words", "tokens", "tokens_ignored"])

    largest = max(token_counts)
    average = sum(token_counts) / len(token_counts)
    print("  tokens per chunk: average", round(average), "| largest", largest, "| limit", limit)

    if len(too_long) == 0:
        print("  every chunk fits: nothing is cut off")
    else:
        share = len(too_long) / len(chunks)
        print("  WARNING:", len(too_long), "chunks are longer than", limit, "tokens",
              "(" + str(round(share * 100, 1)) + " %); their ends are not searchable.")
        print("  If this is more than a few percent, lower CHUNK_WORDS in chunk_documents.py.")

    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)
    too_long.to_csv(TOKEN_CHECK_FILE, index=False)
    return too_long


# ==================================================================
# The step
# ==================================================================

def save_results(vectors, chunk_ids, chunks_md5, too_long_count):
    RAG_INDEX_FOLDER.mkdir(parents=True, exist_ok=True)

    float_vectors = vectors.astype(np.float32)
    np.save(EMBEDDINGS_FILE, float_vectors)

    with open(CHUNK_IDS_FILE, "w", encoding="utf-8") as file:
        json.dump(chunk_ids, file)

    now = datetime.datetime.now()
    info = {
        "model": MODEL_NAME,
        "dimensions": int(vectors.shape[1]),
        "chunks": int(vectors.shape[0]),
        "normalised": True,
        "chunks_md5": chunks_md5,
        "chunks_longer_than_model_limit": int(too_long_count),
        "made_on": now.strftime("%Y-%m-%d %H:%M"),
    }
    with open(INFO_FILE, "w", encoding="utf-8") as file:
        json.dump(info, file, indent=2)

    print("Saved", EMBEDDINGS_FILE.relative_to(PROJECT_ROOT), "-", vectors.shape[0], "x", vectors.shape[1])
    print("Saved", CHUNK_IDS_FILE.relative_to(PROJECT_ROOT))
    print("Saved", INFO_FILE.relative_to(PROJECT_ROOT))


def embed_chunks(model=None):
    """Step 33: embed every chunk and save the vectors.

    model can be passed in (for testing); normally it is loaded here.
    """

    chunks = read_chunks()
    chunks_md5 = compute_md5(CHUNKS_FILE)
    print(len(chunks), "chunks to embed")

    if model is None and is_up_to_date(chunks_md5):
        print("Embeddings are up to date (same chunks, same model): nothing to do.")
        return

    if model is None:
        model = load_model()

    too_long = check_token_counts(model, chunks)

    texts = []
    chunk_ids = []
    for chunk in chunks:
        texts.append(chunk["embedding_text"])
        chunk_ids.append(chunk["chunk_id"])

    print("Embedding ...")
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # Quick sanity check: every vector should have length 1
    lengths = np.linalg.norm(vectors, axis=1)
    smallest_length = lengths.min()
    largest_length = lengths.max()
    print("  vector lengths between", round(smallest_length, 4), "and", round(largest_length, 4), "(should be 1)")

    save_results(vectors, chunk_ids, chunks_md5, len(too_long))