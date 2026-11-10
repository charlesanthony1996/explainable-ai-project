"""
Inference pathway, block 2B, step 10: retrieve sources.

Searches the FAISS index from training (steps 32 - 34) with the user's question
and returns the top 4 chunks with their sources.

The question is embedded with the SAME model that embedded the chunks (read from
rag_index/embedding_info.json) and normalised the same way, so the FAISS score
is the cosine similarity (1 = same meaning, 0 = unrelated).

Run a quick test with:
    uv run python -m explainable_ai_project.rag "Which article requires human oversight?"
"""

import json
import sys
from pathlib import Path

import numpy as np


THIS_FILE = Path(__file__).resolve()
PACKAGE_FOLDER = THIS_FILE.parent
SRC_FOLDER = PACKAGE_FOLDER.parent
PROJECT_ROOT = SRC_FOLDER.parent

RAG_INDEX_FOLDER = PROJECT_ROOT / "rag_index"
INDEX_FILE = RAG_INDEX_FOLDER / "faiss.index"
INDEX_CHUNKS_FILE = RAG_INDEX_FOLDER / "index_chunks.json"
EMBEDDING_INFO_FILE = RAG_INDEX_FOLDER / "embedding_info.json"

TOP_K = 4

_index = None
_chunks = None
_embedder = None


def load_index():
    """The FAISS index and the chunk details, loaded once."""

    global _index, _chunks
    if _index is None:
        import faiss

        for file_path in [INDEX_FILE, INDEX_CHUNKS_FILE, EMBEDDING_INFO_FILE]:
            if not file_path.exists():
                raise FileNotFoundError("Missing " + str(file_path.relative_to(PROJECT_ROOT))
                                        + ". Run the training pathway first: uv run main.py")

        _index = faiss.read_index(str(INDEX_FILE))
        with open(INDEX_CHUNKS_FILE, "r", encoding="utf-8") as file:
            _chunks = json.load(file)

        if _index.ntotal != len(_chunks):
            raise ValueError("The index and index_chunks.json do not match. Run steps 33 and 34 again.")
    return _index, _chunks


def load_embedder():
    """The embedding model that made the index (loaded once)."""

    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        with open(EMBEDDING_INFO_FILE, "r", encoding="utf-8") as file:
            info = json.load(file)
        _embedder = SentenceTransformer(info["model"])
    return _embedder


def search(question, k=TOP_K):
    """Step 10: the k chunks closest to the question, best first.

    Each hit: chunk_id, doc_id, title, section, page_start, page_end, text, score.
    """

    index, chunks = load_index()
    embedder = load_embedder()

    vector = embedder.encode([question], normalize_embeddings=True, convert_to_numpy=True)
    vector = np.asarray(vector, dtype="float32")
    scores, positions = index.search(vector, k)

    hits = []
    for score, position in zip(scores[0], positions[0]):
        if position < 0:
            continue
        hit = dict(chunks[position])
        hit["score"] = round(float(score), 4)
        hits.append(hit)
    return hits


def source_label(hit):
    """'EU AI Act ..., Article 14, p. 52' - how a source is shown to the user."""

    parts = [hit["title"]]
    if hit.get("section"):
        parts.append(hit["section"])
    page_start = hit.get("page_start")
    page_end = hit.get("page_end")
    if page_start:
        if page_end and page_end != page_start:
            parts.append("pp. " + str(page_start) + "-" + str(page_end))
        else:
            parts.append("p. " + str(page_start))
    return ", ".join(parts)


if __name__ == "__main__":
    test_question = " ".join(sys.argv[1:]) or "Which article requires human oversight?"
    print("question:", test_question)
    for rank, found in enumerate(search(test_question), start=1):
        print(str(rank) + ".", "score", found["score"], "-", source_label(found))
        print("   ", found["text"][:200].replace("\n", " "), "...")
