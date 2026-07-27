"""
Hybrid retrieval: combines vector (semantic) search and BM25 (keyword)
search using Reciprocal Rank Fusion, with automatic part-number detection
so a query naming a specific part gets filtered to that part's chunks
before ranking - this directly fixes the cross-contamination we saw in
testing (a TPS61030 chart label outranking real LM317 content for an
LM317-specific question).

Usage:
    python src/hybrid_query.py "what is the maximum output current of the LM317"
    python src/hybrid_query.py "quiescent current" --top-k 8
"""

import argparse
import pickle
import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from build_bm25_index import tokenize, BM25_INDEX_PATH
from chunk_utils import get_known_part_numbers

VECTOR_DB_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "semiconductor_datasheets"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

CANDIDATE_POOL_SIZE = 20  # how many results each system contributes before fusion
RRF_K = 60  # standard Reciprocal Rank Fusion damping constant


def detect_part_number(question: str, known_parts: list[str]) -> str | None:
    """Check if the question names a specific part number (case-insensitive,
    whole-word match). Returns the first match, or None if no part is named
    - in which case search runs across the whole corpus."""
    question_lower = question.lower()
    for part in known_parts:
        if re.search(rf"\b{re.escape(part.lower())}\b", question_lower):
            return part
    return None


def vector_search(model, collection, question: str, part_number: str | None, n: int):
    query_embedding = model.encode(QUERY_INSTRUCTION + question).tolist()
    where_filter = {"part_number": part_number} if part_number else None
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        where=where_filter,
    )
    # Return ranked list of chunk IDs (Chroma's own ids, matching our shared scheme)
    return results["ids"][0]


def bm25_search(bm25, chunks, question: str, part_number: str | None, n: int):
    tokenized_query = tokenize(question)
    scores = bm25.get_scores(tokenized_query)

    # Pair each score with its chunk, optionally restricting to one part
    scored = [
        (score, chunk)
        for score, chunk in zip(scores, chunks)
        if part_number is None or chunk["part_number"] == part_number
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk["id"] for _, chunk in scored[:n]]


def reciprocal_rank_fusion(*ranked_id_lists, k: int = RRF_K):
    """Fuse multiple ranked lists of IDs into one combined ranking. An item
    that ranks well in EITHER system (not just one) scores highly - this is
    what lets BM25 rescue an exact keyword/number match that vector search
    ranked low, and vice versa."""
    scores = {}
    for ranked_ids in ranked_id_lists:
        for rank, chunk_id in enumerate(ranked_ids):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def load_retrieval_backend():
    """Load everything retrieval needs once: embedding model, vector store,
    BM25 index, and the known part-number list. Returns a dict so callers
    (the CLI here, and generate_answer.py) share one implementation."""
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        raise RuntimeError(
            f"Vector collection '{COLLECTION_NAME}' not found. Run src/build_vector_store.py first."
        )

    if not BM25_INDEX_PATH.exists():
        raise RuntimeError(f"BM25 index not found at {BM25_INDEX_PATH}. Run src/build_bm25_index.py first.")

    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_data = pickle.load(f)

    return {
        "model": model,
        "collection": collection,
        "bm25": bm25_data["bm25"],
        "chunks": bm25_data["chunks"],
        "chunk_by_id": {c["id"]: c for c in bm25_data["chunks"]},
        "known_parts": get_known_part_numbers(),
    }


def retrieve_chunks(backend: dict, question: str, top_k: int = 5):
    """Run hybrid retrieval and return a plain list of result dicts (content
    + metadata + fusion diagnostics) - no printing, so this can be reused by
    both the CLI below and the answer-generation script."""
    part_number = detect_part_number(question, backend["known_parts"])

    vector_ids = vector_search(backend["model"], backend["collection"], question, part_number, CANDIDATE_POOL_SIZE)
    bm25_ids = bm25_search(backend["bm25"], backend["chunks"], question, part_number, CANDIDATE_POOL_SIZE)
    fused = reciprocal_rank_fusion(vector_ids, bm25_ids)[:top_k]

    results = []
    for rank, (chunk_id, score) in enumerate(fused, start=1):
        chunk = backend["chunk_by_id"].get(chunk_id)
        if not chunk:
            continue
        sources = []
        if chunk_id in vector_ids:
            sources.append(f"vector#{vector_ids.index(chunk_id) + 1}")
        if chunk_id in bm25_ids:
            sources.append(f"bm25#{bm25_ids.index(chunk_id) + 1}")
        results.append({
            "rank": rank,
            "fused_score": score,
            "sources": sources,
            "part_number": chunk["part_number"],
            "section": chunk["section"],
            "page_number": chunk["page_number"],
            "type": chunk["type"],
            "content": chunk["content"],
        })

    return {"part_number_filter": part_number, "results": results}


def hybrid_query(question: str, top_k: int = 5):
    print("Loading models and indexes...")
    try:
        backend = load_retrieval_backend()
    except RuntimeError as e:
        print(e)
        return

    print(f"\nQuery: {question}")
    outcome = retrieve_chunks(backend, question, top_k)
    print(f"Detected part number filter: {outcome['part_number_filter'] or '(none - searching all parts)'}\n")

    results = outcome["results"]
    print(f"Top {len(results)} results (fused ranking):\n")
    for r in results:
        print(f"--- Result {r['rank']} (fused score: {r['fused_score']:.4f}, from: {', '.join(r['sources'])}) ---")
        print(f"Part: {r['part_number']} | Section: {r['section']} | Page: {r['page_number']} | Type: {r['type']}")
        print(r["content"][:300])
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid (vector + BM25) retrieval test")
    parser.add_argument("question", help="Question to search for")
    parser.add_argument("--top-k", type=int, default=5, help="Number of fused results to return")
    args = parser.parse_args()

    hybrid_query(args.question, top_k=args.top_k)