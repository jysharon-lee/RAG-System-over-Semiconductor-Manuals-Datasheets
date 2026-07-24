"""
Builds a BM25 keyword index from all processed chunks, saved alongside the
vector store. This is the "hybrid" half of hybrid retrieval - vector search
alone struggles with exact part numbers, specific parameter names, and
numeric values (see the LM317/TPS61030 cross-contamination example from
testing pure vector search). BM25 handles literal keyword/number matches
that semantic embeddings can blur past.

Usage:
    python src/build_bm25_index.py
"""

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from chunk_utils import load_all_chunks_with_ids

BM25_INDEX_PATH = Path(__file__).parent.parent / "data" / "bm25_index.pkl"

# Simple word tokenizer: lowercase, split on anything that isn't a letter,
# digit, or decimal point (keeps numbers like "3.3" and units like "mA"
# intact as single tokens, which matters for spec-value matching).
TOKEN_PATTERN = re.compile(r"[a-z0-9.]+")


def tokenize(text: str):
    return TOKEN_PATTERN.findall(text.lower())


def build_bm25_index():
    print("Loading processed chunks...")
    chunks = load_all_chunks_with_ids()
    if not chunks:
        return

    print(f"Total chunks to index: {len(chunks)}")

    print("Tokenizing...")
    tokenized_corpus = [tokenize(c["content"]) for c in chunks]

    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)

    # Save the BM25 model AND the chunks list together - the index alone is
    # just term statistics, we need the parallel chunks list (same order)
    # to map a scored position back to actual content and metadata.
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    print(f"\nDone. BM25 index saved to {BM25_INDEX_PATH}")


if __name__ == "__main__":
    build_bm25_index()