"""
Usage:
    python src/query_rag.py "what is the maximum output current of the LM317"
    python src/query_rag.py "TPS62840 quiescent current" --part TPS62840
    python src/query_rag.py "electrical characteristics" --top-k 10
"""

import argparse
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

VECTOR_DB_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "semiconductor_datasheets"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def query(question: str, top_k: int = 5, part_number: str | None = None):
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        print(f"Collection '{COLLECTION_NAME}' not found.")
        print("Run src/build_vector_store.py first.")
        return

    query_embedding = model.encode(QUERY_INSTRUCTION + question).tolist()

    where_filter = {"part_number": part_number} if part_number else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
    )

    print(f"\nQuery: {question}")
    if part_number:
        print(f"Filtered to part: {part_number}")
    print(f"Top {top_k} results:\n")

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
        similarity = 1 - dist  # Chroma returns cosine distance by default
        print(f"--- Result {i} (similarity: {similarity:.3f}) ---")
        print(f"Part: {meta['part_number']} | Section: {meta['section']} | Page: {meta['page_number']} | Type: {meta['type']}")
        print(doc[:300])
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test retrieval against the datasheet vector store")
    parser.add_argument("question", help="Question to search for")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--part", type=str, default=None, help="Filter results to a specific part number")
    args = parser.parse_args()

    query(args.question, top_k=args.top_k, part_number=args.part)