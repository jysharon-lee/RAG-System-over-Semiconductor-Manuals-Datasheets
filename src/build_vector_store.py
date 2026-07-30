"""
Loads every data/processed_chunks/*.json file, embeds each chunk's content, and upserts it into a persistent Chroma
collection with metadata attached

Usage:
    python src/build_vector_store.py
"""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from chunk_utils import load_all_chunks_with_ids

VECTOR_DB_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "semiconductor_datasheets"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE models were trained with an instruction prefix on the DOCUMENT side for asymmetric retrieval setups in some variants
BATCH_SIZE = 64


def build_vector_store():
    print("Loading processed chunks...")
    chunks = load_all_chunks_with_ids()
    if not chunks:
        return

    print(f"\nTotal chunks to embed: {len(chunks)}")

    print(f"\nLoading embedding model ({EMBEDDING_MODEL_NAME})...")
    print("(first run downloads the model - subsequent runs use the local cache)")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("\nConnecting to Chroma...")
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    # Recreate the collection each run so stale/removed chunks don't linger
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    print(f"\nEmbedding and inserting {len(chunks)} chunks in batches of {BATCH_SIZE}...")
    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start:batch_start + BATCH_SIZE]

        texts = [c["content"] for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        ids = [c["id"] for c in batch]
        metadatas = [
            {
                "part_number": c["part_number"],
                "section": c["section"],
                "page_number": c["page_number"],
                "type": c["type"],
            }
            for c in batch
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        done = min(batch_start + BATCH_SIZE, len(chunks))
        print(f"  {done}/{len(chunks)} embedded")

    print(f"\nDone. Vector store saved to {VECTOR_DB_DIR}")
    print(f"Collection '{COLLECTION_NAME}' now has {collection.count()} chunks.")


if __name__ == "__main__":
    build_vector_store()