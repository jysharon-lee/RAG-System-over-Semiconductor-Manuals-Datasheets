"""
Builds the vector store: loads every data/processed_chunks/*.json file,
embeds each chunk's content, and upserts it into a persistent Chroma
collection with metadata (part_number, section, page_number, type) attached
so results can be filtered/cited later.

Embedding model: BAAI/bge-small-en-v1.5 (~130MB, runs fully locally, no API
key or cost). Good accuracy-per-size tradeoff for a portfolio project - see
README for why this was chosen over a larger model.

Usage:
    python src/build_vector_store.py

Re-running this script is safe: chunk IDs are deterministic
(part_number + chunk index), so re-running after re-parsing a datasheet
will just overwrite that datasheet's old vectors rather than duplicating.
"""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

PROCESSED_CHUNKS_DIR = Path(__file__).parent.parent / "data" / "processed_chunks"
VECTOR_DB_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "semiconductor_datasheets"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE models were trained with an instruction prefix on the DOCUMENT side
# for asymmetric retrieval setups in some variants, but bge-small-en-v1.5
# specifically only needs the prefix on the QUERY side at search time (see
# query_rag.py) - documents are embedded as-is.
BATCH_SIZE = 64


def load_all_chunks():
    """Load every processed chunk JSON file into one flat list."""
    all_chunks = []
    json_files = sorted(PROCESSED_CHUNKS_DIR.glob("*.json"))

    if not json_files:
        print(f"No processed chunk files found in {PROCESSED_CHUNKS_DIR}")
        print("Run src/parse_datasheet.py on your PDFs first.")
        return []

    for json_file in json_files:
        with open(json_file) as f:
            chunks = json.load(f)
        print(f"  loaded {len(chunks)} chunks from {json_file.name}")
        all_chunks.extend(chunks)

    return all_chunks


def build_vector_store():
    print("Loading processed chunks...")
    chunks = load_all_chunks()
    if not chunks:
        return

    print(f"\nTotal chunks to embed: {len(chunks)}")

    print(f"\nLoading embedding model ({EMBEDDING_MODEL_NAME})...")
    print("(first run downloads the model - subsequent runs use the local cache)")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("\nConnecting to Chroma...")
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    # Recreate the collection each run so stale/removed chunks don't linger
    # from a previous parse of the same datasheet.
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

        ids = [f"{c['part_number']}_{batch_start + i}" for i, c in enumerate(batch)]
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