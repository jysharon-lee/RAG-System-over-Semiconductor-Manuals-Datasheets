"""
Shared chunk-loading logic used by both build_vector_store.py and
build_bm25_index.py. Having ONE canonical place that assigns chunk IDs
matters a lot here: hybrid retrieval fuses rankings from two separate
systems (Chroma and BM25) by matching on chunk ID, so if the two build
scripts ever computed IDs differently, fusion would silently misalign
results without erroring.
"""

import json
from pathlib import Path

PROCESSED_CHUNKS_DIR = Path(__file__).parent.parent / "data" / "processed_chunks"


def load_all_chunks_with_ids():
    """Load every processed chunk JSON file into one flat list, each with
    a unique 'id' field attached (used as the join key between the vector
    store and the BM25 index)."""
    all_chunks = []
    json_files = sorted(PROCESSED_CHUNKS_DIR.glob("*.json"))

    if not json_files:
        print(f"No processed chunk files found in {PROCESSED_CHUNKS_DIR}")
        print("Run src/parse_datasheet.py on your PDFs first.")
        return []

    for json_file in json_files:
        with open(json_file) as f:
            chunks = json.load(f)
        all_chunks.extend(chunks)

    for i, chunk in enumerate(all_chunks):
        chunk["id"] = f"{chunk['part_number']}_{i}"

    return all_chunks


def get_known_part_numbers():
    """Distinct part numbers present in the processed chunk data - used to
    detect which part a query is asking about, for metadata filtering."""
    chunks = load_all_chunks_with_ids()
    return sorted({c["part_number"] for c in chunks})