"""
Shared chunk-loading logic used by both build_vector_store.py and
build_bm25_index.py
"""

import json
from pathlib import Path

PROCESSED_CHUNKS_DIR = Path(__file__).parent.parent / "data" / "processed_chunks"

CAPTION_SECTION_MARKER = "typical characteristics"
CAPTION_MAX_LENGTH = 80


def _is_chart_caption(chunk: dict) -> bool:
    if chunk["type"] != "text":
        return False
    if CAPTION_SECTION_MARKER not in chunk["section"].lower():
        return False
    return len(chunk["content"].strip()) < CAPTION_MAX_LENGTH


def load_all_chunks_with_ids():
    """Load every processed chunk JSON file into one flat list, each with
    a unique 'id' field attached (used as the join key between the vector
    store and the BM25 index). Filters out low-value chart-caption chunks."""
    all_chunks = []
    json_files = sorted(PROCESSED_CHUNKS_DIR.glob("*.json"))

    if not json_files:
        print(f"No processed chunk files found in {PROCESSED_CHUNKS_DIR}")
        print("Run src/parse_datasheet.py on your PDFs first.")
        return []

    caption_filtered = 0
    for json_file in json_files:
        with open(json_file) as f:
            chunks = json.load(f)
        for chunk in chunks:
            if _is_chart_caption(chunk):
                caption_filtered += 1
                continue
            all_chunks.append(chunk)

    if caption_filtered:
        print(f"  filtered {caption_filtered} chart-caption chunks (Typical Characteristics figure labels)")

    for i, chunk in enumerate(all_chunks):
        chunk["id"] = f"{chunk['part_number']}_{i}"

    return all_chunks


def get_known_part_numbers():
    """Distinct part numbers present in the processed chunk data - used to
    detect which part a query is asking about, for metadata filtering."""
    chunks = load_all_chunks_with_ids()
    return sorted({c["part_number"] for c in chunks})