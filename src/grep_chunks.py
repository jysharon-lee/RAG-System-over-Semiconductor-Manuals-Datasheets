"""
Searches the raw processed_chunks JSON directly for keyword matches bypasses retrieval entirely

Usage:
    python src/grep_chunks.py TPS61030 "voltage"
    python src/grep_chunks.py TPS61030 "absolute maximum" --type table_row
"""

import argparse
import json
from pathlib import Path

PROCESSED_CHUNKS_DIR = Path(__file__).parent.parent / "data" / "processed_chunks"


def grep_chunks(part_number: str, keyword: str, chunk_type: str = None):
    json_path = PROCESSED_CHUNKS_DIR / f"{part_number}.json"
    if not json_path.exists():
        print(f"No processed chunks found for {part_number} at {json_path}")
        return

    chunks = json.load(open(json_path))
    keyword_lower = keyword.lower()

    matches = [
        c for c in chunks
        if keyword_lower in c["content"].lower()
        and (chunk_type is None or c["type"] == chunk_type)
    ]

    print(f"Found {len(matches)} chunks in {part_number}.json containing '{keyword}'"
          f"{f' (type={chunk_type})' if chunk_type else ''}\n")

    for c in matches:
        print(f"--- page {c['page_number']} | {c['section']} | {c['type']} ---")
        print(c["content"][:200])
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search raw processed chunks for a keyword")
    parser.add_argument("part_number", help="Part number, e.g. TPS61030")
    parser.add_argument("keyword", help="Keyword to search for (case-insensitive)")
    parser.add_argument("--type", type=str, default=None, choices=["text", "table_row"], help="Restrict to a chunk type")
    args = parser.parse_args()

    grep_chunks(args.part_number, args.keyword, args.type)