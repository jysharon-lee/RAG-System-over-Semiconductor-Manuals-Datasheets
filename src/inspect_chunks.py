"""
inspect chunk counts, section detection, and raw content
for a parsed datasheet. Helps debug over-fragmented or mis-sectioned output

Usage:
    python src/inspect_chunks.py data/processed_chunks/LM317.json
"""

import json
import sys
from collections import Counter
from pathlib import Path


def inspect(json_path: Path):
    data = json.load(open(json_path))
    text_chunks = [c for c in data if c["type"] == "text"]
    table_chunks = [c for c in data if c["type"] == "table_row"]

    sections = Counter(c["section"] for c in text_chunks)
    print(f"File: {json_path.name}")
    print(f"Total text chunks: {len(text_chunks)}")
    print(f"Total table-row chunks: {len(table_chunks)}\n")

    print("Top sections by chunk count:")
    for sec, count in sections.most_common(10):
        print(f"  {count:4d}  {sec}")

    print("\nFirst 8 text chunks:")
    for c in text_chunks[:8]:
        section = c["section"][:40]
        content = repr(c["content"][:80])
        print(f"  page {c['page_number']:>3} | {section:40s} | {content}")

    print("\nShortest 8 text chunks (likely fragments/noise):")
    shortest = sorted(text_chunks, key=lambda c: len(c["content"]))[:8]
    for c in shortest:
        print(f"  page {c['page_number']:>3} | {repr(c['content'])}")

    print("\nSample of 10 table-row chunks (spread across the document):")
    if table_chunks:
        step = max(1, len(table_chunks) // 10)
        for c in table_chunks[::step][:10]:
            print(f"  page {c['page_number']:>3} | {c['content']}")
    else:
        print("  (none found - table extraction may have failed for this PDF)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/inspect_chunks.py <path_to_json>")
        sys.exit(1)
    inspect(Path(sys.argv[1]))