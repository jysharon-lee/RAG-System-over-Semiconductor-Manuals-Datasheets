"""
Dumps raw PyMuPDF text blocks for a page range, unfiltered, so can see
exactly how a given page's headings and text are being extracted before
any section-detection or noise-filtering logic touches them

Usage:
    python src/dump_raw_blocks.py data/raw_pdfs/LM317.pdf 2 4
    (dumps pages 2 through 4, 1-indexed)
"""

import sys
from pathlib import Path

import fitz


def dump(pdf_path: Path, start_page: int, end_page: int):
    with fitz.open(pdf_path) as doc:
        for page_num in range(start_page, end_page + 1):
            if page_num > len(doc):
                break
            page = doc[page_num - 1]
            blocks = page.get_text("blocks")
            blocks.sort(key=lambda b: (b[1], b[0]))

            print(f"\n{'='*70}")
            print(f"PAGE {page_num}  ({len(blocks)} blocks)")
            print(f"{'='*70}")
            for i, block in enumerate(blocks):
                text = block[4]
                print(f"--- block {i} ---")
                print(repr(text[:200]))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python src/dump_raw_blocks.py <pdf_path> <start_page> <end_page>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    start_page = int(sys.argv[2])
    end_page = int(sys.argv[3])
    dump(pdf_path, start_page, end_page)