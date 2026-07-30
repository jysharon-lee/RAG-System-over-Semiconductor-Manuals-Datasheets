"""
Tests pdfplumber's table detection with different strategies on one page
to diagnose why a real table (e.g. Absolute Maximum Ratings) didn't get
extracted

Usage:
    python src/debug_table_extraction.py data/raw_pdfs/TPS61030.pdf 4
"""

import sys
from pathlib import Path

import pdfplumber

STRATEGIES = {
    "default (lines)": {},
    "text-based": {"vertical_strategy": "text", "horizontal_strategy": "text"},
    "lines_strict + text fallback": {"vertical_strategy": "lines_strict", "horizontal_strategy": "text"},
}


def debug_page(pdf_path: Path, page_num: int):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num - 1]

        print(f"Page {page_num} raw text (first 500 chars):")
        print(repr(page.extract_text()[:500]))
        print()

        for name, settings in STRATEGIES.items():
            print(f"{'=' * 60}")
            print(f"Strategy: {name}  (settings: {settings})")
            print(f"{'=' * 60}")
            tables = page.extract_tables(table_settings=settings)
            print(f"Found {len(tables)} table(s)")
            for t_idx, table in enumerate(tables):
                print(f"  Table {t_idx}: {len(table)} rows")
                for row in table[:5]:
                    print(f"    {row}")
            print()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/debug_table_extraction.py <pdf_path> <page_number>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    page_num = int(sys.argv[2])
    debug_page(pdf_path, page_num)