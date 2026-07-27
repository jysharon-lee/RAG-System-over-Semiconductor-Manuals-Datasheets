"""Diagnostic: dump what pdfplumber actually extracts for tables on a given page."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pdfplumber

pdf_path = sys.argv[1]
pages = [int(p) for p in sys.argv[2:]] if len(sys.argv) > 2 else None

table_settings = {"text_x_tolerance": 3}

with pdfplumber.open(pdf_path) as pdf:
    for page_num, page in enumerate(pdf.pages, start=1):
        if pages and page_num not in pages:
            continue
        found = page.find_tables(table_settings=table_settings)
        if not found:
            continue
        print(f"{'='*70}")
        print(f"PAGE {page_num}  ({len(found)} tables)")
        print(f"{'='*70}")
        for ti, tobj in enumerate(found):
            table = tobj.extract()
            bbox = tobj.bbox
            print(f"\n--- Table {ti} (bbox top_y={bbox[1]:.1f}) ---")
            print(f"  Rows: {len(table)}, Cols: {max(len(r) for r in table)}")
            for ri, row in enumerate(table):
                label = "HDR" if ri == 0 else f"R{ri:02d}"
                cells = [repr(c) if c else "''" for c in row]
                print(f"  [{label}] {' | '.join(cells)}")
