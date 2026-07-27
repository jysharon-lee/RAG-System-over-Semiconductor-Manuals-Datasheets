"""
Parses a datasheet PDF into section-aware chunks with page-level metadata.

This is deliberately NOT a naive "extract all text, split every 500 chars"
parser. Datasheets have structure (Absolute Maximum Ratings, Electrical
Characteristics, Pin Description, etc.) and dense tables that a flat text
split destroys. This script:

  1. Extracts text per page with layout awareness (PyMuPDF)
  2. Detects section headers using common datasheet heading patterns
  3. Extracts tables separately (pdfplumber) and serializes each row into
     a readable sentence instead of a flattened grid
  4. Groups everything into chunks keyed by (section, page) so each chunk
     stays semantically coherent and small-enough to embed cleanly

Usage:
    python src/parse_datasheet.py data/raw_pdfs/TPS7A4700.pdf

Output:
    data/processed_chunks/TPS7A4700.json
"""

import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
import wordninja

# Some datasheet PDFs (notably TI's newer template) embed table-cell text
# with no space characters between words at all - pdfplumber's cell-text
# reconstruction then produces "PARTNUMBER" instead of "PART NUMBER". General
# dictionary-based word segmentation (wordninja) fixes most of this, but it
# doesn't know semiconductor-specific acronyms and will wrongly split them
# (e.g. "PWM" -> "PW M"). This protected list is checked first so acronyms
# survive intact; wordninja only runs on the text between them. Kept to 3+
# letter terms - 2-letter acronyms (EN, FB, IC, SW) are too likely to
# collide with ordinary English substrings (e.g. "IC" inside "METRIC").
PROTECTED_ACRONYMS = sorted({
    "GND", "PGND", "VBAT", "VOUT", "VIN", "LBI", "LBO", "SYNC",
    "IOUT", "TJA", "TSTG", "PWM", "ESD", "ROHS", "LDO", "ADJ", "CADJ", "MSL",
    "SPQ", "NOPB", "QFN", "TSSOP", "HTSSOP", "DGR", "YBG", "DLC", "E96",
    "E24", "IQ", "VFB", "VIH", "VIL", "SON", "DCY", "KTT", "KCT", "KCS",
    "PWP", "RSA",
}, key=len, reverse=True)

_PROTECTED_SET = set(PROTECTED_ACRONYMS)
_PROTECTED_SPLIT = re.compile("(" + "|".join(re.escape(t) for t in PROTECTED_ACRONYMS) + ")")
_LONG_ALPHA_RUN = re.compile(r"[A-Za-z]{6,}")


def _segment_run(run: str) -> str:
    parts = _PROTECTED_SPLIT.split(run)
    out = []
    for part in parts:
        if not part:
            continue
        out.append(part if part in _PROTECTED_SET else " ".join(wordninja.split(part)))
    return " ".join(out)


def desegment_cell(text: str) -> str:
    """Insert spaces into concatenated table-cell text, e.g. 'PARTNUMBER'
    -> 'PART NUMBER', while keeping known pin/package acronyms intact."""
    return _LONG_ALPHA_RUN.sub(lambda m: _segment_run(m.group(0)), text)

# Common datasheet section headers - extend this list as you see more
# real datasheets; manufacturers are fairly consistent about these, but TI
# numbers every heading (e.g. "6.1 Absolute Maximum Ratings", "4 Device
# Comparison Table"), so we allow an optional leading number/decimal prefix.
SECTION_TITLES = [
    "Absolute Maximum Ratings",
    "ESD Ratings",
    "Recommended Operating Conditions",
    "Thermal Information",
    "Thermal Characteristics",
    "Thermal Resistance",
    "Electrical Characteristics",
    "Typical Characteristics",
    "Pin Configuration and Functions",
    "Pin Functions",
    "Pin Description",
    "Device Comparison Table",
    "Detailed Description",
    "Application and Implementation",
    "Application Information",
    "Typical Application",
    "Power Supply Recommendations",
    "Layout",
    "Device and Documentation Support",
    "Ordering Information",
    "Package Information",
    "Package Option Addendum",
    "Functional Block Diagram",
    "Specifications",
]

NUMERIC_PREFIX = r"^\s*(?:\d+(?:\.\d+)*\s+)?"
SECTION_REGEX = re.compile(
    NUMERIC_PREFIX + "(?:" + "|".join(re.escape(t) for t in SECTION_TITLES) + ")",
    re.IGNORECASE,
)

# Standalone fragments like pin numbers on a pinout diagram ("1", "2", "3.3V"
# axis ticks on a graph, lone "+"/"-" symbols), plus control characters and
# stray glyph artifacts (checkbox icons, table border remnants) that some
# PDF renderers leave behind as their own tiny text blocks.
NOISE_PATTERN = re.compile(r"^[\d\.\-–+/,%]{1,3}$")
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
# Short alphabetic-only fragments: dimension callouts on mechanical drawings
# (W, H, L) and graph axis unit labels (dB, VI). Real text, but a standalone
# block of just "dB" has no retrievable meaning on its own.
SHORT_LABEL_PATTERN = re.compile(r"^[A-Za-z°ΩµμΔ]{1,3}$")
# Table of Contents entries like "Related Links ........................ 19"
# - dot leaders ending in a page number. These previously only blocked
# false-positive header detection; they were still being kept as ordinary
# body text afterward, which let irrelevant ToC noise leak into retrieval
# (e.g. outranking real content for a generic query like "electrical
# characteristics" since the ToC line contains that exact phrase).
TOC_LINE_PATTERN = re.compile(r"\.{2,}\s*\d+\s*$")


def is_noise(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if CONTROL_CHAR_PATTERN.search(stripped):
        return True
    if NOISE_PATTERN.match(stripped) or SHORT_LABEL_PATTERN.match(stripped):
        return True
    if TOC_LINE_PATTERN.search(stripped):
        return True
    # Single stray symbol with no alphanumeric content at all (e.g. "_", "|")
    if len(stripped) <= 2 and not any(ch.isalnum() for ch in stripped):
        return True
    return False


BARE_NUMBER_LINE = re.compile(r"^\d+(?:\.\d+)*$")


TOC_DOT_LEADER = re.compile(r"\.{2,}")


def _is_real_header(candidate: str) -> bool:
    """Check candidate text against the section regex, rejecting sentences
    that merely start with a section title word (e.g. "Specifications are
    for TJ=25C...") rather than being an actual heading, and Table of
    Contents dot-leader lines (e.g. "Device Support..........35")."""
    match = SECTION_REGEX.search(candidate)
    if not match or len(candidate) >= 80:
        return False
    trailing = candidate[match.end():].strip()
    looks_like_sentence = trailing and trailing[0].islower()
    looks_like_toc_entry = bool(TOC_DOT_LEADER.search(trailing))
    return not (looks_like_sentence or looks_like_toc_entry)


def _section_for_y(section_positions: list, y: float) -> str:
    """Given a sorted list of (y_pos, section_name) tuples and a target
    y-coordinate, return the section whose header is nearest above (or at) y.
    This replaces the old page-level approximation with position-accurate
    attribution - critical when multiple sections share a single page."""
    best = "General / Overview"
    for header_y, section_name in section_positions:
        if header_y <= y:
            best = section_name
        else:
            break
    return best


def extract_text_chunks(pdf_path: Path, part_number: str):
    """Extract text per page, tagging each chunk with the current section.

    Scans line-by-line within each block (not just the first line) because
    some datasheet templates split a section number and its title across
    two separate lines within one block, e.g.:
        "6\\nDevice Comparison Table (1)\\n"
    A bare numeric line is merged with the following line before testing
    for a header match to catch this case.

    Also builds and returns a page_section_positions map: for each page,
    a sorted list of (y_coordinate, section_name) pairs recording the
    vertical position of every section header found by PyMuPDF. Table
    extraction runs as a completely separate pass (pdfplumber, not PyMuPDF)
    and has no concept of "section" on its own - without this map, every
    table row got tagged with a generic "Table N (page P)" placeholder,
    which meant a table row's content had zero lexical/semantic connection
    to its actual section name (e.g. "Absolute Maximum Ratings"). That
    caused real retrieval misses. Originally this was a page-level
    approximation (one section per page), but that mislabeled every table
    on multi-section pages - e.g. TPS61030 page 4 has four sections each
    with their own table, and the page-level shortcut stamped all of them
    with the last section. The position-based approach matches each
    table's vertical position against the nearest header above it on the
    same page, fixing this entirely.
    """
    chunks = []
    current_section = "General / Overview"
    noise_filtered = 0
    page_section_positions = {}  # page_num -> [(y_pos, section_name), ...]

    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, ...)
            blocks.sort(key=lambda b: (b[1], b[0]))  # reading order: top-to-bottom, left-to-right

            # Seed this page with the section inherited from the previous
            # page so tables appearing before the first header get the
            # correct label.
            page_section_positions[page_num] = [(-1.0, current_section)]

            for block in blocks:
                raw = block[4].strip()
                block_y = block[1]  # y0: top edge of this block
                if not raw:
                    continue

                lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
                segment_lines = []
                i = 0
                while i < len(lines):
                    line = lines[i]
                    candidate = line
                    consumed = 1

                    # A bare number line ("6", "8.5") followed by a title on
                    # the next line - merge them before testing for a header.
                    if BARE_NUMBER_LINE.match(line) and i + 1 < len(lines):
                        merged = f"{line} {lines[i + 1]}"
                        if _is_real_header(merged):
                            candidate = merged
                            consumed = 2

                    if _is_real_header(candidate):
                        # Flush accumulated body text under the OLD section
                        # before switching to the new one.
                        if segment_lines:
                            content = "\n".join(segment_lines).strip()
                            if is_noise(content):
                                noise_filtered += 1
                            else:
                                chunks.append({
                                    "part_number": part_number,
                                    "page_number": page_num,
                                    "section": current_section,
                                    "type": "text",
                                    "content": content,
                                })
                            segment_lines = []
                        current_section = candidate
                        page_section_positions[page_num].append((block_y, current_section))
                        i += consumed
                        continue

                    if is_noise(line):
                        noise_filtered += 1
                    else:
                        segment_lines.append(line)
                    i += 1

                if segment_lines:
                    content = "\n".join(segment_lines).strip()
                    if is_noise(content):
                        noise_filtered += 1
                    else:
                        chunks.append({
                            "part_number": part_number,
                            "page_number": page_num,
                            "section": current_section,
                            "type": "text",
                            "content": content,
                        })


    if noise_filtered:
        print(f"  filtered {noise_filtered} noise fragments (pin/graph labels)")

    return chunks, page_section_positions


MAX_ROW_LENGTH = 350  # a real electrical-characteristics row is short; a
                       # row this long is almost always pdfplumber mistaking
                       # a graph/chart region for a table grid.

NAV_CHROME_TERMS = [
    "productfolder", "clickhere", "sample&buy", "sample & buy",
    "technicaldocuments", "tools&software", "support&community",
]


def _is_junk_table_row(sentence: str) -> bool:
    if len(sentence) > MAX_ROW_LENGTH:
        return True
    lowered = sentence.lower().replace(" ", "").replace("\n", "")
    nav_hits = sum(1 for term in NAV_CHROME_TERMS if term.replace(" ", "") in lowered)
    if nav_hits >= 2:
        return True
    return False


# Labels identifying spec-value columns in datasheet tables.  When these
# appear in the header row, we know the table has structured min/typ/max
# data and can apply the merged-cell redistribution logic below.
SPEC_LABELS = {'MIN', 'TYP', 'MAX', 'NOM'}

_NUMERIC_LIKE = re.compile(r'^[\u2013\u2212-]?\d')  # starts with optional minus then digit


def _looks_numeric(token: str) -> bool:
    """Does this token look like a numeric value (possibly with unit suffix)?"""
    return bool(_NUMERIC_LIKE.match(token.strip('(),')))


def _has_numeric_value(cell: str) -> bool:
    """Does this cell contain at least one standalone numeric token?"""
    if not cell:
        return False
    for token in cell.replace('\n', ' ').split():
        if _looks_numeric(token):
            return True
    return False


def _merge_multi_row_header(table: list) -> tuple:
    """Detect multi-row headers (common in thermal/package tables where
    sub-headers like 'PWP / 16 PINS' span rows 1-2 beneath a generic row 0)
    and merge them vertically into a single compound header.

    Returns (merged_header, first_data_row_index).

    Only activates when the first header row does NOT already contain
    standard spec-column labels (MIN/TYP/MAX/NOM) - if it does, the
    header is already well-formed and rows below it are data."""
    if not table:
        return [], 0

    header = [str(h).strip() if h else '' for h in table[0]]

    # If the header already has standard spec columns, it's complete.
    if {h.upper() for h in header} & SPEC_LABELS:
        return header, 1

    # Otherwise, check subsequent rows for sub-headers (no numeric data).
    data_start = 1
    for i in range(1, min(len(table), 4)):
        row = [str(c).strip() if c else '' for c in table[i]]
        if any(_has_numeric_value(c) for c in row):
            break  # first row with actual numeric data
        # Merge this sub-header row into the compound header
        for j in range(min(len(header), len(row))):
            if row[j]:
                header[j] = (header[j] + ' ' + row[j]).strip()
        data_start = i + 1

    return header, data_start


def _redistribute_merged_cells(header: list, row: list) -> list:
    """When pdfplumber merges MIN/TYP/MAX values into a single cell
    (e.g. '-0.3 3.6' under the MIN column while MAX is empty), split
    the space-separated tokens back out across the correct columns.

    Handles two common patterns:
      2 tokens across 2 spec cols:  MIN=-0.3, MAX=3.6
      2 tokens across 3 spec cols:  MIN=1.8, MAX=5.5 (TYP/NOM stays empty)
      3 tokens across 3 spec cols:  MIN=490, TYP=500, MAX=510
    """
    spec_indices = [i for i, h in enumerate(header) if h.upper().strip() in SPEC_LABELS]
    if len(spec_indices) < 2:
        return row  # need at least 2 spec columns

    spec_start = spec_indices[0]
    first_cell = row[spec_start] if spec_start < len(row) else ''
    if not first_cell:
        return row

    # Only redistribute if the remaining spec columns are all empty
    if not all((row[i] if i < len(row) else '') == '' for i in spec_indices[1:]):
        return row  # values already in separate cells

    tokens = first_cell.replace('\n', ' ').split()
    if len(tokens) < 2:
        return row  # single value, nothing to split

    # All tokens must look numeric - otherwise it's a text value that
    # happens to contain spaces (e.g. '0.4×I\nSW').
    if not all(_looks_numeric(t) for t in tokens):
        return row

    new_row = list(row)
    spec_count = len(spec_indices)

    if len(tokens) == spec_count:
        # Perfect match: one token per spec column (MIN/TYP/MAX)
        for j, idx in enumerate(spec_indices):
            new_row[idx] = tokens[j]
    elif len(tokens) == 2 and spec_count >= 2:
        # Two tokens → first and last spec column (MIN and MAX),
        # middle columns (TYP/NOM) stay empty.
        new_row[spec_indices[0]] = tokens[0]
        new_row[spec_indices[-1]] = tokens[1]
        for idx in spec_indices[1:-1]:
            new_row[idx] = ''
    else:
        return row  # unexpected count, don't touch

    return new_row


def extract_table_chunks(pdf_path: Path, part_number: str, page_section_positions: dict = None):
    """Extract tables and serialize each row as a readable sentence."""
    chunks = []
    junk_filtered = 0
    page_section_positions = page_section_positions or {}

    # text_x_tolerance widened from pdfplumber's default: some datasheet
    # PDFs (notably TI's newer template) embed characters with tighter
    # spacing metadata than usual, causing pdfplumber's default cell-text
    # reconstruction to merge adjacent words with no space between them
    # (e.g. "PART NUMBER" -> "PARTNUMBER"). A wider tolerance treats small
    # gaps as word boundaries again.
    table_settings = {
        "text_x_tolerance": 3,
    }

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            section_positions = page_section_positions.get(page_num, [])
            found_tables = page.find_tables(table_settings=table_settings)
            for table_idx, table_obj in enumerate(found_tables):
                table = table_obj.extract()
                if not table or len(table) < 2:
                    continue

                # Use the table's vertical position to find the correct
                # section header above it, instead of the old page-level
                # approximation that mislabeled every table on multi-section
                # pages (e.g. TPS61030 page 4 has 4 sections with 4 tables).
                table_top_y = table_obj.bbox[1]
                section = _section_for_y(section_positions, table_top_y)

                header, data_start = _merge_multi_row_header(table)
                header = [desegment_cell(h) for h in header]
                for row in table[data_start:]:
                    row = [desegment_cell(str(c).strip()) if c else "" for c in row]
                    row = _redistribute_merged_cells(header, row)
                    if not any(row):
                        continue

                    # Serialize as "Header1: val1, Header2: val2, ..." instead
                    # of a flattened grid - this reads far better for both
                    # embedding and for the LLM at generation time.
                    #
                    # Some datasheet tables (notably Absolute Maximum Ratings
                    # and ESD Ratings) don't print a header label above the
                    # Symbol/Parameter columns at all - only MIN/MAX/UNIT get
                    # a header. Requiring a non-empty header here used to
                    # silently DROP those columns entirely, discarding the
                    # actual parameter name (e.g. "Input voltage on LBI")
                    # and leaving only "MIN: -0.3 3.6" with no idea what it's
                    # a rating for. Now: use "Header: value" when a header
                    # exists, otherwise keep the value on its own so the
                    # information survives.
                    pairs = []
                    for h, v in zip(header, row):
                        if not v:
                            continue
                        pairs.append(f"{h}: {v}" if h else v)
                    if not pairs:
                        continue

                    # Section name baked into the indexed text itself (not
                    # just metadata) - without this, a query like "absolute
                    # maximum input voltage" has no lexical/semantic anchor
                    # distinguishing this row from a similarly-worded row in
                    # a totally different table (e.g. Recommended Operating
                    # Conditions), since the raw parameter/value text alone
                    # doesn't say which table it came from.
                    sentence = f"[{part_number}, {section}, page {page_num}] " + "; ".join(pairs)

                    if _is_junk_table_row(sentence):
                        junk_filtered += 1
                        continue

                    chunks.append({
                        "part_number": part_number,
                        "page_number": page_num,
                        "section": section,
                        "type": "table_row",
                        "content": sentence,
                    })

    if junk_filtered:
        print(f"  filtered {junk_filtered} junk table rows (graph mis-parses / nav chrome)")

    return chunks


def parse_datasheet(pdf_path: Path):
    part_number = pdf_path.stem
    print(f"Parsing {part_number}...")

    text_chunks, page_section_positions = extract_text_chunks(pdf_path, part_number)
    table_chunks = extract_table_chunks(pdf_path, part_number, page_section_positions)

    print(f"  {len(text_chunks)} text chunks, {len(table_chunks)} table-row chunks")

    all_chunks = text_chunks + table_chunks

    output_dir = pdf_path.parent.parent / "processed_chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{part_number}.json"

    with open(output_path, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"  saved -> {output_path}")
    return all_chunks


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/parse_datasheet.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    parse_datasheet(pdf_path)