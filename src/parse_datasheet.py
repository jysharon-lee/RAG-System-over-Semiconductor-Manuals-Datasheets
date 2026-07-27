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


def extract_text_chunks(pdf_path: Path, part_number: str):
    """Extract text per page, tagging each chunk with the current section.

    Scans line-by-line within each block (not just the first line) because
    some datasheet templates split a section number and its title across
    two separate lines within one block, e.g.:
        "6\\nDevice Comparison Table (1)\\n"
    A bare numeric line is merged with the following line before testing
    for a header match to catch this case.
    """
    chunks = []
    current_section = "General / Overview"
    noise_filtered = 0

    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, ...)
            blocks.sort(key=lambda b: (b[1], b[0]))  # reading order: top-to-bottom, left-to-right

            for block in blocks:
                raw = block[4].strip()
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

    return chunks


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


def extract_table_chunks(pdf_path: Path, part_number: str):
    """Extract tables and serialize each row as a readable sentence."""
    chunks = []
    junk_filtered = 0

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
            tables = page.extract_tables(table_settings=table_settings)
            for table_idx, table in enumerate(tables):
                if not table or len(table) < 2:
                    continue

                header = [desegment_cell(str(h).strip()) if h else "" for h in table[0]]
                for row in table[1:]:
                    row = [desegment_cell(str(c).strip()) if c else "" for c in row]
                    if not any(row):
                        continue

                    # Serialize as "Header1: val1, Header2: val2, ..." instead
                    # of a flattened grid - this reads far better for both
                    # embedding and for the LLM at generation time.
                    pairs = [
                        f"{h}: {v}" for h, v in zip(header, row) if h and v
                    ]
                    if not pairs:
                        continue

                    sentence = f"[{part_number}, table on page {page_num}] " + "; ".join(pairs)

                    if _is_junk_table_row(sentence):
                        junk_filtered += 1
                        continue

                    chunks.append({
                        "part_number": part_number,
                        "page_number": page_num,
                        "section": f"Table {table_idx + 1} (page {page_num})",
                        "type": "table_row",
                        "content": sentence,
                    })

    if junk_filtered:
        print(f"  filtered {junk_filtered} junk table rows (graph mis-parses / nav chrome)")

    return chunks


def parse_datasheet(pdf_path: Path):
    part_number = pdf_path.stem
    print(f"Parsing {part_number}...")

    text_chunks = extract_text_chunks(pdf_path, part_number)
    table_chunks = extract_table_chunks(pdf_path, part_number)

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