"""
Downloads datasheet PDFs listed in data/manifest.csv into data/raw_pdfs/.

Usage:
    python src/download_datasheets.py

Notes:
- Run this on a machine, not in a restricted sandbox as manufacturer sites are not always reachable from locked-down network environments
- Manufacturer sites occasionally change their PDF URL structure or block scripted requests
"""

import csv
import time
from pathlib import Path

import requests

MANIFEST_PATH = Path(__file__).parent.parent / "data" / "manifest.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw_pdfs"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

REQUEST_DELAY_SECONDS = 2


def download_datasheet(part_number: str, url: str) -> bool:
    """Download a single datasheet PDF. Returns True on success."""
    if not url:
        print(f"  [skip] {part_number}: no URL in manifest yet")
        return False

    dest = OUTPUT_DIR / f"{part_number}.pdf"
    if dest.exists():
        print(f"  [skip] {part_number}: already downloaded")
        return True

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "pdf" not in content_type.lower() and not response.content[:4] == b"%PDF":
            print(f"  [fail] {part_number}: response was not a PDF (got {content_type})")
            return False

        dest.write_bytes(response.content)
        print(f"  [ok]   {part_number}: saved to {dest}")
        return True

    except requests.RequestException as exc:
        print(f"  [fail] {part_number}: {exc}")
        return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"Found {len(rows)} datasheets in manifest\n")

    results = {"ok": 0, "skip": 0, "fail": 0}
    for row in rows:
        part_number = row["part_number"]
        url = row["url"].strip()
        success = download_datasheet(part_number, url)
        if success:
            results["ok"] += 1
        elif not url:
            results["skip"] += 1
        else:
            results["fail"] += 1
        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\nDone. {results['ok']} downloaded, {results['fail']} failed, {results['skip']} skipped (no URL yet).")
    if results["fail"] or results["skip"]:
        print("For failed/skipped parts: open the manufacturer product page in a browser, ")
        print("find the current datasheet PDF link, and update data/manifest.csv, then re-run.")


if __name__ == "__main__":
    main()
