"""
collect_text_pdfs.py

Scans a directory of student folders, detects text-based PDFs (vs. scanned/image),
copies valid ones to an output directory with student-prefixed filenames, and saves
a JSON mapping of student_id → list of copied PDF filenames.

Each PDF is classified per-page into one of three categories:
  text    — all non-blank pages pass quality gates  → copied to <output>/
  mixed   — some pages pass, some don't             → copied to <output>/mixed/
  scanned — no pages pass                           → skipped

Usage:
    python collect_text_pdfs.py --input <students_dir> --output <output_dir> [--min-chars 100] [--min-alpha 0.6]

Dependencies:
    pip install pymupdf
"""

import argparse
import json
import shutil
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Per-page quality gates:
#   MIN_CHARS       — a page must have at least this many non-whitespace content
#                     characters to be considered a text page.
#   MIN_ALPHA_RATIO — at least this fraction of those characters must be alphabetic.
DEFAULT_MIN_CHARS = 100
DEFAULT_MIN_ALPHA_RATIO = 0.6

# Pages with fewer than this many non-whitespace chars are treated as blank
# (cover pages, intentionally empty pages) and excluded from classification.
BLANK_PAGE_CHARS = 20

# Unicode categories that are invisible noise in PDF text streams.
# Excluded from both the count and the alpha ratio computation.
#   Cc — control chars, Cf — format/bidi chars (common in Hebrew/Arabic PDFs),
#   Co — private-use glyphs, Cs — surrogates
_UNICODE_NOISE = frozenset({"Cc", "Cf", "Co", "Cs"})

# Classification labels
TEXT = "text"
MIXED = "mixed"
SCANNED = "scanned"


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def classify_text(text: str) -> Tuple[int, float]:
    """
    Return (non_ws_count, alpha_ratio) for a string, ignoring whitespace and
    Unicode noise characters (bidi marks, control chars, private-use glyphs).
    alpha_ratio is 0.0 when non_ws_count == 0.
    """
    content = [
        c for c in text
        if not c.isspace() and unicodedata.category(c) not in _UNICODE_NOISE
    ]
    non_ws_count = len(content)
    if non_ws_count == 0:
        return 0, 0.0
    alpha_ratio = sum(c.isalpha() for c in content) / non_ws_count
    return non_ws_count, alpha_ratio


def classify_pdf(
    pdf_path: Path,
    min_chars: int = DEFAULT_MIN_CHARS,
    min_alpha_ratio: float = DEFAULT_MIN_ALPHA_RATIO,
) -> str:
    """
    Classify a PDF by checking each page independently.

    Page classification:
      blank   — non_ws_count < BLANK_PAGE_CHARS  (ignored, not counted)
      text    — non_ws_count >= min_chars AND alpha_ratio >= min_alpha_ratio
      scanned — everything else

    Document classification from non-blank pages:
      text    — all non-blank pages are text pages
      mixed   — mix of text and scanned pages
      scanned — all non-blank pages are scanned (or file is unreadable)
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        print(f"  [ERROR] Could not open {pdf_path.name}: {exc}")
        return SCANNED

    text_pages = 0
    scanned_pages = 0

    with doc:
        for page in doc:
            page_text = page.get_text()
            non_ws_count, alpha_ratio = classify_text(page_text)

            if non_ws_count < BLANK_PAGE_CHARS:
                continue  # blank page — skip

            if non_ws_count >= min_chars and alpha_ratio >= min_alpha_ratio:
                text_pages += 1
            else:
                scanned_pages += 1

    total_meaningful = text_pages + scanned_pages

    if total_meaningful == 0:
        label = SCANNED  # entirely blank or unreadable
    elif scanned_pages == 0:
        label = TEXT
    elif text_pages == 0:
        label = SCANNED
    else:
        label = MIXED

    print(
        f"    text_pages={text_pages}  scanned_pages={scanned_pages}"
        f"  -> {label.upper()}"
    )
    return label


def copy_pdf(src: Path, dest_dir: Path, student_id: str) -> Path:
    """Copy src to dest_dir renamed as <student_id>_<original_name>."""
    dest_name = f"{student_id}_{src.name}"
    dest_path = dest_dir / dest_name
    shutil.copy2(src, dest_path)
    return dest_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_student_folders(
    input_dir: Path,
    output_dir: Path,
    min_chars: int,
    min_alpha_ratio: float,
) -> Dict[str, List[dict]]:
    """
    Walk input_dir, classify each PDF per-page, and route to the right destination:
      text    → output_dir/
      mixed   → output_dir/mixed/
      scanned → skipped

    Returns a mapping { student_id: [ {file, category}, ... ] }.
    """
    text_dir = output_dir / "text"
    mixed_dir = output_dir / "mixed"
    output_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    mixed_dir.mkdir(parents=True, exist_ok=True)

    counts = {TEXT: 0, MIXED: 0, SCANNED: 0}
    total = 0
    student_map: Dict[str, List[dict]] = {}

    student_dirs = sorted(p for p in input_dir.iterdir() if p.is_dir())
    if not student_dirs:
        print(f"No subdirectories found in {input_dir}")
        return student_map

    for student_dir in student_dirs:
        student_id = student_dir.name
        pdfs = sorted(student_dir.rglob("*.pdf"))

        for pdf_path in pdfs:
            total += 1
            print(f"  Checking: {student_id}/{pdf_path.relative_to(student_dir)}")

            label = classify_pdf(pdf_path, min_chars, min_alpha_ratio)
            counts[label] += 1

            if label == TEXT:
                dest = copy_pdf(pdf_path, text_dir, student_id)
                student_map.setdefault(student_id, []).append(
                    {"file": dest.name, "category": TEXT}
                )
                print(f"    -> copied to output/text/")

            elif label == MIXED:
                dest = copy_pdf(pdf_path, mixed_dir, student_id)
                student_map.setdefault(student_id, []).append(
                    {"file": dest.name, "category": MIXED}
                )
                print(f"    -> copied to output/mixed/")

            else:
                print(f"    -> skipped (scanned / unreadable)")

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  Total PDFs found : {total}")
    print(f"  Text (copied)    : {counts[TEXT]}   →  {text_dir}")
    print(f"  Mixed (flagged)  : {counts[MIXED]}  →  {mixed_dir}")
    print(f"  Scanned (skipped): {counts[SCANNED]}")
    print("=" * 50)

    return student_map


def save_mapping(student_map: Dict[str, List[dict]], output_dir: Path) -> None:
    """Save the student_id → file list mapping as JSON."""
    mapping_path = output_dir / "student_pdf_mapping.json"
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(student_map, f, indent=2, ensure_ascii=False)
    print(f"\nMapping saved to: {mapping_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect text-based PDFs from student folders."
    )
    parser.add_argument(
        "--input", "-i", required=True, type=Path,
        help="Root directory containing one subfolder per student.",
    )
    parser.add_argument(
        "--output", "-o", required=True, type=Path,
        help="Directory where text-based PDFs will be copied.",
    )
    parser.add_argument(
        "--min-chars", "-c", type=int, default=DEFAULT_MIN_CHARS,
        help=f"Min non-whitespace chars per page to count as a text page (default: {DEFAULT_MIN_CHARS}).",
    )
    parser.add_argument(
        "--min-alpha", "-a", type=float, default=DEFAULT_MIN_ALPHA_RATIO,
        help=f"Min alphabetic ratio per page (default: {DEFAULT_MIN_ALPHA_RATIO}). Range: 0.0–1.0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir: Path = args.input.resolve()
    output_dir: Path = args.output.resolve()

    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    print(f"Input       : {input_dir}")
    print(f"Output      : {output_dir}")
    print(f"Text        : {output_dir / 'text'}")
    print(f"Mixed       : {output_dir / 'mixed'}")
    print(f"Min chars   : {args.min_chars}  (per page)")
    print(f"Min alpha   : {args.min_alpha}  (per page)\n")

    student_map = process_student_folders(
        input_dir, output_dir, args.min_chars, args.min_alpha
    )
    save_mapping(student_map, output_dir)


if __name__ == "__main__":
    main()
