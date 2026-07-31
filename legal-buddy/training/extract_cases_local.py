"""Extract text from Bangladesh court case PDFs using PyMuPDF.

Runs locally — no GPU or API key needed. Produces a JSONL file
that can be uploaded to the remote GPU machine for processing.

Usage:
    cd legal-buddy
    uv run python training/extract_cases_local.py
"""
import fitz  # PyMuPDF
import json
import sys
from pathlib import Path

# Resolve paths relative to the repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
CASES_DIR = REPO_ROOT.parent / "legal_dataset" / "cases"
OUTPUT_FILE = REPO_ROOT / "training" / "parsed_cases.jsonl"


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF file using PyMuPDF."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text.strip())
    doc.close()
    return "\n\n".join(pages)


def parse_case_metadata_from_filename(filename: str) -> dict:
    """Extract rough case metadata from the filename."""
    stem = filename.replace(".pdf", "")
    parts = stem.split("_", 1)
    return {
        "case_id": parts[0] if parts else stem,
        "case_description": parts[1] if len(parts) > 1 else "",
    }


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not CASES_DIR.exists():
        print(f"Cases directory not found: {CASES_DIR}")
        sys.exit(1)

    pdf_files = sorted(CASES_DIR.glob("*.pdf"))
    total = len(pdf_files)
    print(f"Found {total} case PDFs in {CASES_DIR}")

    success = 0
    skipped = 0
    errors = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, pdf_path in enumerate(pdf_files):
            try:
                text = extract_text_from_pdf(pdf_path)
                if len(text) < 200:
                    skipped += 1
                    continue

                meta = parse_case_metadata_from_filename(pdf_path.name)
                record = {
                    "filename": pdf_path.name,
                    "case_id": meta["case_id"],
                    "case_description": meta["case_description"],
                    "text": text,
                    "char_count": len(text),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                success += 1

            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  Error: {pdf_path.name}: {e}")

            if (i + 1) % 200 == 0:
                print(
                    f"  Progress: {i + 1}/{total} "
                    f"(✓ {success} | ⏭ {skipped} | ✗ {errors})"
                )

    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"\nDone!")
    print(f"  Extracted: {success}")
    print(f"  Skipped (too short): {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Output: {OUTPUT_FILE} ({file_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
