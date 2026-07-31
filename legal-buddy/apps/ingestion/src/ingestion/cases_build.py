"""Build: data/cases/*.pdf -> data/cases_json/<case_id>.json via page-level
hybrid extraction (``cases_hybrid``).

Every PDF is read in one pass: each page goes to PyMuPDF text if it is clean
English, or to EasyOCR if it is Bengali / legacy-font garbled / scanned. So a
pure-English judgment never loads the GPU; a mixed file keeps its English pages
pristine and OCRs only its Bengali pages, instead of embedding those pages as
mojibake.

    EMBEDDING_DEVICE=cuda .venv/bin/python -m ingestion.cases_build   # whole corpus (GPU)
    CASES_LIMIT=30 ...                                                # smoke
    CASES_HYBRID_RESUME=1 ...                                         # skip already-written
    CASES_HYBRID_MAX_OCR_PAGES=400 ...                               # per-file OCR budget

Must run on the Fedora GPU box (OCR is inline). Heavy deps imported lazily.
"""
import json
import os
import time
from collections import Counter
from pathlib import Path

from ingestion.cases_hybrid import extract_document
from ingestion.cases_structure import parse_case

ROOT = Path(__file__).resolve().parents[4]
CASES_DIR = ROOT / "data" / "cases"
OUT_DIR = ROOT / "data" / "cases_json"

# Bound OCR on a single pathological bundle (e.g. the 16k-page death-reference
# paper book) while still reading every page's text layer.
DEFAULT_MAX_OCR_PAGES = 400


def _lang(ben: float) -> str:
    if ben >= 0.15:
        return "bengali"
    if ben >= 0.03:
        return "mixed"
    return "english"


def build(limit: int | None = None) -> dict:
    gpu = os.getenv("EMBEDDING_DEVICE", "cpu").lower() == "cuda"
    resume = os.getenv("CASES_HYBRID_RESUME", "").lower() in ("1", "true")
    mo_env = os.getenv("CASES_HYBRID_MAX_OCR_PAGES", "").strip()
    max_ocr = int(mo_env) if mo_env.isdigit() else DEFAULT_MAX_OCR_PAGES
    pc_env = os.getenv("CASES_HYBRID_PAGE_CAP", "").strip()
    page_cap = int(pc_env) if pc_env.isdigit() else None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(CASES_DIR.glob("*.pdf"))
    if limit:
        pdfs = pdfs[:limit]

    # lazy shared EasyOCR reader: built on the first OCR page across the whole run,
    # so a corpus slice that is all English never pays the GPU init.
    _reader = {}
    def get_reader():
        if "r" not in _reader:
            from ingestion.cases_hybrid import _make_reader
            print("[cases-build] building EasyOCR reader (first OCR page)", flush=True)
            _reader["r"] = _make_reader(["en", "bn"], gpu)
        return _reader["r"]

    print(f"[cases-build] gpu={gpu} resume={resume} max_ocr={max_ocr} "
          f"page_cap={page_cap} files={len(pdfs)}", flush=True)

    stats = Counter()
    lang_dist = Counter()
    total_ocr_pages = total_tier1_pages = files_with_ocr = 0
    written = skipped = 0
    failed = []
    t0 = time.time()

    for i, pdf in enumerate(pdfs, 1):
        out_path = OUT_DIR / f"{pdf.stem}.json"
        if resume and out_path.exists():
            skipped += 1
            continue
        try:
            res = extract_document(pdf, get_reader, page_cap=page_cap, max_ocr_pages=max_ocr)
        except Exception as exc:
            failed.append({"file": pdf.name, "error": str(exc)[:200]})
            continue
        if not res.text.strip():
            failed.append({"file": pdf.name, "error": "no text after hybrid"})
            continue

        rec = parse_case(pdf.stem, res.text, header_text=res.header_text)
        ben = res.bengali_ratio
        rec.update(
            extraction_tier="hybrid",
            lang=_lang(ben),
            bengali_ratio=ben,
            n_pages=res.n_pages,
            pages_read=res.pages_read,
            pages_tier1=res.pages_tier1,
            pages_ocr=res.pages_ocr,
            pages_empty=res.pages_empty,
            route_counts=res.route_counts,
            ocr_capped=res.ocr_capped,
            char_count=res.char_count,
        )
        out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        written += 1

        lang_dist[rec["lang"]] += 1
        total_ocr_pages += res.pages_ocr
        total_tier1_pages += res.pages_tier1
        if res.pages_ocr:
            files_with_ocr += 1
        if not rec["full_case_ref"]:
            stats["no_case_ref"] += 1
        if res.ocr_capped:
            stats["ocr_capped"] += 1

        if i % 100 == 0 or i == len(pdfs):
            el = time.time() - t0
            done = i - skipped
            rate = el / max(done, 1)
            eta = rate * (len(pdfs) - i) / 60
            print(f"[cases-build] {i}/{len(pdfs)} written={written} skip={skipped} "
                  f"ocr_pages={total_ocr_pages} ({rate:.2f}s/file, ETA {eta:.0f}m)",
                  flush=True)

    manifest = {
        "total_pdfs": len(pdfs),
        "written": written, "skipped": skipped, "failed": len(failed),
        "total_tier1_pages": total_tier1_pages,
        "total_ocr_pages": total_ocr_pages,
        "files_with_ocr": files_with_ocr,
        "lang_dist": dict(lang_dist),
        "flags": dict(stats),
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "failures": failed[:200],
    }
    (OUT_DIR / "_build_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[cases-build] done: {json.dumps({k: v for k, v in manifest.items() if k != 'failures'})}",
          flush=True)
    return manifest


def main() -> None:
    env = os.getenv("CASES_LIMIT", "").strip()
    limit = int(env) if env.isdigit() and int(env) > 0 else None
    build(limit=limit)


if __name__ == "__main__":
    main()
