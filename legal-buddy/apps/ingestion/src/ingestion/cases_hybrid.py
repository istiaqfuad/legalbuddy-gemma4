"""Page-level hybrid extraction: tier-1 PyMuPDF text per page, OCR only the pages
that need it.

Two facts drive the design:
  * tier-1 PyMuPDF text is perfect for clean English pages,
  * but is unusable for Bengali/legacy-font pages -- whether the text layer comes
    out as mojibake Latin (``q¡C-L``) or as scrambled Bengali codepoints
    (``আমভ স``), both need OCR of the rendered bitmap.

Whole-file OCR (the old tier-2) fixed the Bengali but *degraded the clean English*
(``Shaishir`` -> ``Slaishir``) and burned GPU on English pages. Page-level hybrid
keeps each page on its best engine:

    for each page:
        t = page.get_text()                      # cheap, always
        route = probe_page(t)                     # ok | garbled | bengali | scanned | empty
        ok            -> keep t                    (pristine English, no GPU)
        garbled|bengali|scanned -> OCR this page   (readable Bengali)
        empty         -> drop                      (blank, nothing to read)

So a pure-English judgment never loads EasyOCR; a 16k-page bundle OCRs only its
Bengali pages; a mixed file keeps its English pages clean and OCRs its Bengali
pages. The EasyOCR reader is built lazily (first OCR page across the whole run)
and reused, so the English-only majority pays zero GPU cost.

No font map: OCR reads the bitmap, so it is font-agnostic (AdarshaLipi /
LipiChameli / Bijoy / SutonnyMJ) and generalizes to any future PDF.
"""
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

# Per-page routing thresholds (page text is shorter/noisier than whole-doc, but
# the same signals hold). A page below this many real chars has no usable text
# layer -> OCR if it carries an image, else it is a blank/separator page.
SCANNED_PAGE_CHARS = 100
PAGE_BENGALI_RATIO = 0.03      # real OR scrambled Bengali present -> OCR
PAGE_NONASCII_RATIO = 0.10     # legacy-font mojibake -> OCR

DEFAULT_DPI = 200

_BENGALI_RE = re.compile(r"[ঀ-৿]")


def tidy_text(text: str) -> str:
    """One line per paragraph; a linebreak only marks a new paragraph.

    Both engines emit paragraphs separated by a blank line (tier-1 from PyMuPDF
    text blocks, OCR from EasyOCR's paragraph grouping). A single newline *inside*
    a paragraph is a wrap, not a break, so it collapses to a space. The result has
    each paragraph on its own line, paragraphs separated by one blank line -- no
    per-visual-line newlines.
    """
    paras = re.split(r"\n\s*\n+", text)            # blank line(s) = paragraph break
    cleaned = [re.sub(r"\s+", " ", p).strip() for p in paras]   # wraps -> spaces
    return "\n\n".join(p for p in cleaned if p)


def _bengali_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if _BENGALI_RE.match(c)) / len(letters)


def _nonascii_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if ord(c) > 127) / len(chars)


def probe_page(text: str, has_image: bool) -> str:
    """Route a single page from its tier-1 text. See module docstring."""
    stripped = text.strip()
    if len(stripped) < SCANNED_PAGE_CHARS:
        # too little text to be a real text layer: OCR if there's an image to read,
        # otherwise it's a genuinely blank/separator page.
        return "scanned" if has_image else "empty"
    if _bengali_ratio(text) >= PAGE_BENGALI_RATIO:
        return "bengali"
    if _nonascii_ratio(text) >= PAGE_NONASCII_RATIO:
        return "garbled"
    return "ok"


@dataclass
class HybridResult:
    source_file: str
    text: str
    n_pages: int
    pages_read: int
    pages_tier1: int
    pages_ocr: int
    pages_empty: int
    ocr_capped: bool                 # hit max_ocr_pages budget
    char_count: int
    bengali_ratio: float
    route_counts: dict = field(default_factory=dict)
    page_routes: list = field(default_factory=list)   # omitted for giant files
    header_text: str = ""   # line-structured top-of-doc text, for header parsing

    def to_meta(self) -> dict:
        d = asdict(self)
        d.pop("text")
        d.pop("header_text", None)
        return d


def _page_text(page) -> str:
    """Tier-1 page text reflowed into paragraphs.

    PyMuPDF sometimes returns one block per visual line, so block-joining alone
    still breaks every line. Instead we take the text lines with their vertical
    positions and re-wrap: consecutive lines join into one paragraph, and a new
    paragraph starts only where the vertical gap exceeds the page's typical line
    leading (a real paragraph gap, not a wrap). So a linebreak marks a new
    paragraph, never a wrap."""
    lines = []
    for blk in page.get_text("dict").get("blocks", []):
        if blk.get("type", 0) != 0:        # skip image blocks
            continue
        for ln in blk.get("lines", []):
            txt = "".join(sp.get("text", "") for sp in ln.get("spans", [])).strip()
            if txt:
                lines.append((ln["bbox"][1], ln["bbox"][0], txt))  # (y0, x0, text)
    if not lines:
        return ""
    lines.sort(key=lambda l: (round(l[0], 1), l[1]))

    gaps = sorted(g for g in (lines[i][0] - lines[i - 1][0]
                              for i in range(1, len(lines))) if g > 0)
    median = gaps[len(gaps) // 2] if gaps else 0.0
    # 1.5x the typical leading = a paragraph gap; below that is a line wrap.
    threshold = median * 1.5 if median else float("inf")

    paras = [lines[0][2]]
    for i in range(1, len(lines)):
        gap = lines[i][0] - lines[i - 1][0]
        if gap > threshold:
            paras.append(lines[i][2])      # new paragraph
        else:
            paras[-1] += " " + lines[i][2]  # same paragraph: wrap -> space
    return "\n\n".join(paras)


def _make_reader(langs: list[str], gpu: bool):
    """Build an EasyOCR reader. Imported lazily so the package stays import-light
    and English-only runs never load torch/easyocr."""
    import easyocr
    return easyocr.Reader(langs, gpu=gpu)


def _ocr_page(page, reader, dpi: int) -> str:
    """Render one page to a bitmap and OCR it (en+bn). paragraph=True returns one
    string per paragraph; join with a blank line so each is its own paragraph."""
    mat = __import__("fitz").Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat)
    lines = reader.readtext(pix.tobytes("png"), detail=0, paragraph=True)
    return "\n\n".join(s.strip() for s in lines if s and s.strip())


def extract_document(
    pdf_path: str | Path,
    get_reader: Callable[[], object],
    dpi: int = DEFAULT_DPI,
    page_cap: int | None = None,
    max_ocr_pages: int | None = None,
    keep_routes_max_pages: int = 400,
) -> HybridResult:
    """Page-level hybrid extraction.

    ``get_reader`` is a zero-arg callable returning a (cached) EasyOCR reader; it is
    only invoked when a page actually needs OCR, so English-only documents never
    build a reader. ``max_ocr_pages`` bounds GPU work on pathological bundles
    (thousands of Bengali pages) while still reading every page's text layer.
    """
    import fitz

    pdf_path = Path(pdf_path)
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return HybridResult(pdf_path.name, "", 0, 0, 0, 0, 0, False, 0, 0.0,
                            {"empty": 1}, [])

    n_pages = doc.page_count
    pages_read = n_pages if page_cap is None else min(n_pages, page_cap)
    parts, routes, header_parts = [], [], []
    n_tier1 = n_ocr = n_empty = 0
    ocr_used = 0
    capped = False

    for i in range(pages_read):
        page = doc[i]
        try:
            t = _page_text(page)
        except Exception:
            t = ""
        if i < 3:
            # raw line-structured text of the head pages, for header field parsing
            # (the reflowed body collapses the line anchors parse_case relies on).
            header_parts.append(page.get_text("text"))
        has_image = bool(page.get_images(full=True))
        route = probe_page(t, has_image)

        if route == "ok":
            parts.append(tidy_text(unicodedata.normalize("NFC", t)))
            n_tier1 += 1
        elif route == "empty":
            n_empty += 1
            # nothing to append
        else:  # garbled | bengali | scanned -> OCR
            if max_ocr_pages is not None and ocr_used >= max_ocr_pages:
                capped = True
                # budget spent: fall back to whatever tier-1 had (better than nothing)
                if t.strip():
                    parts.append(tidy_text(unicodedata.normalize("NFC", t)))
                routes.append(route + "+capped")
                continue
            otext = _ocr_page(page, get_reader(), dpi)
            ocr_used += 1
            if otext.strip():
                parts.append(tidy_text(unicodedata.normalize("NFC", otext)))
                n_ocr += 1
            else:
                n_empty += 1
        routes.append(route)

    doc.close()

    text = tidy_text("\n\n".join(p for p in parts if p))
    counts: dict = {}
    for r in routes:
        counts[r] = counts.get(r, 0) + 1

    return HybridResult(
        source_file=pdf_path.name, text=text, n_pages=n_pages, pages_read=pages_read,
        pages_tier1=n_tier1, pages_ocr=n_ocr, pages_empty=n_empty, ocr_capped=capped,
        char_count=len(text), bengali_ratio=round(_bengali_ratio(text), 4),
        route_counts=counts,
        page_routes=routes if n_pages <= keep_routes_max_pages else [],
        header_text=unicodedata.normalize("NFC", "\n".join(header_parts)).strip(),
    )
