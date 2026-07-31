"""Frozen snapshot of the CURRENT notebooks/ingest_qdrant.py chunking logic.

Kept here so the baseline arm of the A/B stays reproducible after the production
file is edited with improvements #1-#4. Logic is copied verbatim; only the
per-section driver (yielding chunks for one section) is factored out so the eval
ingest wrapper can attach identity tags.
"""
import re

SECTION_INDEX_RE = re.compile(r"^[\s\"'\[\]]*\[?([0-9০-৯]+[a-zA-Z]*)[.।৷\-\s]")
FOOTNOTE_MARKER_RE = re.compile(r"\d+\[(.*?)\]")
VOID_SECTION_RE = re.compile(
    r"\[\s*(Omitted|Repealed?|Rep\.)\s+by" r"|\[\s*Repeal\.\-" r"|\[\s*Omit\.\-",
    re.IGNORECASE,
)
SUBSECTION_SPLIT_RE = re.compile(r"(?=(?:\(\d+\)|\([a-zA-Z]+\)))")

MAX_CHARS_PER_CHUNK = 1200
CHUNK_OVERLAP = 100


def extract_section_index(section_content: str) -> str:
    if not section_content:
        return "Unknown"
    match = SECTION_INDEX_RE.search(section_content)
    return (match.group(1).strip() if match else "Unknown") or "Unknown"


def clean_section_content(section_content: str) -> str:
    if not section_content:
        return ""
    return FOOTNOTE_MARKER_RE.sub(r"\1", section_content).strip()


def chunk_section_content(text: str, max_chars: int = 1200, overlap: int = 100) -> list[str]:
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    def slice_with_overlap(value: str) -> list[str]:
        chunks, start = [], 0
        while start < len(value):
            end = min(start + max_chars, len(value))
            chunks.append(value[start:end].strip())
            if end >= len(value):
                break
            start = max(0, end - overlap)
        return [chunk for chunk in chunks if chunk]

    parts = [part.strip() for part in SUBSECTION_SPLIT_RE.split(text) if part.strip()]
    if len(parts) > 1:
        merged, current = [], ""
        for part in parts:
            candidate = f"{current} {part}".strip() if current else part
            if len(candidate) <= max_chars:
                current = candidate
                continue
            if current:
                merged.append(current)
                current = ""
            if len(part) <= max_chars:
                current = part
                continue
            merged.extend(slice_with_overlap(part))
        if current:
            merged.append(current)
        if merged:
            return merged
    return slice_with_overlap(text)


def build_embedding_text(act_title: str, section_index: str, chunk_part: int, chunk_text: str) -> str:
    return f"Act: {act_title}\nSection {section_index} (Part {chunk_part}): {chunk_text}"


def section_records(act_obj: dict, section: dict, model=None) -> list[dict]:
    """Yield baseline records for one section: {embedding_text, payload_extra}.

    payload_extra holds the chunk-specific payload fields the baseline pipeline
    stores. The eval wrapper adds identity tags (act_file, section_ord, etc.).
    Returns [] for void/empty sections (skipped, exactly like production).
    """
    raw_content = (section or {}).get("section_content", "")
    if VOID_SECTION_RE.search(raw_content or ""):
        return []
    cleaned = clean_section_content(raw_content)
    if not cleaned:
        return []

    section_index = extract_section_index(cleaned)
    act_title = act_obj.get("act_title", "Unknown Act")
    records = []
    for chunk_part, chunk_text in enumerate(
        chunk_section_content(cleaned, MAX_CHARS_PER_CHUNK, CHUNK_OVERLAP), start=1
    ):
        records.append(
            {
                "embedding_text": build_embedding_text(
                    act_title, section_index, chunk_part, chunk_text
                ),
                "payload_extra": {
                    "section_index": section_index,
                    "chunk_part": chunk_part,
                    "section_content_clean": chunk_text,
                },
            }
        )
    return records
