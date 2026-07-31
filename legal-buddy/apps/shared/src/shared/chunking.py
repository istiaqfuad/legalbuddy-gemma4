"""Token-aware, structure-aware chunking for Bangladesh legal acts.

Shared by ingestion (build the index) and available to anything that needs to
reproduce the exact chunking. The chunk is sized to the embedding model's REAL
max_seq_length so the whole chunk is actually embedded (no silent truncation);
the contextual header and "passage:" prefix are reserved out of the budget.
"""
import json
import re
from pathlib import Path

from shared.embedding import model_max_tokens, passage_prefix

TOKEN_OVERLAP = 24  # ~20% of a 128-token window

SECTION_INDEX_RE = re.compile(r"^[\s\"'\[\]]*\[?([0-9০-৯]+[a-zA-Z]*)[.।৷\-\s]")
FOOTNOTE_MARKER_RE = re.compile(r"\d+\[(.*?)\]")
VOID_SECTION_RE = re.compile(
    r"\[\s*(Omitted|Repealed?|Rep\.)\s+by" r"|\[\s*Repeal\.\-" r"|\[\s*Omit\.\-",
    re.IGNORECASE,
)
SUBSECTION_SPLIT_RE = re.compile(r"(?=(?:\(\d+\)|\([a-zA-Z]+\)))")


def extract_section_index(section_content: str) -> str:
    if not section_content:
        return "Unknown"
    match = SECTION_INDEX_RE.search(section_content)
    return (match.group(1).strip() if match else "Unknown") or "Unknown"


def clean_section_content(section_content: str) -> str:
    if not section_content:
        return ""
    return FOOTNOTE_MARKER_RE.sub(r"\1", section_content).strip()


# ---- token-aware splitting ----------------------------------------------------

def _token_len(model, text: str) -> int:
    return len(model.tokenizer.encode(text, add_special_tokens=False))


def _token_slice(model, text: str, budget: int, overlap: int) -> list[str]:
    """Slice a too-long string into <=budget-token windows with token overlap."""
    ids = model.tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= budget:
        return [text.strip()] if text.strip() else []
    out, start = [], 0
    stride = max(1, budget - overlap)
    while start < len(ids):
        window = ids[start : start + budget]
        piece = model.tokenizer.decode(window).strip()
        if piece:
            out.append(piece)
        if start + budget >= len(ids):
            break
        start += stride
    return out


def chunk_section_tokens(model, text: str, budget: int, overlap: int) -> list[str]:
    """Split-then-merge on subsection boundaries, measured in tokens."""
    if not text:
        return []
    if _token_len(model, text) <= budget:
        return [text]
    parts = [p.strip() for p in SUBSECTION_SPLIT_RE.split(text) if p.strip()]
    if len(parts) > 1:
        merged, current = [], ""
        for part in parts:
            candidate = f"{current} {part}".strip() if current else part
            if _token_len(model, candidate) <= budget:
                current = candidate
                continue
            if current:
                merged.append(current)
                current = ""
            if _token_len(model, part) <= budget:
                current = part
                continue
            merged.extend(_token_slice(model, part, budget, overlap))
        if current:
            merged.append(current)
        if merged:
            return merged
    return _token_slice(model, text, budget, overlap)


# ---- contextual header --------------------------------------------------------

def context_header(act_title: str, section_title: str | None, section_index: str) -> str:
    bits = [f"Act: {act_title}"]
    if section_title:
        bits.append(f"Title: {section_title}")
    bits.append(f"Section {section_index}")
    return " | ".join(bits)


def build_embedding_text(header: str, part_no: int, n_parts: int, chunk_text: str) -> str:
    # Every part carries the act/title/section header, so split parts 2..n keep
    # their context instead of being a headless tail.
    cont = f" (part {part_no}/{n_parts})" if n_parts > 1 else ""
    return f"{header}{cont}\n{chunk_text}"


def section_records(act_obj: dict, section: dict, model, e5: bool) -> list[dict]:
    """Yield records for one section: {embedding_text, payload_extra}.

    Adds section_title to the header/payload, sizes chunks to the model window,
    repeats the header on every split part, and carries section_full for
    parent-document retrieval. Returns [] for void/empty sections.
    """
    raw_content = (section or {}).get("section_content", "")
    if VOID_SECTION_RE.search(raw_content or ""):
        return []
    cleaned = clean_section_content(raw_content)
    if not cleaned:
        return []

    section_title = (section or {}).get("section_title")
    section_index = extract_section_index(cleaned)
    act_title = act_obj.get("act_title", "Unknown Act")
    header = context_header(act_title, section_title, section_index)

    max_tokens = model_max_tokens(model)
    # +12 covers the "passage:" prefix, the "(part k/n)" marker, and the newline
    # that sit alongside the header but aren't in `header` itself.
    reserved = _token_len(model, passage_prefix(header, e5)) + 12
    budget = max(32, min(max_tokens - reserved, max_tokens - 4))
    overlap = min(TOKEN_OVERLAP, max(4, budget // 4))

    parts = chunk_section_tokens(model, cleaned, budget, overlap)
    n_parts = len(parts)
    records = []
    for part_no, chunk_text in enumerate(parts, start=1):
        records.append(
            {
                "embedding_text": build_embedding_text(header, part_no, n_parts, chunk_text),
                "payload_extra": {
                    "section_title": section_title,
                    "section_index": section_index,
                    "chunk_part": part_no,
                    "n_parts": n_parts,
                    "section_content_clean": chunk_text,
                    "section_full": cleaned,  # parent-document text
                },
            }
        )
    return records


def collect_all_records(acts_dir: Path, model, e5: bool) -> tuple[list[dict], dict]:
    """Walk acts_dir, chunk every non-repealed section, return (records, stats)."""
    records = []
    stats = {
        "files_seen": 0,
        "acts_skipped_repealed": 0,
        "acts_skipped_no_sections": 0,
        "sections_seen": 0,
        "sections_skipped_void_or_empty": 0,
        "chunks_created": 0,
    }

    for file_path in sorted(Path(acts_dir).glob("act-print-*.json")):
        stats["files_seen"] += 1
        with open(file_path, "r", encoding="utf-8") as f:
            act_obj = json.load(f)

        if act_obj.get("csv_metadata", {}).get("is_repealed") is True:
            stats["acts_skipped_repealed"] += 1
            continue

        sections = act_obj.get("sections") or []
        if not sections:
            stats["acts_skipped_no_sections"] += 1
            continue

        for section_ord, section in enumerate(sections):
            stats["sections_seen"] += 1
            recs = section_records(act_obj, section, model, e5)
            if not recs:
                stats["sections_skipped_void_or_empty"] += 1
                continue
            for rec in recs:
                payload = {
                    "act_file": file_path.stem,
                    "section_ord": section_ord,
                    "section_uid": f"{file_path.stem}#{section_ord}",
                    "act_title": act_obj.get("act_title"),
                    "act_no": act_obj.get("act_no"),
                    "act_year": (
                        int(act_obj["act_year"])
                        if str(act_obj.get("act_year", "")).isdigit()
                        else None
                    ),
                    "language": act_obj.get("language"),
                    "govt_system": act_obj.get("government_context", {}).get("govt_system"),
                    "source_url": act_obj.get("source_url"),
                    **rec["payload_extra"],
                }
                records.append({"embedding_text": rec["embedding_text"], "payload": payload})

    stats["chunks_created"] = len(records)
    return records, stats
