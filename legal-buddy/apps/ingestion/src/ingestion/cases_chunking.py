"""Chunk structured case records (data/cases_json/*.json) for embedding.

Judgments are long prose, not the numbered sections statutes have, so we split on
paragraph boundaries and merge to the embedding model's token window with
overlap -- the prose analogue of ``shared.chunking.chunk_section_tokens``. Every
chunk carries a one-line **case header** (the cross-document analogue of the
act/section header) so a mid-judgment chunk still says which case it is from.
Unlike the acts pipeline we do NOT store the whole judgment per chunk -- a single
copy would blow Qdrant's 32MB request limit; neighbouring context is fetched on
demand by (case_uid, chunk_part), with the full text in data/cases_json/.

Token primitives are reused from ``shared.chunking`` so cases and acts measure
length identically against the same model.
"""
import json
import re
from pathlib import Path

from shared.chunking import _token_len, _token_slice
from shared.embedding import model_max_tokens, passage_prefix

TOKEN_OVERLAP = 32
_PARA_SPLIT_RE = re.compile(r"\n\s*\n+")


def case_header(rec: dict) -> str:
    """One-line context header prepended to every chunk of a case."""
    bits = [f"Case: {rec.get('full_case_ref') or rec.get('case_type') or 'Judgment'}"]
    court = rec.get("division") or rec.get("court")
    if court:
        bits.append(court.title() if court.isupper() else court)
    parties = _short_parties(rec)
    if parties:
        bits.append(parties)
    return " | ".join(bits)


def _short_parties(rec: dict) -> str | None:
    def first_party(raw: str | None) -> str | None:
        if not raw:
            return None
        line = next((l.strip() for l in raw.splitlines()
                     if l.strip() and not l.strip().startswith(("-", ".", "="))), None)
        return (line[:60] if line else None)

    pet, resp = first_party(rec.get("petitioners_raw")), first_party(rec.get("respondents_raw"))
    if pet and resp:
        return f"{pet} v. {resp}"
    return pet or resp


def chunk_prose_tokens(model, text: str, budget: int, overlap: int) -> list[str]:
    """Merge paragraphs up to ``budget`` tokens; slice any single over-long one."""
    if not text:
        return []
    # Whole-text shortcut only for small bodies; for a multi-thousand-page bundle,
    # tokenizing all ~13M chars in one call just to learn "too long" is wasteful,
    # so go straight to paragraph splitting.
    if len(text) < 20_000 and _token_len(model, text) <= budget:
        return [text.strip()]
    paras = [p.strip() for p in _PARA_SPLIT_RE.split(text) if p.strip()]
    merged, current = [], ""
    for para in paras:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if _token_len(model, candidate) <= budget:
            current = candidate
            continue
        if current:
            merged.append(current)
        if _token_len(model, para) <= budget:
            current = para
        else:
            merged.extend(_token_slice(model, para, budget, overlap))
            current = ""
    if current:
        merged.append(current)
    return merged or _token_slice(model, text, budget, overlap)


def case_records(rec: dict, model, e5: bool) -> list[dict]:
    """Yield {embedding_text, payload} chunk records for one case JSON record."""
    body = rec.get("body_text") or ""
    if not body.strip():
        return []

    header = case_header(rec)
    max_tokens = model_max_tokens(model)
    reserved = _token_len(model, passage_prefix(header, e5)) + 12
    budget = max(64, min(max_tokens - reserved, max_tokens - 4))
    overlap = min(TOKEN_OVERLAP, max(8, budget // 4))

    parts = chunk_prose_tokens(model, body, budget, overlap)
    n_parts = len(parts)
    case_uid = str(rec.get("case_id"))
    base_payload = {
        "case_id": rec.get("case_id"),
        "case_uid": case_uid,
        "source_file": rec.get("source_file"),
        "court": rec.get("court"),
        "division": rec.get("division"),
        "jurisdiction": rec.get("jurisdiction"),
        "case_type": rec.get("case_type"),
        "case_no": rec.get("case_no"),
        "case_year": rec.get("case_year"),
        "full_case_ref": rec.get("full_case_ref"),
        "judges": rec.get("judges"),
        "judgment_date": rec.get("judgment_date"),
        "disposition": rec.get("disposition"),
        "district": rec.get("district"),
        "source_type": "precedent",
    }

    # NB: unlike the acts pipeline we do NOT store the whole judgment per chunk.
    # A judgment is long (10KB-500KB); duplicating it across ~8 chunks blows
    # Qdrant's 32MB request limit and bloats storage ~10x. The retrievable unit
    # is the chunk + metadata; neighbouring context is fetched on demand by
    # (case_uid, chunk_part), and data/cases_json/<case_id>.json keeps the full
    # text for parent-document reconstruction when needed.
    records = []
    for i, chunk_text in enumerate(parts, start=1):
        cont = f" (part {i}/{n_parts})" if n_parts > 1 else ""
        records.append({
            "embedding_text": f"{header}{cont}\n{chunk_text}",
            "payload": {
                **base_payload,
                "chunk_part": i,
                "n_parts": n_parts,
                "chunk_text": chunk_text,
            },
        })
    return records


def collect_case_records(cases_json_dir: Path, model, e5: bool,
                         limit: int | None = None,
                         only_ids: set[str] | None = None) -> tuple[list[dict], dict]:
    """Walk cases_json_dir, chunk every case record, return (records, stats).

    ``only_ids`` restricts to the given case_ids (stems) -- used by incremental
    append ingests (e.g. the OCR'd hard tail) so an update touches only its cases.
    """
    records = []
    stats = {"files_seen": 0, "cases_empty": 0, "chunks_created": 0}
    files = sorted(p for p in Path(cases_json_dir).glob("*.json")
                   if not p.name.startswith("_"))
    if only_ids is not None:
        files = [p for p in files if p.stem in only_ids]
    if limit:
        files = files[:limit]

    for path in files:
        stats["files_seen"] += 1
        rec = json.loads(path.read_text(encoding="utf-8"))
        recs = case_records(rec, model, e5)
        if not recs:
            stats["cases_empty"] += 1
            continue
        records.extend(recs)

    stats["chunks_created"] = len(records)
    return records, stats
