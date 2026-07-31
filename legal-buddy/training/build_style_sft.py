"""Build a grounded Law Buddy SFT dataset from parsed cases and acts.

This script is intended to run in Molab from /marimo/training. It creates
RAG-style chat examples for fine-tuning: the user prompt contains citable statute
sources and optional precedent background, and the assistant answer cites only
the supplied statute sources.
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BASE = Path("/marimo/training")
ACTS_DIR = BASE / "acts"
CASES_PATH = BASE / "parsed_cases.jsonl"
OUT_DIR = BASE / "style_sft_prod"
SEED = 20260730
TRAIN_TARGET = 1000
EVAL_TARGET = 50
SYSTEM_PROMPT = (
    "You are Law Buddy, a Bangladesh legal assistant. Answer plainly and "
    "directly. Use only the provided statute sources for legal claims and cite "
    "them exactly as [Source N]. Case-law background is for reasoning only; do "
    "not cite or name cases unless they are listed as citable sources. If the "
    "sources are insufficient or the question is too vague, ask a short "
    "clarifying question or say what source is missing."
)


SECTION_REF_RE = re.compile(
    r"(?i)\b(?:under|u/s|section|sections|sec\.?)\s+"
    r"([0-9A-Za-z][0-9A-Za-z/(),.\-\s]{0,50})\s+of\s+the\s+"
    r"([A-Z][A-Za-z .,'()\-]{2,90}?(?:Act|Code|Order|Ordinance|Rules|Regulation|Constitution)(?:,\s*\d{4})?)"
)
SECTION_START_RE = re.compile(r"^\s*([0-9A-Za-z]+(?:\([^)]+\))?)\s*[.।)]")
BAD_TEXT_RE = re.compile(r"[¡¢¤£¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿]")
QUESTION_RE = re.compile(r"\?")


@dataclass(frozen=True)
class Source:
    act_title: str
    act_year: str
    section: str
    text: str
    url: str


def clean_text(text: str, limit: int | None = None) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([a-z])-\s+([a-z])", r"\1\2", text)
    if limit and len(text) > limit:
        return text[:limit].rsplit(" ", 1)[0] + "..."
    return text


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def display_title(title: str) -> str:
    title = re.sub(r"^\d+\s*", "", title).strip()
    title = re.sub(r"^\d+\[", "[", title).strip()
    return title


GENERIC_ACT_PHRASES = {
    "act", "the act", "said act", "this act", "code", "the code", "said code",
    "ordinance", "the ordinance", "rules", "the rules", "law", "the law",
}


def distinctive_tokens(s: str) -> set[str]:
    stop = {
        "the", "act", "code", "ordinance", "order", "rules", "regulation",
        "of", "and", "bangladesh", "no", "section", "sections", "said", "this",
    }
    return {t for t in norm(s).split() if t not in stop and not t.isdigit()}


def section_no(section_content: str, fallback: int) -> str:
    m = SECTION_START_RE.search(section_content or "")
    return m.group(1) if m else str(fallback)


def load_acts() -> tuple[list[Source], dict[str, list[Source]]]:
    all_sources: list[Source] = []
    by_act_tokens: dict[str, list[Source]] = defaultdict(list)
    for path in sorted(ACTS_DIR.glob("*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        title = display_title(str(obj.get("act_title") or obj.get("csv_metadata", {}).get("act_title_from_csv") or ""))
        year = str(obj.get("act_year") or obj.get("csv_metadata", {}).get("act_year_from_csv") or "")
        url = str(obj.get("source_url") or "")
        if not title or obj.get("csv_metadata", {}).get("is_repealed") is True:
            continue
        for idx, sec in enumerate(obj.get("sections") or [], 1):
            content = clean_text(str(sec.get("section_content") or ""), 1200)
            if len(content) < 40 or "repealed" in content.lower()[:80]:
                continue
            source = Source(title, year, section_no(content, idx), content, url)
            all_sources.append(source)
            title_norm = norm(title)
            by_act_tokens[title_norm].append(source)
            short = re.sub(r"\b(the|act|code|ordinance|order|rules|regulation|of|bangladesh)\b", " ", title_norm)
            short = re.sub(r"\s+", " ", short).strip()
            if short:
                by_act_tokens[short].append(source)
    return all_sources, by_act_tokens


def parse_section_tokens(raw: str) -> list[str]:
    out: list[str] = []
    for part in re.split(r"[/,;&]| and | or ", raw, flags=re.I):
        m = re.search(r"\d+[A-Za-z]?(?:\([^)]+\))?", part)
        if m:
            out.append(m.group(0))
    return out[:4]


def find_sources_for_case(text: str, by_act: dict[str, list[Source]]) -> list[Source]:
    found: list[Source] = []
    seen = set()
    for m in SECTION_REF_RE.finditer(text[:25000]):
        sections = parse_section_tokens(m.group(1))
        act_phrase = norm(m.group(2))
        phrase_tokens = distinctive_tokens(act_phrase)
        if act_phrase in GENERIC_ACT_PHRASES or len(phrase_tokens) < 1:
            continue
        candidates = []
        for key, vals in by_act.items():
            key_tokens = distinctive_tokens(key)
            if not key_tokens:
                continue
            overlap = phrase_tokens & key_tokens
            if (
                (len(phrase_tokens) == 1 and phrase_tokens <= key_tokens)
                or (len(overlap) >= min(2, len(phrase_tokens)) and overlap == phrase_tokens)
                or (act_phrase in key and len(phrase_tokens) >= 2)
                or (key in act_phrase and len(key_tokens) >= 2)
            ):
                candidates.extend(vals)
        if not candidates:
            continue
        for sec in sections:
            for source in candidates:
                if source.section.lower() == sec.lower() or source.text.lower().startswith(sec.lower()):
                    k = (source.act_title, source.section)
                    if k not in seen:
                        seen.add(k)
                        found.append(source)
                    break
        if len(found) >= 3:
            break
    return found[:3]


def case_ref(filename: str, description: str) -> str:
    stem = filename[:-4] if filename.lower().endswith(".pdf") else filename
    label = description or stem
    label = re.sub(r"[_]+", " ", label)
    label = re.sub(r"(?i)\bno\.?", " No. ", label)
    label = re.sub(r"\s+", " ", label).strip(" .-")
    return label[:120] or stem


def is_bad_case(text: str) -> bool:
    if len(text) < 1500:
        return True
    sample = text[:8000]
    bad = len(BAD_TEXT_RE.findall(sample))
    letters = max(1, sum(ch.isalpha() for ch in sample))
    return bad / letters > 0.015


def extract_background(text: str) -> tuple[str, str, str]:
    cleaned = clean_text(text)
    paras = [clean_text(p) for p in re.split(r"\n\s*\n+", text) if len(clean_text(p)) > 80]
    facts = next((p for p in paras if re.search(r"(?i)\bfacts?|prosecution case|plaintiff|petitioner|complainant", p)), paras[0] if paras else cleaned[:500])
    reasoning = next((p for p in reversed(paras) if re.search(r"(?i)\bfind|appears|considered|held|therefore|accordingly|result", p)), paras[-1] if paras else cleaned[-500:])
    tail = cleaned[-1800:].lower()
    disposition = "disposed of"
    for cue, label in [
        ("rule is made absolute", "rule made absolute"),
        ("rule made absolute", "rule made absolute"),
        ("appeal is dismissed", "appeal dismissed"),
        ("rule is discharged", "rule discharged"),
        ("rule discharged", "rule discharged"),
        ("appeal is allowed", "appeal allowed"),
        ("proceeding is quashed", "proceeding quashed"),
        ("conviction and sentence", "conviction/sentence considered"),
    ]:
        if cue in tail:
            disposition = label
            break
    return clean_text(facts, 420), clean_text(reasoning, 420), disposition


def source_block(sources: list[Source]) -> str:
    blocks = []
    for i, src in enumerate(sources, 1):
        blocks.append(
            "\n".join([
                f"[Source {i}]",
                f"Act: {src.act_title}",
                f"Year: {src.act_year or 'Unknown'}",
                f"Section: {src.section}",
                f"Text: {src.text}",
                f"URL: {src.url or 'N/A'}",
            ])
        )
    return "\n\n".join(blocks)


def answer_for(question_kind: str, sources: list[Source], bg: tuple[str, str, str]) -> str:
    facts, reasoning, disposition = bg
    first = sources[0]
    cite_all = " ".join(f"[Source {i}]" for i in range(1, len(sources) + 1))
    if question_kind == "normal":
        return (
            f"The main rule comes from {first.act_title}, section {first.section}. "
            f"On the supplied facts, the answer should be framed through that rule "
            f"and any connected sections provided in the sources {cite_all}.\n\n"
            f"The precedent background points to this practical treatment: {reasoning} "
            f"So the safer answer is to apply the statutory elements first, then use "
            f"the court background only as supporting reasoning, not as a separate citation."
        )
    if question_kind == "situation":
        return (
            f"Yes, this situation may fall within the supplied legal rule if the facts "
            f"match the statutory elements. Section {first.section} of {first.act_title} "
            f"is the starting point {cite_all}.\n\n"
            f"The case background suggests the court focused on facts like these: {facts} "
            f"The important point is whether those facts prove the ingredient stated in "
            f"the source. What exact act or event happened in your situation?"
        )
    if question_kind == "banglish":
        return (
            f"Ei facts-e prothome {first.act_title} er section {first.section} dekhte "
            f"hobe {cite_all}. Source-e je rule deya ache, answer ta oi rule-er upor "
            f"grounded hobe, case background shudhu reasoning help kore.\n\n"
            f"Short answer: facts jodi source-er legal ingredients meet kore, tahole "
            f"eta oi section-er moddhe porte pare. Specific result depend korbe proof "
            f"and exact facts-er upor."
        )
    raise AssertionError(question_kind)


def make_example(kind: str, rec: dict, sources: list[Source], idx: int) -> dict:
    ref = case_ref(str(rec.get("filename") or ""), str(rec.get("case_description") or ""))
    bg = extract_background(str(rec.get("text") or ""))
    facts, reasoning, disposition = bg
    if kind == "normal":
        question = f"What legal rule applies in a case like {ref}?"
    elif kind == "situation":
        question = f"I have a situation similar to this: {facts} What does Bangladesh law say?"
    elif kind == "banglish":
        question = f"Amar case ta erokom: {facts[:220]} Bangladesh law e eta ki?"
    else:
        raise AssertionError(kind)
    user = (
        f"Question / situation: {question}\n\n"
        f"Statute sources (citable):\n{source_block(sources)}\n\n"
        "Precedent background (reasoning only, not citable):\n"
        f"Case: {ref}\n"
        f"Disposition: {disposition}\n"
        f"Facts/background: {facts}\n"
        f"Court reasoning/outcome excerpt: {reasoning}\n\n"
        "Answer directly. Cite only the statute sources as [Source N]."
    )
    return {
        "id": f"{kind}_{idx:04d}",
        "kind": kind,
        "source_case": rec.get("filename"),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer_for(kind, sources, bg)},
        ],
    }


def make_clarify(idx: int) -> dict:
    prompts = [
        "Is it legal?", "Can I sue?", "What will happen now?",
        "Is this allowed in Bangladesh?", "Can they do this to me?",
        "What remedy do I have?", "Can the police take action?",
        "Can I stop this?", "What court should I go to?",
        "Can I challenge this decision?", "Do I need a lawyer?",
        "Can I recover my money?", "What evidence will I need?",
        "Is there a deadline to complain?", "Can I make a written complaint?",
        "Can a notice be challenged?", "What should I do first?",
        "Can I get an injunction?", "Is this a criminal matter?",
        "Does this affect my property?", "Can I appeal this order?",
    ]
    q = prompts[idx % len(prompts)]
    responses = [
        "I need a little more detail before I can answer. What happened, and which area of Bangladesh law do you want to understand?",
        "Please describe the key facts, the people involved, and what outcome you want. I also need to know whether this concerns property, family, crime, work, or a contract.",
        "I cannot tell from that alone. Please explain what was done, when it happened, where it happened, and whether you have any notice, agreement, or court paper.",
        "To give a useful Bangladesh-law answer, please provide the event, the other party's role, any documents, and what you want to challenge or obtain.",
    ]
    return {
        "id": f"clarify_{idx:04d}",
        "kind": "clarify",
        "source_case": None,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question / situation: {q}\n\nStatute sources (citable):\n(none)\n\nAnswer directly."},
            {"role": "assistant", "content": responses[idx % len(responses)]},
        ],
    }


def make_abstain(source: Source, idx: int) -> dict:
    q = [
        "What is the immigration rule for moving to Canada?",
        "How do I file taxes in California?",
        "What does Indian law say about this employment contract?",
        "Can you predict the result of a private business dispute with no facts?",
        "What are the rules for a divorce in England?",
        "Can you draft a United States patent application?",
        "Which Australian visa should I apply for?",
        "What is the law of a French commercial lease?",
        "How is a Singapore company incorporated?",
        "Can I appeal a decision of the European Court of Human Rights?",
        "What are the inheritance rules in Japan?",
        "How do I contest a traffic ticket in New York?",
        "What is the VAT rate in Germany?",
        "Can you decide who is right without any facts or documents?",
        "What does Sri Lankan law say about this land dispute?",
    ][idx % 15]
    user = (
        f"Question / situation: {q}\n\n"
        f"Statute sources (citable):\n{source_block([source])}\n\n"
        "Answer directly. Cite only the statute sources as [Source N]."
    )
    return {
        "id": f"abstain_{idx:04d}",
        "kind": "abstain",
        "source_case": None,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": "The supplied Bangladesh statute source does not answer that question. Please provide the relevant Bangladesh legal issue or a matching source before I give a legal answer."},
        ],
    }


def validate_example(ex: dict) -> tuple[bool, str]:
    messages = ex.get("messages") or []
    if len(messages) != 3:
        return False, "bad_message_count"
    user = messages[1]["content"]
    assistant = messages[2]["content"]
    if BAD_TEXT_RE.search(user) or BAD_TEXT_RE.search(assistant):
        return False, "garbled_text"
    source_ids = {int(x) for x in re.findall(r"\[Source (\d+)\]", user)}
    cited = {int(x) for x in re.findall(r"\[Source (\d+)\]", assistant)}
    if cited - source_ids:
        return False, "invalid_citation"
    if ex["kind"] not in {"clarify", "abstain"} and not cited:
        return False, "missing_citation"
    if "as an ai" in assistant.lower() or "legal advice disclaimer" in assistant.lower():
        return False, "bad_phrase"
    if len(QUESTION_RE.findall(assistant)) > 1:
        return False, "too_many_questions"
    if len(assistant) > 1800:
        return False, "answer_too_long"
    return True, "ok"


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n")


def main() -> None:
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_sources, by_act = load_acts()
    if not all_sources:
        raise RuntimeError(f"No usable act sources found under {ACTS_DIR}")

    cases = []
    with CASES_PATH.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text = str(rec.get("text") or "")
            if not is_bad_case(text):
                srcs = find_sources_for_case(text, by_act)
                if srcs:
                    cases.append((rec, srcs))

    random.shuffle(cases)
    rows: list[dict] = []
    rejected: list[dict] = []
    counts = Counter()
    desired = {"normal": 630, "situation": 170, "banglish": 50}
    cursor = 0
    for kind, need in desired.items():
        made = 0
        while made < need and cursor < len(cases):
            rec, srcs = cases[cursor]
            cursor += 1
            ex = make_example(kind, rec, srcs, len(rows) + 1)
            ok, reason = validate_example(ex)
            if ok:
                rows.append(ex)
                counts[kind] += 1
                made += 1
            else:
                rejected.append({"id": ex.get("id"), "reason": reason, "source_case": ex.get("source_case")})

    for i in range(125):
        ex = make_clarify(i)
        ok, reason = validate_example(ex)
        if ok:
            rows.append(ex)
            counts["clarify"] += 1
        else:
            rejected.append({"id": ex["id"], "reason": reason})

    random.shuffle(all_sources)
    abstain_made = 0
    abstain_cursor = 0
    while abstain_made < 75 and abstain_cursor < len(all_sources):
        ex = make_abstain(all_sources[abstain_cursor], abstain_cursor)
        abstain_cursor += 1
        ok, reason = validate_example(ex)
        if ok:
            rows.append(ex)
            counts["abstain"] += 1
            abstain_made += 1
        else:
            rejected.append({"id": ex["id"], "reason": reason})
    if abstain_made < 75:
        raise RuntimeError("Not enough clean statute sources for abstention examples")

    if len(rows) < TRAIN_TARGET + EVAL_TARGET:
        raise RuntimeError(f"Only built {len(rows)} valid rows; need {TRAIN_TARGET + EVAL_TARGET}")

    random.shuffle(rows)
    eval_rows = rows[:EVAL_TARGET]
    train_rows = rows[EVAL_TARGET:EVAL_TARGET + TRAIN_TARGET]

    write_jsonl(OUT_DIR / "train.jsonl", train_rows)
    write_jsonl(OUT_DIR / "eval.jsonl", eval_rows)
    with (OUT_DIR / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for row in train_rows + eval_rows:
            f.write(json.dumps({k: v for k, v in row.items() if k != "messages"}, ensure_ascii=False) + "\n")
    with (OUT_DIR / "rejected.jsonl").open("w", encoding="utf-8") as f:
        for row in rejected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    qa_rows = []
    for split, split_rows in [("train", train_rows), ("eval", eval_rows)]:
        for row in split_rows:
            ok, reason = validate_example(row)
            qa_rows.append({
                "split": split,
                "id": row["id"],
                "kind": row["kind"],
                "source_case": row.get("source_case") or "",
                "valid": ok,
                "reason": reason,
                "assistant_chars": len(row["messages"][2]["content"]),
                "user_chars": len(row["messages"][1]["content"]),
            })
    with (OUT_DIR / "validation.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(qa_rows[0].keys()))
        writer.writeheader()
        writer.writerows(qa_rows)

    manifest = {
        "seed": SEED,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "valid_rows_total": len(rows),
        "rejected_rows": len(rejected),
        "kind_counts_all_valid": dict(counts),
        "case_candidates_with_sources": len(cases),
        "acts_loaded_sections": len(all_sources),
        "system_prompt": SYSTEM_PROMPT,
        "notes": [
            "Generated from parsed case PDFs plus real act JSON sources.",
            "Rows are RAG-style and citation-validated.",
            "Garbled legacy-font case text is rejected by heuristic.",
            "Append tokenizer.eos_token during training formatting.",
        ],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "preview.json").write_text(json.dumps((train_rows + eval_rows)[:10], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
