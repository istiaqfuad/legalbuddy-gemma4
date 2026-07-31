"""Generate a chunking-independent gold set with Gemini.

Samples sections from the subset, asks Gemini to write ONE natural legal question
whose answer lives in that section, and records gold = (act_file, section_ord).
Writes eval/goldset.json: [{question, act_file, section_ord, gold_key, act_title,
section_index, section_preview}].
"""
import json
import random
import re
import time

from common import ACTS_DIR, GEMINI_API_KEY, GOLDSET_PATH, SUBSET_PATH, gold_key

TARGET = 60
SAMPLE = 90  # oversample; some generations fail/are filtered
MODEL = "gemini-2.5-flash-lite"  # separate free-tier daily bucket from gemini-2.5-flash
SEED = 42

FOOTNOTE_RE = re.compile(r"\d+\[(.*?)\]")
VOID_RE = re.compile(r"\[\s*(Omitted|Repealed?|Rep\.)\s+by|\[\s*Repeal\.\-|\[\s*Omit\.\-", re.I)

PROMPT = """You are building a retrieval benchmark for Bangladesh statutory law.
Below is one section from the act titled "{title}".

Write exactly ONE natural-language question that a lawyer or citizen might ask,
whose answer is contained in this section. Rules:
- Do NOT mention the section number or the word "section".
- Do NOT quote the text verbatim; phrase it as a real question.
- Make it specific enough that this section is clearly the right answer.
- Output ONLY the question, nothing else.

SECTION:
{content}
"""


def clean(t: str) -> str:
    return FOOTNOTE_RE.sub(r"\1", t or "").strip()


def gather_candidates():
    cands = []
    subset = json.loads(SUBSET_PATH.read_text(encoding="utf-8"))
    for stem in subset:
        try:
            obj = json.loads((ACTS_DIR / f"{stem}.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        title = obj.get("act_title", "Unknown Act")
        for ord_, sec in enumerate(obj.get("sections") or []):
            raw = (sec or {}).get("section_content", "") or ""
            if VOID_RE.search(raw):
                continue
            c = clean(raw)
            if not (250 <= len(c) <= 3500):
                continue
            cands.append(
                {
                    "act_file": stem,
                    "section_ord": ord_,
                    "act_title": title,
                    "section_title": (sec or {}).get("section_title"),
                    "content": c,
                }
            )
    return cands


def main():
    if not GEMINI_API_KEY:
        raise SystemExit("GEMINI_API_KEY not set")
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)

    cands = gather_candidates()
    random.Random(SEED).shuffle(cands)
    print(f"candidate sections: {len(cands)}; sampling up to {SAMPLE}")

    gold = []
    for c in cands[:SAMPLE]:
        if len(gold) >= TARGET:
            break
        prompt = PROMPT.format(title=c["act_title"], content=c["content"][:3500])
        q = None
        for attempt in range(3):
            try:
                resp = client.models.generate_content(model=MODEL, contents=prompt)
                q = (resp.text or "").strip().strip('"')
                break
            except Exception as e:
                msg = str(e)
                if "RESOURCE_EXHAUSTED" in msg and attempt < 2:
                    time.sleep(8)
                    continue
                print(f"  gen fail {c['act_file']}#{c['section_ord']}: {msg[:80]}")
                break
        if not q:
            continue
        if not q or len(q) < 10 or "\n" in q.strip() and len(q) > 400:
            continue
        q = q.split("\n")[0].strip()
        gold.append(
            {
                "question": q,
                "act_file": c["act_file"],
                "section_ord": c["section_ord"],
                "gold_key": gold_key(c["act_file"], c["section_ord"]),
                "act_title": c["act_title"],
                "section_title": c["section_title"],
                "section_preview": c["content"][:160],
            }
        )
        if len(gold) % 10 == 0:
            print(f"  generated {len(gold)}/{TARGET}")
        time.sleep(0.5)

    GOLDSET_PATH.write_text(
        json.dumps(gold, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {len(gold)} gold questions -> {GOLDSET_PATH}")


if __name__ == "__main__":
    main()
