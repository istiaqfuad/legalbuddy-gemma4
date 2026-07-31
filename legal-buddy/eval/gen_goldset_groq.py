"""Expand the gold set with Groq (OpenAI-compatible API) — generous free tier, so
this is what lifts n from ~26 to a credible size. Appends to eval/goldset.json,
dedupes by gold_key, skips sections already used.
"""
import json
import random
import time

from common import GOLDSET_PATH, GROQ_API_KEY, gold_key
from gen_goldset import PROMPT, gather_candidates

TARGET_TOTAL = 120
MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
BASE_URL = "https://api.groq.com/openai/v1"


def main():
    if not GROQ_API_KEY:
        raise SystemExit("GROQ_API_KEY not set")
    from openai import OpenAI

    client = OpenAI(api_key=GROQ_API_KEY, base_url=BASE_URL)
    gold = json.loads(GOLDSET_PATH.read_text(encoding="utf-8"))
    used = {g["gold_key"] for g in gold}
    print(f"existing: {len(gold)}; target {TARGET_TOTAL}")

    cands = [
        c for c in gather_candidates()
        if gold_key(c["act_file"], c["section_ord"]) not in used
    ]
    random.Random(13).shuffle(cands)

    model_idx = 0
    for c in cands:
        if len(gold) >= TARGET_TOTAL:
            break
        gk = gold_key(c["act_file"], c["section_ord"])
        if gk in used:
            continue
        prompt = PROMPT.format(title=c["act_title"], content=c["content"][:3500])
        q = None
        for attempt in range(4):
            model = MODELS[min(model_idx, len(MODELS) - 1)]
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=120,
                )
                q = (resp.choices[0].message.content or "").strip().strip('"')
                break
            except Exception as e:
                msg = str(e)
                if "rate_limit" in msg or "429" in msg:
                    time.sleep(3)
                    continue
                if ("model" in msg.lower() and "decommission" in msg.lower()) or "400" in msg:
                    model_idx += 1  # fall back to next model
                    continue
                print(f"  fail {gk}: {msg[:90]}")
                break
        if not q:
            continue
        q = q.split("\n")[0].strip()
        if len(q) < 10:
            continue
        used.add(gk)
        gold.append(
            {
                "question": q,
                "act_file": c["act_file"],
                "section_ord": c["section_ord"],
                "gold_key": gk,
                "act_title": c["act_title"],
                "section_title": c["section_title"],
                "section_preview": c["content"][:160],
                "gen_model": "groq:" + MODELS[min(model_idx, len(MODELS) - 1)],
            }
        )
        if len(gold) % 15 == 0:
            print(f"  total {len(gold)}")
            GOLDSET_PATH.write_text(
                json.dumps(gold, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        time.sleep(0.4)

    GOLDSET_PATH.write_text(
        json.dumps(gold, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {len(gold)} gold questions")


if __name__ == "__main__":
    main()
