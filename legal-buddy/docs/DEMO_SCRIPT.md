# LegalBuddy — Demo Script

For the ≤10-minute demo video and the live presentation (guideline: video 30
pts, live presentation 30 pts). Timings target the video; the live talk
compresses segments 3–5 and leaves room for Q&A.

**Before recording, verify the stack:** `docker compose -f
docker-compose.full.yml ps` — all four services healthy. Have
<http://localhost:8000/docs> open in a second tab (architecture segment) and
this script's queries typed in a scratch file for clean pasting.

---

## 0. One-liner (memorize)

> "LegalBuddy lets any Bangladeshi ask a legal question in their own language
> and get an answer grounded in the actual statute — every claim cites an act
> and section you can open and read."

## 1. The problem — 1.5 min

- Bangladesh: ~1,000 acts, 35,000+ sections, written in English legalese; most
  citizens speak Bengali or Banglish.
- Statutes are scattered across government portals with no plain-language
  search; a lawyer consult costs BDT 2,000–10,000+.
- General AI tools hallucinate section numbers with no citation trail.
- **Say the one-liner.**

## 2. How it works — 2 min

Screen: the architecture diagram in `TECHNICAL_DOCUMENTATION.md` §1 (or the
API docs tab). Walk the request flow in five beats:

1. Question comes in — English, Bengali, or Banglish.
2. Follow-ups are rewritten into standalone queries (conversation memory).
3. The question is embedded and searched against 35,000 indexed statute
   sections in Qdrant; retrieval collapses chunks to **whole sections**.
4. A **locally-served, fine-tuned Gemma model** writes the answer — and it is
   only allowed to cite the sources it was given. No API, no proprietary model.
5. Confidence gating: garbage questions never reach the model; borderline ones
   get flagged so the model asks instead of guessing.

One number worth saying aloud: retrieval Recall@10 **0.575 → 0.883** after the
measured embedding-model swap (eval harness in `eval/REPORT.md`).

## 3. Core demo — citation grounding — 3 min

**Query 1 (English, direct):**
> What is the punishment for theft under the Penal Code?

Expected: answer citing the Penal Code theft section; **click an inline [N]
citation mark** — the source panel opens on that statute with the full section
text and the bdlaws.minlaw.gov.bd link. Say: "The model can't invent law —
everything it cites is right here."

**Query 2 (Bengali, same capability):**
> আমার প্রতিবেশী আমাকে হুমকি দিচ্ছে। আইনে এর বিধান কী?

Expected: retrieves the criminal-intimidation provisions (e.g. Penal Code
§503/§506), answer in Bengali or English with citations. Point out the Bengali
query matched the **English** statute corpus — that's the multilingual
embedding model at work.

**Query 3 (Banglish, situational):**
> amar barite dhuke churi korar chesta korlo, ki korte pari?

Expected: situational answer grounded in the relevant sections, possibly with
one short follow-up question (the fine-tuned clarification behavior).

## 4. Robustness — the guardrails — 2 min

**Query 4 (off-topic — hard floor):**
> What is the best pizza recipe?

Expected: no LLM call, no fake citations — the system asks the user to ask a
legal question. Say: "Below the confidence floor, retrieval itself refuses —
there's nothing to ground an answer in."

**Query 5 (follow-up memory):** continue the theft conversation with
> শাস্তি কত বছর?

Expected: the follow-up is rewritten into a standalone query using the
conversation, then answered with citations.

**Point at the disclaimer** under the composer: "General information, not
legal advice." — information side of the line, by design (system prompt +
training data + UI).

## 5. Under the hood — 1 min

Screen: `training/train_prod.log` tail + `style_sft_prod/manifest.json`.

- 1,000-example fine-tuning set: 60% grounded Q&A, 16% situational, 5%
  Banglish, 12% clarification turns, 7% abstention — the model is *trained* to
  ask and to decline.
- Every training citation was machine-validated against the example's sources.
- Training loss curves are committed (`train_prod.log.json`).

## 6. Impact + close — 0.5 min

- First-contact legal information for 170M people, at NGO-deployable cost
  (4-bit model runs on an 18 GB GPU or high-RAM CPU).
- Everything is open and self-hostable: one `docker compose up` brings the
  whole stack.
- **Close with the one-liner again.**

---

## Live-presentation notes

- Required by guideline: 1-minute member introduction with responsibilities.
  Single-member team — keep it to ~30 s: problem discovery, corpus +
  ingestion pipeline, embedding/retrieval evals, fine-tuning, full-stack app.
- Expect jury questions on: hallucination mitigation (answer: retrieval-only
  grounding + citation validation + abstention training), corpus freshness
  (re-ingest pipeline is in the repo), and why not an API model (cost,
  sovereignty, offline deployability — whitepaper §1.3).
- Fallback if the demo environment fails: the demo queries above are also in
  the UI's empty-state examples; a recorded clip of segment 3 covers the core.

## Demo query cheat-sheet

| # | query | demonstrates |
|---|---|---|
| 1 | What is the punishment for theft under the Penal Code? | grounding + expandable citation cards |
| 2 | আমার প্রতিবেশী আমাকে হুমকি দিচ্ছে। আইনে এর বিধান কী? | Bengali → English corpus cross-lingual retrieval |
| 3 | amar barite dhuke churi korar chesta korlo, ki korte pari? | Banglish situational + clarification behavior |
| 4 | What is the best pizza recipe? | hard confidence floor — no hallucinated legal answer |
| 5 | শাস্তি কত বছর? (as a follow-up) | conversation memory / query rewriting |
