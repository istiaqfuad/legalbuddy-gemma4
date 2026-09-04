# LegalBuddy — 10-Minute Demo Video Script

Script for the final-round demo video (guideline: max 10 minutes, English or
English-subtitled; scored on functionality demonstration 15 pts + presentation
quality 15 pts).

Every interaction below was verified against the live deployment at
**lb.irflab.tech** on 2026-09-04 — the timings, source hits, and behaviors
described are what the site actually does, not expectations.

**Recording setup (do this first):**
- 1920×1080, browser at ~110% zoom, clean profile (no bookmarks bar)
- Pre-type the queries in a scratch file; paste cleanly, never retype on camera
- Record the system audio-free; narration is added after, or read live if you
  prefer one take
- Close the settings drawer; keep the window at a fixed size for all segments
- Pre-typing tip: keep all five queries in a scratch file and paste — never
  retype on camera

For the live presentation (not the video), use
[`PRESENTATION_SCRIPT.md`](PRESENTATION_SCRIPT.md) — same material, arranged
for speaking with Q&A prep.

---

## Segment 1 — The problem (0:00 – 0:45)

**Screen:** the lb.irflab.tech landing page, slowly scrolled. Then cut to a
plain text card if you want the numbers on screen:

```
1,000 acts · 35,312 sections · written in English legalese
170 million people governed by them
BDT 2,000–10,000 per lawyer consultation
```

**Narration:**
> This is Bangladesh's law: a thousand acts, thirty-five thousand sections,
> written in English legalese, on portals you can only search if you already
> know which act you need. A lawyer costs two to ten thousand taka a session.
> Most people never find out what the law actually says about their situation.
> LegalBuddy is built to change that.

## Segment 2 — Ask in your own language (0:45 – 2:30)

**Screen:** the landing page. Click the first example.

**Query 1 (English):** `What is the punishment for theft under the Penal Code?`

Let the answer stream completely. Verified behavior: retrieval returns Penal
Code §379 as the top source (score 0.87) and the answer cites it inline as
[Source N].

**Narration:**
> The answer streams in plain language — and every claim carries a citation.
> Theft is Section 379: up to three years, or a fine, or both. That's not the
> model remembering law — that text was retrieved for this question.

**On camera:** click one of the small [N] citation marks. The source panel
opens on that exact statute, with the full section text.

> Clicking a citation opens the actual section — and the link to the official
> bdlaws.minlaw.gov.bd page, so anyone can verify the answer against the law
> itself.

**Query 2 (Bengali):** `আমার প্রতিবেশী আমাকে হুমকি দিচ্ছে। আইনে এর বিধান কী?`

**Narration:**
> Now in Bengali — a person describing a real problem: my neighbor is
> threatening me. The statute corpus is in English. The retrieval is
> cross-lingual, so the Bengali question still finds the right sections —
> criminal intimidation, Sections 503 and 506 of the Penal Code.

**Query 3 (Banglish):** `amar barite dhuke churi korar chesta korlo, ki korte pari?`

**Narration:**
> And Banglish — Bengali written in English letters, the way people actually
> type. Same grounded answer, and notice it may end with a short follow-up
> question — that's trained behavior: when a fact would change the answer,
> it asks.

## Segment 3 — The guardrails (2:30 – 3:30)

**Query 4 (off-topic):** `What is the best pizza recipe?`

Verified behavior: the top retrieval score falls below the 0.79 floor, so the
question never reaches the model — the reply is a clarifying question asking
for a legal topic.

**Narration:**
> What about questions that have nothing to do with law? The retrieval
> confidence here is below the hard floor — so the model is never called.
> There's nothing to ground an answer in, and LegalBuddy doesn't invent one.
> It asks what legal topic you meant.

**Query 5 (follow-up memory):** in the same conversation as Query 1, type
`শাস্তি কত বছর?`

Verified behavior: the follow-up is rewritten into a standalone query using
the conversation, then answered with citations.

**Narration:**
> And it remembers the conversation. After the theft question, "শাস্তি কত
> বছর?" — how many years is the punishment — is understood in context,
> rewritten into a standalone query behind the scenes, and answered with
> citations again.

**On camera:** point at the disclaimer under the composer.

> One thing this is not: legal advice. That line is enforced in the system
> prompt, in the training data, and in the interface.

## Segment 4 — The fine-tuned model (3:30 – 5:15)

This is the segment that shows the model itself. See "Showing the model
without hosting it" below — record whichever option you choose, and use this
narration.

**Narration (over the recorded material):**
> Everything so far ran on the deployed endpoint. The system's actual answer
> model is a Gemma 4 31B that we fine-tuned ourselves — 16-bit LoRA, a
> thousand examples, run start to finish on a single GPU.
>
> The backend speaks the standard OpenAI-compatible protocol, so the
> application doesn't care where the model runs. Here is the same code,
> with one environment variable changed, answering from the fine-tuned
> model served locally.

**On camera (recorded segment):** show, in order —
1. The `.env` diff: `GROQ_BASE_URL` pointing at the local llama.cpp server,
   `GROQ_MODEL=lawbuddy-gemma4` — one variable, no code change.
2. `docker compose -f docker-compose.full.yml up` — services coming healthy.
3. The same theft question asked again, answered by the local Gemma 4 with
   the same citation behavior.

> Same application, same grounding, same citations — the only difference is
> where the model is served. For an NGO deployment, that's the difference
> between a monthly API bill and an eighteen-gigabyte file on their own
> machine.

## Segment 5 — Under the hood (5:15 – 7:00)

**Screen (B-roll, no live typing):**
- `training/train_gemma4.log` tail: "Final train loss: 0.6546 / Best eval
  loss: 0.3869 / Duration: 40.4 minutes"
- The loss-curve figure (slide 8 of the deck) or the raw log scrolling
- `eval/REPORT.md` results table

**Narration:**
> The fine-tune took forty minutes on one 96-gigabyte GPU. Training loss went
> from seven point three to zero point six five; the held-out evaluation loss
> fell at every checkpoint — no overfitting. We ran the identical setup twice,
> once on a 27B model and once on Gemma 4: evaluation loss zero point five one
> versus zero point three nine, so the gain is the base model.
>
> The fine-tuning data teaches behavior, not law: when to cite, when to ask a
> clarifying question, and when to decline. Every citation in every training
> answer was machine-checked against the sources in that example — sixteen
> rows failed and were dropped.
>
> Retrieval was chosen the same way — by measurement. A hundred and twenty
> Bengali questions with known answers, four configurations: recall at ten
> went from fifty-eight percent to eighty-eight. The original embedding model
> was silently truncating its input at a hundred and twenty-eight tokens —
> most of every statute chunk was never being read at all.

## Segment 6 — Architecture and deployment (7:00 – 8:15)

**Screen:** the architecture figure (slide 4 of the deck), then a terminal
showing `docker compose ps` output if you have it recorded.

**Narration:**
> The whole system is four services — web, API, vector store, model server —
> started with one command. The corpus pipeline ingested a thousand acts and
> two thousand court judgments into thirty-five thousand indexed sections.
>
> The model serves at three precisions: sixty-two gigabytes at full precision,
> thirty-two at eight-bit, eighteen at four-bit — that's the tier that runs on
> a consumer GPU or a high-RAM CPU. Monthly cost on the CPU path is thirteen
> to twenty-six thousand taka.

## Segment 7 — Impact and close (8:15 – 10:00)

**Screen:** back to the live site, one final question of your choice
streaming.

**Narration:**
> LegalBuddy is a first-contact tool: it tells people what the law says
> before they need a lawyer, in the language they think in, with citations
> they can check. For legal-aid organizations it's a statutory reference that
> takes seconds instead of minutes; for law students, a way to see how
> provisions connect.
>
> The plan is free access first, pilots through legal-aid organizations, then
> on-premise packages for NGOs — their machines, their data.
>
> Any Bangladeshi can ask the law a question — and check the answer against
> the law itself.
>
> This is live at lb.irflab.tech, and everything — code, evaluation harness,
> training logs — is at github.com/istiaqfuad/legalbuddy-gemma4. Thank you.

---

## Showing the model without hosting it

The deployed site uses a cloud endpoint; the fine-tuned model currently has
no permanent host. Options, ranked:

**Option A — one free Molab session (recommended).** Start a GPU session on
Molab (the same free platform the training ran on). The merged checkpoint is
safe on Google Drive. On the session: download it, build llama.cpp, convert
and quantize to Q4_K_M (~18 GB), run `llama-server`, and point a local copy
of the app at it — the `.env` diff shown in Segment 4. Record the segment,
then let the session expire. Cost: zero. Bonus: the Q4 GGUF becomes a
permanent artifact on Drive, ready for any future real deployment. Roughly
two hours of session time; the conversion is scriptable and I can run the
whole thing when you open the session.

**Option B — one rented GPU hour (~$1).** Any RTX 3090/4090-class rental
(RunPod, Vast) fits the 18 GB Q4 model. Download the GGUF produced in
Option A, serve, record, destroy. Only worth it if Molab is unavailable.

**Option C — the CPU path.** The 4-bit model runs on any 32 GB+ RAM machine
— the whitepaper's low-cost tier. Generation is visibly slower (seconds per
sentence), but that honestly demonstrates the NGO-deployable cost story. If
you record this, keep the on-camera typing minimal and let the narration
carry the segment.

**Option D — artifacts only (fallback).** If no serve is possible at
recording time: show the training log tail, the loss curves, the LangSmith
trace of a real request, and the merged checkpoint on Drive, and say the
model is deployment-ready with the committed pipeline. Weaker than a live
answer, but honest and complete — and Segment 4's narration still works with
"here is the configuration and the artifacts it produced."

**Do not** claim the live site is running the fine-tuned model if it isn't —
juries ask exactly this question, and the env-switch demo (Option A/B/C) is a
stronger answer than a claim.

---

## Demo query cheat-sheet

| # | query | demonstrates |
|---|---|---|
| 1 | What is the punishment for theft under the Penal Code? | grounding + expandable citation cards |
| 2 | আমার প্রতিবেশী আমাকে হুমকি দিচ্ছে। আইনে এর বিধান কী? | Bengali → English corpus cross-lingual retrieval |
| 3 | amar barite dhuke churi korar chesta korlo, ki korte pari? | Banglish situational + clarification behavior |
| 4 | What is the best pizza recipe? | hard confidence floor — no hallucinated legal answer |
| 5 | শাস্তি কত বছর? (as a follow-up in the Query 1 conversation) | conversation memory / query rewriting |
