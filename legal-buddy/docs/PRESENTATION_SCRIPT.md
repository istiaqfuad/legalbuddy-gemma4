# LegalBuddy — Presentation Script

Spoken narration for the 12-slide deck, slide by slide. Written to be said
out loud, not read silently — short sentences, one breath per line. Total
speaking time without the live demo: about 7 minutes; with the demo, 9 to 10.
Timings are per slide and assume a relaxed pace.

Paired with: [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) (the query-level demo
walkthrough) and `LegalBuddy_Presentation.pptx` (in `docs/submission_materials/`).

---

## Slide 1 — Title (30 s)

> Good morning. I'm here with LegalBuddy — a legal AI assistant for
> Bangladesh, built on a Gemma 4 model we fine-tuned ourselves.
>
> One sentence before anything else: it lets anyone ask a legal question in
> Bengali, Banglish, or English — and every claim in the answer points to the
> actual act and section it came from.

*(If the jury expects the 1-minute member introduction here instead of slide
11, jump to the intro block below and come back.)*

## Slide 2 — The problem (45 s)

> Bangladesh has about a thousand acts and thirty-five thousand sections of
> law. They're written in English legalese. Most of the hundred and seventy
> million people they govern speak Bengali.
>
> The laws sit on a government portal you can only search if you already know
> which act you're looking for. And asking a lawyer costs two to ten thousand
> taka per session — which rules out most of the country.
>
> So people face police stations, landlords, and courts without knowing what
> the law actually says about their situation. This hits rural communities and
> women the hardest.

## Slide 3 — The solution (45 s)

> LegalBuddy is a first-contact tool. You ask in your own language. You get a
> plain-language answer — and every legal claim in it carries a citation.
> Click the citation, and the full text of that section opens, with the link
> to the official portal.
>
> Three things make it work. The retrieval is cross-lingual — a Bengali
> question finds the right English statute. The model only explains text that
> was retrieved for that question — it doesn't answer from memory. And it runs
> entirely on open weights, so an NGO can deploy it on one machine with no API
> bill.

## Slide 4 — Architecture (40 s)

> The system is four services, started with one docker compose command.
>
> The web chat talks to the API. The API owns the pipeline: it embeds the
> question, searches Qdrant — which holds all thirty-five thousand sections —
> and sends the retrieved sections with the question to the Gemma 4 model we
> fine-tuned, served by llama.cpp.
>
> On the left is the corpus pipeline that built the index: a thousand acts
> plus two thousand court judgments, chunked and embedded.
>
> One detail worth noting: the API only reports healthy after the embedding
> model is warm and the statute collection is verified. The stack never serves
> traffic half-booted.

## Slide 5 — How grounding works (60 s)

> One question moves through five steps.
>
> If it's a follow-up — like "how many years is the punishment?" — it's first
> rewritten into a standalone query using the conversation.
>
> Then it's embedded and searched. The hits are collapsed to whole sections,
> never fragments, so the model always sees the section in context.
>
> Then the confidence gate. If the best match scores below zero point seven
> nine, the question never reaches the model at all — the system just asks
> for specifics. Between the two thresholds, the model gets the sources plus a
> warning, and it decides whether to answer or ask.
>
> The prompt labels each source. The model must cite only what it was given —
> and citation numbers are validated before anything reaches the user.

## Slide 6 — Retrieval (40 s)

> We didn't guess the embedding model — we measured it. A hundred and twenty
> Bengali questions with known correct sections, four configurations.
>
> The baseline got twenty-six percent on Recall at one. The configuration we
> ship gets fifty-seven — and eighty-eight on Recall at ten. Most of that gain
> came from one finding: the original model was silently truncating its input
> at a hundred and twenty-eight tokens, so most of every statute chunk was
> never even embedded.
>
> All of this is in the repository — the gold set, the harness, and the
> per-variant numbers.

## Slide 7 — Fine-tuning (40 s)

> The fine-tune teaches behavior, not law. A thousand examples across five
> situations: direct questions with citations, personal situations, Banglish,
> turns where the right move is to ask a clarifying question, and turns where
> the right move is to decline.
>
> Every citation in every training answer was machine-checked against the
> sources in that example's own prompt. Sixteen rows failed the check and were
> logged and dropped.

## Slide 8 — Training results (40 s)

> We ran the identical fine-tune twice — same data, same settings — once on a
> 27B model as a development run, once on Gemma 4 31B as the production run.
>
> The training loss went from seven point three down to zero point six five,
> and the held-out evaluation loss fell at every single checkpoint — no
> overfitting. Final evaluation loss: zero point three nine on Gemma 4, versus
> zero point five one on the 27B. Since only the base model differs, that
> improvement comes from the stronger base.
>
> The whole run takes forty minutes on one GPU, and the loss curves you're
> looking at are the actual committed logs.

## Slide 9 — Serving (35 s)

> The merged model serves at three precisions: full precision at around
> sixty-two gigabytes for institutions, eight-bit at thirty-two, and four-bit
> at eighteen — that's the tier the demo runs on, and it fits a consumer GPU
> or a high-RAM CPU box.
>
> The backend speaks the OpenAI-compatible protocol, so the local model,
> Groq, and vLLM are interchangeable through environment variables alone.
> Structured output is enforced — answer, citations, limitations — and
> invalid citation numbers are dropped before display.

## Slide 10 — Impact and roadmap (35 s)

> The rollout plan has three phases. First, a free web app with pilots
> through legal-aid organizations and university law clinics. Second,
> on-premise packages for NGOs and law faculties — their hardware, their data.
> Third, a mobile app, messaging integrations, and an API.
>
> On cost: the CPU path runs at roughly thirteen to twenty-six thousand taka
> a month. Break-even is around two to three institutional licenses.

## Slide 11 — Scope of work (30–60 s)

*(The guideline asks for a one-minute introduction of each member and their
responsibilities. Use this block — here, or on the title slide.)*

> I'm Istiaqur Rahman Fuad, from the Department of Computer Science and
> Engineering at the University of Rajshahi, and this is a one-person project.
>
> I built the corpus pipeline — parsing the acts and the judgment PDFs. I
> built the retrieval evaluation — the gold set and the harness that chose the
> embedding model. I constructed and validated the fine-tuning dataset and ran
> both training runs. And I built the application — the API, the chat
> interface, and the deployment.

## Slide 12 — Closing + live demo (2–3 min)

> The one-liner, once more: any Bangladeshi can ask the law a question — and
> check the answer against the law itself.
>
> Let me show it live.

*(Switch to <https://lb.irflab.tech> and run the five queries from
`DEMO_SCRIPT.md` §3–4: English theft question → click a citation; the Bengali
threat question; the Banglish situation; the pizza-recipe question that hits
the confidence floor; and the Bengali follow-up that shows conversation
memory. Say the one-liner again at the end.)*

> Code, technical documentation, and every evaluation artifact are public at
> github.com/istiaqfuad/legalbuddy-gemma4.
>
> Thank you — I'm happy to take questions.

---

## Q&A preparation

Likely jury questions and the short answers:

- **How do you stop hallucinated citations?** The model can only cite the
  sources in its prompt; numbers are validated against that list at request
  time; and 19% of the fine-tuning data trains clarify/decline behavior.
- **What if the law changes?** The corpus is a snapshot; re-running the
  ingestion pipeline rebuilds the index. The UI states the limitation.
- **Why not just use an API model?** Cost per token at NGO scale, data
  sovereignty for on-premise deployments, and no dependency on a foreign
  provider's availability or pricing.
- **Is it legal advice?** No — information, not advice, enforced in the
  system prompt, the training data, and the UI disclaimer.
- **What's the weakest part?** The confidence thresholds are a calibration,
  not a classifier — a cross-encoder reranker is the identified next step.
  End-to-end answer quality on labeled conversations isn't measured yet.
