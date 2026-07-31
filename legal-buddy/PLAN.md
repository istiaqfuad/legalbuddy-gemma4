# Law Buddy Gemma 4 Style Fine-Tuning Plan

## Summary

Law Buddy should use Gemma 4 as the only LLM while keeping the existing RAG
architecture for legal correctness. Statutes and cases remain retrieved at query
time from Qdrant; fine-tuning is used only to teach Gemma 4 the Law Buddy answer
style: plain, direct, Bangladesh-community friendly, concise, and citation-safe.

The submission story is:

> Qdrant RAG supplies Bangladesh legal sources. Gemma 4 LoRA turns those sources
> into accessible, cited legal explanations for ordinary users.

Do not fine-tune Gemma 4 to memorize acts or judgments. That would weaken source
grounding and increase hallucination risk. Fine-tune it on examples where the
retrieved context is already present in the prompt.

## Goals

- Make Gemma 4 visibly central to the application and competition writeup.
- Preserve the current statute and case-law retrieval pipeline.
- Improve answer style without sacrificing citation accuracy.
- Demonstrate a clear before/after improvement: base Gemma 4 vs Law Buddy tuned
  Gemma 4 on the same retrieved sources.
- Keep the implementation reproducible enough for a public code repository or
  Kaggle notebook.

## Non-Goals

- Do not train on raw statutes or raw case PDFs as memorization data.
- Do not use Gemini, Groq, OpenAI, Claude, or any other LLM in the final demo.
- Do not rely on fine-tuning for factual legal recall.
- Do not add long legal disclaimers inside every answer; the UI already shows a
  general-information notice.

## Current App State

The application already has the right foundation:

- FastAPI backend with a legal chat pipeline.
- Next.js chat frontend.
- Qdrant retrieval for Bangladesh statutes.
- Case-law retrieval support through a separate `legal_cases` collection.
- Prompting that separates citable statute sources from reasoning-only precedent
  background.
- Evaluation notes showing `intfloat/multilingual-e5-base` gives strong retrieval
  gains.

The main competition risk is that the current code and docs still mention
Gemini/Groq. The final submission must make Gemma 4 the only LLM used by the app.

## Architecture Target

```text
User question
  -> optional conversation rewrite using Gemma 4
  -> embedding model creates query vector
  -> Qdrant retrieves statute sections
  -> Qdrant retrieves similar case-law chunks
  -> prompt builder formats retrieved context
  -> fine-tuned Gemma 4 generates answer
  -> UI shows answer and statute citations
```

Fine-tuned Gemma 4 is responsible for style, structure, citation discipline, and
clarification behavior. Qdrant is responsible for legal facts.

## Gemma 4 Integration Plan

Use a Gemma-serving path that exposes an OpenAI-compatible chat completions API.
This keeps the backend implementation simple and avoids tying the app to one
runtime.

Recommended runtime choices:

- `vLLM` or `SGLang` for a hosted GPU deployment.
- `Ollama` or `LM Studio` for a local demo if it can expose a compatible endpoint.
- Kaggle/Colab notebook serving is acceptable for the recorded demo if hosting is
  not practical during the hackathon.

Backend configuration should use Gemma-specific environment variables:

```env
GEMMA_BASE_URL=http://localhost:8001/v1
GEMMA_API_KEY=not-needed-for-local
GEMMA_MODEL=gemma-4-lawbuddy-lora
GEMMA_CONDENSE_MODEL=gemma-4-lawbuddy-lora
```

Public UI settings should not expose non-Gemma providers. Development knobs can
remain hidden or removed.

## Fine-Tuning Objective

Train the model to produce this answer behavior:

- Start with the direct answer.
- Use simple language understandable by non-lawyers.
- Explain legal terms briefly when needed.
- Cite only provided statute sources as `[Source N]`.
- Never invent act names, section numbers, case names, or source ids.
- Use precedent background only to improve reasoning unless the UI later exposes
  precedent sources.
- Ask one short clarifying question when important facts are missing.
- Say the retrieved sources do not answer the question when context is
  insufficient.
- Avoid AI jargon, generic disclaimers, robotic preambles, and fixed templates.

## Law Buddy Voice Guide

Use this style guide for dataset creation and the system prompt:

- Plain and direct.
- Helpful but not overconfident.
- Bangladesh-specific where the source supports it.
- Short paragraphs or compact bullets.
- No "As an AI language model".
- No "It is important to note" filler.
- No broad procedural advice unless retrieved sources support it.
- No invented next steps such as filing an FIR, going to court, or collecting
  evidence unless the source context supports that advice.
- One follow-up question at most, only when facts would materially change the
  answer.

## Training Data Format

Use chat-style supervised fine-tuning JSONL. Each row should contain one complete
conversation.

Example shape:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are Law Buddy, a Bangladesh legal assistant. Answer plainly, cite only provided statute sources as [Source N], and do not invent legal facts."
    },
    {
      "role": "user",
      "content": "Question: My neighbor threatened to kill me. What does the law say?\n\nStatute sources (citable):\n[Source 1]\nAct: The Penal Code, 1860\nSection: 503\nText: ...\n\n[Source 2]\nAct: The Penal Code, 1860\nSection: 506\nText: ...\n\nPrecedent background (reasoning only):\n- Court treatment of similar facts:\n  Outcome: ...\n  Reasoning: ..."
    },
    {
      "role": "assistant",
      "content": "This may fall under criminal intimidation. Section 503 covers threats of injury made to alarm a person or force them to do or not do something [Source 1]. Section 506 sets the punishment for that offence [Source 2].\n\nThe key fact is whether the words or conduct were serious enough to cause alarm. What exactly did the neighbor say or do?"
    }
  ]
}
```

## Dataset Composition

Target size:

- Minimum useful set: 150-200 reviewed examples.
- Strong hackathon set: 300-500 reviewed examples.
- Ambitious set: 800-1000 examples if generation and validation are automated.

Quality matters more than volume. A smaller reviewed set is better than a large
noisy generated set.

Recommended distribution:

- 60% normal answer examples with good retrieved statute context.
- 15% situational examples with statute plus precedent background.
- 10% vague-question examples that should ask for clarification.
- 10% insufficient-source examples that should abstain or ask for better facts.
- 5% multilingual examples in Bangla and Banglish.

Legal topic coverage:

- Theft and property offences.
- Criminal intimidation and assault.
- Dowry and violence against women.
- Landlord-tenant and property possession.
- Family law basics.
- Contract questions.
- Land/property disputes.
- Labor/employment questions.
- Writs and constitutional remedies.
- Bail/criminal procedure at a high level only when retrieved context supports it.

## Dataset Generation Workflow

1. Start from existing eval questions and common user scenarios.
2. Run the current retrieval pipeline to attach statute sources and precedent
   background.
3. Draft target answers manually or with a temporary helper model outside the
   final app.
4. Review every target answer for citation accuracy and style.
5. Save accepted samples as JSONL under `training/style_sft/`.
6. Keep rejected or questionable examples out of the training file.

Any generated examples must be treated as drafts, not ground truth.

## Dataset Validation Rules

Add a validator before training. It should fail samples when:

- The assistant answer contains `[Source N]` where `N` is not present in the
  user context.
- The answer contains no citation despite making a legal claim.
- The answer uses banned phrases such as "As an AI language model".
- The answer cites precedent if precedent is marked reasoning-only.
- The answer invents a section number not present in the context.
- The answer is empty, too long, or mostly copied source text.
- The answer has more than one follow-up question.

Also sample manually for legal plausibility; automated validation cannot catch all
unsupported claims.

## Fine-Tuning Method

Use LoRA or QLoRA supervised fine-tuning.

Default choices:

- Base: the Gemma 4 instruction model available for the competition.
- Method: QLoRA if VRAM is limited, LoRA if enough GPU memory is available.
- Max sequence length: enough to include the question plus retrieved context,
  preferably 4k-8k tokens depending on model/runtime limits.
- Epochs: start with 1-3; stop if the model becomes verbose or overfits.
- Learning rate: conservative SFT default for LoRA/QLoRA.
- Output: adapter artifacts separate from the base model.

Training environment is intentionally flexible: Kaggle GPU, local GPU, or cloud
GPU are all acceptable. The repository should document the exact command or
notebook used once the environment is chosen.

## Inference Integration

After training:

- Serve the base Gemma 4 model with the Law Buddy LoRA adapter loaded.
- Point the backend to the tuned model using `GEMMA_MODEL`.
- Use the tuned Gemma endpoint for answer generation.
- Use Gemma for conversation rewrite as well, or replace rewrite with a small
  deterministic fallback if latency is too high.
- Keep retrieval, source formatting, and citation rendering unchanged unless a
  test shows the tuned model needs a simpler prompt.

The final demo must not route any app LLM call to Gemini, Groq, or another
non-Gemma model.

## Prompt Alignment

The production prompt and fine-tuning examples must teach the same behavior.

Keep the production prompt shorter after fine-tuning:

- Identify the assistant as Law Buddy for Bangladesh law.
- State that statute sources are the only citable sources.
- State that precedent background is reasoning-only.
- Require exact `[Source N]` citation format.
- Require clarification or abstention when context is insufficient.
- Avoid long repeated instructions that fight the trained style.

## Case-Law Use

The current safest behavior is:

- Statutes are binding and user-facing.
- Cases are reasoning background.
- The answer can use case-law context to reason about likely treatment.
- The answer should not name or cite specific cases unless the UI exposes those
  cases clearly as sources.

Optional later improvement:

- Add a "Similar court decisions" panel in the UI.
- If that panel is implemented, update training data so the model can cite case
  sources separately from statute sources.

For this fine-tuning plan, keep the existing distinction to reduce hallucination
risk.

## Evaluation Plan

Create a fixed style evaluation set of 30-50 examples before training. Keep this
set separate from training data.

Evaluate base Gemma 4 and tuned Gemma 4 on the exact same retrieved contexts.

Metrics and checks:

- Citation validity: every citation id exists.
- Grounding: legal claims are supported by provided sources.
- Style: direct, plain, non-robotic.
- Concision: simple questions receive short answers.
- Clarification: vague prompts ask instead of guessing.
- Abstention: irrelevant context does not produce confident legal advice.
- Language: Bangla and Banglish prompts are handled naturally.

The demo and writeup should show at least three before/after examples.

## Demo Scenarios

Use these scenarios for the recorded demo or live walkthrough:

1. Direct statute lookup:
   - "What is the punishment for theft under the Penal Code?"
2. Situational criminal law:
   - "My neighbor threatened to kill me. What does the law say?"
3. Property or tenancy:
   - "My landlord removed me without notice even though I paid rent."
4. Vague question:
   - "Is it legal?"
   - Expected behavior: asks what action or situation the user means.
5. Bangla or Banglish:
   - "Amar neighbor amake mere felar humki dise. Bangladesh law e eta ki?"
6. Insufficient retrieval:
   - A question outside Bangladesh law or outside the indexed corpus.
   - Expected behavior: no confident legal answer.

## Repository Deliverables

Add or update:

- `PLAN.md`: this plan.
- `README.md`: Gemma 4-only setup, usage, and demo instructions.
- `.env.example`: Gemma environment variables; remove public Gemini/Groq defaults.
- `training/style_sft/README.md`: dataset format and validation rules.
- `training/style_sft/*.jsonl`: curated train/eval data if it can be public.
- Training notebook or script for LoRA/QLoRA.
- Evaluation script or notebook for base-vs-tuned comparison.

If the fine-tuning data contains copyrighted source text, keep only small
excerpts and document how to regenerate the dataset locally from the legal corpus.

## Submission Writeup Points

Emphasize:

- Gemma 4 is the only LLM powering the assistant.
- Fine-tuning is used for tone and response structure, not legal memory.
- RAG retrieves Bangladesh statutes and Supreme Court case-law context.
- Statute citations are traceable to retrieved sources.
- The project addresses a real access-to-justice problem for Bangladeshi users.
- The tuned model reduces AI jargon and makes legal explanations easier to
  understand.

Suggested writeup phrase:

> We intentionally did not fine-tune Gemma 4 to memorize law. Instead, we use
> retrieval for legal facts and fine-tune Gemma 4 to communicate those facts in a
> clear, citation-disciplined Law Buddy style for Bangladeshi users.

## Acceptance Criteria

The project is submission-ready when:

- The app uses Gemma 4 for all LLM calls.
- The tuned model can answer from retrieved statutes with valid `[Source N]`
  citations.
- The tuned model avoids obvious AI jargon.
- The tuned model handles at least one Bangla or Banglish query.
- The tuned model asks for clarification on vague prompts.
- The demo shows base-vs-tuned improvement.
- README and Kaggle writeup clearly explain Gemma 4 integration.

## Implementation Order

1. Remove non-Gemma LLM provider exposure from UI and docs.
2. Add Gemma-compatible backend provider configuration.
3. Build 30-50 example eval set.
4. Build 150-500 example style SFT training set.
5. Add dataset validator.
6. Train LoRA/QLoRA adapter.
7. Serve Gemma 4 with the adapter.
8. Run base-vs-tuned evaluation.
9. Record demo and update README/writeup.

## Assumptions

- Gemma 4 serving details will be chosen later based on available GPU/runtime.
- LoRA/QLoRA is preferred over full fine-tuning.
- Existing statute and case retrieval remain the source of truth.
- Statutes remain the only user-facing citations for the first submission.
- Case-law remains reasoning-only unless a visible precedent source UI is added.
