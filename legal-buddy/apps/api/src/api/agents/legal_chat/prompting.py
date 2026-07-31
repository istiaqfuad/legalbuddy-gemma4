from api.api.models import ChatMessage, SourceItem


def _format_history(history: list[ChatMessage]) -> str:
    labels = {"user": "User", "assistant": "Assistant"}
    return "\n".join(
        f"{labels.get(m.role, m.role.title())}: {m.content.strip()}" for m in history
    )


def _format_statute(source: SourceItem) -> str:
    return "\n".join(
        [
            f"[Source {source.citation_id}]",
            f"Act: {source.act_title or 'Unknown'}",
            f"Year: {source.act_year if source.act_year is not None else 'Unknown'}",
            f"Section: {source.section_index or 'Unknown'}",
            f"Text: {source.excerpt}",
            f"URL: {source.source_url or 'N/A'}",
        ]
    )


def build_grounded_prompt(
    question: str,
    statutes: list[SourceItem],
    history: list[ChatMessage] | None = None,
    low_confidence: bool = False,
) -> list[dict]:
    history = history or []
    statute_blocks = [_format_statute(s) for s in statutes]

    system_prompt = (
        "You are an expert legal assistant for Bangladesh law. The user may ask a "
        "plain statutory question OR describe a real situation and ask what the law "
        "says or what the likely outcome is. Lead with the direct answer, then add "
        "only what the question needs. Choose whatever format fits best — a short "
        "paragraph, a few bullets, or a single line — and vary it with the question "
        "instead of forcing one template. Be concise and don't repeat yourself. "
        "Ground "
        "every claim in the statute sources — do NOT add generic court-process, "
        "evidence, recovery, or sentencing-factor commentary (e.g. 'the court may "
        "consider the value/intent/prior record', 'the police may recover the "
        "property', 'prosecution depends on the evidence') unless a source actually "
        "states it; omit what the sources do not support.\n"
        "Use the STATUTE SOURCES as the binding law and your only citable "
        "material. Identify the governing act and section number(s), state the "
        "rule, and cite every source you rely on — synthesize across all that bear "
        "on the answer, not just one. Write each citation as its own bracket with "
        "a single number, e.g. [Source 2] [Source 5]; never combine them as "
        "[Source 2, Source 5] or [Source 2 and 5].\n"
        "Reason from the general rule to the specific facts: when the user "
        "describes a situation, work out which offence or rule it constitutes and "
        "apply the relevant section even if that exact scenario is not named "
        "(e.g. theft of any movable property — including an animal — is governed by "
        "the general theft and theft-punishment sections). Do NOT reply that the "
        "sources 'do not specify this exact case' when a general statutory rule "
        "plainly covers it.\n"
        "Cite ONLY statute sources as [Source N]. Do not fabricate statutes, "
        "sections, facts, or outcomes. Only flag the sources as insufficient when "
        "NO retrieved statute governs the situation even by general application — "
        "not merely because the precise fact pattern is unnamed; then say so in one "
        "bullet and state what additional legal text is needed.\n"
        "If the question lacks key facts needed to identify the governing law, or "
        "is too vague to answer reliably, do NOT guess — ask 1-2 short clarifying "
        "questions instead of answering. A bare question with no concrete action or "
        "subject ('is it legal?', 'what does the law say?') is too vague: ask what "
        "action, offence, or area of law they mean, even if some generic sources "
        "happened to match. When the user describes their OWN real "
        "situation (first person — 'my car', 'someone stole my…', 'I was…'), answer "
        "first, then end with ONE short follow-up question (a single sentence) "
        "about a fact that would refine or change the answer. Skip the follow-up on "
        "a plain statutory lookup. Prefer answering whenever the sources are "
        "sufficient.\n"
        "When earlier conversation is provided, use it to interpret the current "
        "question (e.g. resolve references like 'it', 'that', or 'the punishment'), "
        "but ground every legal claim ONLY in the statute sources below."
    )

    user_parts: list[str] = []
    if history:
        user_parts.append("Conversation so far:")
        user_parts.append(_format_history(history))
        user_parts.append("")
    user_parts.append(f"Question / situation: {question}")
    if low_confidence:
        user_parts.append(
            "(Retrieval confidence is low — if NONE of these sources plausibly "
            "apply to the question even by general principle, ask a brief "
            "clarifying question instead of answering; otherwise reason from them.)"
        )
    user_parts.append("")
    user_parts.append("Statute sources (citable):")
    user_parts.append(chr(10).join(statute_blocks) if statute_blocks else "(none)")
    user_parts.append("")
    user_parts.append(
        "Answer the question directly, grounding every claim in the statute sources "
        "and citing each as [Source N] (one number per bracket). Structure it "
        "however best fits this question — prose, a few bullets, or a single line — "
        "there is no fixed template; let simple questions stay short. Don't repeat "
        "the same point or citation. If the user describes their own situation, you "
        "may end with one short follow-up question about a fact that would change "
        "the answer. Stick to what the sources support — no generic procedure, "
        "recovery, evidence, or sentencing speculation — and do not add a "
        "legal-advice disclaimer; the interface shows one."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


def build_clarify_prompt(
    question: str, history: list[ChatMessage] | None = None
) -> list[dict]:
    """Prompt for when retrieval found no usable sources.

    There is nothing to ground an answer in, so instead of dead-ending, ask the
    user for the specifics that would make the question searchable. Plain text,
    no citations.
    """
    history = history or []
    system_prompt = (
        "You are a legal assistant for Bangladesh law. No matching statute was "
        "found for the user's question — it is likely too vague, off-topic, or "
        "missing key facts. Do NOT answer or guess at the law. Instead ask 1-2 "
        "short clarifying questions that would let you find the right statute — "
        "e.g. the area of law, the specific act, the key facts, or the "
        "jurisdiction. Be brief and friendly; do not cite anything."
    )
    user_parts: list[str] = []
    if history:
        user_parts.append("Conversation so far:")
        user_parts.append(_format_history(history))
        user_parts.append("")
    user_parts.append(f"Question / situation: {question}")
    user_parts.append("")
    user_parts.append("Ask 1-2 short clarifying questions:")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


def build_condense_messages(
    question: str, history: list[ChatMessage]
) -> list[dict]:
    """Prompt that rewrites a follow-up into a standalone retrieval query."""
    system_prompt = (
        "You rewrite a user's follow-up into a standalone search query for a "
        "Bangladesh legal statute database. Use the conversation only to resolve "
        "references (pronouns, ellipsis, 'the punishment', 'that offence'). Keep the "
        "user's intent and legal terms. CARRY FORWARD the aspect the conversation is "
        "asking about — if the previous turn was about the punishment/penalty (or "
        "the procedure, the definition, etc.), keep that word in the rewrite. E.g. "
        "after 'punishment for theft', the follow-up 'what if it was my car' becomes "
        "'punishment for theft of a car', NOT 'theft of a car'. If the follow-up is "
        "already self-contained, return it unchanged. Output ONLY the rewritten "
        "query — no preamble, no quotes, no explanation."
    )
    user_content = (
        f"Conversation so far:\n{_format_history(history)}\n\n"
        f"Follow-up question: {question}\n\n"
        "Standalone search query:"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
