import { NextResponse } from "next/server";

// Server-side only. In docker-compose this resolves to the api service;
// for local `next dev` outside docker, set API_URL=http://localhost:8000.
const API_URL = process.env.API_URL ?? "http://api:8000";
const REQUEST_TIMEOUT_MS = 120_000;

export async function POST(request: Request) {
  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body." }, { status: 400 });
  }

  const question = typeof body.question === "string" ? body.question.trim() : "";
  if (!question) {
    return NextResponse.json({ error: "Question is required." }, { status: 400 });
  }
  const num = (v: unknown) =>
    typeof v === "number" && Number.isFinite(v) ? v : undefined;
  const topK = num(body.top_k);
  // Testing knobs forwarded as-is to the API (remove with the UI controls in prod).
  const provider =
    body.provider === "gemini" || body.provider === "groq" ? body.provider : undefined;
  const model =
    typeof body.model === "string" && body.model.trim() ? body.model.trim() : undefined;
  const temperature = num(body.temperature);
  const maxTokens = num(body.max_tokens);
  const clarifyScoreFloor = num(body.clarify_score_floor);
  const lowConfidenceFloor = num(body.low_confidence_floor);

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const upstream = await fetch(`${API_URL}/rag/legal/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        top_k: topK,
        provider,
        model,
        temperature,
        max_tokens: maxTokens,
        clarify_score_floor: clarifyScoreFloor,
        low_confidence_floor: lowConfidenceFloor,
      }),
      signal: controller.signal,
    });

    const payload = await upstream.json().catch(() => null);

    if (!upstream.ok) {
      const detail = extractDetail(payload) ?? `Request failed (${upstream.status}).`;
      return NextResponse.json({ error: detail }, { status: upstream.status });
    }
    return NextResponse.json(payload, { status: 200 });
  } catch (err) {
    const aborted = err instanceof Error && err.name === "AbortError";
    return NextResponse.json(
      {
        error: aborted
          ? "The request timed out. Please try again."
          : "Could not reach the legal assistant service.",
      },
      { status: 502 },
    );
  } finally {
    clearTimeout(timeout);
  }
}

// FastAPI errors come back as { detail: string | [{ msg, ... }] }.
function extractDetail(payload: unknown): string | null {
  if (!payload || typeof payload !== "object") return null;
  const detail = (payload as { detail?: unknown }).detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d) => (d && typeof d === "object" && "msg" in d ? String(d.msg) : String(d)))
      .join("; ");
  }
  return null;
}
