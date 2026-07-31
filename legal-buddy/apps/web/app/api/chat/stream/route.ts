import { NextResponse } from "next/server";

// Stream the answer token-by-token. Mirrors app/api/chat/route.ts but proxies the
// backend SSE stream straight through to the browser.
export const dynamic = "force-dynamic";

const API_URL = process.env.API_URL ?? "http://api:8000";
const CONNECT_TIMEOUT_MS = 120_000;

type Msg = { role: "user" | "assistant"; content: string };

function parseHistory(raw: unknown): Msg[] {
  if (!Array.isArray(raw)) return [];
  const out: Msg[] = [];
  for (const item of raw) {
    if (!item || typeof item !== "object") continue;
    const rec = item as Record<string, unknown>;
    if (typeof rec.content !== "string") continue;
    out.push({
      role: rec.role === "assistant" ? "assistant" : "user",
      content: rec.content,
    });
  }
  return out;
}

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
  const history = parseHistory(body.history);
  const provider =
    body.provider === "gemini" || body.provider === "groq" ? body.provider : undefined;
  const model =
    typeof body.model === "string" && body.model.trim() ? body.model.trim() : undefined;

  // Abort only guards the initial connection; once the stream is flowing we let it run.
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), CONNECT_TIMEOUT_MS);

  try {
    const upstream = await fetch(`${API_URL}/rag/legal/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        history,
        top_k: num(body.top_k),
        provider,
        model,
        temperature: num(body.temperature),
        max_tokens: num(body.max_tokens),
        clarify_score_floor: num(body.clarify_score_floor),
        low_confidence_floor: num(body.low_confidence_floor),
      }),
      signal: controller.signal,
    });

    if (!upstream.ok || !upstream.body) {
      const detail = await upstream.text().catch(() => "");
      return NextResponse.json(
        { error: detail || `Request failed (${upstream.status}).` },
        { status: upstream.status || 502 },
      );
    }

    return new Response(upstream.body, {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
      },
    });
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
