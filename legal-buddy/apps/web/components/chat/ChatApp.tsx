"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowRight, Settings2, X } from "lucide-react";
import {
  type ChatSettings,
  type Source,
  type Turn,
  DEFAULT_SETTINGS,
} from "@/lib/types";

const HISTORY_WINDOW_TURNS = 6;
import { Message } from "./Message";
import { Composer } from "./Composer";
import { SidebarContent } from "./Sidebar";
import { Logo } from "@/components/ui/Logo";
import { cn } from "@/lib/utils";

const EXAMPLES = [
  "What is the punishment for theft under the Penal Code?",
  "আমার প্রতিবেশী আমাকে হুমকি দিচ্ছে। আইনে এর বিধান কী?",
  "amar barite dhuke churi korar chesta korlo, ki korte pari?",
  "How is culpable homicide defined?",
];

export function ChatApp() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [settings, setSettings] = useState<ChatSettings>(DEFAULT_SETTINGS);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, loading]);

  // Close the settings panel on Escape.
  useEffect(() => {
    if (!drawerOpen) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setDrawerOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [drawerOpen]);

  const send = useCallback(
    async (raw?: string) => {
      const question = (raw ?? input).trim();
      if (!question || loading) return;

      // Conversation memory: send prior turns (hard last-N-turn window) so the
      // backend can resolve follow-ups. Built before appending the new user turn.
      const history = turns
        .filter((t) => !t.error)
        .map((t) => ({ role: t.role, content: t.content }))
        .slice(-HISTORY_WINDOW_TURNS);

      setTurns((t) => [...t, { id: uid(), role: "user", content: question }]);
      setInput("");
      setLoading(true);

      // The assistant turn is created lazily on the first event so the "Thinking"
      // indicator shows until tokens start arriving.
      const assistantId = uid();
      let assistantAdded = false;
      const ensureAssistant = () => {
        if (assistantAdded) return;
        assistantAdded = true;
        setTurns((t) => [
          ...t,
          { id: assistantId, role: "assistant", content: "", sources: [] },
        ]);
      };
      const patch = (fn: (turn: Turn) => Turn) =>
        setTurns((t) => t.map((x) => (x.id === assistantId ? fn(x) : x)));

      const handleEvent = (event: string, data: string) => {
        let payload: { text?: string; sources?: Source[]; error?: string };
        try {
          payload = JSON.parse(data);
        } catch {
          return;
        }
        if (event === "sources") {
          ensureAssistant();
          patch((x) => ({ ...x, sources: payload.sources ?? [] }));
        } else if (event === "delta") {
          ensureAssistant();
          setLoading(false);
          patch((x) => ({ ...x, content: x.content + (payload.text ?? "") }));
        } else if (event === "error") {
          ensureAssistant();
          patch((x) => ({
            ...x,
            content: payload.error ?? "Something went wrong.",
            error: true,
          }));
        }
      };

      try {
        const res = await fetch("/api/chat/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question,
            history,
            top_k: settings.topK,
            provider: settings.provider || undefined, // "" -> server env default
            model: settings.model || undefined,
            temperature: settings.temperature,
            max_tokens: settings.maxTokens ?? undefined,
            clarify_score_floor: settings.clarifyScoreFloor,
            low_confidence_floor: settings.lowConfidenceFloor,
          }),
        });

        if (!res.ok || !res.body) {
          const data = await res.json().catch(() => null);
          ensureAssistant();
          patch((x) => ({
            ...x,
            content: data?.error ?? "Something went wrong.",
            error: true,
          }));
          return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          let sep: number;
          while ((sep = buffer.indexOf("\n\n")) !== -1) {
            const frame = buffer.slice(0, sep);
            buffer = buffer.slice(sep + 2);
            let event = "message";
            const dataLines: string[] = [];
            for (const line of frame.split("\n")) {
              if (line.startsWith("event:")) event = line.slice(6).trim();
              else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
            }
            if (dataLines.length) handleEvent(event, dataLines.join("\n"));
          }
        }
      } catch {
        ensureAssistant();
        patch((x) => ({
          ...x,
          content: x.content || "Network error — please try again.",
          error: true,
        }));
      } finally {
        setLoading(false);
      }
    },
    [input, loading, settings, turns],
  );

  const isEmpty = turns.length === 0;

  return (
    <div className="flex h-dvh flex-col overflow-hidden bg-bg">
      {/* Top bar */}
      <header className="shrink-0 border-b border-line bg-bg/80 backdrop-blur">
        <div className="mx-auto flex max-w-3xl items-center justify-between px-4 py-2.5 sm:px-6">
          <div className="flex items-center gap-2.5">
            <Logo className="h-7 w-7" />
            <div className="flex items-baseline gap-2">
              <span className="text-[15px] font-semibold tracking-tight text-text">LegalBuddy</span>
              <span className="hidden text-[13px] text-muted sm:inline">
                Bangladesh statutes &amp; acts
              </span>
            </div>
          </div>
          <button
            onClick={() => setDrawerOpen(true)}
            aria-label="Settings"
            aria-expanded={drawerOpen}
            className="flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[13px] font-medium text-muted transition-colors hover:bg-accent-soft hover:text-accent-strong focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <Settings2 className="h-4 w-4" />
            <span className="hidden sm:inline">Settings</span>
          </button>
        </div>
      </header>

      {/* Messages */}
      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl px-4 sm:px-6">
          {isEmpty ? (
            <EmptyState onPick={(q) => send(q)} />
          ) : (
            <div className="space-y-6 py-6">
              {turns.map((turn) => (
                <Message key={turn.id} turn={turn} />
              ))}
              {loading && <Thinking />}
              <div ref={bottomRef} />
            </div>
          )}
        </div>
      </main>

      {/* Composer */}
      <div className="shrink-0 bg-bg">
        <div className="mx-auto max-w-3xl px-4 py-3 sm:px-6">
          <Composer value={input} onChange={setInput} onSend={() => send()} loading={loading} />
          <p className="mt-2 text-center text-xs text-faint">
            General information, not legal advice — consult a lawyer.
          </p>
        </div>
      </div>

      {/* Settings — right slide-over */}
      {drawerOpen && (
        <div className="fixed inset-0 z-50">
          <div
            className="fade-in absolute inset-0 bg-text/30"
            onClick={() => setDrawerOpen(false)}
            aria-hidden
          />
          <aside
            aria-label="Settings"
            className="panel-in absolute right-0 top-0 h-full w-[25rem] max-w-[90vw] border-l border-line bg-surface shadow-2xl"
          >
            <button
              onClick={() => setDrawerOpen(false)}
              aria-label="Close settings"
              className="absolute right-3 top-4 z-10 flex h-7 w-7 items-center justify-center rounded-md text-muted transition-colors hover:bg-accent-soft hover:text-accent-strong"
            >
              <X className="h-4 w-4" />
            </button>
            <SidebarContent
              settings={settings}
              onChange={setSettings}
              onClear={() => {
                setTurns([]);
                setDrawerOpen(false);
              }}
              canClear={!isEmpty}
            />
          </aside>
        </div>
      )}
    </div>
  );
}

function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="mx-auto flex min-h-full max-w-xl flex-col justify-center py-16">
      <div className="rise">
        <h1 className="text-[1.7rem] font-semibold leading-tight tracking-tight text-text">
          Legal answers, cited to the section.
        </h1>
        <p className="mt-2 text-[15px] leading-relaxed text-muted">
          Ask about Bangladesh statutory law. Answers are grounded in the indexed acts, with
          sources you can check.
        </p>

        <div className="mt-7">
          <p className="mb-2 text-xs font-medium uppercase tracking-wider text-faint">Try asking</p>
          <div className="overflow-hidden rounded-xl border border-line bg-surface">
            {EXAMPLES.map((q, i) => (
              <button
                key={q}
                onClick={() => onPick(q)}
                className={cn(
                  "group flex w-full items-center gap-3 px-4 py-3 text-left transition-colors hover:bg-accent-soft",
                  i > 0 && "border-t border-line",
                )}
              >
                <span className="text-[14.5px] text-text">{q}</span>
                <ArrowRight className="ml-auto h-4 w-4 shrink-0 -translate-x-1 text-faint opacity-0 transition-all group-hover:translate-x-0 group-hover:text-accent group-hover:opacity-100" />
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function Thinking() {
  return (
    <div className="flex items-center gap-1.5 py-1" role="status" aria-label="Thinking">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="h-2 w-2 rounded-full bg-accent"
          style={{ animation: "pulse-dot 1.2s ease-in-out infinite", animationDelay: `${i * 0.15}s` }}
        />
      ))}
    </div>
  );
}

function uid(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return Math.random().toString(36).slice(2);
}
