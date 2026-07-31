"use client";

import { useRef, useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { AlertCircle } from "lucide-react";
import type { Turn } from "@/lib/types";
import { SourceList } from "./SourceList";

export function Message({ turn }: { turn: Turn }) {
  // Hover previews a citation↔source link; a click pins it. Hover wins while the
  // mouse is over a mark, so you can preview other sources without losing the pin.
  const [hoverCite, setHoverCite] = useState<number | null>(null);
  const [pinnedCite, setPinnedCite] = useState<number | null>(null);
  // The source panel's open/closed state lives here (not in SourceList) so it
  // survives streaming re-renders and a citation click can force it open.
  // Collapsed by default (like the earlier UI): show just the "N sources" button
  // until the user expands it or clicks an inline citation.
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const activeCite = hoverCite ?? pinnedCite;
  const rootRef = useRef<HTMLDivElement>(null);

  if (turn.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] whitespace-pre-wrap break-words rounded-2xl rounded-br-md bg-surface px-4 py-2.5 text-[15px] leading-relaxed text-text shadow-sm ring-1 ring-line">
          {turn.content}
        </div>
      </div>
    );
  }

  if (turn.error) {
    return (
      <div className="flex items-start gap-2.5 rounded-xl border border-danger/30 bg-danger/5 px-4 py-3 text-[15px] text-danger">
        <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
        <span>{turn.content}</span>
      </div>
    );
  }

  const sources = turn.sources ?? [];
  const linked = linkifyCitations(turn.content);

  // Click an inline [N]: open the panel if collapsed, pin the highlight, and
  // bring that source into view. The external link lives in the panel itself,
  // so inline marks never navigate away.
  const revealSource = (n: number) => {
    setPinnedCite(n);
    setSourcesOpen(true);
    requestAnimationFrame(() => {
      rootRef.current
        ?.querySelector(`[data-source="${n}"]`)
        ?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
  };

  const components: Components = {
    a({ href, children }) {
      const text = String(Array.isArray(children) ? children.join("") : children ?? "").trim();
      const n = /^\d+$/.test(text) ? Number(text) : null;

      // Real links in the answer body still open out; only numeric citation
      // marks become in-place pins.
      if (n === null) {
        return (
          <a
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            className="text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent"
          >
            {children}
          </a>
        );
      }

      return (
        <button
          type="button"
          className="cite"
          data-active={activeCite === n ? "true" : undefined}
          aria-label={`Source ${n}`}
          onMouseEnter={() => setHoverCite(n)}
          onMouseLeave={() => setHoverCite(null)}
          onFocus={() => setHoverCite(n)}
          onBlur={() => setHoverCite(null)}
          onClick={() => revealSource(n)}
        >
          {n}
        </button>
      );
    },
  };

  return (
    <div ref={rootRef} className="rise flex flex-col">
      <div className="answer-prose max-w-full break-words text-text">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
          {linked}
        </ReactMarkdown>
      </div>
      <SourceList
        sources={sources}
        activeCite={activeCite}
        onHover={setHoverCite}
        open={sourcesOpen}
        onOpenChange={setSourcesOpen}
      />
    </div>
  );
}

// Render the API's "[Source N]" citations as compact "[N]" reference marks that
// pin to the source panel — never an external link (the panel carries those).
// The "cite:" href is stripped by React-Markdown, so each mark falls through to
// the pin button in `components.a`. Eat any leading space so the mark hugs the
// word it annotates (otherwise it can wrap onto its own line), and emit each
// number as its own marker so "[Source 2]", "[Source 2, 5]", and "[Sources 2 and
// 5]" all split into separate marks.
function linkifyCitations(answer: string): string {
  return answer.replace(/[ \t]*\[Sources?\b[^\]]*\]/gi, (block) => {
    const nums = block.match(/\d+/g);
    if (!nums) return block;
    return nums.map((n) => `[${n}](cite:${n})`).join("");
  });
}
