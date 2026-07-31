"use client";

import { useState } from "react";
import { ChevronDown, ExternalLink } from "lucide-react";
import type { Source } from "@/lib/types";
import { cn } from "@/lib/utils";

export function SourceList({
  sources,
  activeCite,
  onHover,
  open,
  onOpenChange,
}: {
  sources: Source[];
  activeCite?: number | null;
  onHover?: (n: number | null) => void;
  // Controlled open/closed state. When omitted, falls back to local state so the
  // component still works standalone.
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}) {
  const [internalOpen, setInternalOpen] = useState(false);
  const isOpen = open ?? internalOpen;
  const toggle = () => (onOpenChange ? onOpenChange(!isOpen) : setInternalOpen((v) => !v));
  if (!sources.length) return null;

  return (
    <div className="mt-4">
      <button
        onClick={toggle}
        className="inline-flex items-center gap-1.5 text-[13px] font-medium text-muted transition-colors hover:text-text"
        aria-expanded={isOpen}
      >
        {sources.length} {sources.length === 1 ? "source" : "sources"}
        <ChevronDown className={cn("h-3.5 w-3.5 transition-transform", isOpen && "rotate-180")} />
      </button>

      {isOpen && (
        <ol className="mt-2 space-y-1.5">
          {sources.map((s) => {
            const active = activeCite === s.citation_id;
            return (
              <li
                key={s.citation_id}
                data-source={s.citation_id}
                onMouseEnter={() => onHover?.(s.citation_id)}
                onMouseLeave={() => onHover?.(null)}
                className={cn(
                  "flex gap-3 rounded-xl border bg-surface p-3 transition-colors",
                  active ? "border-accent/40 bg-accent-soft" : "border-line",
                )}
              >
                <span
                  className={cn(
                    "mt-0.5 flex h-5 min-w-5 shrink-0 items-center justify-center rounded-md px-1 text-xs font-semibold tabular-nums transition-colors",
                    active ? "bg-accent text-white" : "bg-accent-soft text-accent",
                  )}
                >
                  {s.citation_id}
                </span>

                <div className="min-w-0 flex-1">
                  <div className="flex items-start justify-between gap-3">
                    <p className="text-[14px] font-medium leading-snug text-text">
                      {cleanTitle(s.act_title)}
                      {s.act_year ? <span className="text-muted"> ({s.act_year})</span> : null}
                    </p>
                    {s.source_url && (
                      <a
                        href={s.source_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="shrink-0 text-faint transition-colors hover:text-accent"
                        aria-label="Open original source"
                      >
                        <ExternalLink className="h-3.5 w-3.5" />
                      </a>
                    )}
                  </div>
                  <p className="mt-0.5 text-[12px] text-faint">
                    Section {s.section_index ?? "—"} · score {s.score.toFixed(3)}
                  </p>
                  {s.excerpt && (
                    <p className="mt-1.5 line-clamp-3 text-[13px] leading-relaxed text-muted">
                      {s.excerpt}
                    </p>
                  )}
                </div>
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}

// Source titles in the data sometimes carry a stray leading digit, e.g. "1The Penal Code".
function cleanTitle(title: string | null): string {
  if (!title) return "Unknown Act";
  return title.replace(/^\d+(?=[A-Z])/, "");
}
