"use client";

import { Trash2 } from "lucide-react";
import { type ChatSettings } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export function SidebarContent({
  settings,
  onChange,
  onClear,
  canClear,
}: {
  settings: ChatSettings;
  onChange: (next: ChatSettings) => void;
  onClear: () => void;
  canClear: boolean;
}) {
  const set = (patch: Partial<ChatSettings>) => onChange({ ...settings, ...patch });

  return (
    <div className="flex h-full flex-col bg-surface">
      <div className="px-6 pb-4 pt-6">
        <h2 className="text-lg font-semibold tracking-tight text-text">Settings</h2>
        <p className="mt-0.5 text-[13px] text-muted">Tune retrieval and the model.</p>
      </div>

      <div className="flex-1 overflow-y-auto px-6">
        {/* Retrieval */}
        <Section title="Retrieval">
          <Row label="Sources to retrieve" value={settings.topK} />
          <input
            type="range"
            min={3}
            max={12}
            step={1}
            value={settings.topK}
            onChange={(e) => set({ topK: Number(e.target.value) })}
            aria-label="Sources to retrieve"
            aria-valuetext={`${settings.topK} sources`}
            className="w-full accent-accent"
          />
          <Hint>Statute sections fed to the model. More = broader & better-cited; fewer = tighter.</Hint>
        </Section>

        {/* Clarification thresholds */}
        <Section title="Clarification" divider>
          <div className="space-y-2">
            <Row label="Clarify floor (hard)" value={settings.clarifyScoreFloor.toFixed(2)} />
            <input
              type="range"
              min={0.7}
              max={0.95}
              step={0.01}
              value={settings.clarifyScoreFloor}
              onChange={(e) => {
                const v = Number(e.target.value);
                set({
                  clarifyScoreFloor: v,
                  lowConfidenceFloor: Math.max(v, settings.lowConfidenceFloor),
                });
              }}
              aria-label="Clarify score floor"
              aria-valuetext={settings.clarifyScoreFloor.toFixed(2)}
              className="w-full accent-accent"
            />
            <Hint>Below this top-match score the answer is replaced by a clarifying question. Lower = answers more; higher = asks more.</Hint>
          </div>

          <div className="space-y-2">
            <Row label="Low-confidence floor (soft)" value={settings.lowConfidenceFloor.toFixed(2)} />
            <input
              type="range"
              min={0.7}
              max={0.95}
              step={0.01}
              value={settings.lowConfidenceFloor}
              onChange={(e) => {
                const v = Number(e.target.value);
                set({
                  lowConfidenceFloor: v,
                  clarifyScoreFloor: Math.min(v, settings.clarifyScoreFloor),
                });
              }}
              aria-label="Low confidence floor"
              aria-valuetext={settings.lowConfidenceFloor.toFixed(2)}
              className="w-full accent-accent"
            />
            <Hint>Between this and the hard floor the model keeps the sources but is nudged to ask if they don&apos;t fit. Kept ≥ hard floor.</Hint>
          </div>
        </Section>


      </div>

      <div className="border-t border-line px-6 py-4">
        <Button variant="outline" size="md" onClick={onClear} disabled={!canClear} className="w-full">
          <Trash2 className="h-4 w-4" />
          Clear conversation
        </Button>
      </div>
    </div>
  );
}

function Section({
  title,
  tag,
  divider,
  children,
}: {
  title: string;
  tag?: string;
  divider?: boolean;
  children: React.ReactNode;
}) {
  return (
    <section className={cn("py-5", divider && "border-t border-line")}>
      <h3 className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-faint">
        {title}
        {tag && (
          <span className="rounded bg-accent-soft px-1.5 py-0.5 text-[10px] font-medium normal-case tracking-normal text-accent">
            {tag}
          </span>
        )}
      </h3>
      <div className="space-y-5">{children}</div>
    </section>
  );
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-sm font-medium text-text">{label}</span>
      <span className="rounded-md bg-accent-soft px-2 py-0.5 text-xs font-medium tabular-nums text-accent">
        {value}
      </span>
    </div>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return <span className="text-sm font-medium text-text">{children}</span>;
}

function Hint({ children }: { children: React.ReactNode }) {
  return <p className="text-[13px] leading-relaxed text-muted">{children}</p>;
}
