// Mirrors the FastAPI response (apps/api/src/api/api/models.py)
export interface Source {
  citation_id: number;
  act_title: string | null;
  act_year: number | null;
  section_index: string | null;
  source_url: string | null;
  excerpt: string;
  score: number;
}

export interface ChatResponse {
  answer: string;
  sources: Source[];
}

// --- Testing knobs (dev-only; remove with the UI controls before production) ---
export type Provider = "gemini" | "groq";

export interface ChatSettings {
  provider: Provider;
  model: string; // "" => let the API pick the provider default
  temperature: number;
  maxTokens: number | null;
  topK: number;
  // Clarify thresholds (top-statute cosine). clarifyScoreFloor = hard no-match
  // floor (below -> deterministic clarify); lowConfidenceFloor = soft hint floor
  // (below -> nudge the model toward asking). Mirror the API config defaults.
  clarifyScoreFloor: number;
  lowConfidenceFloor: number;
}

export const PROVIDER_MODELS: Record<Provider, string[]> = {
  gemini: ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"],
  groq: ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-120b"],
};

export const DEFAULT_SETTINGS: ChatSettings = {
  provider: "groq",
  model: "llama-3.3-70b-versatile",
  temperature: 0.2,
  maxTokens: null,
  topK: 6,
  clarifyScoreFloor: 0.79,
  lowConfidenceFloor: 0.83,
};

export type Role = "user" | "assistant";

export interface Turn {
  id: string;
  role: Role;
  content: string;
  sources?: Source[];
  error?: boolean;
}

// Prior turns sent to the backend for multi-turn memory (mirrors api ChatMessage).
export interface ChatMessage {
  role: Role;
  content: string;
}
