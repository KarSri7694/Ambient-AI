export type JsonRecord = Record<string, unknown>;

export interface RuntimeLog {
  id: number;
  timestamp: string;
  logger: string;
  level: string;
  message: string;
}

export interface ChatSession {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  last_message?: string;
  preview?: string;
}

export interface ChatMessage {
  id: string;
  session_id: string;
  role: "user" | "assistant";
  content: string;
  status: string;
  error_text?: string | null;
  message_kind?: string;
  metadata?: JsonRecord;
  created_at: string;
  updated_at: string;
}

export interface NormalizedMessage {
  role: string;
  content: string;
}

export interface InteractionInput {
  protected: boolean;
  request: NormalizedMessage | null;
  context_messages: NormalizedMessage[];
  malformed: boolean;
}

export interface Interaction {
  interaction_id: string;
  interaction_run_id?: string | null;
  created_at: string;
  completed_at?: string | null;
  source: string;
  model: string;
  duration_ms?: number | null;
  input: InteractionInput;
  response_text?: string | null;
  error_text?: string | null;
  reasoning_text?: string | null;
  tools?: unknown;
  tool_calls?: unknown;
  metadata: JsonRecord;
  report?: unknown;
  has_image: boolean;
  image_url?: string | null;
}

export interface InteractionPage {
  items: Interaction[];
  pagination: { limit: number; offset: number; total: number; has_more: boolean };
  sort: "newest" | "oldest";
  date_from?: string | null;
  date_to?: string | null;
}
