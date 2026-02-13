// request and response types that match backend endpoints and payloads

export type TimelineEvent = {
  id: number;
  event_type: "exposure" | "symptom";
  user_id: number;
  timestamp: string | null;
  item_id?: number | null;
  item_name?: string | null;
  route?: string | null;
  symptom_id?: number | null;
  symptom_name?: string | null;
  severity?: number | null;
};

export type CreateEventRequest = {
  event_type: "exposure" | "symptom";
  user_id: number;
  timestamp?: string;
  time_range_start?: string;
  time_range_end?: string;
  time_confidence?: "exact" | "approx" | "backfilled";
  raw_text?: string;
  item_id?: number | null;
  item_name?: string | null;
  route?: string | null;
  symptom_id?: number | null;
  severity?: number | null;
};

export type TextIngestRequest = {
  user_id: number;
  raw_text: string;
};

export type TextIngestResponse = {
  status: string;
  event_type?: "exposure" | "symptom";
  resolution?: string;
};

export type CreateEventResponse = {
  id: number;
  event_type: "exposure" | "symptom";
  user_id: number;
  timestamp?: string | null;
  time_range_start?: string | null;
  time_range_end?: string | null;
  time_confidence?: "exact" | "approx" | "backfilled" | null;
  raw_text?: string | null;
  item_id?: number | null;
  route?: string | null;
  symptom_id?: number | null;
  severity?: number | null;
  status?: string | null;
  resolution?: string | null;
};

export type InsightCitation = {
  title?: string | null;
  url?: string | null;
  snippet?: string | null;
  evidence_polarity_and_strength?: number | null;
};

export type Insight = {
  id: number;
  user_id: number;
  item_id: number;
  item_name: string;
  symptom_id: number;
  symptom_name: string;
  model_probability?: number | null;
  evidence_strength_score?: number | null;
  evidence_summary?: string | null;
  display_decision_reason?: string | null;
  display_status?: "supported" | "insufficient_evidence" | "suppressed" | null;
  created_at?: string | null;
  citations: InsightCitation[];
};
