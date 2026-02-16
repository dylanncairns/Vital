// request and response types that match backend endpoints and payloads

export type TimelineEvent = {
  id: number;
  event_type: "exposure" | "symptom";
  user_id: number;
  timestamp: string | null;
  time_range_start?: string | null;
  time_range_end?: string | null;
  time_confidence?: "exact" | "approx" | "backfilled" | null;
  raw_text?: string | null;
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
  timezone_offset_minutes?: number;
};

export type TextIngestResponse = {
  status: string;
  event_type?: "exposure" | "symptom" | "multi";
  resolution?: string;
};

export type UpdateEventRequest = {
  timestamp?: string | null;
  time_range_start?: string | null;
  time_range_end?: string | null;
  time_confidence?: "exact" | "approx" | "backfilled" | null;
  raw_text?: string | null;
  item_id?: number | null;
  item_name?: string | null;
  route?: string | null;
  symptom_id?: number | null;
  symptom_name?: string | null;
  severity?: number | null;
};

export type EventMutationResponse = {
  status: string;
  event_id: number;
  event_type: "exposure" | "symptom";
  jobs_queued: number;
};

export type RecurringExposureRule = {
  id: number;
  user_id: number;
  item_id: number;
  item_name: string;
  route: string;
  start_at: string;
  interval_hours: number;
  time_confidence?: "exact" | "approx" | "backfilled" | null;
  is_active: number;
  last_generated_at?: string | null;
  notes?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
};

export type CreateRecurringExposureRequest = {
  user_id: number;
  item_id?: number | null;
  item_name?: string | null;
  route: string;
  start_at: string;
  interval_hours: number;
  time_confidence?: "exact" | "approx" | "backfilled";
  notes?: string | null;
};

export type PatchRecurringExposureRequest = {
  route?: string | null;
  start_at?: string | null;
  interval_hours?: number | null;
  time_confidence?: "exact" | "approx" | "backfilled" | null;
  is_active?: boolean | null;
  notes?: string | null;
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
  source?: string | null;
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
  source_ingredient_id?: number | null;
  source_ingredient_name?: string | null;
  symptom_id: number;
  symptom_name: string;
  model_probability?: number | null;
  evidence_strength_score?: number | null;
  evidence_quality_score?: number | null;
  penalty_score?: number | null;
  overall_confidence_score?: number | null;
  evidence_summary?: string | null;
  display_decision_reason?: string | null;
  display_status?: "supported" | "insufficient_evidence" | "suppressed" | null;
  user_verified?: boolean;
  user_rejected?: boolean;
  created_at?: string | null;
  citations: InsightCitation[];
};

export type EventInsightLink = {
  event_type: "exposure" | "symptom";
  event_id: number;
  insight_id: number;
};
