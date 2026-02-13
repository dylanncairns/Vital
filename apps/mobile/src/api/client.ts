import {
  TimelineEvent,
  Insight,
  CreateEventRequest,
  CreateEventResponse,
  TextIngestRequest,
  TextIngestResponse,
} from "../models/events";

// Local server address
const BASE_URL = "http://127.0.0.1:8000";

// Request to GET/events with user id specified
export async function fetchEvents(userId: number): Promise<TimelineEvent[]> {
  const res = await fetch(`${BASE_URL}/events?user_id=${userId}`);
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Failed to fetch events: ${res.status} ${text}`);
  }
  return res.json();
}

// Sends JSON payload to POST/events 
export async function createEvent(payload: CreateEventRequest): Promise<CreateEventResponse> {
  const res = await fetch(`${BASE_URL}/events`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to create event: ${res.status} ${text}`);
  }
  return res.json();
}

// Route used if text blurb entry (later speech-to-text button) - to ingestion pipeline
export async function ingestTextEvent(payload: TextIngestRequest): Promise<TextIngestResponse> {
  const res = await fetch(`${BASE_URL}/events/ingest_text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to ingest text: ${res.status} ${text}`);
  }
  return res.json();
}

export async function fetchInsights(userId: number, includeSuppressed = false): Promise<Insight[]> {
  const res = await fetch(
    `${BASE_URL}/insights?user_id=${userId}&include_suppressed=${includeSuppressed ? "true" : "false"}`
  );
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to fetch insights: ${res.status} ${text}`);
  }
  return res.json();
}
