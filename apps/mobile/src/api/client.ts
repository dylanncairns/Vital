import {
  TimelineEvent,
  Insight,
  CreateEventRequest,
  CreateEventResponse,
  TextIngestRequest,
  TextIngestResponse,
  UpdateEventRequest,
  EventMutationResponse,
  RecurringExposureRule,
  CreateRecurringExposureRequest,
  PatchRecurringExposureRequest,
  EventInsightLink,
} from "../models/events";

// API base URL (Expo env in production, localhost fallback in dev)
const envBaseUrl = process.env.EXPO_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";
const BASE_URL = envBaseUrl.replace(/\/+$/, "");
let AUTH_TOKEN: string | null = null;
const REQUEST_TIMEOUT_MS = 15000;

export type AuthUser = { id: number; username: string; name?: string | null };
export type AuthResponse = { token: string; user: AuthUser };

export function setAuthToken(token: string | null) {
  AUTH_TOKEN = token;
}

function authHeaders(extra: Record<string, string> = {}): Record<string, string> {
  const headers: Record<string, string> = { ...extra };
  if (AUTH_TOKEN) {
    headers.Authorization = `Bearer ${AUTH_TOKEN}`;
  }
  return headers;
}

async function fetchWithTimeout(url: string, init: RequestInit = {}, timeoutMs = REQUEST_TIMEOUT_MS): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } catch (err: any) {
    if (err?.name === "AbortError") {
      throw new Error(`Request timed out after ${Math.round(timeoutMs / 1000)}s`);
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

function parseApiErrorDetail(text: string): string {
  const trimmed = (text || "").trim();
  if (!trimmed) return "";
  try {
    const parsed = JSON.parse(trimmed);
    if (parsed && typeof parsed === "object" && typeof parsed.detail === "string") {
      return parsed.detail.trim();
    }
  } catch {
    // no-op; fall through to raw text
  }
  return trimmed;
}

export async function register(payload: { username: string; password: string; name?: string }): Promise<AuthResponse> {
  const res = await fetch(`${BASE_URL}/auth/register`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const detailRaw = parseApiErrorDetail(await res.text());
    const detail = detailRaw.toLowerCase();
    if (detail.includes("username")) {
      throw new Error("Failed to register: username is already taken");
    }
    if (detail.includes("password")) {
      throw new Error("Failed to register: password must be at least 8 characters");
    }
    if (detailRaw) {
      throw new Error(`Failed to register: ${detailRaw}`);
    }
    throw new Error("Failed to register: please check your input and try again");
  }
  const json = await res.json();
  return json;
}

export async function login(payload: { username: string; password: string }): Promise<AuthResponse> {
  const res = await fetch(`${BASE_URL}/auth/login`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    if (res.status === 401) {
      throw new Error("Failed to login: invalid username or password");
    }
    const text = await res.text();
    throw new Error(`Failed to login: ${res.status} ${text}`);
  }
  return res.json();
}

export async function logout(): Promise<void> {
  const res = await fetch(`${BASE_URL}/auth/logout`, {
    method: "POST",
    headers: authHeaders(),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to logout: ${res.status} ${text}`);
  }
}

export async function fetchMe(): Promise<AuthUser> {
  const res = await fetch(`${BASE_URL}/auth/me`, {
    headers: authHeaders(),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to fetch me: ${res.status} ${text}`);
  }
  return res.json();
}

export async function updateMe(payload: { name: string }): Promise<AuthUser> {
  const res = await fetch(`${BASE_URL}/auth/me`, {
    method: "PATCH",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to update account: ${res.status} ${text}`);
  }
  return res.json();
}

export async function deleteMe(): Promise<void> {
  const res = await fetchWithTimeout(`${BASE_URL}/auth/me`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to delete account: ${res.status} ${text}`);
  }
}

// Request to GET/events with user id specified
export async function fetchEvents(userId: number): Promise<TimelineEvent[]> {
  const res = await fetch(`${BASE_URL}/events?user_id=${userId}`, {
    headers: authHeaders(),
  });
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
    headers: authHeaders({ "Content-Type": "application/json" }),
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
  const withTz = payload.timezone_offset_minutes == null
    ? { ...payload, timezone_offset_minutes: new Date().getTimezoneOffset() }
    : payload;
  const res = await fetch(`${BASE_URL}/events/ingest_text`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(withTz),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to ingest text: ${res.status} ${text}`);
  }
  return res.json();
}

export async function fetchInsights(userId: number, includeSuppressed = false): Promise<Insight[]> {
  const res = await fetch(
    `${BASE_URL}/insights?user_id=${userId}&include_suppressed=${includeSuppressed ? "true" : "false"}`,
    { headers: authHeaders() }
  );
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to fetch insights: ${res.status} ${text}`);
  }
  return res.json();
}

export async function fetchEventInsightLinks(
  userId: number,
  supportedOnly = true
): Promise<EventInsightLink[]> {
  const res = await fetch(
    `${BASE_URL}/events/insight_links?user_id=${userId}&supported_only=${supportedOnly ? "true" : "false"}`,
    { headers: authHeaders() }
  );
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to fetch event insight links: ${res.status} ${text}`);
  }
  return res.json();
}

export async function setInsightVerification(
  insightId: number,
  userId: number,
  verified: boolean
): Promise<{status: string; insight_id: number; user_id: number; item_id: number; symptom_id: number; verified: boolean}> {
  const res = await fetch(`${BASE_URL}/insights/${insightId}/verify`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ user_id: userId, verified }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to set insight verification: ${res.status} ${text}`);
  }
  return res.json();
}

export async function setInsightRejection(
  insightId: number,
  userId: number,
  rejected: boolean
): Promise<{status: string; insight_id: number; user_id: number; item_id: number; symptom_id: number; verified: boolean; rejected: boolean}> {
  const res = await fetch(`${BASE_URL}/insights/${insightId}/reject`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ user_id: userId, rejected }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to set insight rejection: ${res.status} ${text}`);
  }
  return res.json();
}

export async function updateEvent(
  eventType: "exposure" | "symptom",
  eventId: number,
  userId: number,
  payload: UpdateEventRequest
): Promise<EventMutationResponse> {
  const res = await fetch(`${BASE_URL}/events/${eventType}/${eventId}?user_id=${userId}`, {
    method: "PATCH",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to update event: ${res.status} ${text}`);
  }
  return res.json();
}

export async function deleteEvent(
  eventType: "exposure" | "symptom",
  eventId: number,
  userId: number
): Promise<EventMutationResponse> {
  const res = await fetchWithTimeout(`${BASE_URL}/events/${eventType}/${eventId}?user_id=${userId}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to delete event: ${res.status} ${text}`);
  }
  return res.json();
}

export async function fetchRecurringExposures(userId: number): Promise<RecurringExposureRule[]> {
  const res = await fetch(`${BASE_URL}/recurring_exposures?user_id=${userId}`, { headers: authHeaders() });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to fetch recurring exposures: ${res.status} ${text}`);
  }
  return res.json();
}

export async function createRecurringExposure(
  payload: CreateRecurringExposureRequest
): Promise<RecurringExposureRule> {
  const res = await fetch(`${BASE_URL}/recurring_exposures`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to create recurring exposure: ${res.status} ${text}`);
  }
  return res.json();
}

export async function patchRecurringExposure(
  ruleId: number,
  userId: number,
  payload: PatchRecurringExposureRequest
): Promise<RecurringExposureRule> {
  const res = await fetch(`${BASE_URL}/recurring_exposures/${ruleId}?user_id=${userId}`, {
    method: "PATCH",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to update recurring exposure: ${res.status} ${text}`);
  }
  return res.json();
}

export async function deleteRecurringExposure(ruleId: number, userId: number): Promise<{status: string; rule_id: number}> {
  const res = await fetch(`${BASE_URL}/recurring_exposures/${ruleId}?user_id=${userId}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to delete recurring exposure: ${res.status} ${text}`);
  }
  return res.json();
}
