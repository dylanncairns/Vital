import { TimelineEvent, CreateEventRequest } from "../models/events";

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
export async function createEvent(payload: CreateEventRequest): Promise<void> {
  const res = await fetch(`${BASE_URL}/events`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error("Failed to create event");
}
