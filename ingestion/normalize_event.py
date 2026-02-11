# validate and normalize the payload sent to /events
# transform into structured dict before attempt to insert into DB
# missing/invalid required fields raise NormalizationError

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol, TypedDict

from api.repositories.resolve import resolve_item_id, resolve_symptom_id

# lets main catch specific validation failures and separates from DB/runtime errors
class NormalizationError(ValueError):
    pass

# expected input shape from EventIn to reduce ambiguity
class EventPayload(Protocol):
    event_type: str
    user_id: int
    timestamp: str | None
    time_range_start: str | None
    time_range_end: str | None
    time_confidence: str | None
    raw_text: str | None
    item_id: int | None
    item_name: str | None
    route: str | None
    symptom_id: int | None
    symptom_name: str | None
    severity: int | None

# output contract to normalize shape of final return value
class NormalizedEvent(TypedDict):
    event_type: str
    user_id: int
    timestamp: str | None
    time_range_start: str | None
    time_range_end: str | None
    time_confidence: str
    raw_text: str | None
    item_id: int | None
    route: str | None
    symptom_id: int | None
    severity: int | None
    ingested_at: str

# normalize timestamps, handling user entry of "3pm" which refers to their local time
def _to_utc_iso(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise NormalizationError(f"invalid datetime format: {value}") from exc
    if parsed.tzinfo is None:
        local_tz = datetime.now().astimezone().tzinfo
        parsed = parsed.replace(tzinfo=local_tz)
    return parsed.astimezone(timezone.utc).isoformat()


def normalize_event(payload: EventPayload) -> NormalizedEvent:
    # require either exact timestamp or a time range & set confidence
    timestamp_value = None
    if payload.timestamp:
        if not payload.timestamp.strip():
            raise NormalizationError("timestamp cannot be empty")
        timestamp_value = _to_utc_iso(payload.timestamp)
    else:
        if not payload.time_range_start or not payload.time_range_end:
            raise NormalizationError("timestamp or time_range_start/time_range_end is required")
    time_range_start = _to_utc_iso(payload.time_range_start)
    time_range_end = _to_utc_iso(payload.time_range_end)
    time_confidence = payload.time_confidence
    if time_confidence is None:
        time_confidence = "exact" if timestamp_value else "approx"

    item_id = payload.item_id
    symptom_id = payload.symptom_id

    if payload.event_type == "exposure":
        # exposure must resolve to an item and include route
        if item_id is None:
            if payload.item_name:
                item_id = resolve_item_id(payload.item_name)
            else:
                raise NormalizationError("exposure event requires item_id or item_name")
        if payload.route is None or not payload.route.strip():
            raise NormalizationError("exposure event requires route")
        # keep mutually exclusive fields clean
        symptom_id = None
    else:
        # symptom must resolve to a symptom id either directly or by name
        if symptom_id is None:
            if payload.symptom_name:
                symptom_id = resolve_symptom_id(payload.symptom_name)
            else:
                raise NormalizationError("symptom event requires symptom_id or symptom_name")
        # keep mutually exclusive fields clean
        item_id = None

    # output normalized dict payload for db insert
    return {
        "event_type": payload.event_type,
        "user_id": payload.user_id,
        "timestamp": timestamp_value,
        "time_range_start": time_range_start,
        "time_range_end": time_range_end,
        "time_confidence": time_confidence,
        "raw_text": payload.raw_text,
        "item_id": item_id,
        "route": payload.route,
        "symptom_id": symptom_id,
        "severity": payload.severity,
        "ingested_at": datetime.now(tz=timezone.utc).isoformat(),
    }
