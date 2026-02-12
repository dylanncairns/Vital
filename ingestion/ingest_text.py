# convert user text to extractable entries
# ie: "I ate yogurt at 8am" converted to exposure as event, yogurt as item, 8:00:00 as timestamp

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib import error, request

from api.db import get_connection
from api.repositories.raw_event_ingest import insert_raw_event_ingest
from ingestion.expand_exposure import expand_exposure_event
from ingestion.normalize_event import normalize_route
from ingestion.time_utils import to_utc_iso
from api.repositories.resolve import resolve_item_id, resolve_symptom_id

# normalized complete parse result to log for ingested input
@dataclass
class ParsedEvent:
    event_type: str
    timestamp: str | None
    time_range_start: str | None
    time_range_end: str | None
    time_confidence: str
    item_id: int | None
    route: str | None
    symptom_id: int | None
    severity: int | None

# Regex for identifying different parts of logged entries within text blurb
_EXPOSURE_VERBS = re.compile(r"\b(ate|eaten|drank|drink|used|apply|applied|took|take|smoked)\b", re.I)
_SEVERITY_RE = re.compile(r"\b(severity|sev|pain)\s*[:=]?\s*(\d)\b", re.I)
_RATING_RE = re.compile(r"\b(\d)\s*/\s*5\b")
_RATING_10_RE = re.compile(r"\b(10|[1-9])\s*/\s*10\b")
_SEVERITY_SUFFIX_RE = re.compile(r"\b(10|[1-9])\s*(?:/10)?\s*(?:severity|sev|pain)\b", re.I)
_TIME_AT_RE = re.compile(r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", re.I)
_RELATIVE_RE = re.compile(
    r"\b("
    r"yesterday morning|yesterday afternoon|yesterday evening|yesterday night|"
    r"this morning|this afternoon|this evening|this night|"
    r"last night|today|yesterday"
    r")\b",
    re.I,
)
_SYMPTOM_CUES_RE = re.compile(r"\b(felt|feel|had|have|having|tired|ache|pain|nausea|headache|stomachache)\b", re.I)
_DATE_TOKEN_RE = re.compile(
    r"\b("
    r"\d{4}-\d{2}-\d{2}|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"today|yesterday|last night|this morning|this afternoon|this evening"
    r")\b",
    re.I,
)

_LOCAL_TZ = datetime.now().astimezone().tzinfo or timezone.utc

# identify time from within text blob (safe for voice-to-text where user states timestamp)
def _parse_time(text: str) -> tuple[str | None, str | None, str | None]:
    # Parse time words as local user-time intent, then convert to UTC for storage.
    now = datetime.now().astimezone()
    match = _RELATIVE_RE.search(text)
    if match:
        token = match.group(1).lower()
        base = now
        if token.startswith("yesterday") or token == "last night":
            base = now - timedelta(days=1)

        # Fixed default times for dayparts (local), then stored as UTC.
        if "morning" in token:
            ts = base.replace(hour=9, minute=0, second=0, microsecond=0)
            return ts.astimezone(timezone.utc).isoformat(), None, None
        if "afternoon" in token:
            ts = base.replace(hour=15, minute=0, second=0, microsecond=0)
            return ts.astimezone(timezone.utc).isoformat(), None, None
        if "evening" in token:
            ts = base.replace(hour=19, minute=0, second=0, microsecond=0)
            return ts.astimezone(timezone.utc).isoformat(), None, None
        if "night" in token:
            ts = base.replace(hour=22, minute=0, second=0, microsecond=0)
            return ts.astimezone(timezone.utc).isoformat(), None, None
        if token == "last night":
            ts = base.replace(hour=22, minute=0, second=0, microsecond=0)
            return ts.astimezone(timezone.utc).isoformat(), None, None
        if token == "yesterday":
            ts = base.replace(hour=12, minute=0, second=0, microsecond=0)
            return ts.astimezone(timezone.utc).isoformat(), None, None
        if token == "today":
            ts = base.replace(hour=12, minute=0, second=0, microsecond=0)
            return ts.astimezone(timezone.utc).isoformat(), None, None
        else:
            ts = base.replace(hour=12, minute=0, second=0, microsecond=0)
            return ts.astimezone(timezone.utc).isoformat(), None, None

    match = _TIME_AT_RE.search(text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        meridiem = (match.group(3) or "").lower()
        if meridiem == "pm" and hour < 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0
        ts = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return ts.astimezone(timezone.utc).isoformat(), None, None

    return None, None, None


def _parse_severity(text: str) -> int | None:
    match = _SEVERITY_RE.search(text)
    if match:
        return int(match.group(2))
    match = _SEVERITY_SUFFIX_RE.search(text)
    if match:
        return int(match.group(1))
    match = _RATING_RE.search(text)
    if match:
        return int(match.group(1))
    match = _RATING_10_RE.search(text)
    if match:
        return int(match.group(1))
    return None


def _guess_event_type(text: str) -> str:
    if _EXPOSURE_VERBS.search(text):
        return "exposure"
    return "symptom"


def _infer_route(text: str) -> str:
    t = text.lower()
    if re.search(r"\b(smoked|vaped|inhaled|inhale)\b", t):
        return "inhalation"
    if re.search(r"\b(applied|apply|rubbed|cream|lotion|ointment|topical|used on)\b", t):
        return "topical"
    if re.search(r"\b(injected|inject|shot|iv)\b", t):
        return "injection"
    if re.search(r"\b(ate|eaten|drank|drink|took|take|swallowed|ingested)\b", t):
        return "ingestion"
    return "unknown"


def _split_exposure_items(text: str) -> list[str]:
    # Extract likely item phrase after ingestion verb and split list-like food entries.
    lower = text.lower()
    match = re.search(r"\b(ate|eaten|drank|drink|took|take|swallowed|ingested)\b", lower)
    if not match:
        return []
    segment = lower[match.end():]
    segment = re.split(r"\b(at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b", segment, maxsplit=1)[0]
    segment = re.split(r"\b(today|yesterday|last night|this morning|this afternoon|this evening)\b", segment, maxsplit=1)[0]
    segment = re.split(r"\b(felt|feel|had|have|having)\b", segment, maxsplit=1)[0]
    parts = re.split(r"\s*(?:,|&|\band\b|\bwith\b)\s*", segment)
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        cleaned = _clean_candidate_text(part)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out

# functions below handle long string inputs with ambiguity in event details or time

def _clean_candidate_text(text: str) -> str:
    value = text.strip().lower()
    value = _TIME_AT_RE.sub(" ", value)
    value = _RELATIVE_RE.sub(" ", value)
    value = re.sub(r"\b(i|we|then|and|at|about|this)\b", " ", value)
    value = re.sub(r"\b(was|were|is|am|been|being)\b", " ", value)
    value = re.sub(r"\b(ate|eaten|drank|drink|used|apply|applied|took|take|smoked|felt|feel|had|have|having)\b", " ", value)
    value = re.sub(r"\b(morning|afternoon|evening|night)\b", " ", value)
    value = re.sub(r"\b(10|[1-9])\s*/\s*10\b", " ", value)
    value = re.sub(r"\b([1-9])\s*/\s*5\b", " ", value)
    value = re.sub(r"\b(severity|sev|pain)\b", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" ,.;")
    # drop leading articles so "a cheeseburger" normalizes to "cheeseburger"
    value = re.sub(r"^(a|an|the)\s+", "", value)
    return value


def _looks_like_multi_event(text: str) -> bool:
    if not _EXPOSURE_VERBS.search(text):
        return False
    if not _SYMPTOM_CUES_RE.search(text):
        return False
    time_mentions = len(_TIME_AT_RE.findall(text)) + len(_RELATIVE_RE.findall(text))
    return time_mentions >= 2


def _force_event_type_from_text(event_type: str, text: str) -> str:
    has_exposure = _EXPOSURE_VERBS.search(text) is not None
    has_symptom = _SYMPTOM_CUES_RE.search(text) is not None
    if has_exposure and not has_symptom:
        return "exposure"
    if has_symptom and not has_exposure:
        return "symptom"
    return event_type


def _coerce_api_timestamp_to_today_if_time_only(text: str, timestamp: str | None) -> str | None:
    if not timestamp:
        return None
    if not _TIME_AT_RE.search(text):
        return timestamp
    if _DATE_TOKEN_RE.search(text):
        return timestamp
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return timestamp
    parsed_local = parsed if parsed.tzinfo is None else parsed.astimezone(_LOCAL_TZ)
    if parsed_local.tzinfo is None:
        parsed_local = parsed_local.replace(tzinfo=_LOCAL_TZ)
    now_local = datetime.now().astimezone(_LOCAL_TZ)
    coerced_local = parsed_local.replace(
        year=now_local.year,
        month=now_local.month,
        day=now_local.day,
    )
    return coerced_local.astimezone(timezone.utc).isoformat()


def _coerce_api_range_to_today_if_time_only(
    text: str,
    time_range_start: str | None,
    time_range_end: str | None,
) -> tuple[str | None, str | None]:
    if not time_range_start and not time_range_end:
        return time_range_start, time_range_end
    if not _TIME_AT_RE.search(text):
        return time_range_start, time_range_end
    if _DATE_TOKEN_RE.search(text):
        return time_range_start, time_range_end

    now_local = datetime.now().astimezone(_LOCAL_TZ)

    def _coerce(value: str | None) -> str | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
        parsed_local = parsed if parsed.tzinfo is None else parsed.astimezone(_LOCAL_TZ)
        if parsed_local.tzinfo is None:
            parsed_local = parsed_local.replace(tzinfo=_LOCAL_TZ)
        coerced_local = parsed_local.replace(
            year=now_local.year,
            month=now_local.month,
            day=now_local.day,
        )
        return coerced_local.astimezone(timezone.utc).isoformat()

    return _coerce(time_range_start), _coerce(time_range_end)


def _override_with_daypart_fixed_time(
    text: str,
    timestamp: str | None,
    time_range_start: str | None,
    time_range_end: str | None,
) -> tuple[str | None, str | None, str | None]:
    fixed_ts, _, _ = _parse_time(text)
    if fixed_ts and _RELATIVE_RE.search(text):
        return fixed_ts, None, None
    return timestamp, time_range_start, time_range_end


# First try to call external parser via OpenAI API
def parse_text_event(text: str) -> ParsedEvent | None:
    # One ParsedEvent cannot safely represent mixed exposure + symptom logs
    if _looks_like_multi_event(text):
        return None

    api_parsed = parse_with_api(text)
    if api_parsed is not None:
        return api_parsed

    # If fails use deterministic regex parser 
    rules_parsed = parse_with_rules(text)
    return rules_parsed


def parse_with_api(text: str) -> ParsedEvent | None:
    # API parsing only enabled when key env var is present
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    model = os.getenv("OPENAI_TEXT_PARSE_MODEL", "gpt-4o-mini")
    timeout = float(os.getenv("OPENAI_TEXT_PARSE_TIMEOUT_SECONDS", "1.5"))
    # Force strict structured output so mapping stays fast and predictable.
    schema = {
        "name": "parsed_event",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "event_type": {"type": "string", "enum": ["exposure", "symptom"]},
                "timestamp": {"type": ["string", "null"]},
                "time_range_start": {"type": ["string", "null"]},
                "time_range_end": {"type": ["string", "null"]},
                "time_confidence": {
                    "type": "string",
                    "enum": ["exact", "approx", "backfilled"],
                },
                "item_id": {"type": ["integer", "null"]},
                "item_name": {"type": ["string", "null"]},
                "route": {"type": ["string", "null"]},
                "symptom_id": {"type": ["integer", "null"]},
                "symptom_name": {"type": ["string", "null"]},
                "severity": {"type": ["integer", "null"]},
                "candidate": {"type": ["string", "null"]},
            },
            "required": [
                "event_type",
                "timestamp",
                "time_range_start",
                "time_range_end",
                "time_confidence",
                "item_id",
                "item_name",
                "route",
                "symptom_id",
                "symptom_name",
                "severity",
                "candidate",
            ],
        },
    }
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 220,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Extract a single health event from user text. "
                    "Use event_type exposure for consumed/used items; symptom for felt symptoms. "
                    "Return only schema fields."
                ),
            },
            {"role": "user", "content": text},
        ],
        "response_format": {"type": "json_schema", "json_schema": schema},
    }
    req = request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
            content = raw["choices"][0]["message"]["content"]
            data = json.loads(content)
    except (error.URLError, error.HTTPError, TimeoutError, KeyError, IndexError, TypeError, json.JSONDecodeError, ValueError):
        # If API parse fails for any reason, fall back to regex parser
        return None

    event_type = data.get("event_type")
    if event_type not in {"exposure", "symptom"}:
        return None
    event_type = _force_event_type_from_text(event_type, text)

    timestamp = _coerce_api_timestamp_to_today_if_time_only(text, data.get("timestamp"))
    time_range_start = data.get("time_range_start")
    time_range_end = data.get("time_range_end")
    time_range_start, time_range_end = _coerce_api_range_to_today_if_time_only(
        text,
        time_range_start,
        time_range_end,
    )
    timestamp = to_utc_iso(timestamp, strict=False)
    time_range_start = to_utc_iso(time_range_start, strict=False)
    time_range_end = to_utc_iso(time_range_end, strict=False)
    timestamp, time_range_start, time_range_end = _override_with_daypart_fixed_time(
        text,
        timestamp,
        time_range_start,
        time_range_end,
    )
    time_confidence = data.get("time_confidence")
    if time_confidence not in {"exact", "approx", "backfilled"}:
        time_confidence = "exact" if timestamp else "approx"

    item_id = data.get("item_id")
    route = data.get("route")
    symptom_id = data.get("symptom_id")
    severity = data.get("severity")

    if event_type == "exposure":
        if item_id is None:
            item_name = data.get("item_name") or data.get("candidate")
            if isinstance(item_name, str) and item_name.strip():
                item_id = resolve_item_id(_clean_candidate_text(item_name))
        if item_id is None:
            return None
        if isinstance(route, str):
            route = normalize_route(route, strict=False)
        else:
            route = "unknown"
        if route == "unknown":
            route = normalize_route(_infer_route(text), strict=False)
        symptom_id = None
        severity = None
    else:
        if symptom_id is None:
            symptom_name = data.get("symptom_name") or data.get("candidate")
            if isinstance(symptom_name, str) and symptom_name.strip():
                symptom_id = resolve_symptom_id(_clean_candidate_text(symptom_name))
        if symptom_id is None:
            return None
        item_id = None
        route = None
        if severity is not None:
            try:
                severity = int(severity)
            except (TypeError, ValueError):
                severity = None

    return ParsedEvent(
        event_type=event_type,
        timestamp=timestamp,
        time_range_start=time_range_start,
        time_range_end=time_range_end,
        time_confidence=time_confidence,
        item_id=item_id,
        route=route,
        symptom_id=symptom_id,
        severity=severity,
    )


def parse_with_rules(text: str) -> ParsedEvent | None:
    event_type = _guess_event_type(text)
    timestamp, range_start, range_end = _parse_time(text)
    time_confidence = "approx" if not timestamp else "exact"
    severity = _parse_severity(text) if event_type == "symptom" else None

    item_id = None
    symptom_id = None
    tokens = [t for t in re.split(r"[,\.;]", text) if t.strip()]
    if not tokens:
        return None
    if event_type == "exposure":
        candidate = _clean_candidate_text(tokens[0])
        if not candidate:
            return None
        item_id = resolve_item_id(candidate)
        route = _infer_route(text)
    else:
        candidate = _clean_candidate_text(tokens[0])
        if not candidate:
            return None
        symptom_id = resolve_symptom_id(candidate)
        route = None

    return ParsedEvent(
        event_type=event_type,
        timestamp=timestamp,
        time_range_start=range_start,
        time_range_end=range_end,
        time_confidence=time_confidence,
        item_id=item_id,
        route=route,
        symptom_id=symptom_id,
        severity=severity,
    )

# DB writing logic after parsing
def ingest_text_event(user_id: int, raw_text: str) -> dict:
    parsed = parse_text_event(raw_text)
    if parsed is None:
        insert_raw_event_ingest(user_id, raw_text, "failed", "unparsed")
        return {"status": "queued", "resolution": "pending"}

    conn = get_connection()
    now = datetime.now(tz=timezone.utc).isoformat()
    if parsed.event_type == "exposure":
        route = parsed.route or _infer_route(raw_text)
        route = normalize_route(route, strict=False)
        split_names = _split_exposure_items(raw_text) if route == "ingestion" else []
        item_ids = [resolve_item_id(name) for name in split_names] if len(split_names) > 1 else [parsed.item_id]
        for item_id in item_ids:
            if item_id is None:
                continue
            cursor = conn.execute(
                """
                INSERT INTO exposure_events (
                    user_id, item_id, timestamp, time_range_start, time_range_end,
                    time_confidence, ingested_at, raw_text, route
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    item_id,
                    parsed.timestamp,
                    parsed.time_range_start,
                    parsed.time_range_end,
                    parsed.time_confidence,
                    now,
                    raw_text,
                    route,
                ),
            )
            expand_exposure_event(cursor.lastrowid, conn=conn)
    else:
        conn.execute(
            """
            INSERT INTO symptom_events (
                user_id, symptom_id, timestamp, time_range_start, time_range_end,
                time_confidence, ingested_at, raw_text, severity
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                parsed.symptom_id,
                parsed.timestamp,
                parsed.time_range_start,
                parsed.time_range_end,
                parsed.time_confidence,
                now,
                raw_text,
                parsed.severity,
            ),
        )
    conn.commit()
    conn.close()
    return {"status": "ingested", "event_type": parsed.event_type}
