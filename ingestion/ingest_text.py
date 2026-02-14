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
from api.repositories.jobs import (
    JOB_RECOMPUTE_CANDIDATE,
    enqueue_background_job,
    maybe_enqueue_model_retrain,
)
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
    source_text: str | None = None

# Regex for identifying different parts of logged entries within text blurb
_EXPOSURE_VERBS = re.compile(r"\b(ate|eaten|drank|drink|used|apply|applied|took|take|smoked)\b", re.I)
_CONTEXT_EXPOSURE_RE = re.compile(
    r"\b(went\s+to|visited|was\s+at|at\s+the|club|party|bar|concert|festival|crowd)\b",
    re.I,
)
_SEVERITY_RE = re.compile(r"\b(severity|sev|pain)\s*[:=]?\s*(\d)\b", re.I)
_RATING_RE = re.compile(r"\b(\d)\s*/\s*5\b")
_RATING_10_RE = re.compile(r"\b(10|[1-9])\s*/\s*10\b")
_SEVERITY_SUFFIX_RE = re.compile(r"\b(10|[1-9])\s*(?:/10)?\s*(?:severity|sev|pain)\b", re.I)
_TIME_AT_RE = re.compile(r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", re.I)
_TIME_RANGE_RE = re.compile(
    r"\b(?:from\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*[-–to]+\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b",
    re.I,
)
_RELATIVE_RE = re.compile(
    r"\b("
    r"yesterday morning|yesterday afternoon|yesterday evening|yesterday night|"
    r"yesterday breakfast|yesterday lunch|yesterday dinner|"
    r"this morning|this afternoon|this evening|this night|"
    r"this breakfast|this lunch|this dinner|"
    r"last night|tonight|today|yesterday|"
    r"breakfast|lunch|dinner|"
    r"morning|afternoon|evening|night"
    r")\b",
    re.I,
)
_DAYS_AGO_RE = re.compile(
    r"\b(?:(\d+)|one|two|three|four|five|six|seven)\s+days?\s+ago\b",
    re.I,
)
_SYMPTOM_CUES_RE = re.compile(r"\b(felt|feel|tired|ache|pain|nausea|headache|stomachache)\b", re.I)
_HAD_OBJECT_RE = re.compile(r"\bhad\s+(?:a|an|some|the)?\s*[a-z]", re.I)
_MEAL_CONTEXT_RE = re.compile(r"\b(breakfast|lunch|dinner|ate|eaten|drank|drink)\b", re.I)
_DATE_TOKEN_RE = re.compile(
    r"\b("
    r"\d{4}-\d{2}-\d{2}|"
    r"(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\s+\d{1,2}(?:,\s*\d{4})?|"
    r"\d{1,2}/\d{1,2}(?:/\d{2,4})?|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"today|yesterday|last night|tonight|this morning|this afternoon|this evening|"
    r"breakfast|lunch|dinner|this breakfast|this lunch|this dinner|"
    r"yesterday breakfast|yesterday lunch|yesterday dinner"
    r")\b",
    re.I,
)

_LOCAL_TZ = datetime.now().astimezone().tzinfo or timezone.utc
_LOW_SIGNAL_TOKENS = {
    "in",
    "the",
    "in the",
    "for",
    "to",
    "of",
    "on",
    "at",
    "from",
    "by",
    "with",
    "and",
    "or",
    "then",
    "after",
    "before",
    "during",
}
_COMMON_SYMPTOM_TERMS_RE = re.compile(
    r"\b("
    r"headache|migraine|nausea|vomit|vomiting|stomachache|stomach pain|"
    r"high blood pressure|blood pressure|hypertension|"
    r"acne|rash|itch|itchy|hives|fatigue|tired|brain fog|"
    r"diarrhea|constipation|bloat|bloating|cramp|cramps|"
    r"dizzy|dizziness|anxiety|insomnia|fever|cough|sore throat"
    r")\b",
    re.I,
)

_ABS_MONTH_DATE_RE = re.compile(
    r"\b(?:on\s+)?("
    r"(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\s+\d{1,2}(?:,\s*\d{4})?"
    r"|"
    r"\d{1,2}/\d{1,2}(?:/\d{2,4})?"
    r")\b",
    re.I,
)


def _parse_days_ago(text: str) -> int | None:
    match = _DAYS_AGO_RE.search(text)
    if not match:
        return None
    raw = match.group(0).lower()
    digit = match.group(1)
    if digit is not None:
        try:
            parsed = int(digit)
            return parsed if parsed >= 0 else None
        except ValueError:
            return None
    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
    }
    for token, days in words.items():
        if token in raw:
            return days
    return None

# identify time from within text blob (safe for voice-to-text where user states timestamp)
def _parse_time(text: str) -> tuple[str | None, str | None, str | None]:
    # Parse time words as local user-time intent, then convert to UTC for storage.
    now = datetime.now().astimezone()
    lower = text.lower()

    days_ago = _parse_days_ago(text)
    absolute_date: datetime | None = None
    abs_match = _ABS_MONTH_DATE_RE.search(text)
    if abs_match:
        raw_date = abs_match.group(1).strip()
        has_explicit_year = bool(re.search(r"\b\d{4}\b", raw_date)) or bool(
            re.search(r"\d{1,2}/\d{1,2}/\d{2,4}$", raw_date)
        )
        if not has_explicit_year:
            if "/" in raw_date:
                raw_date = f"{raw_date}/{now.year}"
            else:
                raw_date = f"{raw_date}, {now.year}"
        candidate_formats = (
            "%B %d, %Y",
            "%b %d, %Y",
            "%B %d %Y",
            "%b %d %Y",
            "%m/%d/%Y",
            "%m/%d/%y",
        )
        for fmt in candidate_formats:
            try:
                parsed = datetime.strptime(raw_date, fmt)
            except ValueError:
                continue
            absolute_date = parsed.replace(tzinfo=now.tzinfo)
            break

    range_match = _TIME_RANGE_RE.search(text)
    if range_match:
        shour = int(range_match.group(1))
        sminute = int(range_match.group(2) or 0)
        smeridiem = (range_match.group(3) or "").lower()
        ehour = int(range_match.group(4))
        eminute = int(range_match.group(5) or 0)
        emeridiem = (range_match.group(6) or "").lower()

        if smeridiem == "pm" and shour < 12:
            shour += 12
        if smeridiem == "am" and shour == 12:
            shour = 0
        if emeridiem == "pm" and ehour < 12:
            ehour += 12
        if emeridiem == "am" and ehour == 12:
            ehour = 0

        if not smeridiem and not emeridiem:
            # Default daytime ranges to 24h style assumptions.
            if shour <= 6 and ehour <= 12:
                shour += 12
                ehour += 12
        elif not smeridiem and emeridiem:
            if emeridiem == "pm" and shour < 12:
                shour += 12
            if emeridiem == "am" and shour == 12:
                shour = 0
        elif smeridiem and not emeridiem:
            if smeridiem == "pm" and ehour < 12:
                ehour += 12
            if smeridiem == "am" and ehour == 12:
                ehour = 0

        base = absolute_date or now
        if days_ago is not None:
            base = now - timedelta(days=days_ago)
        rel = _RELATIVE_RE.search(text)
        if rel:
            token = rel.group(1).lower()
            if token.startswith("yesterday") or token == "last night" or token == "yesterday":
                base = now - timedelta(days=1)
        start_ts = base.replace(hour=shour % 24, minute=sminute, second=0, microsecond=0)
        end_ts = base.replace(hour=ehour % 24, minute=eminute, second=0, microsecond=0)
        if end_ts < start_ts:
            end_ts = end_ts + timedelta(days=1)
        return None, start_ts.astimezone(timezone.utc).isoformat(), end_ts.astimezone(timezone.utc).isoformat()

    match = _RELATIVE_RE.search(text)
    if match:
        token = match.group(1).lower()
        base = absolute_date or now
        if days_ago is not None:
            base = now - timedelta(days=days_ago)
        if token.startswith("yesterday") or token == "last night":
            base = now - timedelta(days=1)

        # Fixed default times for dayparts (local), then stored as UTC.
        if "morning" in token:
            ts = base.replace(hour=9, minute=0, second=0, microsecond=0)
            return ts.astimezone(timezone.utc).isoformat(), None, None
        if "breakfast" in token:
            ts = base.replace(hour=8, minute=0, second=0, microsecond=0)
            return ts.astimezone(timezone.utc).isoformat(), None, None
        if "lunch" in token:
            ts = base.replace(hour=13, minute=0, second=0, microsecond=0)
            return ts.astimezone(timezone.utc).isoformat(), None, None
        if "dinner" in token:
            ts = base.replace(hour=19, minute=0, second=0, microsecond=0)
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
        if token == "tonight":
            ts = base.replace(hour=21, minute=0, second=0, microsecond=0)
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

    if absolute_date is not None:
        ts = absolute_date.replace(hour=12, minute=0, second=0, microsecond=0)
        return ts.astimezone(timezone.utc).isoformat(), None, None

    if days_ago is not None:
        base = now - timedelta(days=days_ago)
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

    if re.search(r"\b(got home|home from work|after work)\b", lower):
        ts = now.replace(hour=18, minute=30, second=0, microsecond=0)
        return ts.astimezone(timezone.utc).isoformat(), None, None

    return None, None, None


def _pick_source_segment(full_text: str, candidate: str, event_type: str) -> str:
    text = (full_text or "").strip()
    if not text:
        return text
    candidate_norm = " ".join((candidate or "").strip().lower().split())
    segments = _split_into_segments(text)
    if not segments:
        segments = [text]
    if candidate_norm:
        for segment in segments:
            if candidate_norm in " ".join(segment.lower().split()):
                return segment
    # Fallback by event signal.
    for segment in segments:
        low = segment.lower()
        if event_type == "symptom" and (_SYMPTOM_CUES_RE.search(low) or _COMMON_SYMPTOM_TERMS_RE.search(low)):
            return segment
        if event_type == "exposure" and (_EXPOSURE_VERBS.search(low) or _CONTEXT_EXPOSURE_RE.search(low)):
            return segment
    return segments[0]


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
    lower = text.lower()
    has_exposure = _EXPOSURE_VERBS.search(lower) is not None
    has_context_exposure = _CONTEXT_EXPOSURE_RE.search(lower) is not None
    has_symptom_cue = _SYMPTOM_CUES_RE.search(lower) is not None
    has_common_symptom = _COMMON_SYMPTOM_TERMS_RE.search(lower) is not None
    has_had_object = _HAD_OBJECT_RE.search(lower) is not None
    has_meal_marker = re.search(r"\b(breakfast|lunch|dinner)\b", lower) is not None

    # Prefer symptom when symptom signal is explicit and no strong exposure verb exists.
    if (has_symptom_cue or has_common_symptom) and not has_exposure and not has_context_exposure:
        return "symptom"
    if has_had_object and not has_common_symptom and not has_symptom_cue:
        return "exposure"
    if has_had_object and has_meal_marker and not has_common_symptom:
        return "exposure"

    if _EXPOSURE_VERBS.search(text):
        return "exposure"
    if _MEAL_CONTEXT_RE.search(text) and re.search(r"\b(had|have|having)\b", text, re.I) and not (
        has_symptom_cue or has_common_symptom
    ):
        return "exposure"
    if _CONTEXT_EXPOSURE_RE.search(text):
        return "exposure"
    return "symptom"


def _infer_route(text: str) -> str:
    t = text.lower()
    if re.search(r"\b(near|nearby|around|exposed|exposure|environment|air quality|pollution|pollen|dust|mold|secondhand|second hand|passive smoke)\b", t):
        return "proximity_environment"
    if re.search(r"\b(went\s+to|visited|was\s+at|club|party|bar|concert|festival|crowd)\b", t):
        return "proximity_environment"
    if re.search(r"\b(smoked|vaped|inhaled|inhale)\b", t):
        return "inhalation"
    if re.search(r"\b(applied|apply|rubbed|cream|lotion|ointment|topical|used on)\b", t):
        return "topical"
    if re.search(r"\b(injected|inject|shot|iv)\b", t):
        return "injection"
    if _MEAL_CONTEXT_RE.search(t) and re.search(r"\b(had|have|having)\b", t):
        return "ingestion"
    if re.search(r"\b(ate|eaten|drank|drink|took|take|swallowed|ingested)\b", t):
        return "ingestion"
    if _HAD_OBJECT_RE.search(t) and _COMMON_SYMPTOM_TERMS_RE.search(t) is None:
        return "ingestion"
    return "other"


def _split_exposure_items(text: str) -> list[str]:
    # Extract likely item phrase after ingestion verb and split list-like food entries.
    lower = text.lower()
    match = re.search(r"\b(ate|eaten|drank|drink|took|take|swallowed|ingested|had|have|having)\b", lower)
    if not match:
        # Handle non-ingestion context exposures ("went to the club", "at a party")
        context_match = re.search(r"\b(?:went\s+to|visited|was\s+at)\s+(.*)$", lower)
        if context_match:
            segment = context_match.group(1)
        else:
            # Fallback for noun lists without explicit verbs (e.g., "chicken and rice").
            segment = lower
        segment = re.split(
            r"\b(today|yesterday|last night|this morning|this afternoon|this evening|breakfast|lunch|dinner|this breakfast|this lunch|this dinner|yesterday breakfast|yesterday lunch|yesterday dinner)\b",
            segment,
            maxsplit=1,
        )[0]
        segment = re.sub(r"\b(?:i|we)\s+(?:had|have|having)\b", " and ", segment)
        segment = re.split(r"\b(felt|feel)\b", segment, maxsplit=1)[0]
        parts = re.split(r"\s*(?:,|&|\band\b|\bwith\b)\s*", segment)
        out: list[str] = []
        seen: set[str] = set()
        for part in parts:
            cleaned = _clean_candidate_text(part)
            if not cleaned or _is_low_signal_candidate(cleaned) or cleaned in seen:
                continue
            seen.add(cleaned)
            out.append(cleaned)
        return out
    segment = lower[match.end():]
    segment = re.sub(r"\b(?:i|we)\s+(?:had|have|having)\b", " and ", segment)
    segment = re.split(r"\b(at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b", segment, maxsplit=1)[0]
    segment = re.split(
        r"\b(today|yesterday|last night|this morning|this afternoon|this evening|breakfast|lunch|dinner|this breakfast|this lunch|this dinner|yesterday breakfast|yesterday lunch|yesterday dinner)\b",
        segment,
        maxsplit=1,
    )[0]
    segment = re.split(r"\b(felt|feel)\b", segment, maxsplit=1)[0]
    parts = re.split(r"\s*(?:,|&|\band\b|\bwith\b)\s*", segment)
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        cleaned = _clean_candidate_text(part)
        if not cleaned or _is_low_signal_candidate(cleaned) or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out

# functions below handle long string inputs with ambiguity in event details or time

def _clean_candidate_text(text: str) -> str:
    value = text.strip().lower()
    # Remove conversational scaffolding while preserving medical terms like "testosterone".
    value = re.sub(r"\b(?:did|do|done)\s+(?:that\s+)?test\b", " ", value)
    value = re.sub(r"\bwent\s+and\b", " ", value)
    # Remove quantity/filler phrases so item resolution keeps the core exposure entity.
    value = re.sub(r"\ba\s+lot\s+of\b", " ", value)
    value = re.sub(r"\blots\s+of\b", " ", value)
    value = re.sub(r"\ba\s+bunch\s+of\b", " ", value)
    value = re.sub(r"\b(plenty|many|much|some)\s+of\b", " ", value)
    value = re.sub(r"\b\d{1,2}\s*-\s*\d{1,2}\b", " ", value)
    value = _TIME_AT_RE.sub(" ", value)
    value = _RELATIVE_RE.sub(" ", value)
    value = re.sub(r"\b(i|we|also|then|so|when|and|at|about|this|that|in|for|on|to|of|from|by|went|visit|visited|did|do|done|after|before|during|got|my)\b", " ", value)
    value = re.sub(r"\b(was|were|is|am|been|being)\b", " ", value)
    value = re.sub(r"\b(ate|eaten|drank|drink|used|apply|applied|took|take|smoked|felt|feel|had|have|having)\b", " ", value)
    value = re.sub(r"\b(morning|afternoon|evening|night|breakfast|lunch|dinner)\b", " ", value)
    value = re.sub(r"\b(10|[1-9])\s*/\s*10\b", " ", value)
    value = re.sub(r"\b([1-9])\s*/\s*5\b", " ", value)
    value = re.sub(r"\b(severity|sev|pain)\b", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" ,.;")
    # drop leading articles so "a cheeseburger" normalizes to "cheeseburger"
    value = re.sub(r"^(a|an|the)\s+", "", value)
    # Drop remaining leading quantity tokens ("lot pizza" -> "pizza").
    value = re.sub(r"^(lot|lots|many|much|some)\s+", "", value)
    # Drop discourse markers that can lead item phrases.
    value = re.sub(r"^(also|just|new)\s+", "", value)
    # Drop conversational trailing particles.
    value = re.sub(r"\s+(a|an|the)$", "", value)
    value = re.sub(r"\s+", " ", value).strip(" ,.;")
    return value


def _normalize_symptom_candidate(candidate: str) -> str:
    value = " ".join(candidate.strip().lower().split())
    if not value:
        return value
    value = re.sub(r"^(also|just|really|very)\s+", "", value)
    value = re.sub(r"^(some|new|a|an|the)\s+", "", value)
    term_match = _COMMON_SYMPTOM_TERMS_RE.search(value)
    if term_match:
        return term_match.group(0).strip().lower()
    return value


def _is_low_signal_candidate(value: str) -> bool:
    normalized = " ".join(value.strip().lower().split())
    if not normalized:
        return True
    if normalized in _LOW_SIGNAL_TOKENS:
        return True
    tokens = normalized.split()
    if all(token in _LOW_SIGNAL_TOKENS for token in tokens):
        return True
    return False


def _is_valid_symptom_candidate(candidate: str, source_text: str) -> bool:
    normalized = _normalize_symptom_candidate(candidate)
    if _is_low_signal_candidate(normalized):
        return False
    # If the candidate itself looks like an exposure phrase, never create symptom rows from it.
    if _EXPOSURE_VERBS.search(normalized) or _CONTEXT_EXPOSURE_RE.search(normalized):
        return False
    # Enforce signal for symptom classification: either symptom cues in sentence or known symptom term.
    if _SYMPTOM_CUES_RE.search(source_text):
        return True
    if _COMMON_SYMPTOM_TERMS_RE.search(normalized):
        return True
    return False


def _looks_like_multi_event(text: str) -> bool:
    if not _EXPOSURE_VERBS.search(text):
        return False
    if _SYMPTOM_CUES_RE.search(text) is None and _COMMON_SYMPTOM_TERMS_RE.search(text) is None:
        return False
    time_mentions = len(_TIME_AT_RE.findall(text)) + len(_RELATIVE_RE.findall(text))
    return time_mentions >= 2


def _force_event_type_from_text(event_type: str, text: str) -> str:
    has_exposure = (
        _EXPOSURE_VERBS.search(text) is not None
        or _CONTEXT_EXPOSURE_RE.search(text) is not None
        or (_MEAL_CONTEXT_RE.search(text) is not None and re.search(r"\b(had|have|having)\b", text, re.I) is not None)
    )
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

    rules_parsed = parse_with_rules(text)
    if rules_parsed is not None:
        return rules_parsed

    # Fallback to API parser only when deterministic parser fails.
    api_parsed = parse_with_api(text)
    if api_parsed is not None:
        return api_parsed
    return None


def _split_into_segments(text: str) -> list[str]:
    # Split long blurbs into sentence-like parts while preserving useful fragments.
    raw_segments = re.split(r"[\n\r]+|(?<=[\.\!\?;])\s+", text)
    segments: list[str] = []
    for segment in raw_segments:
        cleaned = segment.strip()
        if len(cleaned) < 3:
            continue
        segments.append(cleaned)
    return segments


def parse_text_events(text: str) -> list[ParsedEvent]:
    api_events = parse_with_api_events(text)
    if api_events:
        return api_events

    segments = _split_into_segments(text)
    if not segments:
        segments = [text.strip()]
    parsed_events: list[ParsedEvent] = []
    seen_keys: set[tuple[str, str | None, str | None, str | None, int | None, int | None, str | None]] = set()

    def _append_if_new(parsed: ParsedEvent) -> None:
        key = (
            parsed.event_type,
            parsed.timestamp,
            parsed.time_range_start,
            parsed.time_range_end,
            parsed.item_id,
            parsed.symptom_id,
            parsed.route,
        )
        if key in seen_keys:
            return
        seen_keys.add(key)
        parsed_events.append(parsed)

    for segment in segments:
        parsed = parse_text_event(segment)
        segment_lower = segment.lower()
        has_multi_exposure_clause = re.search(
            r"\band\s+for\s+(?:breakfast|lunch|dinner)\b",
            segment_lower,
        ) is not None
        has_mixed_signal = (
            _EXPOSURE_VERBS.search(segment_lower) is not None
            and (_SYMPTOM_CUES_RE.search(segment_lower) is not None or _COMMON_SYMPTOM_TERMS_RE.search(segment_lower) is not None)
        )

        if parsed is not None:
            # For multi-exposure clauses, prefer clause parsing over whole-segment parse to avoid blended artifacts.
            if not has_multi_exposure_clause:
                _append_if_new(parsed)
            if not has_mixed_signal and not has_multi_exposure_clause:
                continue
        # Fallback for mixed blurbs in one sentence: split into clause-like chunks.
        clauses = [
            part.strip()
            for part in re.split(
                r"\s*(?:,|\band then\b|\bthen\b|\band after\b|\bafter\b|\band\b|\band for (?:breakfast|lunch|dinner)\b)\s*",
                segment,
                flags=re.I,
            )
            if part.strip()
        ]
        for clause in clauses:
            clause_parsed = parse_text_event(clause)
            if clause_parsed is not None:
                _append_if_new(clause_parsed)
    return parsed_events


def _api_event_to_parsed_event(entry: dict, full_text: str) -> ParsedEvent | None:
    source_text = str(entry.get("source_text") or "").strip()
    candidate = str(entry.get("candidate") or "").strip()
    event_type = entry.get("event_type")
    if event_type not in {"exposure", "symptom"}:
        return None
    if not source_text:
        source_text = _pick_source_segment(full_text, candidate, str(event_type))

    source_text = source_text.strip()
    cleaned_candidate = _clean_candidate_text(candidate) if candidate else ""
    timestamp, time_range_start, time_range_end = _parse_time(source_text)
    time_confidence = entry.get("time_confidence")
    if time_confidence not in {"exact", "approx", "backfilled"}:
        time_confidence = "exact" if timestamp else "approx"

    if event_type == "exposure":
        item_id = None
        source_for_split = source_text
        if candidate and candidate.lower() not in source_text.lower():
            source_for_split = f"{source_text} {candidate}".strip()

        item_tokens = _split_exposure_items(source_for_split)
        if not item_tokens and cleaned_candidate:
            item_tokens = _split_exposure_items(cleaned_candidate) or [cleaned_candidate]
        for token in item_tokens:
            if not token or _is_low_signal_candidate(token):
                continue
            item_id = resolve_item_id(token)
            if item_id is not None:
                break
        if item_id is None:
            return None
        route = entry.get("route")
        if isinstance(route, str):
            route = normalize_route(route, strict=False)
        else:
            route = normalize_route(_infer_route(source_text), strict=False)
        if route in {"unknown", "other"}:
            route = normalize_route(_infer_route(source_text), strict=False)
        return ParsedEvent(
            event_type="exposure",
            timestamp=timestamp,
            time_range_start=time_range_start,
            time_range_end=time_range_end,
            time_confidence=time_confidence,
            item_id=item_id,
            route=route,
            symptom_id=None,
            severity=None,
            source_text=source_for_split,
        )

    symptom_id = None
    normalized_symptom = _normalize_symptom_candidate(cleaned_candidate)
    if normalized_symptom and _is_valid_symptom_candidate(normalized_symptom, source_text):
        symptom_id = resolve_symptom_id(normalized_symptom)
    if symptom_id is None:
        return None
    severity = entry.get("severity")
    if severity is not None:
        try:
            severity = int(severity)
        except (TypeError, ValueError):
            severity = None
    return ParsedEvent(
        event_type="symptom",
        timestamp=timestamp,
        time_range_start=time_range_start,
        time_range_end=time_range_end,
        time_confidence=time_confidence,
        item_id=None,
        route=None,
        symptom_id=symptom_id,
        severity=severity,
        source_text=source_text,
    )


def parse_with_api_events(text: str) -> list[ParsedEvent]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return []
    # Prefer API parsing on long/multi-clause blurbs where deterministic parser is weakest.
    if len(text.strip()) < 40:
        return []

    model = os.getenv("OPENAI_TEXT_PARSE_MODEL", "gpt-4o-mini")
    timeout = float(os.getenv("OPENAI_TEXT_PARSE_TIMEOUT_SECONDS", "2.5"))
    schema = {
        "name": "parsed_events",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "event_count": {"type": "integer", "minimum": 0},
                "events": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "source_text": {"type": "string"},
                            "event_type": {"type": "string", "enum": ["exposure", "symptom"]},
                            "candidate": {"type": ["string", "null"]},
                            "route": {
                                "type": ["string", "null"],
                                "enum": [None, "ingestion", "dermal", "inhalation", "injection", "proximity_environment", "other"],
                            },
                            "severity": {"type": ["integer", "null"]},
                            "time_confidence": {"type": ["string", "null"], "enum": [None, "exact", "approx", "backfilled"]},
                        },
                        "required": ["source_text", "event_type", "candidate", "route", "severity", "time_confidence"],
                    },
                }
            },
            "required": ["event_count", "events"],
        },
    }
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 1200,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a clinical event extraction engine. "
                    "Extract atomic events from user text. "
                    "Rules: "
                    "1) Return one event per exposure or symptom phrase. "
                    "2) For exposures, candidate must be only the item/place name, no filler words. "
                    "3) For symptoms, candidate must be symptom phrase only. "
                    "4) Never output meaningless candidates like 'in the', 'for', 'when got home'. "
                    "5) Keep source_text as the smallest phrase needed for that event."
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
        return []

    events = data.get("events")
    event_count = data.get("event_count")
    if not isinstance(events, list):
        return []
    if not isinstance(event_count, int):
        return []
    if event_count != len(events):
        return []

    out: list[ParsedEvent] = []
    seen_keys: set[tuple[str, str | None, str | None, str | None, int | None, int | None, str | None]] = set()
    for entry in events[:20]:
        if not isinstance(entry, dict):
            continue
        parsed = _api_event_to_parsed_event(entry, text)
        if parsed is None:
            continue
        key = (
            parsed.event_type,
            parsed.timestamp,
            parsed.time_range_start,
            parsed.time_range_end,
            parsed.item_id,
            parsed.symptom_id,
            parsed.route,
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append(parsed)
    return out


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
                cleaned_item = _clean_candidate_text(item_name)
                if not _is_low_signal_candidate(cleaned_item):
                    item_id = resolve_item_id(cleaned_item)
        if item_id is None:
            return None
        if isinstance(route, str):
            route = normalize_route(route, strict=False)
        else:
            route = "other"
        if route in {"unknown", "other"}:
            route = normalize_route(_infer_route(text), strict=False)
        symptom_id = None
        severity = None
    else:
        if symptom_id is None:
            symptom_name = data.get("symptom_name") or data.get("candidate")
            if isinstance(symptom_name, str) and symptom_name.strip():
                cleaned_symptom = _clean_candidate_text(symptom_name)
                normalized_symptom = _normalize_symptom_candidate(cleaned_symptom)
                if _is_valid_symptom_candidate(normalized_symptom, text):
                    symptom_id = resolve_symptom_id(normalized_symptom)
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
    cleaned_tokens = [
        cleaned
        for cleaned in (_clean_candidate_text(token) for token in tokens)
        if cleaned and not _is_low_signal_candidate(cleaned)
    ]
    if event_type == "exposure":
        split_items = _split_exposure_items(text)
        candidate = split_items[0] if split_items else (cleaned_tokens[0] if cleaned_tokens else "")
        if not candidate:
            return None
        item_id = resolve_item_id(candidate)
        route = _infer_route(text)
    else:
        candidate = _normalize_symptom_candidate(cleaned_tokens[0]) if cleaned_tokens else ""
        if not candidate or not _is_valid_symptom_candidate(candidate, text):
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
        source_text=text,
    )

# DB writing logic after parsing
def ingest_text_event(user_id: int, raw_text: str) -> dict:
    parsed_events = parse_text_events(raw_text)
    if not parsed_events:
        insert_raw_event_ingest(user_id, raw_text, "failed", "unparsed")
        return {"status": "queued", "resolution": "pending"}

    conn = get_connection()
    now = datetime.now(tz=timezone.utc).isoformat()
    jobs_queued = 0
    events_written = 0
    queued_pairs: set[tuple[int, int]] = set()
    seen_exposure_keys: set[tuple[int, str | None, str | None, str | None, str, str | None]] = set()
    seen_timed_item_route: set[tuple[int, str]] = set()
    seen_symptom_keys: set[tuple[int, str | None, str | None, str | None, str | None]] = set()
    # Prefer timed exposure parses before unknown-time parses so ambiguous duplicates get dropped.
    def _event_priority(parsed: ParsedEvent) -> tuple[int, int]:
        if parsed.event_type != "exposure":
            return (2, 0)
        has_time = bool(parsed.timestamp or parsed.time_range_start or parsed.time_range_end)
        return (0 if has_time else 1, 0)

    for parsed in sorted(parsed_events, key=_event_priority):
        if parsed.event_type == "exposure":
            route = parsed.route or _infer_route(raw_text)
            route = normalize_route(route, strict=False)
            split_source = parsed.source_text or raw_text
            split_names = _split_exposure_items(split_source) if route == "ingestion" else []
            item_ids = [resolve_item_id(name) for name in split_names] if len(split_names) > 1 else [parsed.item_id]
            written_item_ids: set[int] = set()
            has_time = bool(parsed.timestamp or parsed.time_range_start or parsed.time_range_end)
            for item_id in item_ids:
                if item_id is None:
                    continue
                item_route_key = (int(item_id), route or "other")
                if not has_time and item_route_key in seen_timed_item_route:
                    # Drop unknown-time duplicate when a timed row for same item/route already exists.
                    continue
                exposure_key = (
                    int(item_id),
                    parsed.timestamp,
                    parsed.time_range_start,
                    parsed.time_range_end,
                    route or "other",
                    parsed.time_confidence,
                )
                if exposure_key in seen_exposure_keys:
                    continue
                seen_exposure_keys.add(exposure_key)
                written_item_ids.add(int(item_id))
                if has_time:
                    seen_timed_item_route.add(item_route_key)
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
                events_written += 1
            symptom_rows = conn.execute(
                "SELECT DISTINCT symptom_id FROM symptom_events WHERE user_id = ?",
                (user_id,),
            ).fetchall()
            symptom_ids = [int(row["symptom_id"]) for row in symptom_rows if row["symptom_id"] is not None]
            for item_id in sorted(written_item_ids):
                for symptom_id in symptom_ids:
                    queued_pairs.add((item_id, symptom_id))
        else:
            symptom_key = (
                int(parsed.symptom_id) if parsed.symptom_id is not None else -1,
                parsed.timestamp,
                parsed.time_range_start,
                parsed.time_range_end,
                str(parsed.severity) if parsed.severity is not None else None,
            )
            if symptom_key in seen_symptom_keys:
                continue
            seen_symptom_keys.add(symptom_key)
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
            events_written += 1
            if parsed.symptom_id is not None:
                item_rows = conn.execute(
                    "SELECT DISTINCT item_id FROM exposure_events WHERE user_id = ?",
                    (user_id,),
                ).fetchall()
                symptom_id = int(parsed.symptom_id)
                item_ids = [int(row["item_id"]) for row in item_rows if row["item_id"] is not None]
                for item_id in item_ids:
                    queued_pairs.add((item_id, symptom_id))
    conn.commit()
    conn.close()
    trigger = "ingest_text_multi"
    for item_id, symptom_id in sorted(queued_pairs):
        job_id = enqueue_background_job(
            user_id=user_id,
            job_type=JOB_RECOMPUTE_CANDIDATE,
            item_id=item_id,
            symptom_id=symptom_id,
            payload={"trigger": trigger},
        )
        if job_id is not None:
            jobs_queued += 1
    maybe_enqueue_model_retrain(trigger_user_id=int(user_id))
    return {"status": "ingested", "event_type": "multi" if events_written > 1 else parsed_events[0].event_type, "jobs_queued": jobs_queued}
