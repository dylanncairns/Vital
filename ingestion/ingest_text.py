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
from api.worker.jobs import (
    JOB_RECOMPUTE_CANDIDATE,
    enqueue_background_job,
    maybe_enqueue_model_retrain,
)
from api.helpers.raw_event_ingest import insert_raw_event_ingest
from ingestion.expand_exposure import expand_exposure_event
from ingestion.normalize_event import normalize_route
from ingestion.time_utils import to_utc_iso
from api.helpers.resolve import resolve_item_id, resolve_symptom_id

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
_EXPOSURE_VERBS = re.compile(
    r"\b("
    r"ate|eat|eaten|eating|snacked|snack|snacking|"
    r"drank|drink|drinking|sipped|sip|sipping|"
    r"used|use|using|apply|applied|applying|rubbed|rub|rubbing|"
    r"took|take|taking|swallowed|swallow|swallowing|ingested|ingest|ingesting|"
    r"smoked|smoke|smoking|vaped|vape|vaping|inhaled|inhale|inhaling|"
    r"injected|inject|injecting"
    r")\b",
    re.I,
)
_CONTEXT_EXPOSURE_RE = re.compile(
    r"\b(went\s+to|visited|was\s+at|at\s+the|club|party|bar|concert|festival|crowd)\b",
    re.I,
)
_CONTEXT_PLACE_RE = re.compile(
    r"\b("
    r"work|office|workplace|school|campus|class|gym|home|hospital|clinic|"
    r"airport|restaurant|cafe|coffee\s+shop|bar|club|party|concert|festival|"
    r"mall|store|subway|bus|train"
    r")\b",
    re.I,
)
_AT_PLACE_EXPOSURE_RE = re.compile(
    r"\b(?:i\s+am\s+at|i'm\s+at|im\s+at|at)\s+(?:the\s+)?"
    r"(work|office|workplace|school|campus|class|gym|home|hospital|clinic|"
    r"airport|restaurant|cafe|coffee\s+shop|bar|club|party|concert|festival|"
    r"mall|store|subway|bus|train)\b",
    re.I,
)
_SEVERITY_RE = re.compile(r"\b(severity|sev|pain)\s*[:=]?\s*(\d)\b", re.I)
_RATING_RE = re.compile(r"\b(\d)\s*/\s*5\b")
_RATING_10_RE = re.compile(r"\b(10|[1-9])\s*/\s*10\b")
_SEVERITY_SUFFIX_RE = re.compile(r"\b(10|[1-9])\s*(?:/10)?\s*(?:severity|sev|pain)\b", re.I)
_TIME_AT_RE = re.compile(
    r"(?:\b(?:at|around|about|by)\s+|@)(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b",
    re.I,
)
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
    r"last night|for the night|tonight|today|yesterday|this week|last week|"
    r"breakfast|lunch|dinner|"
    r"morning|afternoon|evening|night"
    r")\b",
    re.I,
)
_DAYS_AGO_RE = re.compile(
    r"\b(?:(\d+)|one|two|three|four|five|six|seven)\s+days?\s+ago\b",
    re.I,
)
_SYMPTOM_CUES_RE = re.compile(
    r"\b("
    r"felt|feel|feeling|tired|fatigued|exhausted|"
    r"ache|pain|hurt|hurts|hurting|sore|burning|pressure|tightness|nausea|headache|stomachache|"
    r"dizzy|dizziness|lightheaded|light-headed|vertigo|faint|fainting|shaky|shakiness|jittery|trembling|tremor|"
    r"anxious|anxiety|panic|"
    r"insomnia|can't sleep|cannot sleep|trouble sleeping|"
    r"brain fog|memory|remember|forget|forgetful|concentrate|focus|"
    r"bloating|bloated|constipation|diarrhea|heartburn|reflux|gerd|acid reflux|indigestion|"
    r"palpitations?|racing heart|fast heart rate|tachycardia|"
    r"shortness of breath|breathless|wheeze|wheezing|"
    r"chest pain|chest tightness|"
    r"congestion|congested|runny nose|stuffy nose|sinus|"
    r"numb|numbness|tingling|pins and needles|"
    r"joint pain|back pain|muscle pain|body aches|"
    r"rash|itch|itchy|hives|acne|breakout|breaking out|"
    r"cough|fever|sore throat|chills"
    r")\b",
    re.I,
)
_HAD_OBJECT_RE = re.compile(r"\bhad\s+(?:a|an|some|the)?\s*[a-z]", re.I)
_MEAL_CONTEXT_RE = re.compile(r"\b(breakfast|lunch|dinner|snack|snacked|ate|eaten|drank|drink)\b", re.I)
_LIFESTYLE_EXPOSURE_RE = re.compile(
    r"\b("
    r"poor sleep|bad sleep|no sleep|sleep deprivation|sleep deprived|insufficient sleep|"
    r"barely slept|hardly slept|didn't sleep|didnt sleep|couldn't sleep at all|couldnt sleep at all|"
    r"long shift|worked a long shift|overnight shift|jet lag|high stress|stressed|overworked|work stress|"
    r"all[- ]?nighter|pulled an all nighter|dehydrated|dehydration|fasting|skipped (?:a )?meal"
    r")\b",
    re.I,
)
_DATE_TOKEN_RE = re.compile(
    r"\b("
    r"\d{4}-\d{2}-\d{2}|"
    r"(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\s+\d{1,2}(?:,\s*\d{4})?|"
    r"\d{1,2}/\d{1,2}(?:/\d{2,4})?|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"today|yesterday|last night|for the night|tonight|this morning|this afternoon|this evening|this week|last week|"
    r"breakfast|lunch|dinner|this breakfast|this lunch|this dinner|"
    r"yesterday breakfast|yesterday lunch|yesterday dinner"
    r")\b",
    re.I,
)
_STRONG_RELATIVE_DATE_RE = re.compile(
    r"\b("
    r"today|yesterday|last night|for the night|tonight|this week|last week|"
    r"this morning|this afternoon|this evening|this night|"
    r"yesterday morning|yesterday afternoon|yesterday evening|yesterday night|"
    r"yesterday breakfast|yesterday lunch|yesterday dinner"
    r")\b",
    re.I,
)

_LOCAL_TZ = datetime.now().astimezone().tzinfo or timezone.utc


def _tz_from_offset_minutes(offset_minutes: int | None):
    if offset_minutes is None:
        return _LOCAL_TZ
    try:
        minutes = int(offset_minutes)
    except (TypeError, ValueError):
        return _LOCAL_TZ
    minutes = max(-14 * 60, min(14 * 60, minutes))
    # JS Date.getTimezoneOffset() semantics: UTC - local.
    return timezone(-timedelta(minutes=minutes))


def _event_has_time(parsed: ParsedEvent) -> bool:
    return bool(parsed.timestamp or parsed.time_range_start or parsed.time_range_end)
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
    "while",
    "because",
    "since",
    "later",
    "earlier",
    "once",
}
_COMMON_SYMPTOM_TERMS_RE = re.compile(
    r"\b("
    r"headache|migraine|nausea|vomit|vomiting|stomachache|stomach pain|"
    r"heartburn|acid reflux|reflux|gerd|indigestion|dyspepsia|"
    r"chest pain|chest tightness|shortness of breath|breathlessness|wheeze|wheezing|"
    r"palpitations?|tachycardia|racing heart|"
    r"congestion|runny nose|stuffy nose|sinus|sinus pain|sinus pressure|"
    r"numbness|tingling|pins and needles|"
    r"joint pain|back pain|muscle pain|body ache|body aches|chills|"
    r"high blood pressure|blood pressure|hypertension|"
    r"acne|rash|itch|itchy|hives|fatigue|tired|brain fog|"
    r"diarrhea|constipation|bloat|bloating|cramp|cramps|"
    r"dizzy|dizziness|shaky|shakiness|jittery|tremor|trembling|anxiety|insomnia|fever|cough|sore throat"
    r")\b",
    re.I,
)

# Common lay phrasing -> canonical symptom labels used for DB resolution.
_SYMPTOM_SYNONYM_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(can'?t|cannot|unable to|unable)\s+(remember|focus|concentrate)\b", re.I), "brain fog"),
    (re.compile(r"\b(can'?t|cannot|unable to)\s+think\s+clearly\b", re.I), "brain fog"),
    (re.compile(r"\b(mental fog|foggy brain|mind feels foggy)\b", re.I), "brain fog"),
    (re.compile(r"\bmemory (issues?|problems?|loss)\b", re.I), "brain fog"),
    (re.compile(r"\bforgetful(?:ness)?\b", re.I), "brain fog"),
    (re.compile(r"\b(light[- ]?headed|lightheadedness)\b", re.I), "dizziness"),
    (re.compile(r"\b(dizzy|dizziness|vertigo|spinning)\b", re.I), "dizziness"),
    (re.compile(r"\b(shaky|shakiness|jittery|trembl(?:e|ing)|tremor)\b", re.I), "dizziness"),
    (re.compile(r"\b(exhausted|worn out|low energy|no energy|drained)\b", re.I), "fatigue"),
    (re.compile(r"\b(tired(?:ness)?|fatigued)\b", re.I), "fatigue"),
    (re.compile(r"\b(can'?t sleep|cannot sleep|unable to sleep|trouble sleeping|difficulty sleeping|poor sleep|sleep is bad)\b", re.I), "insomnia"),
    (re.compile(r"\b(waking up a lot|keep waking up|wake up constantly)\b", re.I), "insomnia"),
    (re.compile(r"\b(anxious|panic(?: attack)?s?)\b", re.I), "anxiety"),
    (re.compile(r"\b(feeling on edge|racing thoughts|restless anxiety)\b", re.I), "anxiety"),
    (re.compile(r"\b(feeling depressed|depressed mood|hopeless|low mood)\b", re.I), "anxiety"),
    (re.compile(r"\b(throwing up|threw up|vomit(?:ing)?)\b", re.I), "vomiting"),
    (re.compile(r"\b(feel|feeling)\s+sick\b", re.I), "nausea"),
    (re.compile(r"\b(queasy|queasiness)\b", re.I), "nausea"),
    (re.compile(r"\b(upset stomach|sour stomach|stomach upset)\b", re.I), "stomachache"),
    (re.compile(r"\b(heartburn|acid reflux|acid[- ]?reflux|gerd)\b", re.I), "heartburn"),
    (re.compile(r"\b(indigestion|dyspepsia)\b", re.I), "heartburn"),
    (re.compile(r"\b(acid(?:ic)?\s+burn(?:ing)?\s+in\s+(?:my\s+)?chest)\b", re.I), "heartburn"),
    (re.compile(r"\b(burning\s+(?:in\s+)?(?:my\s+)?chest(?:\s+after eating)?)\b", re.I), "heartburn"),
    (re.compile(r"\b(reflux(?:ing)?|regurgitation)\b", re.I), "heartburn"),
    (re.compile(r"\b(chest pain|pain in (?:my\s+)?chest)\b", re.I), "chest pain"),
    (re.compile(r"\b(chest tightness|tight chest)\b", re.I), "chest tightness"),
    (re.compile(r"\b(short of breath|shortness of breath|breathless(?:ness)?)\b", re.I), "shortness of breath"),
    (re.compile(r"\b(wheez(?:e|ing)|wheezy)\b", re.I), "shortness of breath"),
    (re.compile(r"\b(heart racing|racing heart|rapid heartbeat|fast heart rate|palpitations?)\b", re.I), "palpitations"),
    (re.compile(r"\b(stuffy nose|runny nose|nasal congestion|congested nose)\b", re.I), "congestion"),
    (re.compile(r"\b(sinus pressure|sinus pain)\b", re.I), "congestion"),
    (re.compile(r"\b(pins and needles|tingly|tingling|numbness|numb)\b", re.I), "tingling"),
    (re.compile(r"\b(lower back pain|upper back pain|back pain)\b", re.I), "back pain"),
    (re.compile(r"\b(joint aches?|joint pain)\b", re.I), "joint pain"),
    (re.compile(r"\b(muscle aches?|muscle pain|body aches?)\b", re.I), "muscle pain"),
    (re.compile(r"\b(stomach pain|abdominal pain|stomach ache)\b", re.I), "stomachache"),
    (re.compile(r"\b(head pain|head ache)\b", re.I), "headache"),
    (re.compile(r"\bhead\s+(?:is\s+)?hurt(?:s|ing)?\b", re.I), "headache"),
    (re.compile(r"\bmy\s+head\s+hurt(?:s|ing)?\b", re.I), "headache"),
    (re.compile(r"\b(acne spots?|breakouts?|breaking out|pimples?|zits?)\b", re.I), "acne"),
    (re.compile(r"\b(bloated|bloating|gassy|gas)\b", re.I), "bloating"),
    (re.compile(r"\b(can'?t poop|hard to poop|haven'?t pooped)\b", re.I), "constipation"),
    (re.compile(r"\b(loose stools?|runny stools?)\b", re.I), "diarrhea"),
    (re.compile(r"\b(itchy|itching)\b", re.I), "itch"),
    (re.compile(r"\b(hives?|welts?)\b", re.I), "hives"),
    (re.compile(r"\b(sore throat|throat pain)\b", re.I), "sore throat"),
    (re.compile(r"\b(high bp|elevated blood pressure|hypertension)\b", re.I), "high blood pressure"),
    (re.compile(r"\b(chills|shivering)\b", re.I), "fever"),
]

_ABS_MONTH_DATE_RE = re.compile(
    r"\b(?:on\s+)?("
    r"(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\s+\d{1,2}(?:,\s*\d{4})?"
    r"|"
    r"\d{1,2}/\d{1,2}(?:/\d{2,4})?"
    r")\b",
    re.I,
)
_ANAPHORIC_DAYPART_RE = re.compile(r"\bthat\s+(morning|afternoon|evening|night|day)\b", re.I)
_NOW_RE = re.compile(r"\b(right now|just now|now)\b", re.I)
_EARLIER_TODAY_RE = re.compile(r"\bearlier today\b", re.I)
_LATER_TODAY_RE = re.compile(r"\blater today\b", re.I)


def _apply_daypart_to_base(base: datetime, daypart: str) -> datetime:
    token = daypart.lower().strip()
    if token == "morning":
        return base.replace(hour=9, minute=0, second=0, microsecond=0)
    if token == "afternoon":
        return base.replace(hour=15, minute=0, second=0, microsecond=0)
    if token == "evening":
        return base.replace(hour=19, minute=0, second=0, microsecond=0)
    if token == "night":
        return base.replace(hour=22, minute=0, second=0, microsecond=0)
    return base.replace(hour=12, minute=0, second=0, microsecond=0)


def _context_anchor_iso(
    *,
    context_timestamp: str | None,
    context_start: str | None,
    context_end: str | None,
) -> str | None:
    return context_timestamp or context_start or context_end


def _resolve_anaphoric_daypart_time(
    text: str,
    *,
    context_timestamp: str | None,
    context_start: str | None,
    context_end: str | None,
    local_tz=None,
) -> str | None:
    match = _ANAPHORIC_DAYPART_RE.search(text)
    if not match:
        return None
    anchor_iso = _context_anchor_iso(
        context_timestamp=context_timestamp,
        context_start=context_start,
        context_end=context_end,
    )
    if not anchor_iso:
        return None
    try:
        anchor = datetime.fromisoformat(anchor_iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    tz = local_tz or _LOCAL_TZ
    anchor_local = anchor.astimezone(tz)
    resolved_local = _apply_daypart_to_base(anchor_local, match.group(1))
    return resolved_local.astimezone(timezone.utc).isoformat()


def _resolve_relative_clause_time(
    text: str,
    *,
    context_timestamp: str | None,
    context_start: str | None,
    context_end: str | None,
    local_tz=None,
) -> str | None:
    has_later = re.search(r"\blater\b", text, re.I) is not None
    has_earlier = re.search(r"\bearlier\b", text, re.I) is not None
    if not has_later and not has_earlier:
        return None
    anchor_iso = _context_anchor_iso(
        context_timestamp=context_timestamp,
        context_start=context_start,
        context_end=context_end,
    )
    if not anchor_iso:
        return None
    try:
        anchor = datetime.fromisoformat(anchor_iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    tz = local_tz or _LOCAL_TZ
    anchor_local = anchor.astimezone(tz)
    if has_later:
        resolved_local = anchor_local + timedelta(hours=2)
    else:
        resolved_local = anchor_local - timedelta(hours=2)
    return resolved_local.astimezone(timezone.utc).isoformat()


def _resolve_bedtime_clause_time(
    text: str,
    *,
    context_timestamp: str | None,
    context_start: str | None,
    context_end: str | None,
    local_tz=None,
) -> str | None:
    if re.search(r"\b(?:before|at)\s+bed(?:time)?\b|\bbefore\s+sleep(?:ing)?\b", text, re.I) is None:
        return None
    anchor_iso = _context_anchor_iso(
        context_timestamp=context_timestamp,
        context_start=context_start,
        context_end=context_end,
    )
    tz = local_tz or _LOCAL_TZ
    if anchor_iso:
        try:
            anchor = datetime.fromisoformat(anchor_iso.replace("Z", "+00:00"))
            anchor_local = anchor.astimezone(tz)
        except ValueError:
            anchor_local = datetime.now().astimezone(tz)
    else:
        anchor_local = datetime.now().astimezone(tz)
    # If anchored to morning/afternoon, "before bed" usually refers to prior night.
    if anchor_local.hour <= 15:
        anchor_local = anchor_local - timedelta(days=1)
    resolved_local = anchor_local.replace(hour=22, minute=0, second=0, microsecond=0)
    return resolved_local.astimezone(timezone.utc).isoformat()


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


def _has_strong_date_anchor(text: str) -> bool:
    if not text:
        return False
    if _ABS_MONTH_DATE_RE.search(text):
        return True
    if _DAYS_AGO_RE.search(text):
        return True
    if _STRONG_RELATIVE_DATE_RE.search(text):
        return True
    return False

# identify time from within text blob (safe for voice-to-text where user states timestamp)
def _parse_time(text: str, *, local_tz=None) -> tuple[str | None, str | None, str | None]:
    # Parse time words as local user-time intent, then convert to UTC for storage.
    tz = local_tz or _LOCAL_TZ
    now = datetime.now().astimezone(tz)
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

    # Prioritize explicit clock time so phrases like "at 10 am yesterday" keep
    # both the clock time and the relative/absolute day anchor.
    match = _TIME_AT_RE.search(text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        meridiem = (match.group(3) or "").lower()
        if meridiem == "pm" and hour < 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0
        # Keep date-anchor behavior consistent with other time branches.
        # If text contains explicit date context, apply it before setting clock time.
        base = absolute_date or now
        if days_ago is not None:
            base = now - timedelta(days=days_ago)
        rel = _RELATIVE_RE.search(text)
        if rel:
            token = rel.group(1).lower()
            if token.startswith("yesterday") or token == "last night" or token == "yesterday":
                base = now - timedelta(days=1)
        ts = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return ts.astimezone(timezone.utc).isoformat(), None, None

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
        if token == "for the night":
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
        if token == "this week":
            week_start = (base - timedelta(days=base.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
            week_end = now.replace(second=0, microsecond=0)
            if week_end < week_start:
                week_end = week_start
            return None, week_start.astimezone(timezone.utc).isoformat(), week_end.astimezone(timezone.utc).isoformat()
        if token == "last week":
            this_week_start = (base - timedelta(days=base.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
            last_week_start = this_week_start - timedelta(days=7)
            last_week_end = this_week_start - timedelta(seconds=1)
            return (
                None,
                last_week_start.astimezone(timezone.utc).isoformat(),
                last_week_end.astimezone(timezone.utc).isoformat(),
            )
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

    if re.search(r"\b(got home|home from work|after work)\b", lower):
        ts = now.replace(hour=18, minute=30, second=0, microsecond=0)
        return ts.astimezone(timezone.utc).isoformat(), None, None

    if _NOW_RE.search(lower):
        ts = now.replace(second=0, microsecond=0)
        return ts.astimezone(timezone.utc).isoformat(), None, None

    if _EARLIER_TODAY_RE.search(lower):
        ts = now.replace(hour=9, minute=0, second=0, microsecond=0)
        if ts > now:
            ts = (now - timedelta(hours=1)).replace(second=0, microsecond=0)
        return ts.astimezone(timezone.utc).isoformat(), None, None

    if _LATER_TODAY_RE.search(lower):
        ts = (now + timedelta(hours=3)).replace(second=0, microsecond=0)
        if ts.date() != now.date():
            ts = now.replace(hour=21, minute=0, second=0, microsecond=0)
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
    has_place_context_exposure = _AT_PLACE_EXPOSURE_RE.search(lower) is not None
    has_symptom_cue = _SYMPTOM_CUES_RE.search(lower) is not None
    has_common_symptom = _COMMON_SYMPTOM_TERMS_RE.search(lower) is not None
    has_had_object = _HAD_OBJECT_RE.search(lower) is not None
    has_meal_marker = re.search(r"\b(breakfast|lunch|dinner)\b", lower) is not None
    has_lifestyle_exposure = _LIFESTYLE_EXPOSURE_RE.search(lower) is not None

    # Prefer symptom when symptom signal is explicit and no strong exposure verb exists.
    if (
        (has_symptom_cue or has_common_symptom)
        and not has_exposure
        and not has_context_exposure
        and not has_place_context_exposure
    ):
        return "symptom"
    if has_had_object and not has_common_symptom and not has_symptom_cue:
        return "exposure"
    if has_had_object and has_meal_marker and not has_common_symptom:
        return "exposure"
    if has_place_context_exposure and not has_common_symptom and not has_symptom_cue:
        return "exposure"
    if has_lifestyle_exposure and not has_common_symptom and not has_symptom_cue:
        return "exposure"

    if _EXPOSURE_VERBS.search(text):
        return "exposure"
    if _MEAL_CONTEXT_RE.search(text) and re.search(r"\b(had|have|having)\b", text, re.I) and not (
        has_symptom_cue or has_common_symptom
    ):
        return "exposure"
    if _CONTEXT_EXPOSURE_RE.search(text):
        return "exposure"
    if _AT_PLACE_EXPOSURE_RE.search(text):
        return "exposure"
    if _LIFESTYLE_EXPOSURE_RE.search(text):
        return "exposure"
    return "symptom"


def _infer_route(text: str) -> str:
    t = text.lower()
    if _LIFESTYLE_EXPOSURE_RE.search(t):
        return "behavioral"
    if re.search(r"\b(near|nearby|around|exposed|exposure|environment|air quality|pollution|pollen|dust|mold|secondhand|second hand|passive smoke)\b", t):
        return "proximity_environment"
    if re.search(r"\b(went\s+to|visited|was\s+at|club|party|bar|concert|festival|crowd)\b", t):
        return "proximity_environment"
    if _AT_PLACE_EXPOSURE_RE.search(t):
        return "proximity_environment"
    if re.search(r"\b(smoked|smoke|vaped|vape|inhaled|inhale|inhaling)\b", t):
        return "inhalation"
    if re.search(r"\b(applied|apply|applying|rubbed|rub|rubbing|cream|lotion|ointment|topical|used on|soap|shampoo|conditioner|cleanser|face wash|detergent|deodorant|perfume|fragrance|sunscreen|makeup)\b", t):
        return "topical"
    if re.search(r"\b(injected|inject|injecting|shot|iv)\b", t):
        return "injection"
    if _MEAL_CONTEXT_RE.search(t) and re.search(r"\b(had|have|having)\b", t):
        return "ingestion"
    if re.search(r"\b(ate|eat|eaten|eating|snacked|snack|snacking|drank|drink|drinking|sipped|sip|sipping|took|take|taking|used|use|using|swallowed|swallow|ingested|ingest)\b", t):
        return "ingestion"
    if _HAD_OBJECT_RE.search(t) and _COMMON_SYMPTOM_TERMS_RE.search(t) is None:
        return "ingestion"
    return "other"


def _split_exposure_items(text: str) -> list[str]:
    # Extract likely item phrase after ingestion verb and split list-like food entries.
    lower = text.lower()
    match = re.search(
        r"\b(ate|eat|eaten|eating|snacked|snack|snacking|drank|drink|drinking|sipped|sip|sipping|took|take|taking|used|use|using|swallowed|swallow|ingested|ingest|smoked|smoke|vaped|vape|inhaled|inhale|had|have|having)\b",
        lower,
    )
    if not match:
        place_match = re.search(
            r"\b(?:i\s+am\s+at|i'm\s+at|im\s+at|at)\s+(?:the\s+)?"
            r"(work|office|workplace|school|campus|class|gym|home|hospital|clinic|"
            r"airport|restaurant|cafe|coffee\s+shop|bar|club|party|concert|festival|"
            r"mall|store|subway|bus|train)\b",
            lower,
        )
        if place_match:
            segment = place_match.group(1)
        else:
        # Handle non-ingestion context exposures ("went to the club", "at a party")
            context_match = re.search(r"\b(?:went\s+to|visited|was\s+at)\s+(.*)$", lower)
            if context_match:
                segment = context_match.group(1)
            else:
                # Fallback for noun lists without explicit verbs (e.g., "chicken and rice").
                segment = lower
        segment = re.split(
            r"\b(today|yesterday|last night|this morning|this afternoon|this evening|morning|afternoon|evening|night|breakfast|lunch|dinner|this breakfast|this lunch|this dinner|yesterday breakfast|yesterday lunch|yesterday dinner)\b",
            segment,
            maxsplit=1,
        )[0]
        segment = _DATE_TOKEN_RE.sub(" ", segment)
        segment = re.sub(r"\b(?:i|we)\s+(?:had|have|having)\b", " and ", segment)
        symptom_match = _SYMPTOM_CUES_RE.search(segment)
        if symptom_match:
            segment = segment[: symptom_match.start()]
        segment = re.split(r"\b(felt|feel)\b", segment, maxsplit=1)[0]
        parts = re.split(r"\s*(?:,|/|&|\band\b|\bwith\b)\s*", segment)
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
    segment = re.split(
        r"(?:\b(?:at|around|about|by)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b|@\d{1,2}(?::\d{2})?\s*(?:am|pm)?)",
        segment,
        maxsplit=1,
    )[0]
    segment = re.split(
        r"\b(today|yesterday|last night|this morning|this afternoon|this evening|morning|afternoon|evening|night|breakfast|lunch|dinner|this breakfast|this lunch|this dinner|yesterday breakfast|yesterday lunch|yesterday dinner)\b",
        segment,
        maxsplit=1,
    )[0]
    segment = _DATE_TOKEN_RE.sub(" ", segment)
    symptom_match = _SYMPTOM_CUES_RE.search(segment)
    if symptom_match:
        segment = segment[: symptom_match.start()]
    segment = re.split(r"\b(felt|feel)\b", segment, maxsplit=1)[0]
    parts = re.split(r"\s*(?:,|/|&|\band\b|\bwith\b)\s*", segment)
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        cleaned = _clean_candidate_text(part)
        if not cleaned or _is_low_signal_candidate(cleaned) or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def _fallback_exposure_candidate_from_text(text: str) -> str | None:
    low = (text or "").lower()
    if re.search(r"\b(vaped|vape|vaping)\b", low):
        return "vaping"
    if re.search(r"\b(smoked|smoke|smoking)\b", low):
        return "smoking"
    if re.search(r"\b(inhaled|inhale|inhaling)\b", low):
        return "inhalation"
    if re.search(r"\b(?:pulled\s+an?\s+)?all[- ]?nighter\b", low) or re.search(
        r"\b(barely slept|hardly slept|didn't sleep|didnt sleep|no sleep|poor sleep)\b", low
    ):
        return "poor sleep"
    if re.search(r"\b(worked\s+(?:a\s+)?long\s+shift|long shift|overnight shift)\b", low):
        return "long shift"
    if _LIFESTYLE_EXPOSURE_RE.search(low):
        # Reuse candidate cleaning normalization for lifestyle phrases.
        cleaned = _clean_candidate_text(low)
        if cleaned and not _is_low_signal_candidate(cleaned):
            return cleaned
    return None

# functions below handle long string inputs with ambiguity in event details or time

def _clean_candidate_text(text: str) -> str:
    value = text.strip().lower()
    # Normalize common behavioral exposure phrases to canonical item-like terms.
    value = re.sub(r"\bworked\s+(?:a\s+)?long\s+shift\b", " long shift ", value)
    value = re.sub(r"\b(?:barely|hardly)\s+slept\b", " poor sleep ", value)
    value = re.sub(r"\b(?:did\s*not|didn't|didnt)\s+sleep\b", " poor sleep ", value)
    value = re.sub(r"\bcould\s*not\s+sleep\s+at\s+all\b|\bcouldn't sleep at all\b|\bcouldnt sleep at all\b", " no sleep ", value)
    value = re.sub(r"\b(?:pulled\s+an?\s+)?all[- ]?nighter\b", " poor sleep ", value)
    value = re.sub(r"\bpoor sleep\s+(?:for\s+work|for\s+school|working|studying)\b", " poor sleep ", value)
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
    value = _DATE_TOKEN_RE.sub(" ", value)
    value = re.sub(r"\b(?:this|last)\s+(?:week|month)\b", " ", value)
    # Remove common temporal phrases that otherwise leak nouns into exposure items
    # (e.g., "face wash before bed" -> "face wash", not "face wash bed").
    value = re.sub(r"\b(?:before|after|at)\s+bed(?:time)?\b", " ", value)
    value = re.sub(r"\b(?:before|after)\s+sleep(?:ing)?\b", " ", value)
    value = re.sub(
        r"\b(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\b",
        " ",
        value,
    )
    value = re.sub(r"\b\d{1,2}(?:st|nd|rd|th)?\b", " ", value)
    value = re.sub(r"\b(i|we|also|then|so|when|and|at|about|this|that|in|for|on|to|of|from|by|went|visit|visited|did|do|done|after|before|during|while|because|since|got|my)\b", " ", value)
    value = re.sub(r"\b(was|were|is|am|been|being)\b", " ", value)
    value = re.sub(
        r"\b(ate|eat|eaten|eating|snacked|snack|snacking|drank|drink|drinking|sipped|sip|sipping|used|use|using|apply|applied|applying|rubbed|rub|rubbing|took|take|taking|smoked|smoke|vaped|vape|inhaled|inhale|felt|feel|had|have|having)\b",
        " ",
        value,
    )
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
    # Remove leading/trailing low-signal tokens repeatedly.
    tokens = [tok for tok in value.split(" ") if tok]
    while tokens and tokens[0] in _LOW_SIGNAL_TOKENS:
        tokens.pop(0)
    while tokens and tokens[-1] in _LOW_SIGNAL_TOKENS:
        tokens.pop()
    value = " ".join(tokens).strip(" ,.;")
    value = re.sub(r"\s+", " ", value).strip(" ,.;")
    return value


def _normalize_symptom_candidate(candidate: str) -> str:
    value = " ".join(candidate.strip().lower().split())
    if not value:
        return value
    value = value.replace("’", "'")
    value = re.sub(r"\bcan't\b", "cannot", value)
    value = re.sub(r"\bwon't\b", "will not", value)
    value = re.sub(r"\bdoesn't\b", "does not", value)
    value = re.sub(r"\bdon't\b", "do not", value)
    # Strip conversational lead-ins and tense wrappers.
    value = re.sub(
        r"^(?:i|we|my|our)\s+(?:have|has|had|am|are|is|was|were|been|be|feel|feeling|felt)\s+",
        "",
        value,
    )
    value = re.sub(
        r"^(?:been|having|dealing with|struggling with|suffering from|experiencing)\s+",
        "",
        value,
    )
    value = re.sub(r"\b(?:recently|lately|for a while|for weeks?|for months?)\b", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"^(also|just|really|very)\s+", "", value)
    value = re.sub(r"^(some|new|a|an|the)\s+", "", value)
    for pattern, canonical in _SYMPTOM_SYNONYM_RULES:
        if pattern.search(value):
            return canonical
    term_match = _COMMON_SYMPTOM_TERMS_RE.search(value)
    if term_match:
        return term_match.group(0).strip().lower()
    return value


_SYMPTOM_RESOLUTION_FALLBACKS: dict[str, list[str]] = {
    "fatigue": ["tired"],
    "dizziness": ["lightheadedness"],
    "palpitations": ["racing heart", "tachycardia"],
    "chest tightness": ["chest pain", "shortness of breath"],
}


def _resolve_symptom_with_fallback(name: str | None) -> int | None:
    if not name:
        return None
    resolved = resolve_symptom_id(name)
    if resolved is not None:
        return resolved
    for alt in _SYMPTOM_RESOLUTION_FALLBACKS.get(str(name).strip().lower(), []):
        resolved = resolve_symptom_id(alt)
        if resolved is not None:
            return resolved
    return None


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
    if _CONTEXT_PLACE_RE.search(normalized):
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
        # exposure-only multi-event blurbs should still split when temporal/context shifts exist
        explicit_context_mentions = len(
            re.findall(r"\b(?:went\s+to|visited|was\s+at|i\s+am\s+at|i'm\s+at|im\s+at)\b", text, re.I)
        )
        exposure_mentions = len(_EXPOSURE_VERBS.findall(text)) + explicit_context_mentions
        temporal_mentions = len(_TIME_AT_RE.findall(text)) + len(_RELATIVE_RE.findall(text)) + len(_NOW_RE.findall(text))
        has_clause_shift = re.search(r"\b(and then|then|after that|afterwards|and now|now)\b", text, re.I) is not None
        return (exposure_mentions >= 2 and temporal_mentions >= 1) or (exposure_mentions >= 2 and has_clause_shift)
    time_mentions = len(_TIME_AT_RE.findall(text)) + len(_RELATIVE_RE.findall(text)) + len(_NOW_RE.findall(text))
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


def _coerce_api_timestamp_to_today_if_time_only(
    text: str,
    timestamp: str | None,
    *,
    local_tz=None,
) -> str | None:
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
    tz = local_tz or _LOCAL_TZ
    parsed_local = parsed if parsed.tzinfo is None else parsed.astimezone(tz)
    if parsed_local.tzinfo is None:
        parsed_local = parsed_local.replace(tzinfo=tz)
    now_local = datetime.now().astimezone(tz)
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
    *,
    local_tz=None,
) -> tuple[str | None, str | None]:
    if not time_range_start and not time_range_end:
        return time_range_start, time_range_end
    if not _TIME_AT_RE.search(text):
        return time_range_start, time_range_end
    if _DATE_TOKEN_RE.search(text):
        return time_range_start, time_range_end

    tz = local_tz or _LOCAL_TZ
    now_local = datetime.now().astimezone(tz)

    def _coerce(value: str | None) -> str | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
        parsed_local = parsed if parsed.tzinfo is None else parsed.astimezone(tz)
        if parsed_local.tzinfo is None:
            parsed_local = parsed_local.replace(tzinfo=tz)
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
    *,
    local_tz=None,
) -> tuple[str | None, str | None, str | None]:
    fixed_ts, _, _ = _parse_time(text, local_tz=local_tz)
    if fixed_ts and _RELATIVE_RE.search(text):
        return fixed_ts, None, None
    return timestamp, time_range_start, time_range_end


# First try to call external parser via OpenAI API
def parse_text_event(
    text: str,
    *,
    allow_api: bool = True,
    local_tz=None,
) -> ParsedEvent | None:
    # One ParsedEvent cannot safely represent mixed exposure + symptom logs
    if _looks_like_multi_event(text):
        return None

    rules_parsed = parse_with_rules(text, local_tz=local_tz)
    if rules_parsed is not None:
        return rules_parsed

    # Fallback to API parser only when deterministic parser fails.
    if allow_api:
        api_parsed = parse_with_api(text, local_tz=local_tz)
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
        parts = re.split(
            r"\s*(?:,\s*)?(?:and then|then|after that|afterwards|later)\s+",
            cleaned,
            flags=re.I,
        )
        for part in parts:
            part_clean = part.strip(" ,")
            if len(part_clean) >= 3:
                segments.append(part_clean)
    return segments


def _has_time_info(parsed: ParsedEvent) -> bool:
    return bool(parsed.timestamp or parsed.time_range_start or parsed.time_range_end)


def _apply_time_context(
    parsed: ParsedEvent,
    *,
    context_timestamp: str | None,
    context_start: str | None,
    context_end: str | None,
) -> ParsedEvent:
    if _has_time_info(parsed):
        return parsed
    if not any([context_timestamp, context_start, context_end]):
        return parsed
    return ParsedEvent(
        event_type=parsed.event_type,
        timestamp=context_timestamp,
        time_range_start=context_start,
        time_range_end=context_end,
        time_confidence="approx",
        item_id=parsed.item_id,
        route=parsed.route,
        symptom_id=parsed.symptom_id,
        severity=parsed.severity,
        source_text=parsed.source_text,
    )


def _override_parsed_timestamp(parsed: ParsedEvent, timestamp: str) -> ParsedEvent:
    return ParsedEvent(
        event_type=parsed.event_type,
        timestamp=timestamp,
        time_range_start=None,
        time_range_end=None,
        time_confidence="approx",
        item_id=parsed.item_id,
        route=parsed.route,
        symptom_id=parsed.symptom_id,
        severity=parsed.severity,
        source_text=parsed.source_text,
    )


def _choose_route_from_api_or_rules(text: str, api_route_value) -> str:
    inferred_route = normalize_route(_infer_route(text), strict=False)
    route = normalize_route(api_route_value, strict=False) if isinstance(api_route_value, str) else "other"
    if route in {"unknown", "other"}:
        return inferred_route
    # Prefer deterministic route for strong, easy lexical cases where API sometimes drifts.
    if route == "proximity_environment" and inferred_route in {"ingestion", "topical", "inhalation", "injection", "behavioral"}:
        return inferred_route
    return route


def _expand_multi_item_exposure(parsed: ParsedEvent, text: str) -> list[ParsedEvent]:
    if parsed.event_type != "exposure":
        return [parsed]
    item_candidates = _split_exposure_items(text)
    # Extra conjunction expansion inside a single ingestion clause.
    # Example: "ate chicken and also mango" should yield both items.
    normalized_clause = " ".join((text or "").strip().lower().split())
    if re.search(r"\b(ate|eat|eating|snacked|snack|snacking|drank|drink|drinking|sipped|sip|sipping|took|take|taking|used|use|using|swallowed|swallow|ingested|ingest|smoked|smoke|vaped|vape|inhaled|inhale)\b", normalized_clause):
        tail_match = re.search(
            r"\b(?:ate|eat|eating|snacked|snack|snacking|drank|drink|drinking|sipped|sip|sipping|took|take|taking|used|use|using|swallowed|swallow|ingested|ingest|smoked|smoke|vaped|vape|inhaled|inhale)\b(.*)$",
            normalized_clause,
        )
        if tail_match:
            tail = tail_match.group(1)
            tail = re.sub(r"\b(?:also|just|really|very)\b", " ", tail)
            tail_parts = re.split(r"\s*(?:,|&|\band\b|\bwith\b)\s*", tail)
            for part in tail_parts:
                cleaned = _clean_candidate_text(part)
                if not cleaned or _is_low_signal_candidate(cleaned):
                    continue
                if cleaned not in item_candidates:
                    item_candidates.append(cleaned)
    if len(item_candidates) <= 1:
        return [parsed]
    out: list[ParsedEvent] = [parsed]
    seen_item_ids: set[int] = set()
    if parsed.item_id is not None:
        seen_item_ids.add(int(parsed.item_id))
    for candidate in item_candidates:
        item_id = resolve_item_id(candidate)
        if item_id is None:
            continue
        item_id = int(item_id)
        if item_id in seen_item_ids:
            continue
        seen_item_ids.add(item_id)
        out.append(
            ParsedEvent(
                event_type=parsed.event_type,
                timestamp=parsed.timestamp,
                time_range_start=parsed.time_range_start,
                time_range_end=parsed.time_range_end,
                time_confidence=parsed.time_confidence,
                item_id=item_id,
                route=parsed.route,
                symptom_id=parsed.symptom_id,
                severity=parsed.severity,
                source_text=parsed.source_text,
            )
        )
    return out


def _expand_multi_symptom_event(parsed: ParsedEvent, text: str) -> list[ParsedEvent]:
    if parsed.event_type != "symptom":
        return [parsed]
    clause = (text or "").strip()
    if not clause:
        return [parsed]
    if not re.search(r"\b(and|/|\+)\b|[+/]", clause, re.I):
        return [parsed]

    parts = [
        part.strip(" ,.;")
        for part in re.split(r"\s*(?:,|/|\+|\band\b)\s*", clause, flags=re.I)
        if part and part.strip(" ,.;")
    ]
    if len(parts) <= 1:
        return [parsed]

    out: list[ParsedEvent] = [parsed]
    seen_symptom_ids: set[int] = set()
    if parsed.symptom_id is not None:
        seen_symptom_ids.add(int(parsed.symptom_id))

    # Preserve the original clause context (time words, etc.) but parse each symptom fragment.
    for part in parts:
        part_lower = part.lower()
        if not (_SYMPTOM_CUES_RE.search(part_lower) or _COMMON_SYMPTOM_TERMS_RE.search(part_lower)):
            continue
        candidate = _normalize_symptom_candidate(_clean_candidate_text(part))
        if not candidate or not _is_valid_symptom_candidate(candidate, clause):
            continue
        symptom_id = _resolve_symptom_with_fallback(candidate)
        if symptom_id is None:
            continue
        symptom_id_int = int(symptom_id)
        if symptom_id_int in seen_symptom_ids:
            continue
        seen_symptom_ids.add(symptom_id_int)
        out.append(
            ParsedEvent(
                event_type="symptom",
                timestamp=parsed.timestamp,
                time_range_start=parsed.time_range_start,
                time_range_end=parsed.time_range_end,
                time_confidence=parsed.time_confidence,
                item_id=None,
                route=None,
                symptom_id=symptom_id_int,
                severity=parsed.severity,
                source_text=clause,
            )
        )
    return out


def _normalize_space(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _segment_time_anchor(
    segment: str,
    *,
    context_timestamp: str | None,
    context_start: str | None,
    context_end: str | None,
    local_tz=None,
) -> tuple[str | None, str | None, str | None]:
    seg_ts, seg_start, seg_end = _parse_time(segment, local_tz=local_tz)
    if _has_strong_date_anchor(segment) and any([seg_ts, seg_start, seg_end]):
        return seg_ts, seg_start, seg_end
    anaphoric_seg_ts = _resolve_anaphoric_daypart_time(
        segment,
        context_timestamp=context_timestamp,
        context_start=context_start,
        context_end=context_end,
        local_tz=local_tz,
    )
    if anaphoric_seg_ts is not None:
        return anaphoric_seg_ts, None, None
    relative_seg_ts = _resolve_relative_clause_time(
        segment,
        context_timestamp=context_timestamp,
        context_start=context_start,
        context_end=context_end,
        local_tz=local_tz,
    )
    if relative_seg_ts is not None:
        return relative_seg_ts, None, None
    return seg_ts, seg_start, seg_end


def _apply_context_to_api_events(
    *,
    text: str,
    api_events: list[ParsedEvent],
    local_tz=None,
) -> list[ParsedEvent]:
    if not api_events:
        return []
    segments = _split_into_segments(text)
    if not segments:
        segments = [text.strip()]

    segment_anchors: list[tuple[str, str | None, str | None, str | None]] = []
    last_context_timestamp: str | None = None
    last_context_start: str | None = None
    last_context_end: str | None = None
    for segment in segments:
        segment_has_strong_date_anchor = _has_strong_date_anchor(segment)
        seg_ts, seg_start, seg_end = _segment_time_anchor(
            segment,
            context_timestamp=last_context_timestamp,
            context_start=last_context_start,
            context_end=last_context_end,
            local_tz=local_tz,
        )
        segment_anchors.append((segment, seg_ts, seg_start, seg_end))
        if any([seg_ts, seg_start, seg_end]):
            last_context_timestamp = seg_ts
            last_context_start = seg_start
            last_context_end = seg_end

    out: list[ParsedEvent] = []
    for parsed in api_events:
        if _event_has_time(parsed):
            out.append(parsed)
            continue
        source = _normalize_space(parsed.source_text or "")
        chosen: tuple[str | None, str | None, str | None] | None = None
        for segment, seg_ts, seg_start, seg_end in segment_anchors:
            if source and source in _normalize_space(segment):
                chosen = (seg_ts, seg_start, seg_end)
                break
        if chosen is None:
            for _segment, seg_ts, seg_start, seg_end in reversed(segment_anchors):
                if any([seg_ts, seg_start, seg_end]):
                    chosen = (seg_ts, seg_start, seg_end)
                    break
        if chosen is None:
            out.append(parsed)
            continue
        out.append(
            _apply_time_context(
                parsed,
                context_timestamp=chosen[0],
                context_start=chosen[1],
                context_end=chosen[2],
            )
        )
    return out


def parse_text_events(text: str, *, local_tz=None) -> list[ParsedEvent]:
    api_events = parse_with_api_events(text, local_tz=local_tz)

    segments = _split_into_segments(text)
    if not segments:
        segments = [text.strip()]
    text_lower = text.lower()
    full_text_mixed_signal = (
        _EXPOSURE_VERBS.search(text_lower) is not None
        and (_SYMPTOM_CUES_RE.search(text_lower) is not None or _COMMON_SYMPTOM_TERMS_RE.search(text_lower) is not None)
    )
    full_text_time_signal_mentions = (
        len(_DATE_TOKEN_RE.findall(text_lower))
        + len(_TIME_AT_RE.findall(text_lower))
        + len(_DAYS_AGO_RE.findall(text_lower))
    )
    should_skip_api_seed = (
        full_text_mixed_signal
        and (
            full_text_time_signal_mentions >= 2
            or re.search(r"\b(and now|later|after that|afterwards)\b", text_lower) is not None
        )
    )
    parsed_events: list[ParsedEvent] = []
    seen_keys: set[tuple[str, str | None, str | None, str | None, int | None, int | None, str | None]] = set()
    last_context_timestamp: str | None = None
    last_context_start: str | None = None
    last_context_end: str | None = None

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

    def _parse_event_without_api(text_value: str) -> ParsedEvent | None:
        try:
            return parse_text_event(text_value, allow_api=False, local_tz=local_tz)
        except TypeError:
            # Preserve compatibility with tests that monkeypatch parse_text_event
            # using the old single-argument signature.
            return parse_text_event(text_value)

    # Prefer API extraction on long blurbs, but reinforce missing time context using
    # deterministic segment anchors so API rows do not lose relative daypart intent.
    if api_events and not should_skip_api_seed:
        for parsed in _apply_context_to_api_events(text=text, api_events=api_events, local_tz=local_tz):
            expanded_rows = _expand_multi_item_exposure(parsed, parsed.source_text or text)
            symptom_expanded_rows: list[ParsedEvent] = []
            for row in expanded_rows:
                symptom_expanded_rows.extend(_expand_multi_symptom_event(row, row.source_text or text))
            for expanded in symptom_expanded_rows:
                _append_if_new(expanded)
                if _event_has_time(expanded):
                    last_context_timestamp = expanded.timestamp
                    last_context_start = expanded.time_range_start
                    last_context_end = expanded.time_range_end

    for segment in segments:
        segment_has_strong_date_anchor = _has_strong_date_anchor(segment)
        seg_ts, seg_start, seg_end = _segment_time_anchor(
            segment,
            context_timestamp=last_context_timestamp,
            context_start=last_context_start,
            context_end=last_context_end,
            local_tz=local_tz,
        )
        if any([seg_ts, seg_start, seg_end]):
            last_context_timestamp, last_context_start, last_context_end = seg_ts, seg_start, seg_end
        parsed = _parse_event_without_api(segment)
        segment_lower = segment.lower()
        has_multi_exposure_clause = re.search(
            r"\band\s+for\s+(?:breakfast|lunch|dinner)\b",
            segment_lower,
        ) is not None
        has_multiple_exposure_mentions = (
            len(
                re.findall(
                    r"\b(ate|eat|eaten|eating|drank|drink|drinking|took|take|taking|smoked|used|use|using|apply|applied|had|have|having)\b",
                    # Include newer common grammar patterns so multi-event detection
                    # keeps pace with exposure classification.
                    segment_lower,
                )
            )
            >= 2
        )
        has_mixed_signal = (
            _EXPOSURE_VERBS.search(segment_lower) is not None
            and (_SYMPTOM_CUES_RE.search(segment_lower) is not None or _COMMON_SYMPTOM_TERMS_RE.search(segment_lower) is not None)
        )
        symptom_signal_mentions = len(_COMMON_SYMPTOM_TERMS_RE.findall(segment_lower)) + len(_SYMPTOM_CUES_RE.findall(segment_lower))
        time_signal_mentions = (
            len(_DATE_TOKEN_RE.findall(segment_lower))
            + len(_TIME_AT_RE.findall(segment_lower))
            + len(_DAYS_AGO_RE.findall(segment_lower))
        )
        has_multi_temporal_symptom_clause = (
            " and " in f" {segment_lower} "
            and symptom_signal_mentions >= 2
            and time_signal_mentions >= 2
        )

        should_trust_whole_segment_parse = not (
            has_mixed_signal
            or has_multi_exposure_clause
            or has_multiple_exposure_mentions
            or has_multi_temporal_symptom_clause
        )

        if parsed is not None:
            parsed = _apply_time_context(
                parsed,
                context_timestamp=seg_ts or last_context_timestamp,
                context_start=seg_start or last_context_start,
                context_end=seg_end or last_context_end,
            )
            # Prefer clause parsing over whole-segment parse for complex mixed/multi-event segments
            # to avoid blended artifacts (e.g., first time phrase incorrectly applied to later symptoms).
            if should_trust_whole_segment_parse:
                expanded_rows = _expand_multi_item_exposure(parsed, segment)
                symptom_expanded_rows: list[ParsedEvent] = []
                for row in expanded_rows:
                    symptom_expanded_rows.extend(_expand_multi_symptom_event(row, segment))
                for expanded in symptom_expanded_rows:
                    _append_if_new(expanded)
            if should_trust_whole_segment_parse:
                continue
        # Fallback for mixed blurbs in one sentence: split into clause-like chunks.
        clauses = [
            part.strip()
            for part in re.split(
                r"\s*(?:,|/|\+|\band then\b|\bthen\b|\band after\b|\bafter\b|\band\b|\band for (?:breakfast|lunch|dinner)\b)\s*",
                segment,
                flags=re.I,
            )
            if part.strip()
        ]
        for clause in clauses:
            clause_ts, clause_start, clause_end = _parse_time(clause, local_tz=local_tz)
            anaphoric_clause_ts = _resolve_anaphoric_daypart_time(
                clause,
                context_timestamp=last_context_timestamp or seg_ts,
                context_start=last_context_start or seg_start,
                context_end=last_context_end or seg_end,
                local_tz=local_tz,
            )
            if anaphoric_clause_ts is not None:
                clause_ts, clause_start, clause_end = anaphoric_clause_ts, None, None
            bedtime_clause_ts = _resolve_bedtime_clause_time(
                clause,
                context_timestamp=last_context_timestamp or seg_ts,
                context_start=last_context_start or seg_start,
                context_end=last_context_end or seg_end,
                local_tz=local_tz,
            )
            if bedtime_clause_ts is not None and clause_ts is None and clause_start is None and clause_end is None:
                clause_ts, clause_start, clause_end = bedtime_clause_ts, None, None
            clause_parsed = _parse_event_without_api(clause)
            if clause_parsed is not None:
                if anaphoric_clause_ts is not None:
                    clause_parsed = _override_parsed_timestamp(clause_parsed, anaphoric_clause_ts)
                elif bedtime_clause_ts is not None and clause_parsed.timestamp is None:
                    clause_parsed = _override_parsed_timestamp(clause_parsed, bedtime_clause_ts)
                elif segment_has_strong_date_anchor and not _has_strong_date_anchor(clause):
                    # Preserve explicit segment-level date anchor (e.g., "On February 10 ...")
                    # for weak daypart-only clauses split from the same sentence.
                    clause_ts, clause_start, clause_end = seg_ts, seg_start, seg_end
                    if seg_ts is not None:
                        clause_parsed = _override_parsed_timestamp(clause_parsed, seg_ts)
                clause_parsed = _apply_time_context(
                    clause_parsed,
                    context_timestamp=clause_ts or last_context_timestamp or seg_ts,
                    context_start=clause_start or last_context_start or seg_start,
                    context_end=clause_end or last_context_end or seg_end,
                )
                expanded_rows = _expand_multi_item_exposure(clause_parsed, clause)
                symptom_expanded_rows: list[ParsedEvent] = []
                for row in expanded_rows:
                    symptom_expanded_rows.extend(_expand_multi_symptom_event(row, clause))
                for expanded in symptom_expanded_rows:
                    _append_if_new(expanded)
                    if _has_time_info(expanded):
                        last_context_timestamp = expanded.timestamp
                        last_context_start = expanded.time_range_start
                        last_context_end = expanded.time_range_end
    return parsed_events


def _api_event_to_parsed_event(
    entry: dict,
    full_text: str,
    *,
    local_tz=None,
) -> ParsedEvent | None:
    source_text = str(entry.get("source_text") or "").strip()
    candidate = str(entry.get("candidate") or "").strip()
    event_type = entry.get("event_type")
    if event_type not in {"exposure", "symptom"}:
        return None
    if not source_text:
        source_text = _pick_source_segment(full_text, candidate, str(event_type))

    source_text = source_text.strip()
    cleaned_candidate = _clean_candidate_text(candidate) if candidate else ""
    timestamp, time_range_start, time_range_end = _parse_time(source_text, local_tz=local_tz)
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
            fallback_item = _fallback_exposure_candidate_from_text(source_for_split)
            if fallback_item:
                item_id = resolve_item_id(fallback_item)
        if item_id is None:
            return None
        route = _choose_route_from_api_or_rules(source_text, entry.get("route"))
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
        symptom_id = _resolve_symptom_with_fallback(normalized_symptom)
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


def parse_with_api_events(text: str, *, local_tz=None) -> list[ParsedEvent]:
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
                                "enum": [None, "ingestion", "dermal", "inhalation", "injection", "proximity_environment", "behavioral", "other"],
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
                    "5) Keep source_text as the smallest phrase needed for that event. "
                    "6) Preserve any explicit relative time phrase in source_text "
                    "(e.g., yesterday morning, this evening, last night, two days ago)."
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
        parsed = _api_event_to_parsed_event(entry, text, local_tz=local_tz)
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


def parse_with_api(text: str, *, local_tz=None) -> ParsedEvent | None:
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

    timestamp = _coerce_api_timestamp_to_today_if_time_only(
        text,
        data.get("timestamp"),
        local_tz=local_tz,
    )
    time_range_start = data.get("time_range_start")
    time_range_end = data.get("time_range_end")
    time_range_start, time_range_end = _coerce_api_range_to_today_if_time_only(
        text,
        time_range_start,
        time_range_end,
        local_tz=local_tz,
    )
    timestamp = to_utc_iso(timestamp, strict=False)
    time_range_start = to_utc_iso(time_range_start, strict=False)
    time_range_end = to_utc_iso(time_range_end, strict=False)
    timestamp, time_range_start, time_range_end = _override_with_daypart_fixed_time(
        text,
        timestamp,
        time_range_start,
        time_range_end,
        local_tz=local_tz,
    )
    # If API extraction missed time fields, fall back to deterministic parser on the raw text.
    if not any([timestamp, time_range_start, time_range_end]):
        fallback_ts, fallback_start, fallback_end = _parse_time(text, local_tz=local_tz)
        timestamp = to_utc_iso(fallback_ts, strict=False)
        time_range_start = to_utc_iso(fallback_start, strict=False)
        time_range_end = to_utc_iso(fallback_end, strict=False)
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
        route = _choose_route_from_api_or_rules(text, route)
        symptom_id = None
        severity = None
    else:
        if symptom_id is None:
            symptom_name = data.get("symptom_name") or data.get("candidate")
            if isinstance(symptom_name, str) and symptom_name.strip():
                cleaned_symptom = _clean_candidate_text(symptom_name)
                normalized_symptom = _normalize_symptom_candidate(cleaned_symptom)
                if _is_valid_symptom_candidate(normalized_symptom, text):
                    symptom_id = _resolve_symptom_with_fallback(normalized_symptom)
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


def parse_with_rules(text: str, *, local_tz=None) -> ParsedEvent | None:
    event_type = _guess_event_type(text)
    timestamp, range_start, range_end = _parse_time(text, local_tz=local_tz)
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
            candidate = _fallback_exposure_candidate_from_text(text) or ""
        if not candidate:
            return None
        item_id = resolve_item_id(candidate)
        if item_id is None:
            fallback_candidate = _fallback_exposure_candidate_from_text(text)
            if fallback_candidate and fallback_candidate != candidate:
                item_id = resolve_item_id(fallback_candidate)
        if item_id is None:
            return None
        route = _infer_route(text)
    else:
        candidate = _normalize_symptom_candidate(cleaned_tokens[0]) if cleaned_tokens else ""
        if not candidate or not _is_valid_symptom_candidate(candidate, text):
            return None
        symptom_id = _resolve_symptom_with_fallback(candidate)
        if symptom_id is None:
            return None
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
def ingest_text_event(
    user_id: int,
    raw_text: str,
    *,
    tz_offset_minutes: int | None = None,
) -> dict:
    local_tz = _tz_from_offset_minutes(tz_offset_minutes)
    parsed_events = parse_text_events(raw_text, local_tz=local_tz)
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
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
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
                expand_exposure_event(int(cursor.fetchone()["id"]), conn=conn)
                events_written += 1
            symptom_rows = conn.execute(
                "SELECT DISTINCT symptom_id FROM symptom_events WHERE user_id = %s",
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
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                    "SELECT DISTINCT item_id FROM exposure_events WHERE user_id = %s",
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
