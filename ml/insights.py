from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from api.db import get_connection
from ingestion.expand_exposure import backfill_missing_exposure_expansions
from ml.evaluator import (
    ROUTE_BUCKETS,
    compute_evidence_quality,
    compute_penalty_score,
    get_decision_thresholds,
    predict_model_probability,
)
from ml.final_score import predict_final_score
from ml.rag import (
    aggregate_evidence,
    build_candidate_query,
    fetch_ingredient_name_map,
    fetch_item_name_map,
    fetch_symptom_name_map,
    generate_user_evidence_summary,
    retrieve_claim_evidence,
)

# generate insight candidates for many possible temporal patterns of exposure
ROLLING_WINDOW = timedelta(days=7)
LAG_BUCKETS = (
    ("0_6h", timedelta(hours=0), timedelta(hours=6)),
    ("6_24h", timedelta(hours=6), timedelta(hours=24)),
    ("24_72h", timedelta(hours=24), timedelta(hours=72)),
    ("72h_7d", timedelta(hours=72), timedelta(days=7)),
)
MAX_LAG_WINDOW = LAG_BUCKETS[-1][2]
_LAG_BUCKET_WIDTH_DAYS = {
    "0_6h": 0.25,
    "6_24h": 0.75,
    "24_72h": 2.0,
    "72h_7d": 4.0,
}
_LOW_SIGNAL_SYMPTOM_TOKENS = {
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
    "went",
}
_COMMON_SYMPTOM_TERM_RE = re.compile(
    r"\b("
    r"headache|migraine|nausea|vomit|vomiting|stomachache|stomach pain|"
    r"acne|rash|itch|itchy|hives|fatigue|tired|brain fog|"
    r"diarrhea|constipation|bloat|bloating|cramp|cramps|"
    r"dizzy|dizziness|anxiety|insomnia|fever|cough|sore throat"
    r")\b",
    re.I,
)
_GENERIC_SYMPTOM_NAMES = {
    "sick",
    "ill",
    "unwell",
    "not feeling well",
    "felt sick",
    "feeling sick",
    "felt bad",
    "feeling bad",
    "bad",
}
_BROAD_GI_SYMPTOM_TERMS = {
    "stomachache",
    "stomach pain",
    "abdominal pain",
    "diarrhea",
    "nausea",
    "vomiting",
    "gastroenteritis",
}
_CONDITIONAL_HAZARD_TERMS = {
    "undercooked",
    "improperly cooked",
    "contaminated",
    "foodborne",
    "infection",
    "pathogen",
    "outbreak",
    "bacterial",
    "campylobacter",
    "salmonella",
    "e coli",
    "listeria",
}
_GENERIC_MONOTONE_ITEM_TERMS = {
    "work",
    "at work",
    "office",
    "workplace",
    "school",
    "class",
    "commute",
    "traffic",
}
_EXPOSURE_QUALIFIER_TERMS = {
    "hard day",
    "stressful",
    "high stress",
    "overtime",
    "long shift",
    "late shift",
    "physically demanding",
    "poor ventilation",
    "toxic",
    "fumes",
    "dusty",
    "loud",
    "dehydrated",
    "didn't sleep",
    "did not sleep",
    "poor sleep",
    "bad sleep",
}


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()

def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None

def _route_key(route: str | None) -> str | None:
    if route is None:
        return None
    return route.strip().lower() or None


def _coerce_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _time_confidence_score(value: str | None) -> float:
    normalized = (value or "").strip().lower()
    if normalized == "exact":
        return 1.0
    if normalized == "approx":
        return 0.7
    if normalized == "backfilled":
        return 0.4
    return 0.55


def _is_low_signal_symptom_name(value: str | None) -> bool:
    normalized = " ".join((value or "").strip().lower().split())
    if not normalized:
        return True
    if normalized in _LOW_SIGNAL_SYMPTOM_TOKENS:
        return True
    tokens = normalized.split()
    if all(token in _LOW_SIGNAL_SYMPTOM_TOKENS for token in tokens):
        return True
    # Reject free-text phrasey entries that have no known symptom term.
    if len(tokens) >= 2 and _COMMON_SYMPTOM_TERM_RE.search(normalized) is None:
        return True
    return False


def _generic_symptom_penalty(value: str | None) -> float:
    normalized = " ".join((value or "").strip().lower().split())
    if not normalized:
        return 0.25
    if normalized in _GENERIC_SYMPTOM_NAMES:
        return 0.25
    if normalized.startswith("felt ") and len(normalized.split()) <= 3:
        return 0.18
    return 0.0


def _conditional_hazard_penalty(
    *,
    item_name: str | None,
    symptom_name: str | None,
    citations: list[dict[str, Any]],
) -> float:
    symptom = " ".join((symptom_name or "").strip().lower().split())
    if not symptom:
        return 0.0
    if not any(term in symptom for term in _BROAD_GI_SYMPTOM_TERMS):
        return 0.0
    item = " ".join((item_name or "").strip().lower().split())
    if any(token in item for token in {"undercooked", "raw", "spoiled", "contaminated"}):
        return 0.0
    if not citations:
        return 0.0

    hazard_hits = 0
    for citation in citations:
        snippet = str(citation.get("snippet") or "").lower()
        title = str(citation.get("title") or "").lower()
        text = f"{title} {snippet}".strip()
        if any(term in text for term in _CONDITIONAL_HAZARD_TERMS):
            hazard_hits += 1
    ratio = hazard_hits / max(1, len(citations))
    if ratio >= 0.5:
        return 0.22
    if ratio >= 0.25:
        return 0.12
    return 0.0


def _merge_claim_rows(*claim_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for claim_list in claim_lists:
        for row in claim_list:
            key = (
                row.get("id"),
                row.get("paper_id"),
                row.get("citation_url"),
                row.get("chunk_index"),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
    return merged


def _build_route_temporal_feature_map(
    *,
    total_cooccurrence: float,
    route_counts: dict[str, int],
    route_min_lag_minutes: dict[str, float],
    route_lag_bucket_counts: dict[str, dict[str, int]],
) -> dict[str, float]:
    total = max(1.0, float(total_cooccurrence))
    out: dict[str, float] = {}
    for route in ROUTE_BUCKETS:
        count = float(route_counts.get(route, 0))
        min_lag = float(route_min_lag_minutes.get(route, 0.0))
        lag_counts = route_lag_bucket_counts.get(route, {})
        early = float(lag_counts.get("0_6h", 0) + lag_counts.get("6_24h", 0))
        delayed = float(lag_counts.get("24_72h", 0) + lag_counts.get("72h_7d", 0))
        denom = max(1.0, count)
        out[f"route_share_{route}"] = count / total
        out[f"route_min_lag_{route}"] = min_lag
        out[f"route_lag_0_24h_share_{route}"] = early / denom
        out[f"route_lag_24h_7d_share_{route}"] = delayed / denom
    return out


def _tokenize_entity_name(value: str | None) -> set[str]:
    normalized = re.sub(r"[^a-z0-9\s]+", " ", (value or "").strip().lower())
    tokens = {token for token in normalized.split() if len(token) >= 3}
    # Keep compact aliases for common short forms if explicitly present.
    if "ph" in normalized:
        tokens.add("ph")
    return tokens


def _claim_mentions_tokens(claim: dict[str, Any], tokens: set[str]) -> bool:
    if not tokens:
        return False
    text = " ".join(
        [
            str(claim.get("summary") or ""),
            str(claim.get("chunk_text") or ""),
            str(claim.get("citation_snippet") or ""),
            str(claim.get("citation_title") or ""),
            str(claim.get("title") or ""),
            str(claim.get("abstract") or ""),
        ]
    ).lower()
    if not text.strip():
        return False
    for token in tokens:
        if re.search(rf"\b{re.escape(token)}\b", text):
            return True
    return False


def _filter_combo_pair_claims(
    claims: list[dict[str, Any]],
    *,
    item_a_name: str | None,
    item_b_name: str | None,
) -> list[dict[str, Any]]:
    # Combo evidence must mention both items in the same claim context.
    tokens_a = _tokenize_entity_name(item_a_name)
    tokens_b = _tokenize_entity_name(item_b_name)
    if not tokens_a or not tokens_b:
        return []
    out: list[dict[str, Any]] = []
    for claim in claims:
        if _claim_mentions_tokens(claim, tokens_a) and _claim_mentions_tokens(claim, tokens_b):
            out.append(claim)
    return out

# each lag is assigned to first matching bucket, converting lag into retrivable categories
def _lag_bucket_label(lag: timedelta) -> str | None:
    for label, start, end in LAG_BUCKETS:
        if lag >= start and lag <= end:
            return label
    return None


def _estimate_temporal_lift(
    *,
    unique_exposure_events: float,
    exposure_count_7d: float,
    symptom_count_7d: float,
    lag_bucket_counts: dict[str, int],
) -> float:
    exposure_total = max(1.0, float(exposure_count_7d))
    observed_prob = min(1.0, max(0.0, float(unique_exposure_events) / exposure_total))
    dominant_lag_bucket = None
    if lag_bucket_counts:
        dominant_lag_bucket = max(lag_bucket_counts.items(), key=lambda row: row[1])[0]
    lag_window_days = _LAG_BUCKET_WIDTH_DAYS.get(str(dominant_lag_bucket), 1.0)
    symptom_daily_rate = max(0.0, float(symptom_count_7d)) / 7.0
    expected_prob = min(0.95, max(0.01, symptom_daily_rate * lag_window_days))
    return min(4.0, max(0.0, observed_prob / expected_prob))


@dataclass
class ExposureEvent:
    event_id: int
    item_id: int
    event_ts: datetime
    route: str | None
    raw_text: str | None
    has_expansion: bool
    time_confidence_score: float

@dataclass
class SymptomEvent:
    event_id: int
    symptom_id: int
    event_ts: datetime
    severity: int | None
    time_confidence_score: float

# define shape of candidate linkage
@dataclass
class CandidateAggregate:
    user_id: int
    item_id: int
    symptom_id: int
    lags_minutes: list[float] = field(default_factory=list)
    symptom_severity_values: list[int] = field(default_factory=list)
    exposure_event_ids: set[int] = field(default_factory=set)
    symptom_event_ids: set[int] = field(default_factory=set)
    routes: set[str] = field(default_factory=set)
    lag_bucket_counts: dict[str, int] = field(default_factory=dict)
    route_counts: dict[str, int] = field(default_factory=dict)
    route_min_lag_minutes: dict[str, float] = field(default_factory=dict)
    route_lag_bucket_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    exposure_with_ingredients_count: int = 0
    time_confidence_values: list[float] = field(default_factory=list)
    latest_symptom_ts: datetime | None = None
    exposure_context_count: int = 0
    exposure_qualifier_hits: int = 0


@dataclass
class IngredientAggregate:
    user_id: int
    ingredient_id: int
    symptom_id: int
    lags_minutes: list[float] = field(default_factory=list)
    symptom_severity_values: list[int] = field(default_factory=list)
    exposure_event_ids: set[int] = field(default_factory=set)
    symptom_event_ids: set[int] = field(default_factory=set)
    routes: set[str] = field(default_factory=set)
    lag_bucket_counts: dict[str, int] = field(default_factory=dict)
    route_counts: dict[str, int] = field(default_factory=dict)
    route_min_lag_minutes: dict[str, float] = field(default_factory=dict)
    route_lag_bucket_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    time_confidence_values: list[float] = field(default_factory=list)
    latest_symptom_ts: datetime | None = None


@dataclass
class ComboAggregate:
    user_id: int
    item_a_id: int
    item_b_id: int
    symptom_id: int
    lags_minutes: list[float] = field(default_factory=list)
    symptom_severity_values: list[int] = field(default_factory=list)
    exposure_event_ids: set[int] = field(default_factory=set)
    symptom_event_ids: set[int] = field(default_factory=set)
    routes: set[str] = field(default_factory=set)
    lag_bucket_counts: dict[str, int] = field(default_factory=dict)
    route_counts: dict[str, int] = field(default_factory=dict)
    route_min_lag_minutes: dict[str, float] = field(default_factory=dict)
    route_lag_bucket_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    time_confidence_values: list[float] = field(default_factory=list)
    latest_symptom_ts: datetime | None = None

    @property
    def combo_key(self) -> str:
        return f"{min(self.item_a_id, self.item_b_id)}:{max(self.item_a_id, self.item_b_id)}"

    @property
    def item_ids_sorted(self) -> tuple[int, int]:
        return (min(self.item_a_id, self.item_b_id), max(self.item_a_id, self.item_b_id))


def _fetch_exposure_ingredients(user_id: int) -> dict[int, set[int]]:
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT e.id AS exposure_event_id, x.ingredient_id AS ingredient_id
            FROM exposure_events e
            JOIN exposure_expansions x ON x.exposure_event_id = e.id
            WHERE e.user_id = ?
              AND x.ingredient_id IS NOT NULL
            """,
            (user_id,),
        ).fetchall()
    finally:
        conn.close()
    mapping: dict[int, set[int]] = {}
    for row in rows:
        event_id = row["exposure_event_id"]
        ingredient_id = row["ingredient_id"]
        if ingredient_id is None:
            continue
        mapping.setdefault(event_id, set()).add(int(ingredient_id))
    return mapping


def _fetch_exposures(user_id: int) -> list[ExposureEvent]:
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT
                e.id AS event_id,
                e.item_id AS item_id,
                COALESCE(e.timestamp, e.time_range_start) AS event_ts,
                e.route AS route,
                e.raw_text AS raw_text,
                e.time_confidence AS time_confidence,
                CASE WHEN COUNT(x.id) > 0 THEN 1 ELSE 0 END AS has_expansion
            FROM exposure_events e
            LEFT JOIN exposure_expansions x ON x.exposure_event_id = e.id
            WHERE e.user_id = ?
              AND COALESCE(e.timestamp, e.time_range_start) IS NOT NULL
            GROUP BY e.id
            ORDER BY event_ts
            """,
            (user_id,),
        ).fetchall()
    finally:
        conn.close()
    out: list[ExposureEvent] = []
    for row in rows:
        parsed_ts = _parse_iso(row["event_ts"])
        if parsed_ts is None:
            continue
        out.append(
            ExposureEvent(
                event_id=row["event_id"],
                item_id=row["item_id"],
                event_ts=parsed_ts,
                route=_route_key(row["route"]),
                raw_text=row["raw_text"],
                has_expansion=bool(row["has_expansion"]),
                time_confidence_score=_time_confidence_score(row["time_confidence"]),
            )
        )
    return out


def _is_generic_monotone_item(item_name: str | None) -> bool:
    normalized = " ".join((item_name or "").strip().lower().split())
    return normalized in _GENERIC_MONOTONE_ITEM_TERMS


def _has_exposure_qualifier(raw_text: str | None) -> bool:
    text = " ".join((raw_text or "").strip().lower().split())
    if not text:
        return False
    return any(term in text for term in _EXPOSURE_QUALIFIER_TERMS)

def _fetch_symptoms(user_id: int) -> list[SymptomEvent]:
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT
                s.id AS event_id,
                s.symptom_id AS symptom_id,
                COALESCE(s.timestamp, s.time_range_start) AS event_ts,
                s.severity AS severity,
                s.time_confidence AS time_confidence
            FROM symptom_events s
            WHERE s.user_id = ?
              AND COALESCE(s.timestamp, s.time_range_start) IS NOT NULL
            ORDER BY event_ts
            """,
            (user_id,),
        ).fetchall()
    finally:
        conn.close()
    out: list[SymptomEvent] = []
    for row in rows:
        parsed_ts = _parse_iso(row["event_ts"])
        if parsed_ts is None:
            continue
        out.append(
            SymptomEvent(
                event_id=row["event_id"],
                symptom_id=row["symptom_id"],
                event_ts=parsed_ts,
                severity=_coerce_int_or_none(row["severity"]),
                time_confidence_score=_time_confidence_score(row["time_confidence"]),
            )
        )
    return out

# builds candidates out of exposures and symptoms
# multiple candidates can be build from same exposure and symptom patterns but with different temporal windows
def _build_candidate_aggregates(
    user_id: int,
) -> tuple[
    dict[tuple[int, int], CandidateAggregate],
    dict[tuple[int, int], IngredientAggregate],
    dict[tuple[int, int, int], ComboAggregate],
    int,
    list[ExposureEvent],
    list[SymptomEvent],
    dict[int, set[int]],
]:
    # Ensure legacy rows or manually seeded exposures are expanded before candidate generation.
    backfill_missing_exposure_expansions(user_id)
    exposures = _fetch_exposures(user_id)
    symptoms = _fetch_symptoms(user_id)
    exposure_ingredients = _fetch_exposure_ingredients(user_id)
    aggregates: dict[tuple[int, int], CandidateAggregate] = {}
    ingredient_aggregates: dict[tuple[int, int], IngredientAggregate] = {}
    combo_aggregates: dict[tuple[int, int, int], ComboAggregate] = {}
    pair_count = 0

    # logic for candidate generation, ensuring realistic potential linkage
    for symptom in symptoms:
        eligible_exposures: list[tuple[ExposureEvent, timedelta, str]] = []
        for exposure in exposures:
            lag = symptom.event_ts - exposure.event_ts
            if lag.total_seconds() < 0:
                continue
            if lag > MAX_LAG_WINDOW:
                continue
            bucket = _lag_bucket_label(lag)
            if bucket is None:
                continue
            eligible_exposures.append((exposure, lag, bucket))

            pair_count += 1
            key = (exposure.item_id, symptom.symptom_id)
            candidate = aggregates.get(key)
            if candidate is None:
                candidate = CandidateAggregate(
                    user_id=user_id,
                    item_id=exposure.item_id,
                    symptom_id=symptom.symptom_id,
                )
                aggregates[key] = candidate

            candidate.lags_minutes.append(lag.total_seconds() / 60.0)
            candidate.exposure_event_ids.add(exposure.event_id)
            candidate.symptom_event_ids.add(symptom.event_id)
            candidate.exposure_context_count += 1
            if _has_exposure_qualifier(exposure.raw_text):
                candidate.exposure_qualifier_hits += 1
            if exposure.route:
                candidate.routes.add(exposure.route)
                candidate.route_counts[exposure.route] = candidate.route_counts.get(exposure.route, 0) + 1
                prev_min = candidate.route_min_lag_minutes.get(exposure.route)
                lag_minutes = lag.total_seconds() / 60.0
                if prev_min is None or lag_minutes < prev_min:
                    candidate.route_min_lag_minutes[exposure.route] = lag_minutes
                route_bucket_counts = candidate.route_lag_bucket_counts.setdefault(exposure.route, {})
                route_bucket_counts[bucket] = route_bucket_counts.get(bucket, 0) + 1
            candidate.lag_bucket_counts[bucket] = candidate.lag_bucket_counts.get(bucket, 0) + 1
            if exposure.has_expansion:
                candidate.exposure_with_ingredients_count += 1
            if symptom.severity is not None:
                candidate.symptom_severity_values.append(symptom.severity)
            candidate.time_confidence_values.append(
                min(exposure.time_confidence_score, symptom.time_confidence_score)
            )
            if candidate.latest_symptom_ts is None or symptom.event_ts > candidate.latest_symptom_ts:
                candidate.latest_symptom_ts = symptom.event_ts

            ingredient_ids = exposure_ingredients.get(exposure.event_id, set())
            for ingredient_id in ingredient_ids:
                ingredient_key = (ingredient_id, symptom.symptom_id)
                ingredient_candidate = ingredient_aggregates.get(ingredient_key)
                if ingredient_candidate is None:
                    ingredient_candidate = IngredientAggregate(
                        user_id=user_id,
                        ingredient_id=ingredient_id,
                        symptom_id=symptom.symptom_id,
                    )
                    ingredient_aggregates[ingredient_key] = ingredient_candidate
                ingredient_candidate.lags_minutes.append(lag.total_seconds() / 60.0)
                ingredient_candidate.exposure_event_ids.add(exposure.event_id)
                ingredient_candidate.symptom_event_ids.add(symptom.event_id)
                if exposure.route:
                    ingredient_candidate.routes.add(exposure.route)
                    ingredient_candidate.route_counts[exposure.route] = (
                        ingredient_candidate.route_counts.get(exposure.route, 0) + 1
                    )
                    prev_min_ing = ingredient_candidate.route_min_lag_minutes.get(exposure.route)
                    lag_minutes_ing = lag.total_seconds() / 60.0
                    if prev_min_ing is None or lag_minutes_ing < prev_min_ing:
                        ingredient_candidate.route_min_lag_minutes[exposure.route] = lag_minutes_ing
                    ing_route_bucket_counts = ingredient_candidate.route_lag_bucket_counts.setdefault(exposure.route, {})
                    ing_route_bucket_counts[bucket] = ing_route_bucket_counts.get(bucket, 0) + 1
                ingredient_candidate.lag_bucket_counts[bucket] = (
                    ingredient_candidate.lag_bucket_counts.get(bucket, 0) + 1
                )
                if symptom.severity is not None:
                    ingredient_candidate.symptom_severity_values.append(symptom.severity)
                ingredient_candidate.time_confidence_values.append(
                    min(exposure.time_confidence_score, symptom.time_confidence_score)
                )
                if (
                    ingredient_candidate.latest_symptom_ts is None
                    or symptom.event_ts > ingredient_candidate.latest_symptom_ts
                ):
                    ingredient_candidate.latest_symptom_ts = symptom.event_ts

        # Pairwise multi-exposure candidates: A + B -> symptom.
        # We only form combos from distinct items observed in the same symptom window.
        if len(eligible_exposures) >= 2:
            for idx in range(len(eligible_exposures)):
                exposure_a, lag_a, bucket_a = eligible_exposures[idx]
                for jdx in range(idx + 1, len(eligible_exposures)):
                    exposure_b, lag_b, bucket_b = eligible_exposures[jdx]
                    if exposure_a.item_id == exposure_b.item_id:
                        continue
                    item_a, item_b = sorted((int(exposure_a.item_id), int(exposure_b.item_id)))
                    combo_key = (item_a, item_b, int(symptom.symptom_id))
                    combo = combo_aggregates.get(combo_key)
                    if combo is None:
                        combo = ComboAggregate(
                            user_id=user_id,
                            item_a_id=item_a,
                            item_b_id=item_b,
                            symptom_id=int(symptom.symptom_id),
                        )
                        combo_aggregates[combo_key] = combo

                    combo.lags_minutes.append(min(lag_a.total_seconds(), lag_b.total_seconds()) / 60.0)
                    combo.exposure_event_ids.add(int(exposure_a.event_id))
                    combo.exposure_event_ids.add(int(exposure_b.event_id))
                    combo.symptom_event_ids.add(int(symptom.event_id))
                    if exposure_a.route:
                        combo.routes.add(exposure_a.route)
                        combo.route_counts[exposure_a.route] = combo.route_counts.get(exposure_a.route, 0) + 1
                        combo_prev_a = combo.route_min_lag_minutes.get(exposure_a.route)
                        lag_minutes_a = lag_a.total_seconds() / 60.0
                        if combo_prev_a is None or lag_minutes_a < combo_prev_a:
                            combo.route_min_lag_minutes[exposure_a.route] = lag_minutes_a
                        combo_route_bucket_a = combo.route_lag_bucket_counts.setdefault(exposure_a.route, {})
                        combo_route_bucket_a[bucket_a] = combo_route_bucket_a.get(bucket_a, 0) + 1
                    if exposure_b.route:
                        combo.routes.add(exposure_b.route)
                        combo.route_counts[exposure_b.route] = combo.route_counts.get(exposure_b.route, 0) + 1
                        combo_prev_b = combo.route_min_lag_minutes.get(exposure_b.route)
                        lag_minutes_b = lag_b.total_seconds() / 60.0
                        if combo_prev_b is None or lag_minutes_b < combo_prev_b:
                            combo.route_min_lag_minutes[exposure_b.route] = lag_minutes_b
                        combo_route_bucket_b = combo.route_lag_bucket_counts.setdefault(exposure_b.route, {})
                        combo_route_bucket_b[bucket_b] = combo_route_bucket_b.get(bucket_b, 0) + 1
                    combo.lag_bucket_counts[bucket_a] = combo.lag_bucket_counts.get(bucket_a, 0) + 1
                    combo.lag_bucket_counts[bucket_b] = combo.lag_bucket_counts.get(bucket_b, 0) + 1
                    if symptom.severity is not None:
                        combo.symptom_severity_values.append(symptom.severity)
                    combo.time_confidence_values.append(
                        min(
                            exposure_a.time_confidence_score,
                            exposure_b.time_confidence_score,
                            symptom.time_confidence_score,
                        )
                    )
                    if combo.latest_symptom_ts is None or symptom.event_ts > combo.latest_symptom_ts:
                        combo.latest_symptom_ts = symptom.event_ts

    return (
        aggregates,
        ingredient_aggregates,
        combo_aggregates,
        pair_count,
        exposures,
        symptoms,
        exposure_ingredients,
    )


def list_rag_sync_candidates(user_id: int) -> list[dict[str, Any]]:
    aggregates, _, combo_aggregates, _, _, _, exposure_ingredients = _build_candidate_aggregates(user_id)
    candidates: list[dict[str, Any]] = []
    for candidate in aggregates.values():
        ingredient_ids: set[int] = set()
        for exposure_event_id in candidate.exposure_event_ids:
            ingredient_ids.update(exposure_ingredients.get(exposure_event_id, set()))
        candidates.append(
            {
                "item_id": candidate.item_id,
                "symptom_id": candidate.symptom_id,
                "ingredient_ids": ingredient_ids,
                "routes": sorted(candidate.routes),
                "lag_bucket_counts": dict(candidate.lag_bucket_counts),
            }
        )
    for combo in combo_aggregates.values():
        candidates.append(
            {
                "item_id": int(combo.item_a_id),
                "secondary_item_id": int(combo.item_b_id),
                "symptom_id": int(combo.symptom_id),
                "ingredient_ids": set(),
                "routes": sorted(combo.routes),
                "lag_bucket_counts": dict(combo.lag_bucket_counts),
            }
        )
    return candidates


def recompute_insights(
    user_id: int,
    *,
    target_pairs: set[tuple[int, int]] | None = None,
) -> dict[str, int]:
    (
        aggregates,
        ingredient_aggregates,
        combo_aggregates,
        pair_count,
        exposures,
        symptoms,
        exposure_ingredients,
    ) = _build_candidate_aggregates(user_id)
    current_pairs = set(aggregates.keys())

    conn = get_connection()
    now_iso = _now_iso()
    thresholds = get_decision_thresholds()
    inserted = 0
    candidates_considered = 0
    pairs_evaluated = (
        sum(len(candidate.lags_minutes) for key, candidate in aggregates.items() if key in target_pairs)
        if target_pairs is not None
        else pair_count
    )
    try:
        item_name_map = fetch_item_name_map(conn)
        ingredient_name_map = fetch_ingredient_name_map(conn)
        symptom_name_map = fetch_symptom_name_map(conn)
        exposure_by_id = {int(event.event_id): event for event in exposures}

        if target_pairs is None:
            conn.execute("DELETE FROM insights WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM derived_features_ingredients WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM derived_features_combos WHERE user_id = ?", (user_id,))
            verification_rows = conn.execute(
                """
                SELECT item_id, symptom_id
                FROM insight_verifications
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchall()
            for row in verification_rows:
                pair = (int(row["item_id"]), int(row["symptom_id"]))
                if pair in current_pairs:
                    continue
                conn.execute(
                    """
                    DELETE FROM insight_verifications
                    WHERE user_id = ? AND item_id = ? AND symptom_id = ?
                    """,
                    (user_id, pair[0], pair[1]),
                )
        else:
            for item_id, symptom_id in sorted(target_pairs):
                conn.execute(
                    "DELETE FROM insights WHERE user_id = ? AND item_id = ? AND symptom_id = ?",
                    (user_id, item_id, symptom_id),
                )
                conn.execute(
                    "DELETE FROM derived_features WHERE user_id = ? AND item_id = ? AND symptom_id = ?",
                    (user_id, item_id, symptom_id),
                )
                conn.execute(
                    """
                    DELETE FROM derived_features_combos
                    WHERE user_id = ? AND symptom_id = ?
                      AND (combo_key LIKE ? OR combo_key LIKE ?)
                    """,
                    (user_id, symptom_id, f"{item_id}:%", f"%:{item_id}"),
                )
                conn.execute(
                    "DELETE FROM retrieval_runs WHERE user_id = ? AND item_id = ? AND symptom_id = ?",
                    (user_id, item_id, symptom_id),
                )
            stale_target_pairs = {
                pair for pair in target_pairs if pair not in current_pairs
            }
            for item_id, symptom_id in sorted(stale_target_pairs):
                conn.execute(
                    """
                    DELETE FROM insight_verifications
                    WHERE user_id = ? AND item_id = ? AND symptom_id = ?
                    """,
                    (user_id, item_id, symptom_id),
                )

        for candidate in aggregates.values():
            pair_key = (candidate.item_id, candidate.symptom_id)
            if target_pairs is not None and pair_key not in target_pairs:
                continue
            symptom_name = symptom_name_map.get(candidate.symptom_id)
            if _is_low_signal_symptom_name(symptom_name):
                continue
            generic_symptom_penalty = _generic_symptom_penalty(symptom_name)
            item_name = item_name_map.get(candidate.item_id)
            generic_monotone_item = _is_generic_monotone_item(item_name)
            qualifier_hits = 0
            qualifier_total = 0
            for exposure_event_id in candidate.exposure_event_ids:
                exposure_event = exposure_by_id.get(int(exposure_event_id))
                if exposure_event is None:
                    continue
                qualifier_total += 1
                if _has_exposure_qualifier(exposure_event.raw_text):
                    qualifier_hits += 1
            qualifier_ratio = (qualifier_hits / qualifier_total) if qualifier_total > 0 else 0.0
            candidates_considered += 1
            if not candidate.lags_minutes:
                continue
            lag_min = min(candidate.lags_minutes)
            lag_avg = sum(candidate.lags_minutes) / len(candidate.lags_minutes)
            cooccurrence_count = len(candidate.lags_minutes)
            cooccurrence_unique_symptom_count = len(candidate.symptom_event_ids)
            unique_exposure_events = float(len(candidate.exposure_event_ids))
            pair_density = (
                cooccurrence_count / cooccurrence_unique_symptom_count
                if cooccurrence_unique_symptom_count > 0
                else None
            )

            if candidate.latest_symptom_ts is None:
                continue
            window_start = candidate.latest_symptom_ts - ROLLING_WINDOW
            exposure_count_7d = sum(
                1
                for event in exposures
                if event.item_id == candidate.item_id
                and window_start <= event.event_ts <= candidate.latest_symptom_ts
            )
            symptom_count_7d = sum(
                1
                for event in symptoms
                if event.symptom_id == candidate.symptom_id
                and window_start <= event.event_ts <= candidate.latest_symptom_ts
            )
            temporal_lift = _estimate_temporal_lift(
                unique_exposure_events=unique_exposure_events,
                exposure_count_7d=float(exposure_count_7d),
                symptom_count_7d=float(symptom_count_7d),
                lag_bucket_counts=candidate.lag_bucket_counts,
            )
            severity_avg_after = (
                sum(candidate.symptom_severity_values) / len(candidate.symptom_severity_values)
                if candidate.symptom_severity_values
                else None
            )
            # writes and updates temporal and coocurrence features keyed by user id, item id, symptom id
            conn.execute(
                """
                INSERT INTO derived_features (
                    user_id, item_id, symptom_id, time_gap_min_minutes, time_gap_avg_minutes,
                    cooccurrence_count, cooccurrence_unique_symptom_count, pair_density,
                    exposure_count_7d, symptom_count_7d, severity_avg_after, computed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, item_id, symptom_id)
                DO UPDATE SET
                    time_gap_min_minutes = excluded.time_gap_min_minutes,
                    time_gap_avg_minutes = excluded.time_gap_avg_minutes,
                    cooccurrence_count = excluded.cooccurrence_count,
                    cooccurrence_unique_symptom_count = excluded.cooccurrence_unique_symptom_count,
                    pair_density = excluded.pair_density,
                    exposure_count_7d = excluded.exposure_count_7d,
                    symptom_count_7d = excluded.symptom_count_7d,
                    severity_avg_after = excluded.severity_avg_after,
                    computed_at = excluded.computed_at
                """,
                (
                    candidate.user_id,
                    candidate.item_id,
                    candidate.symptom_id,
                    lag_min,
                    lag_avg,
                    cooccurrence_count,
                    cooccurrence_unique_symptom_count,
                    pair_density,
                    exposure_count_7d,
                    symptom_count_7d,
                    severity_avg_after,
                    now_iso,
                ),
            )
            ingredient_ids: set[int] = set()
            for exposure_event_id in candidate.exposure_event_ids:
                ingredient_ids.update(exposure_ingredients.get(exposure_event_id, set()))
            query_text = build_candidate_query(
                item_name=item_name_map.get(candidate.item_id),
                symptom_name=symptom_name_map.get(candidate.symptom_id),
                routes=candidate.routes,
                lag_bucket_counts=candidate.lag_bucket_counts,
            )
            retrieved_claims = retrieve_claim_evidence(
                conn,
                ingredient_ids=ingredient_ids,
                item_id=candidate.item_id,
                symptom_id=candidate.symptom_id,
                query_text=query_text,
                top_k=max(5, int(os.getenv("RAG_DB_TOP_K", "12"))),
            )
            evidence = aggregate_evidence(
                retrieved_claims,
                item_name=item_name_map.get(candidate.item_id),
                symptom_name=symptom_name,
            )
            evidence_quality = compute_evidence_quality(evidence)

            exposure_with_ingredients_ratio = (
                candidate.exposure_with_ingredients_count / len(candidate.exposure_event_ids)
                if candidate.exposure_event_ids
                else 0.0
            )
            feature_map = {
                "time_gap_min_minutes": lag_min,
                "time_gap_avg_minutes": lag_avg,
                "cooccurrence_count": float(cooccurrence_count),
                "cooccurrence_unique_symptom_count": float(cooccurrence_unique_symptom_count),
                "pair_density": float(pair_density or 0.0),
                "exposure_count_7d": float(exposure_count_7d),
                "symptom_count_7d": float(symptom_count_7d),
                "severity_avg_after": float(severity_avg_after or 0.0),
                "route_count": float(len(candidate.routes)),
                "lag_bucket_diversity": float(len(candidate.lag_bucket_counts)),
                "exposure_with_ingredients_ratio": float(exposure_with_ingredients_ratio),
                "evidence_strength_score": float(evidence.get("evidence_strength_score") or 0.0),
                "evidence_score_signed": float(evidence.get("evidence_score") or 0.0),
                "citation_count": float(evidence_quality["citation_count"]),
                "support_ratio": float(evidence_quality["support_ratio"]),
                "contradict_ratio": float(evidence_quality["contradict_ratio"]),
                "neutral_ratio": float(evidence_quality["neutral_ratio"]),
                "avg_relevance": float(evidence_quality["avg_relevance"]),
                "study_quality_score": float(evidence_quality["study_quality_score"]),
                "population_match": float(evidence_quality["population_match"]),
                "temporality_match": float(evidence_quality["temporality_match"]),
                "risk_of_bias": float(evidence_quality["risk_of_bias"]),
                "llm_confidence": float(evidence_quality["llm_confidence"]),
                "time_confidence_score": (
                    sum(candidate.time_confidence_values) / len(candidate.time_confidence_values)
                    if candidate.time_confidence_values
                    else 0.0
                ),
                "temporal_lift": float(temporal_lift),
                "symptom_specificity_score": 1.0 - generic_symptom_penalty,
            }
            feature_map.update(
                _build_route_temporal_feature_map(
                    total_cooccurrence=float(cooccurrence_count),
                    route_counts=candidate.route_counts,
                    route_min_lag_minutes=candidate.route_min_lag_minutes,
                    route_lag_bucket_counts=candidate.route_lag_bucket_counts,
                )
            )

            model_probability = predict_model_probability(feature_map, use_calibration=True)
            conditional_hazard_penalty = _conditional_hazard_penalty(
                item_name=item_name_map.get(candidate.item_id),
                symptom_name=symptom_name,
                citations=evidence.get("citations") or [],
            )
            penalty_score = min(
                1.0,
                compute_penalty_score(feature_map)
                + generic_symptom_penalty
                + conditional_hazard_penalty,
            )
            monotone_penalty = 0.0
            if generic_monotone_item:
                min_qualifier_ratio = float(thresholds.get("generic_monotone_min_qualifier_ratio", 0.34))
                if qualifier_ratio < min_qualifier_ratio:
                    monotone_penalty = min(
                        0.22,
                        float(thresholds.get("generic_monotone_penalty", 0.14))
                        * (1.0 + (min_qualifier_ratio - qualifier_ratio)),
                    )
                    penalty_score = min(1.0, penalty_score + monotone_penalty)
            final_confidence = predict_final_score(
                model_probability=model_probability,
                evidence_quality=float(evidence_quality["score"]),
                penalty_score=penalty_score,
                citation_count=float(evidence_quality["citation_count"]),
                contradict_ratio=float(evidence_quality["contradict_ratio"]),
            )
            if generic_symptom_penalty >= 0.20:
                final_confidence = min(float(final_confidence), 0.35)
            if conditional_hazard_penalty >= 0.20:
                final_confidence = min(float(final_confidence), 0.65)

            min_recurrence_for_supported = float(thresholds.get("min_cooccurrence_for_supported", 2.0))
            min_unique_exposure_events_for_supported = float(
                thresholds.get("min_unique_exposure_events_for_supported", 2.0)
            )
            min_temporal_lift = float(thresholds.get("min_temporal_lift", 1.05))
            single_exposure_override_min_evidence_strength = float(
                thresholds.get("single_exposure_override_min_evidence_strength", 0.92)
            )
            single_exposure_override_min_citations = float(
                thresholds.get("single_exposure_override_min_citations", 4.0)
            )
            single_exposure_override_ok = (
                unique_exposure_events >= 1.0
                and float(evidence.get("evidence_strength_score") or 0.0)
                >= single_exposure_override_min_evidence_strength
                and float(evidence_quality.get("citation_count") or 0.0)
                >= single_exposure_override_min_citations
                and float(evidence_quality.get("support_ratio") or 0.0) >= 0.85
                and float(evidence_quality.get("contradict_ratio") or 0.0) <= 0.15
                and float(evidence.get("evidence_score") or 0.0) >= max(
                    0.20, float(thresholds.get("min_support_direction", 0.10))
                )
            )
            # Recurrence exception for high-signal one-offs:
            # allow support when model/evidence/final all agree strongly.
            recurrence_exception_ok = (
                unique_exposure_events >= 1.0
                and bool(evidence["citations"])
                and float(evidence_quality.get("citation_count") or 0.0) >= 2.0
                and float(evidence_quality.get("support_ratio") or 0.0) >= 0.70
                and float(evidence_quality.get("contradict_ratio") or 0.0) <= 0.25
                and float(evidence.get("evidence_score") or 0.0) >= float(thresholds.get("min_support_direction", 0.10))
                and float(evidence.get("evidence_strength_score") or 0.0) >= float(thresholds["min_evidence_strength"])
                and float(model_probability) >= float(thresholds["min_model_probability"])
                and float(final_confidence) >= float(thresholds["min_overall_confidence"])
            )

            dominant_lag_bucket = None
            if candidate.lag_bucket_counts:
                dominant_lag_bucket = max(
                    candidate.lag_bucket_counts.items(),
                    key=lambda row: row[1],
                )[0]
            evidence_summary = (
                generate_user_evidence_summary(
                    item_name=item_name_map.get(candidate.item_id),
                    symptom_name=symptom_name,
                    citations=evidence.get("citations") or [],
                    evidence_score=float(evidence.get("evidence_score") or 0.0),
                )
                or str(evidence["evidence_summary"])
            )
            if dominant_lag_bucket:
                evidence_summary = f"{evidence_summary} Dominant lag window: {dominant_lag_bucket}."

            if generic_symptom_penalty >= 0.20:
                decision_reason = "suppressed_generic_symptom"
            elif (
                generic_monotone_item
                and qualifier_ratio < float(thresholds.get("generic_monotone_min_qualifier_ratio", 0.34))
                and unique_exposure_events
                < float(thresholds.get("generic_monotone_min_unique_exposures_for_supported", 2.0))
            ):
                decision_reason = "suppressed_generic_monotone_context"
            elif float(evidence.get("evidence_score") or 0.0) < float(thresholds.get("min_support_direction", 0.10)):
                decision_reason = "suppressed_non_supportive_direction"
            elif (
                float(cooccurrence_count) < min_recurrence_for_supported
                or unique_exposure_events < min_unique_exposure_events_for_supported
            ) and not (single_exposure_override_ok or recurrence_exception_ok):
                decision_reason = "suppressed_insufficient_recurrence"
            elif float(temporal_lift) < min_temporal_lift and not (single_exposure_override_ok or recurrence_exception_ok):
                decision_reason = "suppressed_low_temporal_lift"
            elif conditional_hazard_penalty >= 0.20 and float(final_confidence) < float(
                thresholds["min_overall_confidence"]
            ):
                decision_reason = "suppressed_conditional_evidence_only"
            elif not evidence["citations"]:
                decision_reason = "suppressed_no_citations"
            elif float(evidence["evidence_strength_score"] or 0.0) < float(
                thresholds["min_evidence_strength"]
            ):
                decision_reason = "suppressed_low_evidence_strength"
            elif float(model_probability) < float(thresholds["min_model_probability"]):
                decision_reason = "suppressed_low_model_probability"
            elif float(final_confidence) < float(thresholds["min_overall_confidence"]):
                decision_reason = "suppressed_low_overall_confidence"
            else:
                decision_reason = "supported"

            insight_cursor = conn.execute(
                """
                INSERT INTO insights (
                    user_id, item_id, source_ingredient_id, symptom_id, model_score, evidence_score, final_score,
                    evidence_summary, evidence_strength_score, evidence_quality_score,
                    model_probability, penalty_score, display_decision_reason, citations_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.user_id,
                    candidate.item_id,
                    None,
                    candidate.symptom_id,
                    model_probability,
                    evidence["evidence_score"],
                    final_confidence,
                    evidence_summary,
                    evidence["evidence_strength_score"],
                    evidence_quality["score"],
                    model_probability,
                    penalty_score,
                    decision_reason,
                    json.dumps(evidence["citations"]),
                    now_iso,
                ),
            )
            insight_id = int(insight_cursor.lastrowid)
            event_link_rows: list[tuple[int, int, str, int, str]] = []
            for exposure_event_id in sorted(candidate.exposure_event_ids):
                event_link_rows.append((candidate.user_id, insight_id, "exposure", int(exposure_event_id), now_iso))
            for symptom_event_id in sorted(candidate.symptom_event_ids):
                event_link_rows.append((candidate.user_id, insight_id, "symptom", int(symptom_event_id), now_iso))
            if event_link_rows:
                conn.executemany(
                    """
                    INSERT INTO insight_event_links (user_id, insight_id, event_type, event_id, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(insight_id, event_type, event_id) DO NOTHING
                    """,
                    event_link_rows,
                )
            
            # allows future retrival system to understand candidate context
            query_key = (
                "item:"
                f"{candidate.item_id}|symptom:{candidate.symptom_id}|"
                f"routes:{','.join(sorted(candidate.routes))}|"
                f"lag_buckets:{json.dumps(candidate.lag_bucket_counts, sort_keys=True)}"
            )
            conn.execute(
                """
                INSERT INTO retrieval_runs (user_id, item_id, symptom_id, query_key, top_k, retrieved_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.user_id,
                    candidate.item_id,
                    candidate.symptom_id,
                    query_key,
                    5,
                    len(retrieved_claims),
                    now_iso,
                ),
            )
            inserted += 1

            # Additional ingredient-derived insights tied to this reference item.
            # These rows represent ingredient-specific evidence while keeping item context for UI grouping.
            for ingredient_id in sorted(ingredient_ids):
                ingredient_name = ingredient_name_map.get(int(ingredient_id))
                if not ingredient_name:
                    continue
                ingredient_query = build_candidate_query(
                    item_name=ingredient_name,
                    symptom_name=symptom_name_map.get(candidate.symptom_id),
                    routes=candidate.routes,
                    lag_bucket_counts=candidate.lag_bucket_counts,
                )
                ingredient_claims = retrieve_claim_evidence(
                    conn,
                    ingredient_ids={int(ingredient_id)},
                    item_id=None,
                    symptom_id=candidate.symptom_id,
                    query_text=ingredient_query,
                    top_k=max(5, int(os.getenv("RAG_DB_TOP_K", "12"))),
                )
                if not ingredient_claims:
                    continue
                ingredient_evidence = aggregate_evidence(
                    ingredient_claims,
                    item_name=ingredient_name,
                    symptom_name=symptom_name,
                )
                ingredient_quality = compute_evidence_quality(ingredient_evidence)
                ingredient_feature_map = dict(feature_map)
                ingredient_feature_map.update(
                    {
                        "evidence_strength_score": float(ingredient_evidence.get("evidence_strength_score") or 0.0),
                        "evidence_score_signed": float(ingredient_evidence.get("evidence_score") or 0.0),
                        "citation_count": float(ingredient_quality["citation_count"]),
                        "support_ratio": float(ingredient_quality["support_ratio"]),
                        "contradict_ratio": float(ingredient_quality["contradict_ratio"]),
                        "neutral_ratio": float(ingredient_quality["neutral_ratio"]),
                        "avg_relevance": float(ingredient_quality["avg_relevance"]),
                        "study_quality_score": float(ingredient_quality["study_quality_score"]),
                        "population_match": float(ingredient_quality["population_match"]),
                        "temporality_match": float(ingredient_quality["temporality_match"]),
                        "risk_of_bias": float(ingredient_quality["risk_of_bias"]),
                        "llm_confidence": float(ingredient_quality["llm_confidence"]),
                    }
                )
                ingredient_model_probability = predict_model_probability(ingredient_feature_map, use_calibration=True)
                ingredient_penalty = min(
                    1.0,
                    compute_penalty_score(ingredient_feature_map)
                    + generic_symptom_penalty,
                )
                ingredient_final = predict_final_score(
                    model_probability=ingredient_model_probability,
                    evidence_quality=float(ingredient_quality["score"]),
                    penalty_score=ingredient_penalty,
                    citation_count=float(ingredient_quality["citation_count"]),
                    contradict_ratio=float(ingredient_quality["contradict_ratio"]),
                )
                ingredient_decision_reason = "supported"
                if generic_symptom_penalty >= 0.20:
                    ingredient_decision_reason = "suppressed_generic_symptom"
                elif float(ingredient_evidence.get("evidence_score") or 0.0) < float(
                    thresholds.get("min_support_direction", 0.10)
                ):
                    ingredient_decision_reason = "suppressed_non_supportive_direction"
                elif (
                    float(cooccurrence_count) < min_recurrence_for_supported
                    or unique_exposure_events < min_unique_exposure_events_for_supported
                ) and not (
                    single_exposure_override_ok
                    or (
                        bool(ingredient_evidence["citations"])
                        and float(ingredient_quality.get("citation_count") or 0.0) >= 2.0
                        and float(ingredient_quality.get("support_ratio") or 0.0) >= 0.70
                        and float(ingredient_quality.get("contradict_ratio") or 0.0) <= 0.25
                        and float(ingredient_evidence.get("evidence_score") or 0.0)
                        >= float(thresholds.get("min_support_direction", 0.10))
                        and float(ingredient_evidence.get("evidence_strength_score") or 0.0)
                        >= float(thresholds["min_evidence_strength"])
                        and float(ingredient_model_probability) >= float(thresholds["min_model_probability"])
                        and float(ingredient_final) >= float(thresholds["min_overall_confidence"])
                    )
                ):
                    ingredient_decision_reason = "suppressed_insufficient_recurrence"
                elif not ingredient_evidence["citations"]:
                    ingredient_decision_reason = "suppressed_no_citations"
                elif float(ingredient_evidence["evidence_strength_score"] or 0.0) < float(
                    thresholds["min_evidence_strength"]
                ):
                    ingredient_decision_reason = "suppressed_low_evidence_strength"
                elif float(ingredient_model_probability) < float(thresholds["min_model_probability"]):
                    ingredient_decision_reason = "suppressed_low_model_probability"
                elif float(ingredient_final) < float(thresholds["min_overall_confidence"]):
                    ingredient_decision_reason = "suppressed_low_overall_confidence"

                ingredient_summary = (
                    generate_user_evidence_summary(
                        item_name=ingredient_name,
                        symptom_name=symptom_name,
                        citations=ingredient_evidence.get("citations") or [],
                        evidence_score=float(ingredient_evidence.get("evidence_score") or 0.0),
                    )
                    or str(ingredient_evidence["evidence_summary"])
                )
                ingredient_cursor = conn.execute(
                    """
                    INSERT INTO insights (
                        user_id, item_id, source_ingredient_id, symptom_id, model_score, evidence_score, final_score,
                        evidence_summary, evidence_strength_score, evidence_quality_score,
                        model_probability, penalty_score, display_decision_reason, citations_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        candidate.user_id,
                        candidate.item_id,
                        int(ingredient_id),
                        candidate.symptom_id,
                        ingredient_model_probability,
                        ingredient_evidence["evidence_score"],
                        ingredient_final,
                        ingredient_summary,
                        ingredient_evidence["evidence_strength_score"],
                        ingredient_quality["score"],
                        ingredient_model_probability,
                        ingredient_penalty,
                        ingredient_decision_reason,
                        json.dumps(ingredient_evidence["citations"]),
                        now_iso,
                    ),
                )
                ingredient_insight_id = int(ingredient_cursor.lastrowid)
                ingredient_event_links: list[tuple[int, int, str, int, str]] = []
                for exposure_event_id in sorted(candidate.exposure_event_ids):
                    if int(ingredient_id) not in exposure_ingredients.get(exposure_event_id, set()):
                        continue
                    ingredient_event_links.append(
                        (candidate.user_id, ingredient_insight_id, "exposure", int(exposure_event_id), now_iso)
                    )
                for symptom_event_id in sorted(candidate.symptom_event_ids):
                    ingredient_event_links.append(
                        (candidate.user_id, ingredient_insight_id, "symptom", int(symptom_event_id), now_iso)
                    )
                if ingredient_event_links:
                    conn.executemany(
                        """
                        INSERT INTO insight_event_links (user_id, insight_id, event_type, event_id, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(insight_id, event_type, event_id) DO NOTHING
                        """,
                        ingredient_event_links,
                    )
                inserted += 1

        for combo in combo_aggregates.values():
            if target_pairs is not None:
                combo_targets = {
                    (int(combo.item_a_id), int(combo.symptom_id)),
                    (int(combo.item_b_id), int(combo.symptom_id)),
                }
                if combo_targets.isdisjoint(target_pairs):
                    continue
            if not combo.lags_minutes:
                continue
            if combo.latest_symptom_ts is None:
                continue

            symptom_name = symptom_name_map.get(combo.symptom_id)
            if _is_low_signal_symptom_name(symptom_name):
                continue
            candidates_considered += 1

            lag_min = min(combo.lags_minutes)
            lag_avg = sum(combo.lags_minutes) / len(combo.lags_minutes)
            cooccurrence_count = len(combo.lags_minutes)
            cooccurrence_unique_symptom_count = len(combo.symptom_event_ids)
            pair_density = (
                cooccurrence_count / cooccurrence_unique_symptom_count
                if cooccurrence_unique_symptom_count > 0
                else None
            )
            window_start = combo.latest_symptom_ts - ROLLING_WINDOW
            combo_items = {int(combo.item_a_id), int(combo.item_b_id)}
            exposure_count_7d = sum(
                1
                for event in exposures
                if event.item_id in combo_items
                and window_start <= event.event_ts <= combo.latest_symptom_ts
            )
            symptom_count_7d = sum(
                1
                for event in symptoms
                if event.symptom_id == combo.symptom_id
                and window_start <= event.event_ts <= combo.latest_symptom_ts
            )
            severity_avg_after = (
                sum(combo.symptom_severity_values) / len(combo.symptom_severity_values)
                if combo.symptom_severity_values
                else None
            )
            combo_key = combo.combo_key
            combo_item_ids_json = json.dumps(list(combo.item_ids_sorted))
            conn.execute(
                """
                INSERT INTO derived_features_combos (
                    user_id, combo_key, item_ids_json, symptom_id,
                    time_gap_min_minutes, time_gap_avg_minutes,
                    cooccurrence_count, cooccurrence_unique_symptom_count, pair_density,
                    exposure_count_7d, symptom_count_7d, severity_avg_after, computed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, combo_key, symptom_id)
                DO UPDATE SET
                    item_ids_json = excluded.item_ids_json,
                    time_gap_min_minutes = excluded.time_gap_min_minutes,
                    time_gap_avg_minutes = excluded.time_gap_avg_minutes,
                    cooccurrence_count = excluded.cooccurrence_count,
                    cooccurrence_unique_symptom_count = excluded.cooccurrence_unique_symptom_count,
                    pair_density = excluded.pair_density,
                    exposure_count_7d = excluded.exposure_count_7d,
                    symptom_count_7d = excluded.symptom_count_7d,
                    severity_avg_after = excluded.severity_avg_after,
                    computed_at = excluded.computed_at
                """,
                (
                    combo.user_id,
                    combo_key,
                    combo_item_ids_json,
                    combo.symptom_id,
                    lag_min,
                    lag_avg,
                    cooccurrence_count,
                    cooccurrence_unique_symptom_count,
                    pair_density,
                    exposure_count_7d,
                    symptom_count_7d,
                    severity_avg_after,
                    now_iso,
                ),
            )

            item_a_name = item_name_map.get(combo.item_a_id) or f"item:{combo.item_a_id}"
            item_b_name = item_name_map.get(combo.item_b_id) or f"item:{combo.item_b_id}"
            combo_item_name = f"{item_a_name} + {item_b_name}"
            combo_query = build_candidate_query(
                item_name=combo_item_name,
                symptom_name=symptom_name_map.get(combo.symptom_id),
                routes=combo.routes,
                lag_bucket_counts=combo.lag_bucket_counts,
            )
            claims_a = retrieve_claim_evidence(
                conn,
                ingredient_ids=set(),
                item_id=combo.item_a_id,
                symptom_id=combo.symptom_id,
                query_text=combo_query,
                top_k=max(5, int(os.getenv("RAG_DB_TOP_K", "12"))),
            )
            claims_b = retrieve_claim_evidence(
                conn,
                ingredient_ids=set(),
                item_id=combo.item_b_id,
                symptom_id=combo.symptom_id,
                query_text=combo_query,
                top_k=max(5, int(os.getenv("RAG_DB_TOP_K", "12"))),
            )
            evidence_a = aggregate_evidence(
                claims_a,
                item_name=item_a_name,
                symptom_name=symptom_name,
            )
            evidence_b = aggregate_evidence(
                claims_b,
                item_name=item_b_name,
                symptom_name=symptom_name,
            )
            evidence_quality_a = compute_evidence_quality(evidence_a)
            evidence_quality_b = compute_evidence_quality(evidence_b)
            combo_claims = _merge_claim_rows(claims_a, claims_b)
            pair_claims = _filter_combo_pair_claims(
                combo_claims,
                item_a_name=item_a_name,
                item_b_name=item_b_name,
            )
            evidence = aggregate_evidence(
                pair_claims,
                item_name=combo_item_name,
                symptom_name=symptom_name,
            )
            evidence_quality = compute_evidence_quality(evidence)
            # Conservative combo evidence: pair-level support must exist, and both sides must have signal.
            combo_evidence_strength = min(
                float(evidence.get("evidence_strength_score") or 0.0),
                float(evidence_a.get("evidence_strength_score") or 0.0),
                float(evidence_b.get("evidence_strength_score") or 0.0),
            )
            combo_evidence_score_signed = min(
                float(evidence_a.get("evidence_score") or 0.0),
                float(evidence_b.get("evidence_score") or 0.0),
            )
            combo_evidence_quality = min(
                float(evidence_quality["score"]),
                float(evidence_quality_a["score"]),
                float(evidence_quality_b["score"]),
            )
            generic_symptom_penalty = _generic_symptom_penalty(symptom_name)
            unique_exposure_events = float(len(combo.exposure_event_ids))
            temporal_lift = _estimate_temporal_lift(
                unique_exposure_events=unique_exposure_events,
                exposure_count_7d=float(exposure_count_7d),
                symptom_count_7d=float(symptom_count_7d),
                lag_bucket_counts=combo.lag_bucket_counts,
            )

            feature_map = {
                "time_gap_min_minutes": lag_min,
                "time_gap_avg_minutes": lag_avg,
                "cooccurrence_count": float(cooccurrence_count),
                "cooccurrence_unique_symptom_count": float(cooccurrence_unique_symptom_count),
                "pair_density": float(pair_density or 0.0),
                "exposure_count_7d": float(exposure_count_7d),
                "symptom_count_7d": float(symptom_count_7d),
                "severity_avg_after": float(severity_avg_after or 0.0),
                "route_count": float(len(combo.routes)),
                "lag_bucket_diversity": float(len(combo.lag_bucket_counts)),
                "exposure_with_ingredients_ratio": 0.0,
                "evidence_strength_score": float(combo_evidence_strength),
                "evidence_score_signed": float(combo_evidence_score_signed),
                "citation_count": float(evidence_quality["citation_count"]),
                "support_ratio": float(evidence_quality["support_ratio"]),
                "contradict_ratio": float(evidence_quality["contradict_ratio"]),
                "neutral_ratio": float(evidence_quality["neutral_ratio"]),
                "avg_relevance": float(evidence_quality["avg_relevance"]),
                "study_quality_score": float(evidence_quality["study_quality_score"]),
                "population_match": float(evidence_quality["population_match"]),
                "temporality_match": float(evidence_quality["temporality_match"]),
                "risk_of_bias": float(evidence_quality["risk_of_bias"]),
                "llm_confidence": float(evidence_quality["llm_confidence"]),
                "time_confidence_score": (
                    sum(combo.time_confidence_values) / len(combo.time_confidence_values)
                    if combo.time_confidence_values
                    else 0.0
                ),
                "temporal_lift": float(temporal_lift),
                "symptom_specificity_score": 1.0 - generic_symptom_penalty,
            }
            feature_map.update(
                _build_route_temporal_feature_map(
                    total_cooccurrence=float(cooccurrence_count),
                    route_counts=combo.route_counts,
                    route_min_lag_minutes=combo.route_min_lag_minutes,
                    route_lag_bucket_counts=combo.route_lag_bucket_counts,
                )
            )

            model_probability = predict_model_probability(feature_map, use_calibration=True)
            penalty_score = min(1.0, compute_penalty_score(feature_map) + generic_symptom_penalty)
            final_confidence = predict_final_score(
                model_probability=model_probability,
                evidence_quality=float(combo_evidence_quality),
                penalty_score=penalty_score,
                citation_count=float(evidence_quality["citation_count"]),
                contradict_ratio=float(evidence_quality["contradict_ratio"]),
            )

            min_combo_recurrence_for_supported = float(
                thresholds.get("min_combo_cooccurrence_for_supported", 3.0)
            )
            min_combo_unique_exposure_events_for_supported = float(
                thresholds.get("min_combo_unique_exposure_events_for_supported", 2.0)
            )
            min_combo_item_citations = float(thresholds.get("min_combo_item_citations", 1.0))
            min_combo_item_support_direction = float(
                thresholds.get("min_combo_item_support_direction", thresholds.get("min_support_direction", 0.10))
            )
            min_combo_pair_citations = float(thresholds.get("min_combo_pair_citations", 1.0))
            min_combo_pair_support_direction = float(
                thresholds.get("min_combo_pair_support_direction", thresholds.get("min_support_direction", 0.10))
            )
            min_combo_temporal_lift = float(thresholds.get("min_combo_temporal_lift", 1.10))
            both_item_evidence_ok = (
                float(evidence_quality_a.get("citation_count") or 0.0) >= min_combo_item_citations
                and float(evidence_quality_b.get("citation_count") or 0.0) >= min_combo_item_citations
                and float(evidence_a.get("evidence_score") or 0.0) >= min_combo_item_support_direction
                and float(evidence_b.get("evidence_score") or 0.0) >= min_combo_item_support_direction
            )
            pair_evidence_ok = (
                float(evidence_quality.get("citation_count") or 0.0) >= min_combo_pair_citations
                and float(evidence.get("evidence_score") or 0.0) >= min_combo_pair_support_direction
            )
            if generic_symptom_penalty >= 0.20:
                decision_reason = "suppressed_generic_symptom"
            elif not pair_evidence_ok:
                decision_reason = "suppressed_combo_no_pair_evidence"
            elif not both_item_evidence_ok:
                decision_reason = "suppressed_combo_unbalanced_evidence"
            elif float(evidence.get("evidence_score") or 0.0) < float(
                thresholds.get("min_support_direction", 0.10)
            ):
                decision_reason = "suppressed_non_supportive_direction"
            elif (
                float(cooccurrence_count) < min_combo_recurrence_for_supported
                or unique_exposure_events < min_combo_unique_exposure_events_for_supported
            ):
                decision_reason = "suppressed_insufficient_recurrence"
            elif float(temporal_lift) < min_combo_temporal_lift:
                decision_reason = "suppressed_low_temporal_lift"
            elif not evidence["citations"]:
                decision_reason = "suppressed_no_citations"
            elif float(evidence["evidence_strength_score"] or 0.0) < float(
                thresholds["min_evidence_strength"]
            ):
                decision_reason = "suppressed_low_evidence_strength"
            elif float(model_probability) < float(thresholds["min_model_probability"]):
                decision_reason = "suppressed_low_model_probability"
            elif float(final_confidence) < float(thresholds["min_overall_confidence"]):
                decision_reason = "suppressed_low_overall_confidence"
            else:
                decision_reason = "supported"

            dominant_lag_bucket = None
            if combo.lag_bucket_counts:
                dominant_lag_bucket = max(combo.lag_bucket_counts.items(), key=lambda row: row[1])[0]
            evidence_summary = (
                generate_user_evidence_summary(
                    item_name=combo_item_name,
                    symptom_name=symptom_name,
                    citations=evidence.get("citations") or [],
                    evidence_score=float(evidence.get("evidence_score") or 0.0),
                )
                or str(evidence["evidence_summary"])
            )
            if dominant_lag_bucket:
                evidence_summary = f"{evidence_summary} Dominant lag window: {dominant_lag_bucket}."

            insight_cursor = conn.execute(
                """
                INSERT INTO insights (
                    user_id, item_id, secondary_item_id, is_combo, combo_key, combo_item_ids_json,
                    source_ingredient_id, symptom_id, model_score, evidence_score, final_score,
                    evidence_summary, evidence_strength_score, evidence_quality_score,
                    model_probability, penalty_score, display_decision_reason, citations_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    combo.user_id,
                    combo.item_a_id,
                    combo.item_b_id,
                    1,
                    combo_key,
                    combo_item_ids_json,
                    None,
                    combo.symptom_id,
                    model_probability,
                    combo_evidence_score_signed,
                    final_confidence,
                    evidence_summary,
                    combo_evidence_strength,
                    combo_evidence_quality,
                    model_probability,
                    penalty_score,
                    decision_reason,
                    json.dumps(evidence["citations"]),
                    now_iso,
                ),
            )
            combo_insight_id = int(insight_cursor.lastrowid)
            combo_event_links: list[tuple[int, int, str, int, str]] = []
            for exposure_event_id in sorted(combo.exposure_event_ids):
                combo_event_links.append((combo.user_id, combo_insight_id, "exposure", int(exposure_event_id), now_iso))
            for symptom_event_id in sorted(combo.symptom_event_ids):
                combo_event_links.append((combo.user_id, combo_insight_id, "symptom", int(symptom_event_id), now_iso))
            if combo_event_links:
                conn.executemany(
                    """
                    INSERT INTO insight_event_links (user_id, insight_id, event_type, event_id, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(insight_id, event_type, event_id) DO NOTHING
                    """,
                    combo_event_links,
                )
            conn.execute(
                """
                INSERT INTO retrieval_runs (user_id, item_id, symptom_id, query_key, top_k, retrieved_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    combo.user_id,
                    combo.item_a_id,
                    combo.symptom_id,
                    f"combo:{combo_key}|symptom:{combo.symptom_id}",
                    5,
                    len(pair_claims),
                    now_iso,
                ),
            )
            inserted += 1

        for ingredient_candidate in ingredient_aggregates.values():
            if target_pairs is not None:
                continue
            if not ingredient_candidate.lags_minutes:
                continue
            if ingredient_candidate.latest_symptom_ts is None:
                continue

            lag_min = min(ingredient_candidate.lags_minutes)
            lag_avg = sum(ingredient_candidate.lags_minutes) / len(ingredient_candidate.lags_minutes)
            cooccurrence_count = len(ingredient_candidate.lags_minutes)
            cooccurrence_unique_symptom_count = len(ingredient_candidate.symptom_event_ids)
            pair_density = (
                cooccurrence_count / cooccurrence_unique_symptom_count
                if cooccurrence_unique_symptom_count > 0
                else None
            )
            window_start = ingredient_candidate.latest_symptom_ts - ROLLING_WINDOW
            exposure_count_7d = sum(
                1
                for event in exposures
                if window_start <= event.event_ts <= ingredient_candidate.latest_symptom_ts
                and ingredient_candidate.ingredient_id in exposure_ingredients.get(event.event_id, set())
            )
            symptom_count_7d = sum(
                1
                for event in symptoms
                if event.symptom_id == ingredient_candidate.symptom_id
                and window_start <= event.event_ts <= ingredient_candidate.latest_symptom_ts
            )
            severity_avg_after = (
                sum(ingredient_candidate.symptom_severity_values)
                / len(ingredient_candidate.symptom_severity_values)
                if ingredient_candidate.symptom_severity_values
                else None
            )
            conn.execute(
                """
                INSERT INTO derived_features_ingredients (
                    user_id, ingredient_id, symptom_id, time_gap_min_minutes, time_gap_avg_minutes,
                    cooccurrence_count, cooccurrence_unique_symptom_count, pair_density,
                    exposure_count_7d, symptom_count_7d, severity_avg_after, computed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, ingredient_id, symptom_id)
                DO UPDATE SET
                    time_gap_min_minutes = excluded.time_gap_min_minutes,
                    time_gap_avg_minutes = excluded.time_gap_avg_minutes,
                    cooccurrence_count = excluded.cooccurrence_count,
                    cooccurrence_unique_symptom_count = excluded.cooccurrence_unique_symptom_count,
                    pair_density = excluded.pair_density,
                    exposure_count_7d = excluded.exposure_count_7d,
                    symptom_count_7d = excluded.symptom_count_7d,
                    severity_avg_after = excluded.severity_avg_after,
                    computed_at = excluded.computed_at
                """,
                (
                    ingredient_candidate.user_id,
                    ingredient_candidate.ingredient_id,
                    ingredient_candidate.symptom_id,
                    lag_min,
                    lag_avg,
                    cooccurrence_count,
                    cooccurrence_unique_symptom_count,
                    pair_density,
                    exposure_count_7d,
                    symptom_count_7d,
                    severity_avg_after,
                    now_iso,
                ),
            )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return {
        "candidates_considered": candidates_considered if target_pairs is not None else len(aggregates),
        "pairs_evaluated": pairs_evaluated,
        "insights_written": inserted,
    }


def list_insights(user_id: int, include_suppressed: bool = True) -> list[dict[str, Any]]:
    conn = get_connection()
    try:
        base_sql = """
            SELECT
                i.id AS id,
                i.user_id AS user_id,
                i.item_id AS item_id,
                it.name AS item_name,
                i.secondary_item_id AS secondary_item_id,
                it2.name AS secondary_item_name,
                COALESCE(i.is_combo, 0) AS is_combo,
                i.combo_key AS combo_key,
                i.combo_item_ids_json AS combo_item_ids_json,
                i.source_ingredient_id AS source_ingredient_id,
                ing.name AS source_ingredient_name,
                i.symptom_id AS symptom_id,
                s.name AS symptom_name,
                i.model_probability AS model_probability,
                i.evidence_score AS evidence_support_score,
                i.evidence_strength_score AS evidence_strength_score,
                i.evidence_quality_score AS evidence_quality_score,
                i.penalty_score AS penalty_score,
                i.final_score AS overall_confidence_score,
                i.evidence_summary AS evidence_summary,
                i.display_decision_reason AS display_decision_reason,
                i.citations_json AS citations_json,
                i.created_at AS created_at,
                COALESCE(v.verified, 0) AS user_verified,
                COALESCE(v.rejected, 0) AS user_rejected
            FROM insights i
            JOIN items it ON it.id = i.item_id
            LEFT JOIN items it2 ON it2.id = i.secondary_item_id
            LEFT JOIN ingredients ing ON ing.id = i.source_ingredient_id
            JOIN symptoms s ON s.id = i.symptom_id
            LEFT JOIN insight_verifications v
                ON v.user_id = i.user_id
               AND v.item_id = i.item_id
               AND v.symptom_id = i.symptom_id
            WHERE i.user_id = ?
        """
        params: list[Any] = [user_id]
        if not include_suppressed:
            base_sql += " AND (i.display_decision_reason IS NULL OR i.display_decision_reason NOT LIKE 'suppressed_%')"
        base_sql += " ORDER BY i.created_at DESC"
        rows = conn.execute(base_sql, tuple(params)).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            if _is_low_signal_symptom_name(row["symptom_name"]):
                continue
            parsed_citations: list[dict[str, Any]] = []
            raw_citations = row["citations_json"]
            if raw_citations:
                try:
                    decoded = json.loads(raw_citations)
                    if isinstance(decoded, list):
                        parsed_citations = [entry for entry in decoded if isinstance(entry, dict)]
                except json.JSONDecodeError:
                    parsed_citations = []
            results.append(
                {
                    "id": row["id"],
                    "user_id": row["user_id"],
                    "item_id": row["item_id"],
                    "item_name": (
                        f"{row['item_name']} + {row['secondary_item_name']}"
                        if int(row["is_combo"] or 0) == 1 and row["secondary_item_name"]
                        else row["item_name"]
                    ),
                    "secondary_item_id": row["secondary_item_id"],
                    "secondary_item_name": row["secondary_item_name"],
                    "is_combo": bool(int(row["is_combo"] or 0)),
                    "combo_key": row["combo_key"],
                    "combo_item_ids": (
                        json.loads(row["combo_item_ids_json"])
                        if row["combo_item_ids_json"]
                        else None
                    ),
                    "source_ingredient_id": row["source_ingredient_id"],
                    "source_ingredient_name": row["source_ingredient_name"],
                    "symptom_id": row["symptom_id"],
                    "symptom_name": row["symptom_name"],
                    "model_probability": row["model_probability"],
                    "evidence_support_score": row["evidence_support_score"],
                    "evidence_strength_score": row["evidence_strength_score"],
                    "evidence_quality_score": row["evidence_quality_score"],
                    "penalty_score": row["penalty_score"],
                    "overall_confidence_score": row["overall_confidence_score"],
                    "evidence_summary": row["evidence_summary"],
                    "display_decision_reason": row["display_decision_reason"],
                    "display_status": (
                        "supported"
                        if row["display_decision_reason"] == "supported"
                        else (
                            "insufficient_evidence"
                            if row["display_decision_reason"] in {
                                "suppressed_no_citations",
                                "suppressed_low_evidence_strength",
                                "suppressed_low_overall_confidence",
                                "suppressed_low_model_probability",
                                "suppressed_generic_symptom",
                                "suppressed_generic_monotone_context",
                                "suppressed_insufficient_recurrence",
                                "suppressed_non_supportive_direction",
                                "suppressed_combo_unbalanced_evidence",
                                "suppressed_combo_no_pair_evidence",
                            }
                            else "suppressed"
                        )
                    ),
                    "user_verified": bool(int(row["user_verified"] or 0)),
                    "user_rejected": bool(int(row["user_rejected"] or 0)),
                    "created_at": row["created_at"],
                    "citations": parsed_citations,
                }
            )
        return results
    finally:
        conn.close()


def list_event_insight_links(user_id: int, *, supported_only: bool = True) -> list[dict[str, Any]]:
    conn = get_connection()
    try:
        sql = """
            SELECT
                l.event_type AS event_type,
                l.event_id AS event_id,
                l.insight_id AS insight_id
            FROM insight_event_links l
            JOIN insights i ON i.id = l.insight_id
            WHERE l.user_id = ?
        """
        params: list[Any] = [user_id]
        if supported_only:
            sql += " AND i.display_decision_reason = 'supported'"
        sql += " ORDER BY l.event_type ASC, l.event_id ASC, i.created_at DESC"
        rows = conn.execute(sql, tuple(params)).fetchall()
        linked = [
            {
                "event_type": row["event_type"],
                "event_id": int(row["event_id"]),
                "insight_id": int(row["insight_id"]),
            }
            for row in rows
        ]
        if linked:
            return linked

        # Fallback for older DB rows created before insight_event_links existed:
        # derive event mappings from current candidate aggregates for the same item+symptom pair.
        insights_sql = """
            SELECT id, item_id, symptom_id
            FROM insights
            WHERE user_id = ?
        """
        insight_params: list[Any] = [user_id]
        if supported_only:
            insights_sql += " AND display_decision_reason = 'supported'"
        insight_rows = conn.execute(insights_sql, tuple(insight_params)).fetchall()
        if not insight_rows:
            return []

        pair_to_insight_ids: dict[tuple[int, int], list[int]] = {}
        for row in insight_rows:
            key = (int(row["item_id"]), int(row["symptom_id"]))
            pair_to_insight_ids.setdefault(key, []).append(int(row["id"]))

        aggregates, _, _, _, _, _, _ = _build_candidate_aggregates(user_id)
        fallback_links: list[dict[str, Any]] = []
        seen: set[tuple[str, int, int]] = set()
        for candidate in aggregates.values():
            pair_key = (candidate.item_id, candidate.symptom_id)
            insight_ids = pair_to_insight_ids.get(pair_key)
            if not insight_ids:
                continue
            for insight_id in insight_ids:
                for exposure_event_id in candidate.exposure_event_ids:
                    dedupe = ("exposure", int(exposure_event_id), int(insight_id))
                    if dedupe in seen:
                        continue
                    seen.add(dedupe)
                    fallback_links.append(
                        {"event_type": "exposure", "event_id": int(exposure_event_id), "insight_id": int(insight_id)}
                    )
                for symptom_event_id in candidate.symptom_event_ids:
                    dedupe = ("symptom", int(symptom_event_id), int(insight_id))
                    if dedupe in seen:
                        continue
                    seen.add(dedupe)
                    fallback_links.append(
                        {"event_type": "symptom", "event_id": int(symptom_event_id), "insight_id": int(insight_id)}
                    )
        return fallback_links
    finally:
        conn.close()


def set_insight_verification(
    *,
    user_id: int,
    insight_id: int,
    verified: bool,
) -> dict[str, Any]:
    conn = get_connection()
    now_iso = _now_iso()
    try:
        insight_row = conn.execute(
            """
            SELECT item_id, symptom_id
            FROM insights
            WHERE id = ? AND user_id = ?
            """,
            (insight_id, user_id),
        ).fetchone()
        if insight_row is None:
            raise ValueError("insight_not_found")

        item_id = int(insight_row["item_id"])
        symptom_id = int(insight_row["symptom_id"])

        if verified:
            conn.execute(
                """
                INSERT INTO insight_verifications (
                    user_id, item_id, symptom_id, verified, rejected, created_at, updated_at
                )
                VALUES (?, ?, ?, 1, 0, ?, ?)
                ON CONFLICT(user_id, item_id, symptom_id)
                DO UPDATE SET
                    verified = 1,
                    rejected = 0,
                    updated_at = excluded.updated_at
                """,
                (user_id, item_id, symptom_id, now_iso, now_iso),
            )
        else:
            existing = conn.execute(
                """
                SELECT COALESCE(rejected, 0) AS rejected
                FROM insight_verifications
                WHERE user_id = ? AND item_id = ? AND symptom_id = ?
                LIMIT 1
                """,
                (user_id, item_id, symptom_id),
            ).fetchone()
            if existing is not None and int(existing["rejected"] or 0) == 1:
                conn.execute(
                    """
                    UPDATE insight_verifications
                    SET verified = 0, updated_at = ?
                    WHERE user_id = ? AND item_id = ? AND symptom_id = ?
                    """,
                    (now_iso, user_id, item_id, symptom_id),
                )
            else:
                conn.execute(
                    """
                    DELETE FROM insight_verifications
                    WHERE user_id = ? AND item_id = ? AND symptom_id = ?
                    """,
                    (user_id, item_id, symptom_id),
                )

        conn.commit()
        return {
            "status": "ok",
            "insight_id": insight_id,
            "user_id": user_id,
            "item_id": item_id,
            "symptom_id": symptom_id,
            "verified": bool(verified),
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def set_insight_rejection(
    *,
    user_id: int,
    insight_id: int,
    rejected: bool,
) -> dict[str, Any]:
    conn = get_connection()
    now_iso = _now_iso()
    try:
        insight_row = conn.execute(
            """
            SELECT item_id, symptom_id
            FROM insights
            WHERE id = ? AND user_id = ?
            """,
            (insight_id, user_id),
        ).fetchone()
        if insight_row is None:
            raise ValueError("insight_not_found")

        item_id = int(insight_row["item_id"])
        symptom_id = int(insight_row["symptom_id"])

        if rejected:
            conn.execute(
                """
                INSERT INTO insight_verifications (
                    user_id, item_id, symptom_id, verified, rejected, created_at, updated_at
                )
                VALUES (?, ?, ?, 0, 1, ?, ?)
                ON CONFLICT(user_id, item_id, symptom_id)
                DO UPDATE SET
                    verified = 0,
                    rejected = 1,
                    updated_at = excluded.updated_at
                """,
                (user_id, item_id, symptom_id, now_iso, now_iso),
            )
        else:
            existing = conn.execute(
                """
                SELECT COALESCE(verified, 0) AS verified
                FROM insight_verifications
                WHERE user_id = ? AND item_id = ? AND symptom_id = ?
                LIMIT 1
                """,
                (user_id, item_id, symptom_id),
            ).fetchone()
            if existing is not None and int(existing["verified"] or 0) == 1:
                conn.execute(
                    """
                    UPDATE insight_verifications
                    SET rejected = 0, updated_at = ?
                    WHERE user_id = ? AND item_id = ? AND symptom_id = ?
                    """,
                    (now_iso, user_id, item_id, symptom_id),
                )
            else:
                conn.execute(
                    """
                    DELETE FROM insight_verifications
                    WHERE user_id = ? AND item_id = ? AND symptom_id = ?
                    """,
                    (user_id, item_id, symptom_id),
                )

        conn.commit()
        return {
            "status": "ok",
            "insight_id": insight_id,
            "user_id": user_id,
            "item_id": item_id,
            "symptom_id": symptom_id,
            "rejected": bool(rejected),
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
