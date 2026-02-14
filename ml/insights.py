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

# each lag is assigned to first matching bucket, converting lag into retrivable categories
def _lag_bucket_label(lag: timedelta) -> str | None:
    for label, start, end in LAG_BUCKETS:
        if lag >= start and lag <= end:
            return label
    return None


@dataclass
class ExposureEvent:
    event_id: int
    item_id: int
    event_ts: datetime
    route: str | None
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
    exposure_with_ingredients_count: int = 0
    time_confidence_values: list[float] = field(default_factory=list)
    latest_symptom_ts: datetime | None = None


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
    time_confidence_values: list[float] = field(default_factory=list)
    latest_symptom_ts: datetime | None = None


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
                has_expansion=bool(row["has_expansion"]),
                time_confidence_score=_time_confidence_score(row["time_confidence"]),
            )
        )
    return out

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
    pair_count = 0

    # logic for candidate generation, ensuring realistic potential linkage
    for symptom in symptoms:
        for exposure in exposures:
            lag = symptom.event_ts - exposure.event_ts
            if lag.total_seconds() < 0:
                continue
            if lag > MAX_LAG_WINDOW:
                continue
            bucket = _lag_bucket_label(lag)
            if bucket is None:
                continue

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
            if exposure.route:
                candidate.routes.add(exposure.route)
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

    return aggregates, ingredient_aggregates, pair_count, exposures, symptoms, exposure_ingredients


def list_rag_sync_candidates(user_id: int) -> list[dict[str, Any]]:
    aggregates, _, _, _, _, exposure_ingredients = _build_candidate_aggregates(user_id)
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
    return candidates


def recompute_insights(
    user_id: int,
    *,
    target_pairs: set[tuple[int, int]] | None = None,
) -> dict[str, int]:
    (
        aggregates,
        ingredient_aggregates,
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

        if target_pairs is None:
            conn.execute("DELETE FROM insights WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM derived_features_ingredients WHERE user_id = ?", (user_id,))
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
            candidates_considered += 1
            if not candidate.lags_minutes:
                continue
            lag_min = min(candidate.lags_minutes)
            lag_avg = sum(candidate.lags_minutes) / len(candidate.lags_minutes)
            cooccurrence_count = len(candidate.lags_minutes)
            cooccurrence_unique_symptom_count = len(candidate.symptom_event_ids)
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
                "symptom_specificity_score": 1.0 - generic_symptom_penalty,
            }

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
            unique_exposure_events = float(len(candidate.exposure_event_ids))
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

            dominant_lag_bucket = None
            if candidate.lag_bucket_counts:
                dominant_lag_bucket = max(
                    candidate.lag_bucket_counts.items(),
                    key=lambda row: row[1],
                )[0]
            evidence_summary = str(evidence["evidence_summary"])
            if dominant_lag_bucket:
                evidence_summary = f"{evidence_summary} Dominant lag window: {dominant_lag_bucket}."

            if generic_symptom_penalty >= 0.20:
                decision_reason = "suppressed_generic_symptom"
            elif float(evidence.get("evidence_score") or 0.0) < float(thresholds.get("min_support_direction", 0.10)):
                decision_reason = "suppressed_non_supportive_direction"
            elif (
                float(cooccurrence_count) < min_recurrence_for_supported
                or unique_exposure_events < min_unique_exposure_events_for_supported
            ) and not single_exposure_override_ok:
                decision_reason = "suppressed_insufficient_recurrence"
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
                ) and not single_exposure_override_ok:
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

                ingredient_summary = str(ingredient_evidence["evidence_summary"])
                ingredient_summary = (
                    f"{ingredient_summary} Ingredient focus: {ingredient_name} (from {item_name_map.get(candidate.item_id)})."
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
                    "item_name": row["item_name"],
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
                                "suppressed_insufficient_recurrence",
                                "suppressed_non_supportive_direction",
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

        aggregates, _, _, _, _, _ = _build_candidate_aggregates(user_id)
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
