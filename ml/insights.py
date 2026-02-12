from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from api.db import get_connection

# generate insight candidates for many possible temporal patterns of exposure
ROLLING_WINDOW = timedelta(days=7)
LAG_BUCKETS = (
    ("0_6h", timedelta(hours=0), timedelta(hours=6)),
    ("6_24h", timedelta(hours=6), timedelta(hours=24)),
    ("24_72h", timedelta(hours=24), timedelta(hours=72)),
    ("72h_7d", timedelta(hours=72), timedelta(days=7)),
)
MAX_LAG_WINDOW = LAG_BUCKETS[-1][2]


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()

def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))

def _route_key(route: str | None) -> str | None:
    if route is None:
        return None
    return route.strip().lower() or None

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

@dataclass
class SymptomEvent:
    event_id: int
    symptom_id: int
    event_ts: datetime
    severity: int | None

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
    latest_symptom_ts: datetime | None = None

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
    return [
        ExposureEvent(
            event_id=row["event_id"],
            item_id=row["item_id"],
            event_ts=_parse_iso(row["event_ts"]),
            route=_route_key(row["route"]),
            has_expansion=bool(row["has_expansion"]),
        )
        for row in rows
    ]

def _fetch_symptoms(user_id: int) -> list[SymptomEvent]:
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT
                s.id AS event_id,
                s.symptom_id AS symptom_id,
                COALESCE(s.timestamp, s.time_range_start) AS event_ts,
                s.severity AS severity
            FROM symptom_events s
            WHERE s.user_id = ?
              AND COALESCE(s.timestamp, s.time_range_start) IS NOT NULL
            ORDER BY event_ts
            """,
            (user_id,),
        ).fetchall()
    finally:
        conn.close()
    return [
        SymptomEvent(
            event_id=row["event_id"],
            symptom_id=row["symptom_id"],
            event_ts=_parse_iso(row["event_ts"]),
            severity=row["severity"],
        )
        for row in rows
    ]

# builds candidates out of exposures and symptoms
# multiple candidates can be build from same exposure and symptom patterns but with different temporal windows
def _build_candidate_aggregates(
    user_id: int,
) -> tuple[dict[tuple[int, int], CandidateAggregate], int, list[ExposureEvent], list[SymptomEvent]]:
    exposures = _fetch_exposures(user_id)
    symptoms = _fetch_symptoms(user_id)
    aggregates: dict[tuple[int, int], CandidateAggregate] = {}
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
            if candidate.latest_symptom_ts is None or symptom.event_ts > candidate.latest_symptom_ts:
                candidate.latest_symptom_ts = symptom.event_ts

    return aggregates, pair_count, exposures, symptoms


def recompute_insights(user_id: int) -> dict[str, int]:
    aggregates, pair_count, exposures, symptoms = _build_candidate_aggregates(user_id)

    conn = get_connection()
    now_iso = _now_iso()
    inserted = 0
    try:
        conn.execute("DELETE FROM insights WHERE user_id = ?", (user_id,))

        for candidate in aggregates.values():
            lag_min = min(candidate.lags_minutes)
            lag_avg = sum(candidate.lags_minutes) / len(candidate.lags_minutes)
            cooccurrence_count = len(candidate.lags_minutes)

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
                    cooccurrence_count, exposure_count_7d, symptom_count_7d, severity_avg_after, computed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, item_id, symptom_id)
                DO UPDATE SET
                    time_gap_min_minutes = excluded.time_gap_min_minutes,
                    time_gap_avg_minutes = excluded.time_gap_avg_minutes,
                    cooccurrence_count = excluded.cooccurrence_count,
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
                    exposure_count_7d,
                    symptom_count_7d,
                    severity_avg_after,
                    now_iso,
                ),
            )
            # placeholder structure
            evidence_strength = 0.0
            model_probability = 0.0
            if candidate.exposure_with_ingredients_count > 0:
                # Temporary signal: ingredient expansion exists but no external evidence indexed yet.
                evidence_strength = 0.1

            conn.execute(
                """
                INSERT INTO insights (
                    user_id, item_id, symptom_id, model_score, evidence_score, final_score,
                    evidence_summary, evidence_strength_score, model_probability, display_decision_reason, citations_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.user_id,
                    candidate.item_id,
                    candidate.symptom_id,
                    model_probability,
                    evidence_strength,
                    (model_probability + evidence_strength) / 2.0,
                    "No literature evidence is indexed yet; this candidate is from temporal co-occurrence only.",
                    evidence_strength,
                    model_probability,
                    "suppressed_pending_rag_and_model",
                    json.dumps([]),
                    now_iso,
                ),
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
                    0,
                    0,
                    now_iso,
                ),
            )
            inserted += 1

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return {
        "candidates_considered": len(aggregates),
        "pairs_evaluated": pair_count,
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
                i.symptom_id AS symptom_id,
                s.name AS symptom_name,
                i.model_probability AS model_probability,
                i.evidence_strength_score AS evidence_strength_score,
                i.evidence_summary AS evidence_summary,
                i.display_decision_reason AS display_decision_reason,
                i.citations_json AS citations_json,
                i.created_at AS created_at
            FROM insights i
            JOIN items it ON it.id = i.item_id
            JOIN symptoms s ON s.id = i.symptom_id
            WHERE i.user_id = ?
        """
        params: list[Any] = [user_id]
        if not include_suppressed:
            base_sql += " AND (i.display_decision_reason IS NULL OR i.display_decision_reason NOT LIKE 'suppressed_%')"
        base_sql += " ORDER BY i.created_at DESC"
        rows = conn.execute(base_sql, tuple(params)).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
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
                    "symptom_id": row["symptom_id"],
                    "symptom_name": row["symptom_name"],
                    "model_probability": row["model_probability"],
                    "evidence_strength_score": row["evidence_strength_score"],
                    "evidence_summary": row["evidence_summary"],
                    "display_decision_reason": row["display_decision_reason"],
                    "created_at": row["created_at"],
                    "citations": parsed_citations,
                }
            )
        return results
    finally:
        conn.close()
