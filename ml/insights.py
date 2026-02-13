from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from api.db import get_connection
from ingestion.expand_exposure import backfill_missing_exposure_expansions
from ml.rag import (
    aggregate_evidence,
    build_candidate_query,
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
MIN_EVIDENCE_STRENGTH = 0.2


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

    conn = get_connection()
    now_iso = _now_iso()
    inserted = 0
    candidates_considered = 0
    pairs_evaluated = (
        sum(len(candidate.lags_minutes) for key, candidate in aggregates.items() if key in target_pairs)
        if target_pairs is not None
        else pair_count
    )
    try:
        item_name_map = fetch_item_name_map(conn)
        symptom_name_map = fetch_symptom_name_map(conn)

        if target_pairs is None:
            conn.execute("DELETE FROM insights WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM derived_features_ingredients WHERE user_id = ?", (user_id,))
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

        for candidate in aggregates.values():
            pair_key = (candidate.item_id, candidate.symptom_id)
            if target_pairs is not None and pair_key not in target_pairs:
                continue
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
                top_k=5,
            )
            evidence = aggregate_evidence(retrieved_claims)
            model_probability = 0.0
            if not evidence["citations"]:
                decision_reason = "suppressed_no_citations"
            elif float(evidence["evidence_strength_score"] or 0.0) < MIN_EVIDENCE_STRENGTH:
                decision_reason = "suppressed_low_evidence_strength"
            else:
                decision_reason = "supported"

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
                    evidence["evidence_score"],
                    (model_probability + evidence["evidence_score"]) / 2.0,
                    evidence["evidence_summary"],
                    evidence["evidence_strength_score"],
                    model_probability,
                    decision_reason,
                    json.dumps(evidence["citations"]),
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
                    5,
                    len(retrieved_claims),
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
                    "display_status": (
                        "supported"
                        if row["display_decision_reason"] == "supported"
                        else ("insufficient_evidence" if row["display_decision_reason"] in {"suppressed_no_citations", "suppressed_low_evidence_strength"} else "suppressed")
                    ),
                    "created_at": row["created_at"],
                    "citations": parsed_citations,
                }
            )
        return results
    finally:
        conn.close()
