from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _time_confidence_score(value: str | None) -> float:
    normalized = (value or "").strip().lower()
    if normalized == "exact":
        return 1.0
    if normalized == "approx":
        return 0.7
    if normalized == "backfilled":
        return 0.4
    return 0.55


def _lag_bucket_minutes(minutes: float) -> str:
    if minutes <= 360.0:
        return "0_6h"
    if minutes <= 1440.0:
        return "6_24h"
    if minutes <= 4320.0:
        return "24_72h"
    return "72h_7d"


@dataclass
class ExposureRow:
    user_id: int
    item_id: int
    ts: datetime
    route: str | None
    time_confidence_score: float
    has_ingredient_expansion: bool


@dataclass
class SymptomRow:
    user_id: int
    symptom_id: int
    ts: datetime
    severity: float
    time_confidence_score: float


def _evidence_lookup_for_pair(conn, *, item_id: int, symptom_id: int) -> dict[str, float]:
    ingredient_rows = conn.execute(
        "SELECT ingredient_id FROM items_ingredients WHERE item_id = ?",
        (item_id,),
    ).fetchall()
    ingredient_ids = [int(row["ingredient_id"]) for row in ingredient_rows if row["ingredient_id"] is not None]
    where_parts = ["(c.item_id = ?)"]
    params: list[Any] = [item_id, symptom_id]
    if ingredient_ids:
        placeholders = ",".join("?" for _ in ingredient_ids)
        where_parts.append(f"(c.ingredient_id IN ({placeholders}))")
        params.extend(ingredient_ids)
    where_clause = " OR ".join(where_parts)
    rows = conn.execute(
        f"""
        SELECT
            c.evidence_polarity_and_strength AS polarity,
            c.study_quality_score AS study_quality_score,
            c.population_match AS population_match,
            c.temporality_match AS temporality_match,
            c.risk_of_bias AS risk_of_bias,
            c.llm_confidence AS llm_confidence
        FROM claims c
        WHERE c.symptom_id = ?
          AND ({where_clause})
        """,
        tuple(params),
    ).fetchall()
    if not rows:
        return {
            "evidence_strength_score": 0.0,
            "evidence_score_signed": 0.0,
            "citation_count": 0.0,
            "support_ratio": 0.0,
            "contradict_ratio": 0.0,
            "neutral_ratio": 0.0,
            "avg_relevance": 0.0,
        }
    support = 0
    contradict = 0
    neutral = 0
    signed_sum = 0.0
    study_quality_sum = 0.0
    population_match_sum = 0.0
    temporality_match_sum = 0.0
    risk_of_bias_sum = 0.0
    llm_confidence_sum = 0.0
    for row in rows:
        polarity = int(row["polarity"] or 0)
        signed_sum += float(polarity)
        if polarity > 0:
            support += 1
        elif polarity < 0:
            contradict += 1
        else:
            neutral += 1
        study_quality_sum += float(row["study_quality_score"] or 0.5)
        population_match_sum += float(row["population_match"] or 0.5)
        temporality_match_sum += float(row["temporality_match"] or 0.5)
        risk_of_bias_sum += float(row["risk_of_bias"] or 0.5)
        llm_confidence_sum += float(row["llm_confidence"] or 0.5)
    n = len(rows)
    signed_score = signed_sum / max(1.0, float(n))
    signed_score = max(-1.0, min(1.0, signed_score))
    return {
        "evidence_strength_score": abs(signed_score),
        "evidence_score_signed": signed_score,
        "citation_count": float(n),
        "support_ratio": support / n,
        "contradict_ratio": contradict / n,
        "neutral_ratio": neutral / n,
        "avg_relevance": 0.55,
        "study_quality_score": max(0.0, min(1.0, study_quality_sum / n)),
        "population_match": max(0.0, min(1.0, population_match_sum / n)),
        "temporality_match": max(0.0, min(1.0, temporality_match_sum / n)),
        "risk_of_bias": max(0.0, min(1.0, risk_of_bias_sum / n)),
        "llm_confidence": max(0.0, min(1.0, llm_confidence_sum / n)),
    }


def _load_user_events(conn, *, user_id: int) -> tuple[list[ExposureRow], list[SymptomRow]]:
    exposure_rows = conn.execute(
        """
        SELECT
            e.user_id AS user_id,
            e.item_id AS item_id,
            COALESCE(e.timestamp, e.time_range_start) AS ts,
            e.route AS route,
            e.time_confidence AS time_confidence,
            CASE WHEN EXISTS (
                SELECT 1 FROM exposure_expansions x WHERE x.exposure_event_id = e.id
            ) THEN 1 ELSE 0 END AS has_ingredient_expansion
        FROM exposure_events e
        WHERE e.user_id = ?
          AND COALESCE(e.timestamp, e.time_range_start) IS NOT NULL
        ORDER BY ts ASC
        """,
        (user_id,),
    ).fetchall()
    symptom_rows = conn.execute(
        """
        SELECT
            s.user_id AS user_id,
            s.symptom_id AS symptom_id,
            COALESCE(s.timestamp, s.time_range_start) AS ts,
            s.severity AS severity,
            s.time_confidence AS time_confidence
        FROM symptom_events s
        WHERE s.user_id = ?
          AND COALESCE(s.timestamp, s.time_range_start) IS NOT NULL
        ORDER BY ts ASC
        """,
        (user_id,),
    ).fetchall()
    exposures: list[ExposureRow] = []
    for row in exposure_rows:
        ts = _parse_iso(row["ts"])
        if ts is None:
            continue
        exposures.append(
            ExposureRow(
                user_id=int(row["user_id"]),
                item_id=int(row["item_id"]),
                ts=ts,
                route=(row["route"] or "").strip().lower() or None,
                time_confidence_score=_time_confidence_score(row["time_confidence"]),
                has_ingredient_expansion=bool(row["has_ingredient_expansion"]),
            )
        )
    symptoms: list[SymptomRow] = []
    for row in symptom_rows:
        ts = _parse_iso(row["ts"])
        if ts is None:
            continue
        symptoms.append(
            SymptomRow(
                user_id=int(row["user_id"]),
                symptom_id=int(row["symptom_id"]),
                ts=ts,
                severity=float(row["severity"] or 0.0),
                time_confidence_score=_time_confidence_score(row["time_confidence"]),
            )
        )
    return exposures, symptoms


def _window_features(
    *,
    exposures: list[ExposureRow],
    symptoms: list[SymptomRow],
    item_id: int,
    symptom_id: int,
    window_end: datetime,
    window_duration: timedelta,
    evidence_features: dict[str, float],
) -> dict[str, float]:
    window_start = window_end - window_duration
    recent_exposures = [
        row
        for row in exposures
        if row.item_id == item_id and window_start <= row.ts <= window_end
    ]
    recent_symptoms = [
        row
        for row in symptoms
        if row.symptom_id == symptom_id and window_start <= row.ts <= window_end
    ]
    if recent_exposures:
        lags = [(window_end - row.ts).total_seconds() / 60.0 for row in recent_exposures]
        lag_min = min(lags)
        lag_avg = sum(lags) / len(lags)
        routes = {row.route for row in recent_exposures if row.route}
        buckets = {_lag_bucket_minutes(lag) for lag in lags}
        exposure_with_ingredients_ratio = sum(
            1.0 for row in recent_exposures if row.has_ingredient_expansion
        ) / len(recent_exposures)
        confidence_scores = [row.time_confidence_score for row in recent_exposures]
    else:
        lag_min = float(window_duration.total_seconds() / 60.0)
        lag_avg = lag_min
        routes = set()
        buckets = set()
        exposure_with_ingredients_ratio = 0.0
        confidence_scores = []

    severity_values = [row.severity for row in recent_symptoms if row.severity > 0]
    symptom_confidences = [row.time_confidence_score for row in recent_symptoms]
    if symptom_confidences:
        confidence_scores.extend(symptom_confidences)
    time_confidence_score = (
        sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.55
    )
    cooccurrence_count = float(len(recent_exposures))
    unique_symptoms = float(max(1, len(recent_symptoms)))
    pair_density = cooccurrence_count / unique_symptoms if unique_symptoms > 0 else 0.0

    row = {
        "time_gap_min_minutes": float(lag_min),
        "time_gap_avg_minutes": float(lag_avg),
        "cooccurrence_count": cooccurrence_count,
        "cooccurrence_unique_symptom_count": unique_symptoms,
        "pair_density": float(pair_density),
        "exposure_count_7d": cooccurrence_count,
        "symptom_count_7d": float(len(recent_symptoms)),
        "severity_avg_after": (
            float(sum(severity_values) / len(severity_values)) if severity_values else 0.0
        ),
        "route_count": float(len(routes)),
        "lag_bucket_diversity": float(len(buckets)),
        "exposure_with_ingredients_ratio": float(exposure_with_ingredients_ratio),
        "time_confidence_score": float(time_confidence_score),
    }
    row.update(evidence_features)
    return row


def build_case_control_training_rows(
    conn,
    *,
    window_hours: int = 24,
    controls_per_case: int = 2,
) -> tuple[list[dict[str, float]], list[int], list[int]]:
    users = conn.execute("SELECT id FROM users ORDER BY id ASC").fetchall()
    window_duration = timedelta(hours=max(1, int(window_hours)))
    all_rows: list[dict[str, float]] = []
    labels: list[int] = []
    groups: list[int] = []

    for user_row in users:
        user_id = int(user_row["id"])
        exposures, symptoms = _load_user_events(conn, user_id=user_id)
        if not exposures or not symptoms:
            continue
        item_ids = sorted({row.item_id for row in exposures})
        symptom_ids = sorted({row.symptom_id for row in symptoms})
        evidence_map: dict[tuple[int, int], dict[str, float]] = {}
        for item_id in item_ids:
            for symptom_id in symptom_ids:
                evidence_map[(item_id, symptom_id)] = _evidence_lookup_for_pair(
                    conn,
                    item_id=item_id,
                    symptom_id=symptom_id,
                )

        symptom_times = sorted(row.ts for row in symptoms)
        first_exposure_ts = min(row.ts for row in exposures)
        for symptom in symptoms:
            # Case windows: before symptom episodes.
            for item_id in item_ids:
                features = _window_features(
                    exposures=exposures,
                    symptoms=symptoms,
                    item_id=item_id,
                    symptom_id=symptom.symptom_id,
                    window_end=symptom.ts,
                    window_duration=window_duration,
                    evidence_features=evidence_map[(item_id, symptom.symptom_id)],
                )
                all_rows.append(features)
                labels.append(1)
                groups.append(user_id)

                for c in range(max(1, controls_per_case)):
                    offset_hours = window_hours * (2 + c)
                    control_end = symptom.ts - timedelta(hours=offset_hours)
                    if control_end <= first_exposure_ts:
                        continue
                    has_nearby_symptom = any(
                        abs((symptom_ts - control_end).total_seconds())
                        <= window_duration.total_seconds()
                        for symptom_ts in symptom_times
                    )
                    if has_nearby_symptom:
                        continue
                    control_features = _window_features(
                        exposures=exposures,
                        symptoms=symptoms,
                        item_id=item_id,
                        symptom_id=symptom.symptom_id,
                        window_end=control_end,
                        window_duration=window_duration,
                        evidence_features=evidence_map[(item_id, symptom.symptom_id)],
                    )
                    all_rows.append(control_features)
                    labels.append(0)
                    groups.append(user_id)

    return all_rows, labels, groups
