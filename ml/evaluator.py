from __future__ import annotations

import json
import math
import os
import pickle
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

try:
    import xgboost as xgb
except Exception:
    xgb = None
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

from api.db import DB_PATH

ROUTE_BUCKETS = (
    "ingestion",
    "inhalation",
    "dermal",
    "injection",
    "proximity_environment",
    "behavioral",
    "other",
    "unknown",
)
LAG_BUCKET_KEYS = ("0_6h", "6_24h", "24_72h", "72h_7d")
ROUTE_TEMPORAL_FEATURES = (
    [f"route_share_{route}" for route in ROUTE_BUCKETS]
    + [f"route_min_lag_{route}" for route in ROUTE_BUCKETS]
    + [f"route_lag_0_24h_share_{route}" for route in ROUTE_BUCKETS]
    + [f"route_lag_24h_7d_share_{route}" for route in ROUTE_BUCKETS]
)

FEATURE_ORDER = [
    "time_gap_min_minutes",
    "time_gap_avg_minutes",
    "cooccurrence_count",
    "cooccurrence_unique_symptom_count",
    "pair_density",
    "exposure_count_7d",
    "symptom_count_7d",
    "severity_avg_after",
    "route_count",
    "lag_bucket_diversity",
    "exposure_with_ingredients_ratio",
    "evidence_strength_score",
    "evidence_score_signed",
    "citation_count",
    "support_ratio",
    "contradict_ratio",
    "neutral_ratio",
    "avg_relevance",
    "study_quality_score",
    "population_match",
    "temporality_match",
    "risk_of_bias",
    "llm_confidence",
    "time_confidence_score",
] + ROUTE_TEMPORAL_FEATURES

DEFAULT_XGBOOST_MODEL_PATH = DB_PATH.parent / "model_artifact.xgb.json"
DEFAULT_CURATED_TRAINING_PATH = DB_PATH.parent / "curated_linkages.json"
DEFAULT_CALIBRATOR_PATH = DB_PATH.parent / "model_calibrator.pkl"
DEFAULT_THRESHOLDS_PATH = DB_PATH.parent / "decision_thresholds.json"
DEFAULT_MAX_MODEL_THRESHOLD = float(os.getenv("MAX_MODEL_THRESHOLD", "0.55"))
DEFAULT_MAX_OVERALL_THRESHOLD = float(os.getenv("MAX_OVERALL_THRESHOLD", "0.65"))


def default_model_path() -> Path:
    return DEFAULT_XGBOOST_MODEL_PATH


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _stable_jitter(key: str, *, amplitude: float = 1.0) -> float:
    digest = sha256(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False) / float(2**64)
    centered = (value * 2.0) - 1.0
    return centered * amplitude


def _joint_recurrence_metrics(
    *,
    cooccurrence_count: float,
    exposure_count_7d: float,
    symptom_count_7d: float,
) -> tuple[float, float, float]:
    shared_pool = max(1.0, min(exposure_count_7d, symptom_count_7d))
    joint_ratio = clamp(cooccurrence_count / shared_pool, 0.0, 1.0)
    unmatched_exposure = max(0.0, exposure_count_7d - cooccurrence_count)
    unmatched_symptom = max(0.0, symptom_count_7d - cooccurrence_count)
    return joint_ratio, unmatched_exposure, unmatched_symptom


def compute_evidence_quality(evidence: dict[str, Any]) -> dict[str, float]:
    citations = evidence.get("citations") or []
    citation_count = len(citations)
    if citation_count == 0:
        return {
            "score": 0.0,
            "citation_count": 0.0,
            "support_ratio": 0.0,
            "contradict_ratio": 0.0,
            "neutral_ratio": 0.0,
            "avg_relevance": 0.0,
            "study_quality_score": 0.0,
            "population_match": 0.0,
            "temporality_match": 0.0,
            "risk_of_bias": 0.0,
            "llm_confidence": 0.0,
        }

    support_count = 0
    contradict_count = 0
    neutral_count = 0
    study_quality_sum = 0.0
    population_match_sum = 0.0
    temporality_match_sum = 0.0
    risk_of_bias_sum = 0.0
    llm_confidence_sum = 0.0
    for citation in citations:
        polarity = clamp(_safe_float(citation.get("evidence_polarity_and_strength"), 0.0), -1.0, 1.0)
        if polarity > 0.15:
            support_count += 1
        elif polarity < -0.15:
            contradict_count += 1
        else:
            neutral_count += 1
        study_quality_sum += clamp(_safe_float(citation.get("study_quality_score"), 0.5), 0.0, 1.0)
        population_match_sum += clamp(_safe_float(citation.get("population_match"), 0.5), 0.0, 1.0)
        temporality_match_sum += clamp(_safe_float(citation.get("temporality_match"), 0.5), 0.0, 1.0)
        risk_of_bias_sum += clamp(_safe_float(citation.get("risk_of_bias"), 0.5), 0.0, 1.0)
        llm_confidence_sum += clamp(_safe_float(citation.get("llm_confidence"), 0.5), 0.0, 1.0)

    support_ratio = support_count / citation_count
    contradict_ratio = contradict_count / citation_count
    neutral_ratio = neutral_count / citation_count
    evidence_strength = clamp(_safe_float(evidence.get("evidence_strength_score")), 0.0, 1.0)
    avg_relevance = clamp(_safe_float(evidence.get("avg_relevance"), 0.0), 0.0, 1.0)
    citation_coverage = clamp(citation_count / 6.0, 0.0, 1.0)
    study_quality = study_quality_sum / citation_count
    population_match = population_match_sum / citation_count
    temporality_match = temporality_match_sum / citation_count
    risk_of_bias = risk_of_bias_sum / citation_count
    llm_confidence = llm_confidence_sum / citation_count

    quality = (
        0.24 * evidence_strength
        + 0.12 * citation_coverage
        + 0.10 * support_ratio
        + 0.10 * avg_relevance
        + 0.12 * study_quality
        + 0.10 * population_match
        + 0.10 * temporality_match
        + 0.08 * llm_confidence
        + 0.04 * (1.0 - contradict_ratio)
        + 0.10 * (1.0 - risk_of_bias)
    )
    quality = clamp(quality, 0.0, 1.0)
    return {
        "score": quality,
        "citation_count": float(citation_count),
        "support_ratio": support_ratio,
        "contradict_ratio": contradict_ratio,
        "neutral_ratio": neutral_ratio,
        "avg_relevance": avg_relevance,
        "study_quality_score": study_quality,
        "population_match": population_match,
        "temporality_match": temporality_match,
        "risk_of_bias": risk_of_bias,
        "llm_confidence": llm_confidence,
    }


def compute_penalty_score(feature_map: dict[str, float]) -> float:
    cooccurrence_count = max(0.0, _safe_float(feature_map.get("cooccurrence_count")))
    exposure_count_7d = max(0.0, _safe_float(feature_map.get("exposure_count_7d")))
    symptom_count_7d = max(0.0, _safe_float(feature_map.get("symptom_count_7d")))
    pair_density = max(0.0, _safe_float(feature_map.get("pair_density")))
    time_confidence = clamp(_safe_float(feature_map.get("time_confidence_score"), 0.0), 0.0, 1.0)
    contradict_ratio = clamp(_safe_float(feature_map.get("contradict_ratio"), 0.0), 0.0, 1.0)
    temporal_lift = max(0.0, _safe_float(feature_map.get("temporal_lift"), 1.0))
    temporal_lift_score = clamp((temporal_lift - 0.5) / 1.5, 0.0, 1.0)
    joint_recurrence_ratio, unmatched_exposure, unmatched_symptom = _joint_recurrence_metrics(
        cooccurrence_count=cooccurrence_count,
        exposure_count_7d=exposure_count_7d,
        symptom_count_7d=symptom_count_7d,
    )

    sparse_penalty = 0.0
    if cooccurrence_count <= 1.0:
        sparse_penalty = 0.16
    elif cooccurrence_count <= 2.0:
        sparse_penalty = 0.08

    baseline_penalty = clamp(unmatched_exposure / 24.0, 0.0, 0.22) * (1.0 - 0.5 * temporal_lift_score)
    symptom_marginal_penalty = clamp(unmatched_symptom / 10.0, 0.0, 0.16) * (1.0 - 0.5 * temporal_lift_score)
    confound_penalty = clamp(max(0.0, pair_density - 1.0) * 0.06, 0.0, 0.18)
    time_penalty = 0.25 * (1.0 - time_confidence)
    contradiction_penalty = 0.22 * contradict_ratio
    weak_joint_signal_penalty = 0.10 * (1.0 - joint_recurrence_ratio) + 0.12 * (1.0 - temporal_lift_score)

    return clamp(
        sparse_penalty
        + baseline_penalty
        + symptom_marginal_penalty
        + confound_penalty
        + time_penalty
        + contradiction_penalty
        + weak_joint_signal_penalty,
        0.0,
        0.75,
    )

# final returned score
def combine_scores(*, model_probability: float, evidence_quality: float, penalty_score: float) -> float:
    # Backward-compatible wrapper around the dedicated fusion scorer.
    from ml.final_score import predict_final_score

    return predict_final_score(
        model_probability=model_probability,
        evidence_quality=evidence_quality,
        penalty_score=penalty_score,
    )


def build_feature_vector(feature_map: dict[str, float], *, feature_order: list[str] | None = None) -> list[float]:
    order = feature_order or FEATURE_ORDER
    return [_safe_float(feature_map.get(name), 0.0) for name in order]


def train_xgboost_model(
    x: list[list[float]],
    y: list[int],
    *,
    rounds: int = 80,
    learning_rate: float = 0.08,
    max_depth: int = 4,
) -> Any:
    if xgb is None:
        raise RuntimeError("xgboost import failed; install xgboost and libomp runtime.")
    if not x or not y or len(x) != len(y):
        raise ValueError("x/y training data missing or mismatched")
    matrix = xgb.DMatrix(x, label=y, feature_names=FEATURE_ORDER)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": float(learning_rate),
        "max_depth": int(max_depth),
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 2.0,
        "alpha": 0.5,
        "min_child_weight": 3.0,
        "gamma": 0.2,
        "max_delta_step": 1.0,
        "seed": 42,
    }
    return xgb.train(params=params, dtrain=matrix, num_boost_round=max(1, int(rounds)))


def save_xgboost_artifact(model: Any, *, path: Path = DEFAULT_XGBOOST_MODEL_PATH) -> None:
    if xgb is None:
        raise RuntimeError("xgboost import failed; install xgboost and libomp runtime.")
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))


_MODEL_CACHE: dict[str, Any] = {}


def _load_model(path: Path) -> Any:
    cache_key = str(path.resolve())
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if xgb is None:
        raise RuntimeError("xgboost import failed; install xgboost and libomp runtime.")
    model = xgb.Booster()
    model.load_model(str(path))
    _MODEL_CACHE[cache_key] = model
    return model


def _ensure_default_model_exists(path: Path) -> None:
    if path.exists():
        return
    x, y = build_training_rows_from_curated_catalog(DEFAULT_CURATED_TRAINING_PATH)
    if len(x) < 10 or sum(y) == 0 or sum(y) == len(y):
        raise RuntimeError("Unable to auto-train default model from curated dataset.")
    model = train_xgboost_model(x, y, rounds=80, learning_rate=0.08)
    save_xgboost_artifact(model, path=path)


def predict_model_probability(
    feature_map: dict[str, float],
    model_path: Path | None = None,
    calibrator_path: Path = DEFAULT_CALIBRATOR_PATH,
    use_calibration: bool = True,
) -> float:
    if xgb is None:
        raise RuntimeError("xgboost import failed; install xgboost and libomp runtime.")
    resolved_path = model_path
    if resolved_path is None:
        resolved_path = DEFAULT_XGBOOST_MODEL_PATH
    _ensure_default_model_exists(resolved_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Model artifact missing: {resolved_path}")
    model = _load_model(resolved_path)
    model_feature_order = getattr(model, "feature_names", None)
    inference_order = (
        [str(name) for name in model_feature_order]
        if isinstance(model_feature_order, list) and model_feature_order
        else FEATURE_ORDER
    )
    vector = build_feature_vector(feature_map, feature_order=inference_order)
    matrix = xgb.DMatrix([vector], feature_names=inference_order)
    preds = model.predict(matrix)
    if len(preds) == 0:
        return 0.0
    raw = clamp(float(preds[0]), 0.0, 1.0)
    calibrated = raw if not use_calibration else apply_probability_calibrator(raw, calibrator_path=calibrator_path)
    return _stabilize_personal_probability(calibrated, feature_map)


def fit_probability_calibrator(raw_probabilities: list[float], labels: list[int]) -> dict[str, Any]:
    if len(raw_probabilities) != len(labels) or not raw_probabilities:
        raise ValueError("Calibration data missing or mismatched")
    x = [[clamp(float(p), 0.0, 1.0)] for p in raw_probabilities]
    y = [1 if int(v) == 1 else 0 for v in labels]
    calibrator = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    calibrator.fit(x, y)
    return {"type": "sigmoid", "model": calibrator}


def save_probability_calibrator(
    calibrator: Any,
    *,
    path: Path = DEFAULT_CALIBRATOR_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(calibrator, handle)


_CALIBRATOR_CACHE: dict[str, Any] = {}


def _load_probability_calibrator(path: Path = DEFAULT_CALIBRATOR_PATH) -> Any | None:
    cache_key = str(path.resolve())
    if cache_key in _CALIBRATOR_CACHE:
        return _CALIBRATOR_CACHE[cache_key]
    if not path.exists():
        _CALIBRATOR_CACHE[cache_key] = None
        return None
    with open(path, "rb") as handle:
        calibrator = pickle.load(handle)
    _CALIBRATOR_CACHE[cache_key] = calibrator
    return calibrator


def apply_probability_calibrator(probability: float, *, calibrator_path: Path = DEFAULT_CALIBRATOR_PATH) -> float:
    calibrator = _load_probability_calibrator(calibrator_path)
    if calibrator is None:
        return clamp(probability, 0.0, 1.0)
    p = clamp(probability, 0.0, 1.0)
    probe = [0.15, 0.30, 0.50, 0.70, 0.85]
    if isinstance(calibrator, dict) and calibrator.get("type") == "sigmoid":
        model = calibrator.get("model")
        try:
            probe_preds = model.predict_proba([[v] for v in probe])[:, 1].tolist()
            if probe_preds:
                spread = max(probe_preds) - min(probe_preds)
                if spread < 0.08:
                    return p
            calibrated = model.predict_proba([[p]])[:, 1]
            if len(calibrated) > 0:
                return clamp(float(calibrated[0]), 0.0, 1.0)
        except Exception:
            return p
        return p
    try:
        probe_preds = [float(v) for v in calibrator.predict(probe)]
        if probe_preds:
            spread = max(probe_preds) - min(probe_preds)
            if spread < 0.08:
                return p
    except Exception:
        return p
    calibrated = calibrator.predict([p])
    if len(calibrated) == 0:
        return p
    return clamp(float(calibrated[0]), 0.0, 1.0)


def _stabilize_personal_probability(probability: float, feature_map: dict[str, float]) -> float:
    cooccurrence = max(0.0, _safe_float(feature_map.get("cooccurrence_count"), 0.0))
    recurrence_factor = clamp(cooccurrence / 4.0, 0.0, 1.0)
    time_confidence = clamp(_safe_float(feature_map.get("time_confidence_score"), 0.0), 0.0, 1.0)
    contradict_ratio = clamp(_safe_float(feature_map.get("contradict_ratio"), 0.0), 0.0, 1.0)
    pair_density = max(0.0, _safe_float(feature_map.get("pair_density"), 0.0))
    time_gap_min = max(0.0, _safe_float(feature_map.get("time_gap_min_minutes"), 0.0))
    symptom_count_7d = max(0.0, _safe_float(feature_map.get("symptom_count_7d"), 0.0))
    exposure_count_7d = max(0.0, _safe_float(feature_map.get("exposure_count_7d"), 0.0))
    temporal_lift = max(0.0, _safe_float(feature_map.get("temporal_lift"), 1.0))
    temporal_lift_score = clamp((temporal_lift - 0.5) / 1.5, 0.0, 1.0)
    joint_recurrence_ratio, unmatched_exposure, unmatched_symptom = _joint_recurrence_metrics(
        cooccurrence_count=cooccurrence,
        exposure_count_7d=exposure_count_7d,
        symptom_count_7d=symptom_count_7d,
    )

    if time_gap_min <= 360.0:
        lag_score = 1.0
    elif time_gap_min <= 1440.0:
        lag_score = 0.85
    elif time_gap_min <= 4320.0:
        lag_score = 0.60
    else:
        lag_score = 0.35

    symptom_noise_penalty = clamp(unmatched_symptom / 8.0, 0.0, 0.40)
    exposure_noise_penalty = clamp(unmatched_exposure / 12.0, 0.0, 0.25)
    temporal_baseline = clamp(
        0.50 * recurrence_factor
        + 0.15 * lag_score
        + 0.12 * time_confidence
        + 0.10 * joint_recurrence_ratio
        + 0.13 * temporal_lift_score
        + 0.05 * (1.0 - symptom_noise_penalty),
        0.0,
        1.0,
    )

    raw_p = clamp(probability, 0.0, 1.0)
    adjusted = (0.40 * raw_p) + (0.60 * temporal_baseline)
    adjusted *= 1.0 - (0.25 * contradict_ratio)
    adjusted *= 1.0 - clamp(max(0.0, pair_density - 1.0) * 0.05, 0.0, 0.20)
    adjusted *= 1.0 - exposure_noise_penalty
    adjusted = clamp(adjusted, 0.0, 1.0)

    # Guardrail: cap personalization drift so global model signal is not overwritten.
    # This keeps personalization meaningful but controlled.
    citation_count = max(0.0, _safe_float(feature_map.get("citation_count"), 0.0))
    max_upshift = 0.12 + (0.06 * recurrence_factor)
    if citation_count < 1.0:
        max_upshift = min(max_upshift, 0.08)
    max_downshift = 0.18
    lower = clamp(raw_p - max_downshift, 0.0, 1.0)
    upper = clamp(raw_p + max_upshift, 0.0, 1.0)
    return clamp(adjusted, lower, upper)


def tune_model_threshold(
    probabilities: list[float],
    labels: list[int],
    *,
    target_precision: float = 0.75,
) -> float:
    if len(probabilities) != len(labels) or not probabilities:
        return 0.35
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if len(thresholds) == 0:
        return 0.35
    best_threshold = 0.35
    best_recall = -1.0
    for idx, threshold in enumerate(thresholds):
        p = float(precision[idx])
        r = float(recall[idx])
        if p >= target_precision and r > best_recall:
            best_recall = r
            best_threshold = float(threshold)
    if best_recall >= 0.0:
        return clamp(best_threshold, 0.05, 0.95)
    # fallback: maximize f1 across thresholds
    best_f1 = -1.0
    for idx, threshold in enumerate(thresholds):
        p = float(precision[idx])
        r = float(recall[idx])
        denom = p + r
        f1 = (2.0 * p * r / denom) if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return clamp(best_threshold, 0.05, 0.95)


def save_decision_thresholds(
    *,
    min_evidence_strength: float,
    min_model_probability: float,
    min_overall_confidence: float,
    min_cooccurrence_for_supported: float = 2.0,
    min_support_direction: float = 0.10,
    min_unique_exposure_events_for_supported: float = 2.0,
    single_exposure_override_min_evidence_strength: float = 0.92,
    single_exposure_override_min_citations: float = 4.0,
    min_combo_item_citations: float = 1.0,
    min_combo_item_support_direction: float = 0.10,
    min_combo_pair_citations: float = 1.0,
    min_combo_pair_support_direction: float = 0.10,
    min_combo_cooccurrence_for_supported: float = 3.0,
    min_combo_unique_exposure_events_for_supported: float = 2.0,
    min_temporal_lift: float = 1.05,
    min_combo_temporal_lift: float = 1.10,
    target_precision: float,
    source: str,
    path: Path = DEFAULT_THRESHOLDS_PATH,
) -> None:
    payload = {
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "source": source,
        "target_precision": clamp(target_precision, 0.5, 0.99),
        "min_evidence_strength": clamp(min_evidence_strength, 0.0, 1.0),
        "min_model_probability": clamp(min_model_probability, 0.0, 1.0),
        "min_overall_confidence": clamp(min_overall_confidence, 0.0, 1.0),
        "min_cooccurrence_for_supported": max(2.0, float(min_cooccurrence_for_supported)),
        "min_support_direction": clamp(min_support_direction, -1.0, 1.0),
        "min_unique_exposure_events_for_supported": max(1.0, float(min_unique_exposure_events_for_supported)),
        "single_exposure_override_min_evidence_strength": clamp(
            single_exposure_override_min_evidence_strength, 0.0, 1.0
        ),
        "single_exposure_override_min_citations": max(1.0, float(single_exposure_override_min_citations)),
        "min_combo_item_citations": max(1.0, float(min_combo_item_citations)),
        "min_combo_item_support_direction": clamp(min_combo_item_support_direction, -1.0, 1.0),
        "min_combo_pair_citations": max(1.0, float(min_combo_pair_citations)),
        "min_combo_pair_support_direction": clamp(min_combo_pair_support_direction, -1.0, 1.0),
        "min_combo_cooccurrence_for_supported": max(1.0, float(min_combo_cooccurrence_for_supported)),
        "min_combo_unique_exposure_events_for_supported": max(
            1.0, float(min_combo_unique_exposure_events_for_supported)
        ),
        "min_temporal_lift": max(0.5, float(min_temporal_lift)),
        "min_combo_temporal_lift": max(0.5, float(min_combo_temporal_lift)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def get_decision_thresholds(path: Path = DEFAULT_THRESHOLDS_PATH) -> dict[str, float]:
    defaults = {
        "min_evidence_strength": 0.2,
        "min_model_probability": 0.35,
        "min_overall_confidence": 0.45,
        "min_cooccurrence_for_supported": 2.0,
        "min_support_direction": 0.10,
        "min_unique_exposure_events_for_supported": 2.0,
        "single_exposure_override_min_evidence_strength": 0.92,
        "single_exposure_override_min_citations": 4.0,
        "min_combo_item_citations": 1.0,
        "min_combo_item_support_direction": 0.10,
        "min_combo_pair_citations": 1.0,
        "min_combo_pair_support_direction": 0.10,
        "min_combo_cooccurrence_for_supported": 3.0,
        "min_combo_unique_exposure_events_for_supported": 2.0,
        "min_temporal_lift": 1.05,
        "min_combo_temporal_lift": 1.10,
    }
    if not path.exists():
        return defaults
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return defaults
    min_evidence_strength = clamp(
        _safe_float(payload.get("min_evidence_strength"), defaults["min_evidence_strength"]),
        0.0,
        1.0,
    )
    min_model_probability = clamp(
        _safe_float(payload.get("min_model_probability"), defaults["min_model_probability"]),
        0.0,
        max(0.0, min(1.0, DEFAULT_MAX_MODEL_THRESHOLD)),
    )
    min_overall_confidence = clamp(
        _safe_float(payload.get("min_overall_confidence"), defaults["min_overall_confidence"]),
        0.0,
        max(0.0, min(1.0, DEFAULT_MAX_OVERALL_THRESHOLD)),
    )
    min_cooccurrence_for_supported = max(
        2.0,
        _safe_float(payload.get("min_cooccurrence_for_supported"), defaults["min_cooccurrence_for_supported"]),
    )
    min_support_direction = clamp(
        _safe_float(payload.get("min_support_direction"), defaults["min_support_direction"]),
        -1.0,
        1.0,
    )
    min_unique_exposure_events_for_supported = max(
        1.0,
        _safe_float(
            payload.get("min_unique_exposure_events_for_supported"),
            defaults["min_unique_exposure_events_for_supported"],
        ),
    )
    single_exposure_override_min_evidence_strength = clamp(
        _safe_float(
            payload.get("single_exposure_override_min_evidence_strength"),
            defaults["single_exposure_override_min_evidence_strength"],
        ),
        0.0,
        1.0,
    )
    single_exposure_override_min_citations = max(
        1.0,
        _safe_float(
            payload.get("single_exposure_override_min_citations"),
            defaults["single_exposure_override_min_citations"],
        ),
    )
    min_combo_item_citations = max(
        1.0,
        _safe_float(
            payload.get("min_combo_item_citations"),
            defaults["min_combo_item_citations"],
        ),
    )
    min_combo_item_support_direction = clamp(
        _safe_float(
            payload.get("min_combo_item_support_direction"),
            defaults["min_combo_item_support_direction"],
        ),
        -1.0,
        1.0,
    )
    min_combo_pair_citations = max(
        1.0,
        _safe_float(
            payload.get("min_combo_pair_citations"),
            defaults["min_combo_pair_citations"],
        ),
    )
    min_combo_pair_support_direction = clamp(
        _safe_float(
            payload.get("min_combo_pair_support_direction"),
            defaults["min_combo_pair_support_direction"],
        ),
        -1.0,
        1.0,
    )
    min_combo_cooccurrence_for_supported = max(
        1.0,
        _safe_float(
            payload.get("min_combo_cooccurrence_for_supported"),
            defaults["min_combo_cooccurrence_for_supported"],
        ),
    )
    min_combo_unique_exposure_events_for_supported = max(
        1.0,
        _safe_float(
            payload.get("min_combo_unique_exposure_events_for_supported"),
            defaults["min_combo_unique_exposure_events_for_supported"],
        ),
    )
    min_temporal_lift = max(
        0.5,
        _safe_float(payload.get("min_temporal_lift"), defaults["min_temporal_lift"]),
    )
    min_combo_temporal_lift = max(
        0.5,
        _safe_float(payload.get("min_combo_temporal_lift"), defaults["min_combo_temporal_lift"]),
    )
    return {
        "min_evidence_strength": min_evidence_strength,
        "min_model_probability": min_model_probability,
        "min_overall_confidence": min_overall_confidence,
        "min_cooccurrence_for_supported": min_cooccurrence_for_supported,
        "min_support_direction": min_support_direction,
        "min_unique_exposure_events_for_supported": min_unique_exposure_events_for_supported,
        "single_exposure_override_min_evidence_strength": single_exposure_override_min_evidence_strength,
        "single_exposure_override_min_citations": single_exposure_override_min_citations,
        "min_combo_item_citations": min_combo_item_citations,
        "min_combo_item_support_direction": min_combo_item_support_direction,
        "min_combo_pair_citations": min_combo_pair_citations,
        "min_combo_pair_support_direction": min_combo_pair_support_direction,
        "min_combo_cooccurrence_for_supported": min_combo_cooccurrence_for_supported,
        "min_combo_unique_exposure_events_for_supported": min_combo_unique_exposure_events_for_supported,
        "min_temporal_lift": min_temporal_lift,
        "min_combo_temporal_lift": min_combo_temporal_lift,
    }


def build_training_rows_from_insights(conn) -> tuple[list[list[float]], list[int]]:
    rows = conn.execute(
        """
        SELECT
            d.time_gap_min_minutes,
            d.time_gap_avg_minutes,
            d.cooccurrence_count,
            d.cooccurrence_unique_symptom_count,
            d.pair_density,
            d.exposure_count_7d,
            d.symptom_count_7d,
            d.severity_avg_after,
            i.evidence_strength_score,
            i.evidence_score,
            i.citations_json,
            i.display_decision_reason
        FROM insights i
        JOIN derived_features d
          ON d.user_id = i.user_id
         AND d.item_id = i.item_id
         AND d.symptom_id = i.symptom_id
        WHERE i.citations_json IS NOT NULL
        """
    ).fetchall()

    x: list[list[float]] = []
    y: list[int] = []
    for row in rows:
        try:
            citations = json.loads(row["citations_json"] or "[]")
        except json.JSONDecodeError:
            citations = []
        evidence = {
            "evidence_strength_score": _safe_float(row["evidence_strength_score"]),
            "evidence_score": _safe_float(row["evidence_score"]),
            "avg_relevance": 0.5,
            "citations": citations if isinstance(citations, list) else [],
        }
        evidence_quality = compute_evidence_quality(evidence)
        feature_map = {
            "time_gap_min_minutes": _safe_float(row["time_gap_min_minutes"]),
            "time_gap_avg_minutes": _safe_float(row["time_gap_avg_minutes"]),
            "cooccurrence_count": _safe_float(row["cooccurrence_count"]),
            "cooccurrence_unique_symptom_count": _safe_float(row["cooccurrence_unique_symptom_count"]),
            "pair_density": _safe_float(row["pair_density"]),
            "exposure_count_7d": _safe_float(row["exposure_count_7d"]),
            "symptom_count_7d": _safe_float(row["symptom_count_7d"]),
            "severity_avg_after": _safe_float(row["severity_avg_after"]),
            "route_count": 1.0,
            "lag_bucket_diversity": 1.0,
            "exposure_with_ingredients_ratio": 0.0,
            "evidence_strength_score": _safe_float(row["evidence_strength_score"]),
            "evidence_score_signed": _safe_float(row["evidence_score"]),
            "citation_count": evidence_quality["citation_count"],
            "support_ratio": evidence_quality["support_ratio"],
            "contradict_ratio": evidence_quality["contradict_ratio"],
            "neutral_ratio": evidence_quality["neutral_ratio"],
            "avg_relevance": evidence_quality["avg_relevance"],
            "study_quality_score": evidence_quality["study_quality_score"],
            "population_match": evidence_quality["population_match"],
            "temporality_match": evidence_quality["temporality_match"],
            "risk_of_bias": evidence_quality["risk_of_bias"],
            "llm_confidence": evidence_quality["llm_confidence"],
            "time_confidence_score": 0.8,
        }
        label = 1 if row["display_decision_reason"] == "supported" else 0
        x.append(build_feature_vector(feature_map))
        y.append(label)
    return x, y


def build_training_rows_from_user_feedback(conn) -> tuple[list[list[float]], list[int], list[int]]:
    rows = conn.execute(
        """
        SELECT
            i.user_id AS user_id,
            d.time_gap_min_minutes,
            d.time_gap_avg_minutes,
            d.cooccurrence_count,
            d.cooccurrence_unique_symptom_count,
            d.pair_density,
            d.exposure_count_7d,
            d.symptom_count_7d,
            d.severity_avg_after,
            i.evidence_strength_score,
            i.evidence_score,
            i.citations_json,
            COALESCE(v.verified, 0) AS verified,
            COALESCE(v.rejected, 0) AS rejected
        FROM insight_verifications v
        JOIN insights i
          ON i.user_id = v.user_id
         AND i.item_id = v.item_id
         AND i.symptom_id = v.symptom_id
        JOIN derived_features d
          ON d.user_id = i.user_id
         AND d.item_id = i.item_id
         AND d.symptom_id = i.symptom_id
        WHERE (COALESCE(v.verified, 0) = 1 OR COALESCE(v.rejected, 0) = 1)
          AND i.id = (
            SELECT i2.id
            FROM insights i2
            WHERE i2.user_id = i.user_id
              AND i2.item_id = i.item_id
              AND i2.symptom_id = i.symptom_id
            ORDER BY i2.created_at DESC, i2.id DESC
            LIMIT 1
          )
        """
    ).fetchall()

    x: list[list[float]] = []
    y: list[int] = []
    groups: list[int] = []
    for row in rows:
        try:
            citations = json.loads(row["citations_json"] or "[]")
        except json.JSONDecodeError:
            citations = []
        evidence = {
            "evidence_strength_score": _safe_float(row["evidence_strength_score"]),
            "evidence_score": _safe_float(row["evidence_score"]),
            "avg_relevance": 0.5,
            "citations": citations if isinstance(citations, list) else [],
        }
        evidence_quality = compute_evidence_quality(evidence)
        feature_map = {
            "time_gap_min_minutes": _safe_float(row["time_gap_min_minutes"]),
            "time_gap_avg_minutes": _safe_float(row["time_gap_avg_minutes"]),
            "cooccurrence_count": _safe_float(row["cooccurrence_count"]),
            "cooccurrence_unique_symptom_count": _safe_float(row["cooccurrence_unique_symptom_count"]),
            "pair_density": _safe_float(row["pair_density"]),
            "exposure_count_7d": _safe_float(row["exposure_count_7d"]),
            "symptom_count_7d": _safe_float(row["symptom_count_7d"]),
            "severity_avg_after": _safe_float(row["severity_avg_after"]),
            "route_count": 1.0,
            "lag_bucket_diversity": 1.0,
            "exposure_with_ingredients_ratio": 0.0,
            "evidence_strength_score": _safe_float(row["evidence_strength_score"]),
            "evidence_score_signed": _safe_float(row["evidence_score"]),
            "citation_count": evidence_quality["citation_count"],
            "support_ratio": evidence_quality["support_ratio"],
            "contradict_ratio": evidence_quality["contradict_ratio"],
            "neutral_ratio": evidence_quality["neutral_ratio"],
            "avg_relevance": evidence_quality["avg_relevance"],
            "study_quality_score": evidence_quality["study_quality_score"],
            "population_match": evidence_quality["population_match"],
            "temporality_match": evidence_quality["temporality_match"],
            "risk_of_bias": evidence_quality["risk_of_bias"],
            "llm_confidence": evidence_quality["llm_confidence"],
            "time_confidence_score": 0.8,
        }
        # verified => positive label, rejected => negative label (rejected wins if inconsistent row)
        label = 0 if int(row["rejected"] or 0) == 1 else 1
        x.append(build_feature_vector(feature_map))
        y.append(label)
        groups.append(_safe_int(row["user_id"], 0))
    return x, y, groups


def build_training_rows_from_curated_catalog(
    path: Path = DEFAULT_CURATED_TRAINING_PATH,
) -> tuple[list[list[float]], list[int]]:
    if not path.exists():
        raise FileNotFoundError(f"Curated training catalog not found: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("Curated training catalog must be a JSON array")

    x: list[list[float]] = []
    y: list[int] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        exposure = str(row.get("exposure", "")).strip().lower()
        symptom = str(row.get("symptom", "")).strip().lower()
        if not exposure or not symptom:
            continue
        label = 1 if int(row.get("label", 0)) == 1 else 0
        strength = clamp(_safe_float(row.get("strength"), 0.7 if label == 1 else 0.15), 0.0, 1.0)
        base_key = f"{exposure}|{symptom}|{label}"

        if label == 1:
            support_ratio = clamp(0.70 + _stable_jitter(base_key + "|support", amplitude=0.20), 0.35, 0.98)
            contradict_ratio = clamp(0.08 + _stable_jitter(base_key + "|contra", amplitude=0.08), 0.0, 0.35)
            neutral_ratio = clamp(1.0 - support_ratio - contradict_ratio, 0.0, 0.55)
            evidence_strength = clamp(strength + _stable_jitter(base_key + "|evs", amplitude=0.08), 0.40, 1.0)
            evidence_signed = clamp(
                evidence_strength * (0.50 + _stable_jitter(base_key + "|dir", amplitude=0.20)),
                0.15,
                1.0,
            )
            cooccurrence_count = clamp(2.0 + _stable_jitter(base_key + "|cooc", amplitude=2.5), 1.0, 8.0)
            exposure_count_7d = clamp(2.0 + _stable_jitter(base_key + "|exp7", amplitude=3.0), 1.0, 12.0)
            pair_density = clamp(1.1 + _stable_jitter(base_key + "|pd", amplitude=0.5), 0.6, 3.0)
            time_confidence = clamp(0.78 + _stable_jitter(base_key + "|tc", amplitude=0.18), 0.4, 1.0)
        else:
            support_ratio = clamp(0.12 + _stable_jitter(base_key + "|support", amplitude=0.15), 0.0, 0.50)
            contradict_ratio = clamp(0.38 + _stable_jitter(base_key + "|contra", amplitude=0.25), 0.05, 0.85)
            neutral_ratio = clamp(1.0 - support_ratio - contradict_ratio, 0.0, 0.85)
            evidence_strength = clamp(strength + _stable_jitter(base_key + "|evs", amplitude=0.10), 0.0, 0.55)
            evidence_signed = clamp(
                -evidence_strength * (0.45 + _stable_jitter(base_key + "|dir", amplitude=0.25)),
                -1.0,
                -0.05,
            )
            cooccurrence_count = clamp(0.7 + _stable_jitter(base_key + "|cooc", amplitude=1.5), 0.0, 4.0)
            exposure_count_7d = clamp(4.0 + _stable_jitter(base_key + "|exp7", amplitude=6.0), 0.0, 18.0)
            pair_density = clamp(0.9 + _stable_jitter(base_key + "|pd", amplitude=0.8), 0.1, 3.5)
            time_confidence = clamp(0.70 + _stable_jitter(base_key + "|tc", amplitude=0.25), 0.25, 1.0)

        citation_count = clamp(
            3.0 + _stable_jitter(base_key + "|cites", amplitude=2.0)
            if label == 1
            else 2.0 + _stable_jitter(base_key + "|cites", amplitude=2.0),
            0.0,
            8.0,
        )
        avg_relevance = clamp(0.70 + _stable_jitter(base_key + "|rel", amplitude=0.2), 0.2, 1.0)
        study_quality_score = clamp(
            0.78 + _stable_jitter(base_key + "|sq", amplitude=0.18)
            if label == 1
            else 0.42 + _stable_jitter(base_key + "|sq", amplitude=0.22),
            0.0,
            1.0,
        )
        population_match = clamp(
            0.72 + _stable_jitter(base_key + "|pm", amplitude=0.2)
            if label == 1
            else 0.45 + _stable_jitter(base_key + "|pm", amplitude=0.2),
            0.0,
            1.0,
        )
        temporality_match = clamp(
            0.70 + _stable_jitter(base_key + "|tm", amplitude=0.2)
            if label == 1
            else 0.38 + _stable_jitter(base_key + "|tm", amplitude=0.2),
            0.0,
            1.0,
        )
        risk_of_bias = clamp(
            0.24 + _stable_jitter(base_key + "|rb", amplitude=0.16)
            if label == 1
            else 0.50 + _stable_jitter(base_key + "|rb", amplitude=0.22),
            0.0,
            1.0,
        )
        llm_confidence = clamp(
            0.84 + _stable_jitter(base_key + "|lc", amplitude=0.12)
            if label == 1
            else 0.58 + _stable_jitter(base_key + "|lc", amplitude=0.2),
            0.0,
            1.0,
        )
        severity_avg = clamp(2.7 + _stable_jitter(base_key + "|sev", amplitude=1.2), 0.0, 5.0)
        lag_min = clamp(90.0 + _stable_jitter(base_key + "|lagmin", amplitude=240.0), 5.0, 2500.0)
        lag_avg = clamp(lag_min + abs(_stable_jitter(base_key + "|lagavg", amplitude=900.0)), lag_min, 8000.0)

        feature_map = {
            "time_gap_min_minutes": lag_min,
            "time_gap_avg_minutes": lag_avg,
            "cooccurrence_count": cooccurrence_count,
            "cooccurrence_unique_symptom_count": clamp(cooccurrence_count - 0.2, 1.0, 8.0),
            "pair_density": pair_density,
            "exposure_count_7d": exposure_count_7d,
            "symptom_count_7d": clamp(1.8 + _stable_jitter(base_key + "|sym7", amplitude=2.5), 0.0, 10.0),
            "severity_avg_after": severity_avg,
            "route_count": 1.0,
            "lag_bucket_diversity": clamp(1.3 + _stable_jitter(base_key + "|lagb", amplitude=1.0), 1.0, 4.0),
            "exposure_with_ingredients_ratio": clamp(
                0.55 + _stable_jitter(base_key + "|eir", amplitude=0.35), 0.0, 1.0
            ),
            "evidence_strength_score": evidence_strength,
            "evidence_score_signed": evidence_signed,
            "citation_count": citation_count,
            "support_ratio": support_ratio,
            "contradict_ratio": contradict_ratio,
            "neutral_ratio": neutral_ratio,
            "avg_relevance": avg_relevance,
            "study_quality_score": study_quality_score,
            "population_match": population_match,
            "temporality_match": temporality_match,
            "risk_of_bias": risk_of_bias,
            "llm_confidence": llm_confidence,
            "time_confidence_score": time_confidence,
        }
        x.append(build_feature_vector(feature_map))
        y.append(label)
    return x, y


def model_metadata(path: Path | None = None) -> dict[str, Any]:
    resolved_path = path or DEFAULT_XGBOOST_MODEL_PATH
    return {
        "path": str(resolved_path),
        "exists": resolved_path.exists(),
        "feature_order": FEATURE_ORDER,
        "backend": "xgboost",
        "calibrator_exists": DEFAULT_CALIBRATOR_PATH.exists(),
        "thresholds_exists": DEFAULT_THRESHOLDS_PATH.exists(),
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
    }
