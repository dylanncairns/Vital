from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from api.db import DB_PATH


DEFAULT_FUSION_MODEL_PATH = DB_PATH.parent / "final_score_fusion.pkl"
DEFAULT_FUSION_CALIBRATOR_PATH = DB_PATH.parent / "final_score_calibrator.pkl"

_FUSION_MODEL_CACHE: dict[str, Any] = {}
_FUSION_CALIBRATOR_CACHE: dict[str, Any] = {}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def compute_evidence_quality_from_features(feature_map: dict[str, float]) -> float:
    evidence_strength = _clamp01(float(feature_map.get("evidence_strength_score", 0.0)))
    citation_coverage = _clamp01(float(feature_map.get("citation_count", 0.0)) / 6.0)
    support_ratio = _clamp01(float(feature_map.get("support_ratio", 0.0)))
    contradict_ratio = _clamp01(float(feature_map.get("contradict_ratio", 0.0)))
    avg_relevance = _clamp01(float(feature_map.get("avg_relevance", 0.0)))
    study_quality = _clamp01(float(feature_map.get("study_quality_score", 0.5)))
    population_match = _clamp01(float(feature_map.get("population_match", 0.5)))
    temporality_match = _clamp01(float(feature_map.get("temporality_match", 0.5)))
    risk_of_bias = _clamp01(float(feature_map.get("risk_of_bias", 0.5)))
    llm_confidence = _clamp01(float(feature_map.get("llm_confidence", 0.5)))
    return _clamp01(
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


def _build_fusion_feature_row(
    *,
    model_probability: float,
    evidence_quality: float,
    penalty_score: float,
    citation_count: float = 0.0,
    contradict_ratio: float = 0.0,
) -> list[float]:
    p = _clamp01(model_probability)
    q = _clamp01(evidence_quality)
    penalty = _clamp01(penalty_score)
    c_count = max(0.0, float(citation_count))
    c_ratio = _clamp01(contradict_ratio)
    return [
        p,
        q,
        penalty,
        p * q,
        q * (1.0 - penalty),
        _clamp01(c_count / 6.0),
        c_ratio,
    ]


def train_fusion_model(
    *,
    rows: list[dict[str, float]],
    labels: list[int],
) -> tuple[LogisticRegression, IsotonicRegression]:
    if len(rows) != len(labels) or len(rows) < 20:
        raise ValueError("Fusion training requires matching rows/labels and at least 20 rows")
    x = [
        _build_fusion_feature_row(
            model_probability=float(row.get("model_probability", 0.0)),
            evidence_quality=float(row.get("evidence_quality", 0.0)),
            penalty_score=float(row.get("penalty_score", 0.0)),
            citation_count=float(row.get("citation_count", 0.0)),
            contradict_ratio=float(row.get("contradict_ratio", 0.0)),
        )
        for row in rows
    ]
    y = [1 if int(v) == 1 else 0 for v in labels]
    model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    model.fit(x, y)
    raw_probs = model.predict_proba(x)[:, 1].tolist()
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw_probs, y)
    return model, calibrator


def save_fusion_artifacts(
    *,
    model: LogisticRegression,
    calibrator: IsotonicRegression,
    model_path: Path = DEFAULT_FUSION_MODEL_PATH,
    calibrator_path: Path = DEFAULT_FUSION_CALIBRATOR_PATH,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as handle:
        pickle.dump(model, handle)
    with open(calibrator_path, "wb") as handle:
        pickle.dump(calibrator, handle)
    _FUSION_MODEL_CACHE.pop(str(model_path.resolve()), None)
    _FUSION_CALIBRATOR_CACHE.pop(str(calibrator_path.resolve()), None)


def _load_fusion_model(path: Path = DEFAULT_FUSION_MODEL_PATH) -> LogisticRegression | None:
    key = str(path.resolve())
    if key in _FUSION_MODEL_CACHE:
        return _FUSION_MODEL_CACHE[key]
    if not path.exists():
        _FUSION_MODEL_CACHE[key] = None
        return None
    with open(path, "rb") as handle:
        model = pickle.load(handle)
    _FUSION_MODEL_CACHE[key] = model
    return model


def _load_fusion_calibrator(path: Path = DEFAULT_FUSION_CALIBRATOR_PATH) -> IsotonicRegression | None:
    key = str(path.resolve())
    if key in _FUSION_CALIBRATOR_CACHE:
        return _FUSION_CALIBRATOR_CACHE[key]
    if not path.exists():
        _FUSION_CALIBRATOR_CACHE[key] = None
        return None
    with open(path, "rb") as handle:
        calibrator = pickle.load(handle)
    _FUSION_CALIBRATOR_CACHE[key] = calibrator
    return calibrator


def predict_final_score(
    *,
    model_probability: float,
    evidence_quality: float,
    penalty_score: float,
    citation_count: float = 0.0,
    contradict_ratio: float = 0.0,
    model_path: Path = DEFAULT_FUSION_MODEL_PATH,
    calibrator_path: Path = DEFAULT_FUSION_CALIBRATOR_PATH,
) -> float:
    model = _load_fusion_model(model_path)
    calibrator = _load_fusion_calibrator(calibrator_path)
    if model is None:
        base = (0.5 * _clamp01(model_probability)) + (0.5 * _clamp01(evidence_quality))
        penalty = _clamp01(penalty_score)
        penalty_weight = 0.25 + (0.75 * penalty)
        penalty_impact = penalty * penalty_weight
        return _clamp01(base - penalty_impact)
    row = _build_fusion_feature_row(
        model_probability=model_probability,
        evidence_quality=evidence_quality,
        penalty_score=penalty_score,
        citation_count=citation_count,
        contradict_ratio=contradict_ratio,
    )
    raw = float(model.predict_proba([row])[0][1])
    if calibrator is None:
        return _clamp01(raw)
    calibrated = calibrator.predict([_clamp01(raw)])
    if len(calibrated) == 0:
        return _clamp01(raw)
    return _clamp01(float(calibrated[0]))
