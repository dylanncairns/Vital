from __future__ import annotations

import hashlib
import json
import pickle
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = DATA_DIR / "models"


DEFAULT_FUSION_MODEL_PATH = MODELS_DIR / "final_score_fusion.pkl"
DEFAULT_FUSION_CALIBRATOR_PATH = MODELS_DIR / "final_score_calibrator.pkl"
DEFAULT_FUSION_MONITOR_PATH = MODELS_DIR / "final_score_monitor.json"
DEFAULT_THRESHOLDS_PATH = MODELS_DIR / "decision_thresholds.json"
DEFAULT_SCORE_GUARDRAILS_PATH = MODELS_DIR / "score_guardrails.json"

_FUSION_MODEL_CACHE: dict[str, Any] = {}
_FUSION_CALIBRATOR_CACHE: dict[str, Any] = {}
_RUNTIME_FUSION_OUTPUTS: deque[float] = deque(maxlen=600)

_DEFAULT_SCORE_GUARDRAILS: dict[str, float] = {
    "runtime_min_window": 120.0,
    "runtime_max_high_conf_ratio": 0.90,
    "runtime_max_mean_probability": 0.88,
    "runtime_min_std_probability": 0.03,
    "runtime_min_guardrail_rows": 100.0,
    "runtime_min_guardrail_positives": 20.0,
    "runtime_min_guardrail_negatives": 20.0,
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _artifact_fingerprint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    stat = path.stat()
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return {
        "exists": True,
        "sha256": hasher.hexdigest(),
        "size_bytes": int(stat.st_size),
        "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def load_score_guardrails(path: Path = DEFAULT_SCORE_GUARDRAILS_PATH) -> dict[str, float]:
    guardrails = dict(_DEFAULT_SCORE_GUARDRAILS)
    if not path.exists():
        return guardrails
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return guardrails
    if not isinstance(payload, dict):
        return guardrails
    for key, default in _DEFAULT_SCORE_GUARDRAILS.items():
        guardrails[key] = max(0.0, _safe_float(payload.get(key), default))
    return guardrails


def _calibrator_is_saturated(calibrator: Any) -> bool:
    if isinstance(calibrator, dict) and calibrator.get("type") == "sigmoid":
        model = calibrator.get("model")
        try:
            probe = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            preds = [float(v) for v in model.predict_proba([[p] for p in probe])[:, 1].tolist()]
        except Exception:
            return False
        if len(preds) != len(probe):
            return False
        mid = preds[2:6]
        near_one_mid = sum(1 for v in mid if v >= 0.98)
        dynamic_range = max(preds) - min(preds)
        unique_bucket_count = len({round(v, 4) for v in preds})
        if near_one_mid >= 2:
            return True
        if dynamic_range <= 0.25 and unique_bucket_count <= 3:
            return True
        return False
    try:
        probe = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        preds = [float(v) for v in calibrator.predict(probe)]
    except Exception:
        return False
    if len(preds) != len(probe):
        return False
    # If calibration maps mid-range raw probabilities to near-1, it's unusable.
    mid = preds[2:6]  # 0.4, 0.5, 0.6, 0.7
    near_one_mid = sum(1 for v in mid if v >= 0.98)
    # Also catch extremely low dynamic range (step-like calibration).
    dynamic_range = max(preds) - min(preds)
    unique_bucket_count = len({round(v, 4) for v in preds})
    if near_one_mid >= 2:
        return True
    if dynamic_range <= 0.25 and unique_bucket_count <= 3:
        return True
    return False


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


def _predict_raw_and_calibrated(
    *,
    model: LogisticRegression,
    calibrator: Any | None,
    rows: list[dict[str, float]],
) -> tuple[list[float], list[float]]:
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
    raw_probs = [float(v) for v in model.predict_proba(x)[:, 1].tolist()]
    if calibrator is None:
        return raw_probs, [_clamp01(v) for v in raw_probs]
    if isinstance(calibrator, dict) and calibrator.get("type") == "sigmoid":
        cal_model = calibrator.get("model")
        try:
            calibrated = [float(v) for v in cal_model.predict_proba([[_clamp01(v)] for v in raw_probs])[:, 1].tolist()]
        except Exception:
            calibrated = [_clamp01(v) for v in raw_probs]
        return raw_probs, [_clamp01(v) for v in calibrated]
    calibrated = [float(v) for v in calibrator.predict([_clamp01(v) for v in raw_probs])]
    return raw_probs, [_clamp01(v) for v in calibrated]


def evaluate_fusion_candidate(
    *,
    model: LogisticRegression,
    calibrator: Any | None,
    rows: list[dict[str, float]],
    labels: list[int],
) -> dict[str, float]:
    if len(rows) != len(labels) or len(rows) < 2:
        raise ValueError("Fusion evaluation requires matching rows/labels and at least 2 rows")
    y = [1 if int(v) == 1 else 0 for v in labels]
    _, probs = _predict_raw_and_calibrated(model=model, calibrator=calibrator, rows=rows)

    positives = int(sum(y))
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        # AUC metrics are undefined on single-class validation slices.
        auc = 0.5
        ap = float(sum(probs) / len(probs)) if probs else 0.0
    else:
        auc = float(roc_auc_score(y, probs))
        ap = float(average_precision_score(y, probs))
    brier = float(brier_score_loss(y, probs))
    mean_prob = float(sum(probs) / len(probs)) if probs else 0.0
    std_prob = float(
        (sum((v - mean_prob) ** 2 for v in probs) / len(probs)) ** 0.5
    ) if probs else 0.0
    high_conf_ratio = float(sum(1 for v in probs if v >= 0.9) / len(probs)) if probs else 0.0
    return {
        "rows": float(len(rows)),
        "positives": float(positives),
        "negatives": float(negatives),
        "auc": _clamp01(auc),
        "average_precision": _clamp01(ap),
        "brier": _clamp01(brier),
        "mean_probability": _clamp01(mean_prob),
        "std_probability": _clamp01(std_prob),
        "high_conf_ratio": _clamp01(high_conf_ratio),
    }


def load_fusion_monitor(path: Path = DEFAULT_FUSION_MONITOR_PATH) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def save_fusion_monitor(
    *,
    metrics: dict[str, float],
    promoted: bool,
    reason: str,
    dataset_source: str,
    model_path: Path = DEFAULT_FUSION_MODEL_PATH,
    calibrator_path: Path = DEFAULT_FUSION_CALIBRATOR_PATH,
    thresholds_path: Path = DEFAULT_THRESHOLDS_PATH,
    guardrails_path: Path = DEFAULT_SCORE_GUARDRAILS_PATH,
    path: Path = DEFAULT_FUSION_MONITOR_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = load_fusion_monitor(path=path) or {}
    history = previous.get("history")
    if not isinstance(history, list):
        history = []

    guardrails = load_score_guardrails(path=guardrails_path)
    payload = {
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "dataset_source": dataset_source,
        "promoted": bool(promoted),
        "reason": str(reason),
        "metrics": metrics,
        "guardrails": guardrails,
        "versions": {
            "fusion_model": _artifact_fingerprint(model_path),
            "fusion_calibrator": _artifact_fingerprint(calibrator_path),
            "decision_thresholds": _artifact_fingerprint(thresholds_path),
            "score_guardrails": _artifact_fingerprint(guardrails_path),
        },
    }
    history.append(
        {
            "created_at": payload["created_at"],
            "dataset_source": dataset_source,
            "promoted": bool(promoted),
            "reason": str(reason),
            "metrics": metrics,
        }
    )
    payload["history"] = history[-30:]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def should_promote_fusion(
    *,
    candidate_metrics: dict[str, float],
    previous_monitor: dict[str, Any] | None,
) -> tuple[bool, str]:
    # Absolute floor checks.
    if float(candidate_metrics.get("auc", 0.0)) < 0.58:
        return False, "guardrail_fail_auc_floor"
    if float(candidate_metrics.get("average_precision", 0.0)) < 0.50:
        return False, "guardrail_fail_ap_floor"
    if float(candidate_metrics.get("brier", 1.0)) > 0.28:
        return False, "guardrail_fail_brier_ceiling"

    # Drift/regression checks against last promoted run.
    if not previous_monitor:
        return True, "promote_no_previous_baseline"
    prev_metrics = previous_monitor.get("metrics")
    prev_promoted = bool(previous_monitor.get("promoted", False))
    if not prev_promoted or not isinstance(prev_metrics, dict):
        return True, "promote_previous_not_usable"

    prev_auc = float(prev_metrics.get("auc", 0.0))
    prev_ap = float(prev_metrics.get("average_precision", 0.0))
    prev_brier = float(prev_metrics.get("brier", 1.0))
    prev_high_conf = float(prev_metrics.get("high_conf_ratio", 0.0))
    new_auc = float(candidate_metrics.get("auc", 0.0))
    new_ap = float(candidate_metrics.get("average_precision", 0.0))
    new_brier = float(candidate_metrics.get("brier", 1.0))
    new_high_conf = float(candidate_metrics.get("high_conf_ratio", 0.0))

    if new_auc < (prev_auc - 0.02):
        return False, "guardrail_fail_auc_regression"
    if new_ap < (prev_ap - 0.03):
        return False, "guardrail_fail_ap_regression"
    if new_brier > (prev_brier + 0.02):
        return False, "guardrail_fail_brier_regression"
    if abs(new_high_conf - prev_high_conf) > 0.25:
        return False, "guardrail_fail_output_drift"

    return True, "promote_passed_guardrails"


def train_fusion_model(
    *,
    rows: list[dict[str, float]],
    labels: list[int],
) -> tuple[LogisticRegression, dict[str, Any]]:
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
    calibrator_model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    calibrator_model.fit([[_clamp01(v)] for v in raw_probs], y)
    return model, {"type": "sigmoid", "model": calibrator_model}


def save_fusion_artifacts(
    *,
    model: LogisticRegression,
    calibrator: Any,
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


def _load_fusion_calibrator(path: Path = DEFAULT_FUSION_CALIBRATOR_PATH) -> Any | None:
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
    fallback_base = (0.5 * _clamp01(model_probability)) + (0.5 * _clamp01(evidence_quality))
    fallback_penalty = _clamp01(penalty_score)
    fallback_penalty_weight = 0.25 + (0.75 * fallback_penalty)
    fallback_penalty_impact = fallback_penalty * fallback_penalty_weight
    fallback_score = _clamp01(fallback_base - fallback_penalty_impact)

    model = _load_fusion_model(model_path)
    calibrator = _load_fusion_calibrator(calibrator_path)
    monitor = load_fusion_monitor()
    guardrails = load_score_guardrails()
    use_fusion = model is not None
    if isinstance(monitor, dict):
        metrics = monitor.get("metrics")
        if isinstance(metrics, dict):
            rows = float(metrics.get("rows", 0.0))
            positives = float(metrics.get("positives", 0.0))
            negatives = float(metrics.get("negatives", 0.0))
            high_conf_ratio = float(metrics.get("high_conf_ratio", 0.0))
            brier = float(metrics.get("brier", 1.0))
            promoted = bool(monitor.get("promoted", False))
            # Require meaningful validation coverage before using learned fusion in production scoring.
            if (
                (not promoted)
                or rows < max(100.0, guardrails["runtime_min_guardrail_rows"])
                or positives < max(20.0, guardrails["runtime_min_guardrail_positives"])
                or negatives < max(20.0, guardrails["runtime_min_guardrail_negatives"])
            ):
                use_fusion = False
            # Guardrail only on statistically meaningful validation runs.
            if rows >= 100.0 and positives >= 20.0 and negatives >= 20.0 and (
                high_conf_ratio > 0.85 or brier > 0.22
            ):
                use_fusion = False
        versions = monitor.get("versions")
        if isinstance(versions, dict):
            monitor_model = versions.get("fusion_model")
            monitor_calibrator = versions.get("fusion_calibrator")
            monitor_thresholds = versions.get("decision_thresholds")
            current_model = _artifact_fingerprint(model_path)
            current_calibrator = _artifact_fingerprint(calibrator_path)
            current_thresholds = _artifact_fingerprint(DEFAULT_THRESHOLDS_PATH)
            if (
                isinstance(monitor_model, dict)
                and isinstance(current_model, dict)
                and monitor_model.get("sha256")
                and current_model.get("sha256")
                and monitor_model.get("sha256") != current_model.get("sha256")
            ):
                use_fusion = False
            if (
                isinstance(monitor_calibrator, dict)
                and isinstance(current_calibrator, dict)
                and monitor_calibrator.get("sha256")
                and current_calibrator.get("sha256")
                and monitor_calibrator.get("sha256") != current_calibrator.get("sha256")
            ):
                use_fusion = False
            if (
                isinstance(monitor_thresholds, dict)
                and isinstance(current_thresholds, dict)
                and monitor_thresholds.get("sha256")
                and current_thresholds.get("sha256")
                and monitor_thresholds.get("sha256") != current_thresholds.get("sha256")
            ):
                use_fusion = False
    else:
        use_fusion = False

    if not use_fusion:
        return fallback_score

    row = _build_fusion_feature_row(
        model_probability=model_probability,
        evidence_quality=evidence_quality,
        penalty_score=penalty_score,
        citation_count=citation_count,
        contradict_ratio=contradict_ratio,
    )
    raw = float(model.predict_proba([row])[0][1])
    if calibrator is None:
        output = _clamp01((0.65 * _clamp01(raw)) + (0.35 * fallback_score))
        _RUNTIME_FUSION_OUTPUTS.append(output)
        return _runtime_guardrail_output(output, fallback_score, guardrails)
    if _calibrator_is_saturated(calibrator):
        output = _clamp01((0.65 * _clamp01(raw)) + (0.35 * fallback_score))
        _RUNTIME_FUSION_OUTPUTS.append(output)
        return _runtime_guardrail_output(output, fallback_score, guardrails)
    if isinstance(calibrator, dict) and calibrator.get("type") == "sigmoid":
        cal_model = calibrator.get("model")
        try:
            calibrated = cal_model.predict_proba([[_clamp01(raw)]])[:, 1]
            if len(calibrated) > 0:
                blended = (0.65 * _clamp01(float(calibrated[0]))) + (0.35 * fallback_score)
                output = _clamp01(blended)
                _RUNTIME_FUSION_OUTPUTS.append(output)
                return _runtime_guardrail_output(output, fallback_score, guardrails)
            output = _clamp01((0.65 * _clamp01(raw)) + (0.35 * fallback_score))
            _RUNTIME_FUSION_OUTPUTS.append(output)
            return _runtime_guardrail_output(output, fallback_score, guardrails)
        except Exception:
            output = _clamp01((0.65 * _clamp01(raw)) + (0.35 * fallback_score))
            _RUNTIME_FUSION_OUTPUTS.append(output)
            return _runtime_guardrail_output(output, fallback_score, guardrails)
    calibrated = calibrator.predict([_clamp01(raw)])
    if len(calibrated) == 0:
        output = _clamp01((0.65 * _clamp01(raw)) + (0.35 * fallback_score))
        _RUNTIME_FUSION_OUTPUTS.append(output)
        return _runtime_guardrail_output(output, fallback_score, guardrails)
    blended = (0.65 * _clamp01(float(calibrated[0]))) + (0.35 * fallback_score)
    output = _clamp01(blended)
    _RUNTIME_FUSION_OUTPUTS.append(output)
    return _runtime_guardrail_output(output, fallback_score, guardrails)


def _runtime_guardrail_output(output: float, fallback_score: float, guardrails: dict[str, float]) -> float:
    min_window = int(max(50.0, guardrails.get("runtime_min_window", 120.0)))
    if len(_RUNTIME_FUSION_OUTPUTS) < min_window:
        return output
    values = list(_RUNTIME_FUSION_OUTPUTS)
    n = float(len(values))
    mean_val = sum(values) / n
    std_val = (sum((v - mean_val) ** 2 for v in values) / n) ** 0.5
    high_conf_ratio = sum(1 for v in values if v >= 0.9) / n
    if high_conf_ratio > guardrails.get("runtime_max_high_conf_ratio", 0.90):
        return fallback_score
    if mean_val > guardrails.get("runtime_max_mean_probability", 0.88):
        return fallback_score
    if std_val < guardrails.get("runtime_min_std_probability", 0.03):
        return fallback_score
    return output
