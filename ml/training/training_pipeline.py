from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from hashlib import sha256
import json
from collections import Counter, defaultdict
from typing import Any

from api.db import get_connection, initialize_database
from ml.evaluator import (
    DEFAULT_CALIBRATOR_PATH,
    DEFAULT_CURATED_TRAINING_PATH,
    DEFAULT_THRESHOLDS_PATH,
    FEATURE_ORDER,
    build_training_rows_from_curated_catalog,
    build_training_rows_from_user_feedback_detailed,
    compute_penalty_score,
    default_model_path,
    fit_probability_calibrator,
    get_decision_thresholds,
    model_metadata,
    apply_probability_calibrator,
    predict_model_probability,
    save_decision_thresholds,
    save_probability_calibrator,
    save_xgboost_artifact,
    train_xgboost_model,
    tune_model_threshold,
)
from ml.final_score import (
    compute_evidence_quality_from_features,
    evaluate_fusion_candidate,
    load_fusion_monitor,
    save_fusion_artifacts,
    save_fusion_monitor,
    should_promote_fusion,
    train_fusion_model,
)
from sklearn.metrics import brier_score_loss

DEFAULT_USER_EVAL_REPORT_PATH = default_model_path().parent / "user_timeline_eval_report.json"
DEFAULT_MODEL_PROMOTION_REPORT_PATH = default_model_path().parent / "model_promotion_report.json"

MIN_FEEDBACK_LABELS_FOR_BLEND = 200
MIN_CLASS_LABELS_FOR_BLEND = 60
USER_WEIGHT_FLOOR = 0.20
USER_WEIGHT_CEIL = 0.80
USER_WEIGHT_MAX_LABELS = 1000
UNSEEN_USER_HOLDOUT_RATIO = 0.20
SEVERE_SYMPTOM_NAMES = {
    "chest pain",
    "fainting",
    "shortness of breath",
    "syncope",
    "vomiting",
}
BAD_PAIR_CONTROLS = {
    ("water", "headache"),
}

def _stable_jitter(key: str, *, amplitude: float) -> float:
    digest = sha256(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False) / float(2**64)
    centered = (value * 2.0) - 1.0
    return centered * amplitude


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _stable_bucket(name: str, *, bucket_count: int = 1000) -> int:
    digest = sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % max(1, int(bucket_count))


def _pick_unseen_holdout_users(user_ids: list[int], *, ratio: float) -> set[int]:
    unique_ids = sorted({int(uid) for uid in user_ids if int(uid) > 0})
    if len(unique_ids) < 5:
        return set()
    target = max(1, int(round(len(unique_ids) * max(0.0, min(0.9, float(ratio))))))
    scored = sorted(((_stable_bucket(f"user:{uid}"), uid) for uid in unique_ids), key=lambda row: (row[0], row[1]))
    return {uid for _, uid in scored[:target]}


def _split_feedback_rows_time_safe(
    feedback_rows: list[dict[str, Any]],
    *,
    unseen_holdout_users: set[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    holdout_rows: list[dict[str, Any]] = []
    remaining_rows: list[dict[str, Any]] = []
    for row in feedback_rows:
        user_id = int(row.get("user_id") or 0)
        if user_id in unseen_holdout_users:
            holdout_rows.append(row)
        else:
            remaining_rows.append(row)

    by_user: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in remaining_rows:
        by_user[int(row.get("user_id") or 0)].append(row)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for user_id, rows in by_user.items():
        _ = user_id
        rows_sorted = sorted(rows, key=lambda r: (str(r.get("feedback_ts") or ""), int(r.get("item_id") or 0)))
        if len(rows_sorted) == 1:
            val_rows.extend(rows_sorted)
            continue
        split_at = max(1, int(len(rows_sorted) * 0.8))
        if split_at >= len(rows_sorted):
            split_at = len(rows_sorted) - 1
        train_rows.extend(rows_sorted[:split_at])
        val_rows.extend(rows_sorted[split_at:])
    return train_rows, val_rows, holdout_rows


def _feedback_blend_weight(total_labels: int) -> float:
    if total_labels <= MIN_FEEDBACK_LABELS_FOR_BLEND:
        return USER_WEIGHT_FLOOR
    span = max(1, USER_WEIGHT_MAX_LABELS - MIN_FEEDBACK_LABELS_FOR_BLEND)
    progress = max(0.0, min(1.0, (float(total_labels) - MIN_FEEDBACK_LABELS_FOR_BLEND) / float(span)))
    return USER_WEIGHT_FLOOR + (USER_WEIGHT_CEIL - USER_WEIGHT_FLOOR) * progress


def _repeat_rows_deterministic(rows: list[dict[str, Any]], *, factor: int) -> list[dict[str, Any]]:
    if factor <= 1 or not rows:
        return list(rows)
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(row)
        for _ in range(factor - 1):
            out.append(dict(row))
    return out


def _rebalance_seed_feedback_rows(
    *,
    seed_rows: list[dict[str, Any]],
    feedback_rows: list[dict[str, Any]],
    target_feedback_weight: float,
) -> tuple[list[dict[str, Any]], float]:
    if not seed_rows and not feedback_rows:
        return [], 0.0
    if not seed_rows:
        return list(feedback_rows), 1.0
    if not feedback_rows:
        return list(seed_rows), 0.0
    target_feedback_weight = max(0.0, min(1.0, float(target_feedback_weight)))
    seed_count = len(seed_rows)
    feedback_count = len(feedback_rows)
    if target_feedback_weight <= 0.0:
        return list(seed_rows), 0.0
    if target_feedback_weight >= 1.0:
        return list(feedback_rows), 1.0
    numerator = target_feedback_weight * float(seed_count)
    denominator = max(1e-6, (1.0 - target_feedback_weight) * float(feedback_count))
    feedback_factor = int(max(1, round(numerator / denominator)))
    weighted_feedback = _repeat_rows_deterministic(feedback_rows, factor=feedback_factor)
    blended = list(seed_rows) + weighted_feedback
    realized_weight = float(len(weighted_feedback)) / float(max(1, len(blended)))
    return blended, realized_weight


def _rows_to_xyg(rows: list[dict[str, Any]], *, fallback_group: int = -9) -> tuple[list[list[float]], list[int], list[int]]:
    x: list[list[float]] = []
    y: list[int] = []
    groups: list[int] = []
    for row in rows:
        x.append(list(row["feature_vector"]))
        y.append(int(row["label"]))
        user_id = int(row.get("user_id") or 0)
        groups.append(user_id if user_id > 0 else int(fallback_group))
    return x, y, groups


def _prepare_curated_rows(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("Curated training catalog must be a JSON array")
    vectors, labels = build_training_rows_from_curated_catalog(path)
    prepared: list[dict[str, Any]] = []
    filtered_idx = 0
    for row in payload:
        if not isinstance(row, dict):
            continue
        exposure = str(row.get("exposure", "")).strip().lower()
        symptom = str(row.get("symptom", "")).strip().lower()
        if not exposure or not symptom:
            continue
        if filtered_idx >= len(vectors):
            break
        prepared.append(
            {
                "feature_vector": list(vectors[filtered_idx]),
                "label": int(labels[filtered_idx]),
                "user_id": -1,
                "item_name": exposure,
                "symptom_name": symptom,
                "feedback_ts": "1970-01-01T00:00:00+00:00",
            }
        )
        filtered_idx += 1
    seed_train: list[dict[str, Any]] = []
    seed_benchmark: list[dict[str, Any]] = []
    for row in prepared:
        key = f"{row['item_name']}|{row['symptom_name']}"
        bucket = _stable_bucket(key, bucket_count=100)
        if bucket < 20:
            seed_benchmark.append(row)
        else:
            seed_train.append(row)
    if not seed_train and seed_benchmark:
        seed_train = seed_benchmark[:-1]
        seed_benchmark = seed_benchmark[-1:]
    return seed_train, seed_benchmark


def _evaluate_threshold_metrics(
    *,
    probabilities: list[float],
    labels: list[int],
    threshold: float,
) -> dict[str, float]:
    metrics = _binary_metrics(probabilities, labels, threshold=threshold)
    if probabilities and labels and len(probabilities) == len(labels):
        try:
            metrics["brier"] = float(brier_score_loss(labels, probabilities))
        except Exception:
            metrics["brier"] = 1.0
    else:
        metrics["brier"] = 1.0
    return metrics


def _severe_recall(
    *,
    rows: list[dict[str, Any]],
    probabilities: list[float],
    threshold: float,
) -> float:
    positives = 0
    true_positives = 0
    for row, prob in zip(rows, probabilities):
        symptom_name = str(row.get("symptom_name") or "").strip().lower()
        if symptom_name not in SEVERE_SYMPTOM_NAMES:
            continue
        label = int(row.get("label") or 0)
        if label != 1:
            continue
        positives += 1
        if float(prob) >= float(threshold):
            true_positives += 1
    if positives == 0:
        return 1.0
    return float(true_positives) / float(positives)


def _bad_pair_false_positive_rate(
    *,
    rows: list[dict[str, Any]],
    probabilities: list[float],
    threshold: float,
) -> float:
    bad_total = 0
    bad_fp = 0
    for row, prob in zip(rows, probabilities):
        pair = (
            str(row.get("item_name") or "").strip().lower(),
            str(row.get("symptom_name") or "").strip().lower(),
        )
        if pair not in BAD_PAIR_CONTROLS:
            continue
        bad_total += 1
        if float(prob) >= float(threshold):
            bad_fp += 1
    if bad_total == 0:
        return 0.0
    return float(bad_fp) / float(bad_total)


def _binary_metrics(probabilities: list[float], labels: list[int], threshold: float) -> dict[str, float]:
    if not probabilities or not labels or len(probabilities) != len(labels):
        return {"count": 0.0, "precision": 0.0, "recall": 0.0}
    tp = fp = fn = 0
    for p, y in zip(probabilities, labels):
        pred = 1 if float(p) >= float(threshold) else 0
        true = 1 if int(y) == 1 else 0
        if pred == 1 and true == 1:
            tp += 1
        elif pred == 1 and true == 0:
            fp += 1
        elif pred == 0 and true == 1:
            fn += 1
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    return {"count": float(len(labels)), "precision": precision, "recall": recall}


def _augment_fusion_rows(
    rows: list[dict[str, float]],
    labels: list[int],
    *,
    factor: int,
) -> tuple[list[dict[str, float]], list[int]]:
    if factor <= 1 or not rows:
        return rows, labels
    augmented_rows: list[dict[str, float]] = []
    augmented_labels: list[int] = []
    for idx, (row, label) in enumerate(zip(rows, labels)):
        base = dict(row)
        augmented_rows.append(base)
        augmented_labels.append(int(label))
        for replica in range(1, int(factor)):
            key = f"{idx}|{replica}|{label}"
            p = _clamp01(float(base.get("model_probability", 0.0)) + _stable_jitter(key + "|p", amplitude=0.035))
            q = _clamp01(float(base.get("evidence_quality", 0.0)) + _stable_jitter(key + "|q", amplitude=0.04))
            pen = _clamp01(float(base.get("penalty_score", 0.0)) + _stable_jitter(key + "|pen", amplitude=0.03))
            cites = max(0.0, float(base.get("citation_count", 0.0)) + _stable_jitter(key + "|c", amplitude=0.8))
            contra = _clamp01(float(base.get("contradict_ratio", 0.0)) + _stable_jitter(key + "|cr", amplitude=0.05))
            augmented_rows.append(
                {
                    "model_probability": p,
                    "evidence_quality": q,
                    "penalty_score": pen,
                    "citation_count": cites,
                    "contradict_ratio": contra,
                }
            )
            augmented_labels.append(int(label))
    return augmented_rows, augmented_labels


def run_training(
    *,
    rounds: int = 30,
    learning_rate: float = 0.2,
    dataset_source: str = "hybrid",
    curated_path: str = str(DEFAULT_CURATED_TRAINING_PATH),
    output: str = "",
    window_hours: int = 24,
    controls_per_case: int = 2,
    target_precision: float = 0.75,
    fusion_augment_factor: int = 10,
    train_xgboost: bool = True,
    train_fusion: bool = True,
) -> dict:
    initialize_database()
    allowed_sources = {"curated", "feedback", "hybrid"}
    if dataset_source not in allowed_sources:
        raise ValueError(f"dataset_source must be one of {sorted(allowed_sources)}")

    curated_seed_rows, curated_benchmark_rows = _prepare_curated_rows(Path(curated_path))
    conn = get_connection()
    try:
        feedback_rows = build_training_rows_from_user_feedback_detailed(conn)
    finally:
        conn.close()
    feedback_label_counter = Counter(int(row.get("label") or 0) for row in feedback_rows)
    feedback_total = int(len(feedback_rows))
    feedback_pos = int(feedback_label_counter.get(1, 0))
    feedback_neg = int(feedback_label_counter.get(0, 0))

    unseen_holdout_users = _pick_unseen_holdout_users(
        [int(row.get("user_id") or 0) for row in feedback_rows],
        ratio=UNSEEN_USER_HOLDOUT_RATIO,
    )
    feedback_train_rows, feedback_val_rows, feedback_holdout_rows = _split_feedback_rows_time_safe(
        feedback_rows,
        unseen_holdout_users=unseen_holdout_users,
    )

    phase = "seed_bootstrap"
    realized_feedback_weight = 0.0
    if dataset_source == "curated":
        train_rows = list(curated_seed_rows)
    elif dataset_source == "feedback":
        phase = "validated_only"
        train_rows = list(feedback_train_rows)
    else:
        if (
            feedback_total >= MIN_FEEDBACK_LABELS_FOR_BLEND
            and feedback_pos >= MIN_CLASS_LABELS_FOR_BLEND
            and feedback_neg >= MIN_CLASS_LABELS_FOR_BLEND
        ):
            phase = "controlled_blend"
            target_feedback_weight = _feedback_blend_weight(feedback_total)
            train_rows, realized_feedback_weight = _rebalance_seed_feedback_rows(
                seed_rows=curated_seed_rows,
                feedback_rows=feedback_train_rows,
                target_feedback_weight=target_feedback_weight,
            )
        else:
            train_rows = list(curated_seed_rows)

    if len(train_rows) < 10:
        raise RuntimeError("Not enough training rows after split/phase gating.")
    train_labels = [int(row.get("label") or 0) for row in train_rows]
    if sum(train_labels) == 0 or sum(train_labels) == len(train_labels):
        raise RuntimeError("Training labels are single-class; gather both verified and rejected examples.")

    val_rows = list(feedback_val_rows) if feedback_val_rows else list(curated_benchmark_rows)
    holdout_rows = list(feedback_holdout_rows) if feedback_holdout_rows else list(val_rows)
    if not val_rows:
        val_rows = list(train_rows[-max(1, len(train_rows) // 5):])
    if not holdout_rows:
        holdout_rows = list(val_rows)

    x, y, _ = _rows_to_xyg(train_rows)

    output_path = Path(output) if output else default_model_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_model_path = output_path.parent / f"{output_path.name}.candidate"
    candidate_calibrator_path = output_path.parent / f"{DEFAULT_CALIBRATOR_PATH.name}.candidate"
    current_thresholds = {
        "min_model_probability": 0.35,
        "min_overall_confidence": 0.45,
    }
    try:
        current_thresholds = get_decision_thresholds(path=DEFAULT_THRESHOLDS_PATH)
    except Exception:
        pass

    promotion_reason = "xgboost_training_disabled"
    promoted_xgboost = False
    if train_xgboost:
        model = train_xgboost_model(
            x,
            y,
            rounds=max(1, int(rounds)),
            learning_rate=max(0.01, float(learning_rate)),
        )
        save_xgboost_artifact(model, path=candidate_model_path)
    elif not output_path.exists():
        raise RuntimeError(f"XGBoost artifact required for fusion-only training: {output_path}")

    tuned_model_threshold: float | None = None
    per_user_eval: dict[str, dict[str, float]] = {}
    promotion_metrics: dict[str, Any] = {}
    if train_xgboost:
        val_x, val_labels, val_groups = _rows_to_xyg(val_rows)
        val_probs_raw: list[float] = []
        for feature_values in val_x:
            feature_map = {name: feature_values[j] for j, name in enumerate(FEATURE_ORDER)}
            val_probs_raw.append(
                predict_model_probability(
                    feature_map,
                    model_path=candidate_model_path,
                    use_calibration=False,
                )
            )
        calibrator = fit_probability_calibrator(val_probs_raw, val_labels)
        save_probability_calibrator(calibrator, path=candidate_calibrator_path)

        holdout_x, holdout_labels, holdout_groups = _rows_to_xyg(holdout_rows)
        holdout_probs_candidate: list[float] = []
        for feature_values in holdout_x:
            feature_map = {name: feature_values[j] for j, name in enumerate(FEATURE_ORDER)}
            holdout_probs_candidate.append(
                predict_model_probability(
                    feature_map,
                    model_path=candidate_model_path,
                    calibrator_path=candidate_calibrator_path,
                    use_calibration=True,
                )
            )
        threshold_labels = holdout_labels if holdout_labels else val_labels
        threshold_probs = holdout_probs_candidate if holdout_labels else [
            float(apply_probability_calibrator(p, calibrator_path=candidate_calibrator_path))
            for p in val_probs_raw
        ]
        tuned_model_threshold = tune_model_threshold(
            threshold_probs,
            threshold_labels,
            target_precision=max(0.5, min(0.99, float(target_precision))),
        )
        eval_threshold = float(tuned_model_threshold or 0.5)
        per_user_probs: dict[int, list[float]] = {}
        per_user_labels: dict[int, list[int]] = {}
        if holdout_labels:
            calibrated_probs = holdout_probs_candidate
            calibrated_groups = holdout_groups
        else:
            calibrated_probs = [
                float(apply_probability_calibrator(p, calibrator_path=candidate_calibrator_path))
                for p in val_probs_raw
            ]
            calibrated_groups = val_groups
        for offset, user_group in enumerate(calibrated_groups):
            if user_group <= 0:
                continue
            per_user_probs.setdefault(user_group, []).append(float(calibrated_probs[offset]))
            per_user_labels.setdefault(user_group, []).append(int(threshold_labels[offset]))
        for user_id in sorted(per_user_probs.keys()):
            metrics = _binary_metrics(
                per_user_probs[user_id],
                per_user_labels.get(user_id, []),
                threshold=eval_threshold,
            )
            metrics["avg_probability"] = (
                sum(per_user_probs[user_id]) / max(1, len(per_user_probs[user_id]))
            )
            per_user_eval[str(user_id)] = metrics

        candidate_eval = _evaluate_threshold_metrics(
            probabilities=calibrated_probs,
            labels=threshold_labels,
            threshold=eval_threshold,
        )
        candidate_severe_recall = _severe_recall(
            rows=holdout_rows if holdout_labels else val_rows,
            probabilities=calibrated_probs,
            threshold=eval_threshold,
        )
        benchmark_x, benchmark_labels, _ = _rows_to_xyg(curated_benchmark_rows)
        benchmark_probs_candidate: list[float] = []
        for feature_values in benchmark_x:
            feature_map = {name: feature_values[j] for j, name in enumerate(FEATURE_ORDER)}
            benchmark_probs_candidate.append(
                predict_model_probability(
                    feature_map,
                    model_path=candidate_model_path,
                    calibrator_path=candidate_calibrator_path,
                    use_calibration=True,
                )
            )
        candidate_bad_pair_fpr = _bad_pair_false_positive_rate(
            rows=curated_benchmark_rows,
            probabilities=benchmark_probs_candidate,
            threshold=eval_threshold,
        )

        current_available = output_path.exists()
        current_eval = {
            "brier": 1.0,
            "severe_recall": 0.0,
            "bad_pair_fpr": 1.0,
        }
        if current_available:
            current_threshold = float(current_thresholds.get("min_model_probability", 0.35))
            current_probs_holdout: list[float] = []
            for feature_values in (holdout_x if holdout_labels else val_x):
                feature_map = {name: feature_values[j] for j, name in enumerate(FEATURE_ORDER)}
                current_probs_holdout.append(
                    predict_model_probability(
                        feature_map,
                        model_path=output_path,
                        calibrator_path=DEFAULT_CALIBRATOR_PATH,
                        use_calibration=True,
                    )
                )
            current_metrics = _evaluate_threshold_metrics(
                probabilities=current_probs_holdout,
                labels=threshold_labels,
                threshold=current_threshold,
            )
            benchmark_probs_current: list[float] = []
            for feature_values in benchmark_x:
                feature_map = {name: feature_values[j] for j, name in enumerate(FEATURE_ORDER)}
                benchmark_probs_current.append(
                    predict_model_probability(
                        feature_map,
                        model_path=output_path,
                        calibrator_path=DEFAULT_CALIBRATOR_PATH,
                        use_calibration=True,
                    )
                )
            current_eval = {
                "brier": float(current_metrics["brier"]),
                "severe_recall": _severe_recall(
                    rows=holdout_rows if holdout_labels else val_rows,
                    probabilities=current_probs_holdout,
                    threshold=current_threshold,
                ),
                "bad_pair_fpr": _bad_pair_false_positive_rate(
                    rows=curated_benchmark_rows,
                    probabilities=benchmark_probs_current,
                    threshold=current_threshold,
                ),
            }

        candidate_eval_full = {
            **candidate_eval,
            "severe_recall": float(candidate_severe_recall),
            "bad_pair_fpr": float(candidate_bad_pair_fpr),
        }
        if not current_available:
            promoted_xgboost = True
            promotion_reason = "bootstrap_no_existing_model"
        else:
            improves_severe_recall = candidate_eval_full["severe_recall"] >= current_eval["severe_recall"]
            improves_bad_fpr = candidate_eval_full["bad_pair_fpr"] <= current_eval["bad_pair_fpr"]
            improves_brier = candidate_eval_full["brier"] <= current_eval["brier"]
            strict_gain = (
                candidate_eval_full["severe_recall"] > current_eval["severe_recall"]
                or candidate_eval_full["bad_pair_fpr"] < current_eval["bad_pair_fpr"]
                or candidate_eval_full["brier"] < current_eval["brier"]
            )
            promoted_xgboost = improves_severe_recall and improves_bad_fpr and improves_brier and strict_gain
            promotion_reason = (
                "promoted_beats_current_on_guardrails"
                if promoted_xgboost
                else "rejected_guardrail_regression"
            )

        if promoted_xgboost:
            save_xgboost_artifact(model, path=output_path)
            save_probability_calibrator(calibrator, path=DEFAULT_CALIBRATOR_PATH)
            save_decision_thresholds(
                min_evidence_strength=0.2,
                min_model_probability=tuned_model_threshold,
                min_overall_confidence=max(0.45, tuned_model_threshold - 0.05),
                target_precision=max(0.5, min(0.99, float(target_precision))),
                source=str(dataset_source),
                severe_symptom_min_model_probability=max(0.10, tuned_model_threshold * 0.75),
                severe_symptom_min_overall_confidence=max(0.35, max(0.45, tuned_model_threshold - 0.05) * 0.8),
                severe_symptom_names=sorted(SEVERE_SYMPTOM_NAMES),
                path=DEFAULT_THRESHOLDS_PATH,
            )
        promotion_metrics = {
            "candidate": candidate_eval_full,
            "current": current_eval,
            "promoted": bool(promoted_xgboost),
            "reason": promotion_reason,
        }
        DEFAULT_MODEL_PROMOTION_REPORT_PATH.write_text(
            json.dumps(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "dataset_source": str(dataset_source),
                    "phase": phase,
                    "promotion_metrics": promotion_metrics,
                },
                indent=2,
            )
        )
        report_payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_source": str(dataset_source),
            "phase": phase,
            "threshold_used": float(tuned_model_threshold or 0.5),
            "users_in_validation": len(per_user_eval),
            "per_user": per_user_eval,
            "promotion": promotion_metrics,
        }
        DEFAULT_USER_EVAL_REPORT_PATH.write_text(json.dumps(report_payload, indent=2))
    if candidate_model_path.exists():
        candidate_model_path.unlink(missing_ok=True)
    if candidate_calibrator_path.exists():
        candidate_calibrator_path.unlink(missing_ok=True)

    fusion_rows: list[dict[str, float]] = []
    for feature_values in x:
        feature_map = {name: feature_values[j] for j, name in enumerate(FEATURE_ORDER)}
        fusion_rows.append(
            {
                "model_probability": predict_model_probability(
                    feature_map,
                    model_path=output_path,
                    use_calibration=False,
                ),
                "evidence_quality": compute_evidence_quality_from_features(feature_map),
                "penalty_score": compute_penalty_score(feature_map),
                "citation_count": float(feature_map.get("citation_count", 0.0)),
                "contradict_ratio": float(feature_map.get("contradict_ratio", 0.0)),
            }
        )

    fusion_train_rows: list[dict[str, float]] = []
    fusion_val_rows: list[dict[str, float]] = []
    fusion_metrics: dict[str, float] = {}
    promote_fusion = False
    fusion_reason = "fusion_training_disabled"
    train_idx = list(range(len(x)))
    val_idx = list(range(max(0, len(x) - max(1, len(x) // 5)), len(x)))
    if train_fusion:
        fusion_train_rows = [fusion_rows[idx] for idx in train_idx]
        fusion_train_labels = [int(y[idx]) for idx in train_idx]
        fusion_val_rows = [fusion_rows[idx] for idx in val_idx]
        fusion_val_labels = [int(y[idx]) for idx in val_idx]
        fusion_train_rows, fusion_train_labels = _augment_fusion_rows(
            fusion_train_rows,
            fusion_train_labels,
            factor=max(1, int(fusion_augment_factor)),
        )

        fusion_model, fusion_calibrator = train_fusion_model(
            rows=fusion_train_rows,
            labels=fusion_train_labels,
        )
        fusion_metrics = evaluate_fusion_candidate(
            model=fusion_model,
            calibrator=fusion_calibrator,
            rows=fusion_val_rows,
            labels=fusion_val_labels,
        )
        previous_monitor = load_fusion_monitor()
        promote_fusion, fusion_reason = should_promote_fusion(
            candidate_metrics=fusion_metrics,
            previous_monitor=previous_monitor,
        )
        if promote_fusion:
            save_fusion_artifacts(model=fusion_model, calibrator=fusion_calibrator)
        save_fusion_monitor(
            metrics=fusion_metrics,
            promoted=promote_fusion,
            reason=fusion_reason,
            dataset_source=str(dataset_source),
            model_path=output_path,
        )

    return {
        "status": "ok",
        "rows": len(x),
        "positives": int(sum(y)),
        "negatives": int(len(y) - sum(y)),
        "artifact": str(output_path),
        "calibrator_artifact": str(DEFAULT_CALIBRATOR_PATH),
        "thresholds_artifact": str(DEFAULT_THRESHOLDS_PATH),
        "rounds": int(rounds),
        "learning_rate": float(learning_rate),
        "dataset_source": str(dataset_source),
        "phase": phase,
        "feedback_labels_total": feedback_total,
        "feedback_labels_positive": feedback_pos,
        "feedback_labels_negative": feedback_neg,
        "feedback_weight_realized": float(realized_feedback_weight),
        "unseen_user_holdout_count": len(unseen_holdout_users),
        "window_hours": int(window_hours),
        "controls_per_case": int(controls_per_case),
        "backend": model_metadata(output_path)["backend"],
        "xgboost_promoted": bool(promoted_xgboost),
        "xgboost_promotion_reason": promotion_reason,
        "xgboost_promotion_metrics": promotion_metrics,
        "fusion_promoted": bool(promote_fusion),
        "fusion_reason": fusion_reason,
        "fusion_metrics": fusion_metrics,
        "fusion_augment_factor": int(fusion_augment_factor),
        "fusion_rows_train": len(fusion_train_rows),
        "fusion_rows_val": len(fusion_val_rows),
        "train_xgboost": bool(train_xgboost),
        "train_fusion": bool(train_fusion),
        "tuned_model_threshold": tuned_model_threshold,
        "per_user_eval_users": len(per_user_eval),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train linkage evaluation gradient-boosted model.")
    parser.add_argument("--rounds", type=int, default=30, help="Boosting rounds")
    parser.add_argument("--learning-rate", type=float, default=0.2, help="Boosting learning rate")
    parser.add_argument(
        "--dataset-source",
        choices=("curated", "feedback", "hybrid"),
        default="hybrid",
        help="Training dataset source",
    )
    parser.add_argument(
        "--curated-path",
        type=str,
        default=str(DEFAULT_CURATED_TRAINING_PATH),
        help="Path to curated linkage JSON catalog",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Path to model artifact JSON",
    )
    parser.add_argument("--window-hours", type=int, default=24, help="Case/control window size in hours")
    parser.add_argument("--controls-per-case", type=int, default=2, help="Control windows per case")
    parser.add_argument(
        "--target-precision",
        type=float,
        default=0.75,
        help="Precision target for threshold tuning",
    )
    parser.add_argument(
        "--fusion-augment-factor",
        type=int,
        default=10,
        help="Multiplier for fusion training rows via deterministic augmentation",
    )
    args = parser.parse_args()
    print(
        run_training(
            rounds=args.rounds,
            learning_rate=args.learning_rate,
            dataset_source=args.dataset_source,
            curated_path=args.curated_path,
            output=args.output,
            window_hours=args.window_hours,
            controls_per_case=args.controls_per_case,
            target_precision=args.target_precision,
            fusion_augment_factor=args.fusion_augment_factor,
        )
    )


if __name__ == "__main__":
    main()
