from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from hashlib import sha256
import json

from api.db import get_connection, initialize_database
from ml.evaluator import (
    DEFAULT_CALIBRATOR_PATH,
    DEFAULT_CURATED_TRAINING_PATH,
    DEFAULT_THRESHOLDS_PATH,
    FEATURE_ORDER,
    build_feature_vector,
    build_training_rows_from_curated_catalog,
    build_training_rows_from_user_feedback,
    build_training_rows_from_insights,
    compute_penalty_score,
    default_model_path,
    fit_probability_calibrator,
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
from ml.training_data import build_case_control_training_rows
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split

DEFAULT_USER_EVAL_REPORT_PATH = default_model_path().parent / "user_timeline_eval_report.json"

def _stable_jitter(key: str, *, amplitude: float) -> float:
    digest = sha256(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False) / float(2**64)
    centered = (value * 2.0) - 1.0
    return centered * amplitude


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


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


def _apply_slight_personal_reweight(
    *,
    x: list[list[float]],
    y: list[int],
    groups: list[int],
    sources: list[str],
) -> tuple[list[list[float]], list[int], list[int], list[str]]:
    # Slightly bias hybrid training toward user-timeline signal while staying conservative.
    if not x:
        return x, y, groups, sources
    out_x = list(x)
    out_y = list(y)
    out_groups = list(groups)
    out_sources = list(sources)
    for idx, source in enumerate(sources):
        source_norm = str(source or "").strip().lower()
        if source_norm == "feedback":
            # +20% effective weight (deterministic)
            if idx % 5 == 0:
                out_x.append(x[idx])
                out_y.append(y[idx])
                out_groups.append(groups[idx] if idx < len(groups) else -3)
                out_sources.append(source)
        elif source_norm == "timeline_case_control":
            # +15% effective weight (deterministic)
            if idx % 7 == 0:
                out_x.append(x[idx])
                out_y.append(y[idx])
                out_groups.append(groups[idx] if idx < len(groups) else -4)
                out_sources.append(source)
    return out_x, out_y, out_groups, out_sources


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
    allowed_sources = {"curated", "insights", "feedback", "timeline_case_control", "hybrid"}
    if dataset_source not in allowed_sources:
        raise ValueError(f"dataset_source must be one of {sorted(allowed_sources)}")

    x: list[list[float]] = []
    y: list[int] = []
    groups: list[int] = []
    sources: list[str] = []

    def _append_rows(
        rows: list[list[float]],
        labels: list[int],
        group_id: int | None = None,
        source: str = "unknown",
    ) -> None:
        x.extend(rows)
        y.extend(labels)
        if group_id is not None:
            groups.extend([group_id] * len(rows))
        else:
            groups.extend([-9] * len(rows))
        sources.extend([source] * len(rows))

    if dataset_source in {"curated", "hybrid"}:
        curated_x, curated_y = build_training_rows_from_curated_catalog(Path(curated_path))
        _append_rows(curated_x, curated_y, group_id=-1, source="curated")

    if dataset_source in {"insights", "hybrid"}:
        conn = get_connection()
        try:
            insights_x, insights_y = build_training_rows_from_insights(conn)
        finally:
            conn.close()
        _append_rows(insights_x, insights_y, group_id=-2, source="insights")

    if dataset_source in {"feedback", "hybrid"}:
        conn = get_connection()
        try:
            feedback_x, feedback_y, feedback_groups = build_training_rows_from_user_feedback(conn)
        finally:
            conn.close()
        x.extend(feedback_x)
        y.extend(feedback_y)
        groups.extend(feedback_groups)
        sources.extend(["feedback"] * len(feedback_x))

    if dataset_source in {"timeline_case_control", "hybrid"}:
        conn = get_connection()
        try:
            timeline_rows, timeline_y, timeline_groups = build_case_control_training_rows(
                conn,
                window_hours=max(1, int(window_hours)),
                controls_per_case=max(1, int(controls_per_case)),
            )
        finally:
            conn.close()
        timeline_x = [build_feature_vector(row) for row in timeline_rows]
        x.extend(timeline_x)
        y.extend(timeline_y)
        groups.extend(timeline_groups)
        sources.extend(["timeline_case_control"] * len(timeline_x))

    if dataset_source == "hybrid":
        x, y, groups, sources = _apply_slight_personal_reweight(
            x=x,
            y=y,
            groups=groups,
            sources=sources,
        )

    if len(x) < 10:
        raise RuntimeError("Not enough training rows. Run insight recomputation first.")
    if sum(y) == 0 or sum(y) == len(y):
        raise RuntimeError("Training labels are single-class; gather mixed supported/suppressed data first.")

    output_path = Path(output) if output else default_model_path()
    if train_xgboost:
        model = train_xgboost_model(
            x,
            y,
            rounds=max(1, int(rounds)),
            learning_rate=max(0.01, float(learning_rate)),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_xgboost_artifact(model, path=output_path)
    elif not output_path.exists():
        raise RuntimeError(f"XGBoost artifact required for fusion-only training: {output_path}")

    # Calibration and threshold tuning on held-out split.
    # Use group split only when we have enough group diversity; otherwise stratified split.
    use_group_split = len(groups) == len(x) and len(set(groups)) >= 10
    if use_group_split:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(splitter.split(x, y, groups=groups))
    else:
        all_idx = list(range(len(x)))
        train_idx, val_idx = train_test_split(
            all_idx,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    train_idx = list(train_idx)
    val_idx = list(val_idx)
    if not val_idx:
        split_at = max(1, int(len(x) * 0.8))
        train_idx = list(range(split_at))
        val_idx = list(range(split_at, len(x)))
    if not train_idx:
        train_idx = list(range(len(x)))

    tuned_model_threshold: float | None = None
    per_user_eval: dict[str, dict[str, float]] = {}
    if train_xgboost:
        val_probs: list[float] = []
        val_labels: list[int] = []
        for idx in val_idx:
            feature_map = {name: x[idx][j] for j, name in enumerate(FEATURE_ORDER)}
            val_probs.append(
                predict_model_probability(
                    feature_map,
                    model_path=output_path,
                    use_calibration=False,
                )
            )
            val_labels.append(int(y[idx]))

        calibrator = fit_probability_calibrator(val_probs, val_labels)
        save_probability_calibrator(calibrator, path=DEFAULT_CALIBRATOR_PATH)

        calibrated_probs = [
            float(apply_probability_calibrator(p, calibrator_path=DEFAULT_CALIBRATOR_PATH))
            for p in val_probs
        ]
        tuned_model_threshold = tune_model_threshold(
            calibrated_probs,
            val_labels,
            target_precision=max(0.5, min(0.99, float(target_precision))),
        )
        eval_threshold = float(tuned_model_threshold or 0.5)
        # Internal-only per-user validation reporting (not shown in UI).
        per_user_probs: dict[int, list[float]] = {}
        per_user_labels: dict[int, list[int]] = {}
        for offset, idx in enumerate(val_idx):
            if idx >= len(groups):
                continue
            user_group = int(groups[idx])
            if user_group <= 0:
                continue
            per_user_probs.setdefault(user_group, []).append(float(calibrated_probs[offset]))
            per_user_labels.setdefault(user_group, []).append(int(val_labels[offset]))
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

        save_decision_thresholds(
            min_evidence_strength=0.2,
            min_model_probability=tuned_model_threshold,
            min_overall_confidence=max(0.45, tuned_model_threshold - 0.05),
            target_precision=max(0.5, min(0.99, float(target_precision))),
            source=str(dataset_source),
            path=DEFAULT_THRESHOLDS_PATH,
        )
        report_payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_source": str(dataset_source),
            "threshold_used": eval_threshold,
            "users_in_validation": len(per_user_eval),
            "per_user": per_user_eval,
        }
        DEFAULT_USER_EVAL_REPORT_PATH.write_text(json.dumps(report_payload, indent=2))

    fusion_rows: list[dict[str, float]] = []
    for idx, feature_values in enumerate(x):
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
        "window_hours": int(window_hours),
        "controls_per_case": int(controls_per_case),
        "backend": model_metadata(output_path)["backend"],
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
        choices=("curated", "insights", "feedback", "timeline_case_control", "hybrid"),
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
