from __future__ import annotations

import argparse
from pathlib import Path

from api.db import get_connection, initialize_database
from ml.evaluator import (
    DEFAULT_CALIBRATOR_PATH,
    DEFAULT_CURATED_TRAINING_PATH,
    DEFAULT_THRESHOLDS_PATH,
    FEATURE_ORDER,
    build_feature_vector,
    build_training_rows_from_curated_catalog,
    build_training_rows_from_insights,
    compute_penalty_score,
    default_model_path,
    fit_probability_calibrator,
    model_metadata,
    predict_model_probability,
    save_decision_thresholds,
    save_probability_calibrator,
    save_xgboost_artifact,
    train_xgboost_model,
    tune_model_threshold,
)
from ml.final_score import (
    compute_evidence_quality_from_features,
    save_fusion_artifacts,
    train_fusion_model,
)
from ml.training_data import build_case_control_training_rows
from sklearn.model_selection import GroupShuffleSplit


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
) -> dict:
    initialize_database()
    allowed_sources = {"curated", "insights", "timeline_case_control", "hybrid"}
    if dataset_source not in allowed_sources:
        raise ValueError(f"dataset_source must be one of {sorted(allowed_sources)}")

    x: list[list[float]] = []
    y: list[int] = []
    groups: list[int] = []

    def _append_rows(rows: list[list[float]], labels: list[int], group_id: int | None = None) -> None:
        x.extend(rows)
        y.extend(labels)
        if group_id is not None:
            groups.extend([group_id] * len(rows))

    if dataset_source in {"curated", "hybrid"}:
        curated_x, curated_y = build_training_rows_from_curated_catalog(Path(curated_path))
        _append_rows(curated_x, curated_y, group_id=-1)

    if dataset_source in {"insights", "hybrid"}:
        conn = get_connection()
        try:
            insights_x, insights_y = build_training_rows_from_insights(conn)
        finally:
            conn.close()
        _append_rows(insights_x, insights_y, group_id=-2)

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

    if len(x) < 10:
        raise RuntimeError("Not enough training rows. Run insight recomputation first.")
    if sum(y) == 0 or sum(y) == len(y):
        raise RuntimeError("Training labels are single-class; gather mixed supported/suppressed data first.")

    model = train_xgboost_model(
        x,
        y,
        rounds=max(1, int(rounds)),
        learning_rate=max(0.01, float(learning_rate)),
    )
    output_path = Path(output) if output else default_model_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_xgboost_artifact(model, path=output_path)

    # Calibration and threshold tuning on held-out split.
    if len(groups) == len(x) and len(set(groups)) > 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        _, val_idx = next(splitter.split(x, y, groups=groups))
    else:
        split_at = max(1, int(len(x) * 0.8))
        val_idx = list(range(split_at, len(x)))
        if not val_idx:
            val_idx = list(range(len(x)))

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

    calibrated_probs = [float(calibrator.predict([p])[0]) for p in val_probs]
    tuned_model_threshold = tune_model_threshold(
        calibrated_probs,
        val_labels,
        target_precision=max(0.5, min(0.99, float(target_precision))),
    )
    save_decision_thresholds(
        min_evidence_strength=0.2,
        min_model_probability=tuned_model_threshold,
        min_overall_confidence=max(0.45, tuned_model_threshold - 0.05),
        target_precision=max(0.5, min(0.99, float(target_precision))),
        source=str(dataset_source),
        path=DEFAULT_THRESHOLDS_PATH,
    )

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

    fusion_model, fusion_calibrator = train_fusion_model(
        rows=fusion_rows,
        labels=[int(v) for v in y],
    )
    save_fusion_artifacts(model=fusion_model, calibrator=fusion_calibrator)

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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train linkage evaluation gradient-boosted model.")
    parser.add_argument("--rounds", type=int, default=30, help="Boosting rounds")
    parser.add_argument("--learning-rate", type=float, default=0.2, help="Boosting learning rate")
    parser.add_argument(
        "--dataset-source",
        choices=("curated", "insights", "timeline_case_control", "hybrid"),
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
        )
    )


if __name__ == "__main__":
    main()
