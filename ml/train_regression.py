from __future__ import annotations

import argparse

from ml.training_pipeline import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fusion regression model only.")
    parser.add_argument(
        "--dataset-source",
        choices=("curated", "insights", "feedback", "timeline_case_control", "hybrid"),
        default="hybrid",
        help="Training dataset source",
    )
    parser.add_argument(
        "--curated-path",
        type=str,
        default="data/models/curated_linkages.json",
        help="Path to curated linkage JSON catalog",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Path to existing model artifact JSON used to generate model_probability",
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
            dataset_source=args.dataset_source,
            curated_path=args.curated_path,
            output=args.output,
            window_hours=args.window_hours,
            controls_per_case=args.controls_per_case,
            target_precision=args.target_precision,
            fusion_augment_factor=args.fusion_augment_factor,
            train_xgboost=False,
            train_fusion=True,
        )
    )


if __name__ == "__main__":
    main()
