from __future__ import annotations

import argparse

from ml.training.training_pipeline import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost linkage model only.")
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
        default="data/models/curated_linkages.json",
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
            train_xgboost=True,
            train_fusion=False,
        )
    )


if __name__ == "__main__":
    main()
