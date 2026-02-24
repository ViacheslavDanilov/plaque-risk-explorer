import argparse
from pathlib import Path

from ml.train import train_model

# Resolve paths relative to the backend package root, regardless of cwd.
_BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the adverse outcome model.")
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=_BACKEND_ROOT / "data" / "features.csv",
        metavar="PATH",
        help="Path to the features CSV file.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=_BACKEND_ROOT / "models",
        metavar="PATH",
        help="Directory where the trained model will be saved.",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=3600,
        metavar="SECONDS",
        help="AutoGluon time budget in seconds (default: 120). "
        "Use 3600+ for highest accuracy on a server.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    train_model(
        features_csv=args.features_csv,
        model_dir=args.model_dir,
        time_limit=args.time_limit,
    )


if __name__ == "__main__":
    main()
