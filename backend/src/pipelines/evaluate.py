import argparse
from pathlib import Path

from ml.eval import evaluate_model

# Resolve paths relative to the backend package root, regardless of cwd.
_BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the adverse outcome model.")
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
        help="Directory where the trained model is saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    evaluate_model(features_csv=args.features_csv, model_dir=args.model_dir)


if __name__ == "__main__":
    main()
