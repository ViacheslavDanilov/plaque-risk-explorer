from pathlib import Path

from ml.train import train_all_models

# Resolve paths relative to the backend package root, regardless of cwd.
_BACKEND_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    train_all_models(
        features_csv=_BACKEND_ROOT / "data" / "features.csv",
        model_dir=_BACKEND_ROOT / "models",
    )


if __name__ == "__main__":
    main()
