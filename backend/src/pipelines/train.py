import warnings
from pathlib import Path

from ml.train import train_adverse_outcome_model, train_unstable_plaque_model

# Resolve paths relative to the backend package root, regardless of cwd.
_BACKEND_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    features_csv = _BACKEND_ROOT / "data" / "features.csv"
    model_dir = _BACKEND_ROOT / "models"

    warnings.warn(
        "Training on n=56 with only 5 positive cases per target. "
        "These models are exploratory only and must not be used for clinical decisions.",
        UserWarning,
        stacklevel=2,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    train_adverse_outcome_model(features_csv=features_csv, model_dir=model_dir)
    train_unstable_plaque_model(features_csv=features_csv, model_dir=model_dir)


if __name__ == "__main__":
    main()
