from pathlib import Path

import pandas as pd


def train_adverse_outcome_model(features_csv: Path, model_path: Path) -> None:
    """Placeholder: train and persist adverse outcome model."""
    df = pd.read_csv(features_csv)

    # Placeholder only: replace with real feature selection and fitting.
    rows, cols = df.shape

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(
        f"placeholder_model\nrows={rows}\ncols={cols}\n",
        encoding="utf-8",
    )
    print(f"Saved placeholder model to {model_path}")
