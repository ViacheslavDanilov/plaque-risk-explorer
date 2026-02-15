from pathlib import Path

import pandas as pd


def build_features(input_csv: Path, output_csv: Path) -> None:
    """Placeholder: transform source data into model-ready features."""
    df = pd.read_csv(input_csv)

    # Example placeholder transformation.
    df = df.copy()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved features to {output_csv}")
