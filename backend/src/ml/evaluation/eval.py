from pathlib import Path

import pandas as pd


def evaluate_adverse_outcome_model(features_csv: Path, model_path: Path) -> None:
    """Placeholder: evaluate model and print simple metrics."""
    df = pd.read_csv(features_csv)
    model_exists = model_path.exists()

    # Placeholder metric values.
    print("Evaluation summary:")
    print(f"- rows: {len(df)}")
    print(f"- model_found: {model_exists}")
    print("- auc: 0.50 (placeholder)")
    print("- f1: 0.00 (placeholder)")
