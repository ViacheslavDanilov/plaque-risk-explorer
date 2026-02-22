from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor

from ml.train import FEATURES


def evaluate_model(features_csv: Path, model_dir: Path) -> None:
    """Load the persisted adverse_outcome model and print its leaderboard."""
    label = "adverse_outcome"
    df = pd.read_csv(features_csv)
    predictor = TabularPredictor.load(str(model_dir / label))
    leaderboard = predictor.leaderboard(df[FEATURES + [label]], silent=True)
    print(f"\n--- {label} leaderboard (roc_auc) ---")
    print(leaderboard.to_string())
