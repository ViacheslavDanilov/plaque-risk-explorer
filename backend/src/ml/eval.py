from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor

from ml.train import CLINICAL_FEATURES


def _evaluate_predictor(df: pd.DataFrame, label: str, model_dir: Path) -> None:
    """Load a saved predictor and print its leaderboard with CV roc_auc scores."""
    target_dir = model_dir / label
    predictor = TabularPredictor.load(str(target_dir))
    leaderboard = predictor.leaderboard(df[CLINICAL_FEATURES + [label]], silent=True)
    print(f"\n--- {label} leaderboard (roc_auc) ---")
    print(leaderboard.to_string())


def evaluate_adverse_outcome_model(features_csv: Path, model_dir: Path) -> None:
    """Evaluate the persisted adverse_outcome model."""
    df = pd.read_csv(features_csv)
    _evaluate_predictor(df, label="adverse_outcome", model_dir=model_dir)


def evaluate_unstable_plaque_model(features_csv: Path, model_dir: Path) -> None:
    """Evaluate the persisted unstable_plaque model."""
    df = pd.read_csv(features_csv)
    _evaluate_predictor(df, label="unstable_plaque", model_dir=model_dir)


def evaluate_all_models(features_csv: Path, model_dir: Path) -> None:
    """Evaluate both persisted models and print their leaderboards."""
    evaluate_adverse_outcome_model(features_csv=features_csv, model_dir=model_dir)
    evaluate_unstable_plaque_model(features_csv=features_csv, model_dir=model_dir)
