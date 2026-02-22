from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor

FEATURES = [
    "gender",
    "age",
    "angina_functional_class",
    "post_infarction_cardiosclerosis",
    "multifocal_atherosclerosis",
    "diabetes_mellitus",
    "hypertension",
    "cholesterol_level",
    "bmi",
    "lvef_percent",
    "syntax_score",
    "ffr",
    "plaque_volume_percent",
    "lumen_area",
    "unstable_plaque",
]


def _fit_predictor(
    df: pd.DataFrame,
    model_dir: Path,
    time_limit: int = 120,
) -> TabularPredictor:
    """Fit a binary TabularPredictor for adverse_outcome and save it under model_dir."""
    label = "adverse_outcome"
    target_dir = model_dir / label
    predictor = TabularPredictor(
        label=label,
        path=str(target_dir),
        eval_metric="roc_auc",
        problem_type="binary",
    ).fit(
        train_data=df[FEATURES + [label]],
        presets="best_quality",
        time_limit=time_limit,
        num_bag_folds=5,
        num_bag_sets=1,
        # Stacking is unreliable at n=56; DyStack corrupts learner state on
        # small datasets, so disable both.
        dynamic_stacking=False,
        num_stack_levels=0,
        verbosity=2,
    )
    return predictor


def train_model(
    features_csv: Path,
    model_dir: Path,
    time_limit: int = 3600,
) -> TabularPredictor:
    """Train and persist the adverse_outcome binary classifier."""
    df = pd.read_csv(features_csv)
    predictor = _fit_predictor(df, model_dir=model_dir, time_limit=time_limit)
    print("\n--- adverse_outcome leaderboard ---")
    print(predictor.leaderboard(silent=True).to_string())
    return predictor
