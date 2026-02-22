from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor

CLINICAL_FEATURES = [
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
]

_SMALL_DATASET_HYPERPARAMETERS = {
    "GBM": [
        {
            "num_leaves": 8,
            "min_child_samples": 3,
            "ag_args": {"name_suffix": "Shallow"},
        },
    ],
    "LR": {},
}


def _fit_predictor(
    df: pd.DataFrame,
    label: str,
    model_dir: Path,
    time_limit: int = 120,
) -> TabularPredictor:
    """Fit a binary TabularPredictor for the given label and save it under model_dir/label."""
    target_dir = model_dir / label
    predictor = TabularPredictor(
        label=label,
        path=str(target_dir),
        eval_metric="roc_auc",
        problem_type="binary",
    ).fit(
        train_data=df[CLINICAL_FEATURES + [label]],
        presets="best_quality",
        time_limit=time_limit,
        num_bag_folds=5,
        num_bag_sets=1,
        # Stacking is unreliable at n=56; DyStack corrupts learner state on
        # small datasets, so disable both.
        dynamic_stacking=False,
        num_stack_levels=0,
        hyperparameters=_SMALL_DATASET_HYPERPARAMETERS,
        verbosity=2,
    )
    return predictor


def train_adverse_outcome_model(
    features_csv: Path,
    model_dir: Path,
    time_limit: int = 120,
) -> TabularPredictor:
    """Train and persist the adverse_outcome binary classifier."""
    df = pd.read_csv(features_csv)
    predictor = _fit_predictor(
        df,
        label="adverse_outcome",
        model_dir=model_dir,
        time_limit=time_limit,
    )
    print("\n--- adverse_outcome leaderboard ---")
    print(predictor.leaderboard(silent=True).to_string())
    return predictor


def train_unstable_plaque_model(
    features_csv: Path,
    model_dir: Path,
    time_limit: int = 120,
) -> TabularPredictor:
    """Train and persist the unstable_plaque binary classifier."""
    df = pd.read_csv(features_csv)
    predictor = _fit_predictor(
        df,
        label="unstable_plaque",
        model_dir=model_dir,
        time_limit=time_limit,
    )
    print("\n--- unstable_plaque leaderboard ---")
    print(predictor.leaderboard(silent=True).to_string())
    return predictor
