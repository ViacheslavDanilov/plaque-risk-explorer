from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor

from ml.train import FEATURES

_LABEL = "adverse_outcome"


def load_predictor(model_dir: Path) -> TabularPredictor:
    """Load the trained adverse_outcome predictor from model_dir."""
    return TabularPredictor.load(str(model_dir / _LABEL))


def predict(predictor: TabularPredictor, features: dict) -> tuple[float, int]:
    """Return (probability, binary prediction) for adverse_outcome.

    Probability is the model's confidence that the outcome is positive (class 1).
    """
    df = pd.DataFrame([features])[FEATURES]
    proba = float(predictor.predict_proba(df).iloc[0][1])
    return round(proba, 3), int(proba >= 0.5)
