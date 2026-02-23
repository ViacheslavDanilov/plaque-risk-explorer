from pathlib import Path
from typing import Literal, TypedDict

import pandas as pd
from autogluon.tabular import TabularPredictor
from pandas.api.types import is_numeric_dtype

from ml.train import FEATURES

_LABEL = "adverse_outcome"
_BOOLEAN_FEATURES = {
    "post_infarction_cardiosclerosis",
    "multifocal_atherosclerosis",
    "diabetes_mellitus",
    "hypertension",
    "unstable_plaque",
}
_DEFAULT_REFERENCE_PROFILE = {
    "gender": "male",
    "age": 62,
    "angina_functional_class": 2,
    "post_infarction_cardiosclerosis": False,
    "multifocal_atherosclerosis": False,
    "diabetes_mellitus": False,
    "hypertension": True,
    "cholesterol_level": 5.2,
    "bmi": 28.0,
    "lvef_percent": 51.0,
    "syntax_score": 18.0,
    "ffr": 0.83,
    "plaque_volume_percent": 60.0,
    "lumen_area": 5.0,
    "unstable_plaque": False,
}

SerializedValue = str | float | int | bool | None
ExplanationDirection = Literal["increase", "decrease", "neutral"]


class FeatureEffect(TypedDict):
    feature: str
    effect: float
    direction: ExplanationDirection
    patient_value: SerializedValue
    reference_value: SerializedValue


class ExplainabilityResult(TypedDict):
    method: Literal["counterfactual_single_feature_delta"]
    baseline_probability: float
    feature_effects: list[FeatureEffect]


def _is_missing(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def _to_serialized_value(value: object) -> SerializedValue:
    if value is None or _is_missing(value):
        return None

    if hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError:
            pass

    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _coerce_bool(value: object) -> bool | None:
    if value is None or _is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return None


def _build_reference_profile(baseline_df: pd.DataFrame) -> dict[str, object]:
    reference_profile: dict[str, object] = {}

    for feature in FEATURES:
        fallback_value = _DEFAULT_REFERENCE_PROFILE[feature]
        if feature not in baseline_df.columns:
            reference_profile[feature] = fallback_value
            continue

        series = baseline_df[feature].dropna()
        if series.empty:
            reference_profile[feature] = fallback_value
            continue

        if feature in _BOOLEAN_FEATURES:
            mode = series.mode(dropna=True)
            candidate = mode.iloc[0] if not mode.empty else series.iloc[0]
            bool_value = _coerce_bool(candidate)
            reference_profile[feature] = (
                fallback_value if bool_value is None else bool_value
            )
            continue

        if is_numeric_dtype(series):
            median = float(series.median())
            if feature in {"age", "angina_functional_class"}:
                reference_profile[feature] = int(round(median))
            elif feature == "ffr":
                reference_profile[feature] = round(median, 2)
            else:
                reference_profile[feature] = round(median, 3)
            continue

        mode = series.mode(dropna=True)
        reference_profile[feature] = mode.iloc[0] if not mode.empty else series.iloc[0]

    return reference_profile


def _positive_class_probability(
    predictor: TabularPredictor,
    row: dict[str, object],
) -> float:
    frame = pd.DataFrame([row], columns=FEATURES)
    predictions = predictor.predict_proba(frame)

    if isinstance(predictions, pd.Series):
        return float(predictions.iloc[0])

    for candidate in (1, "1", True):
        if candidate in predictions.columns:
            return float(predictions[candidate].iloc[0])

    if len(predictions.columns) < 2:
        raise RuntimeError("predict_proba did not return a positive-class column.")
    return float(predictions.iloc[0, 1])


def load_predictor(model_dir: Path) -> tuple[TabularPredictor, dict[str, object]]:
    """Load predictor and build a baseline reference profile used for explanations."""
    target_dir = model_dir / _LABEL
    predictor = TabularPredictor.load(str(target_dir))

    baseline_csv = target_dir / "baseline.csv"
    if baseline_csv.exists():
        baseline_df = pd.read_csv(baseline_csv)
    else:
        baseline_df = pd.DataFrame([_DEFAULT_REFERENCE_PROFILE])

    reference_profile = _build_reference_profile(baseline_df)
    return predictor, reference_profile


def predict(
    predictor: TabularPredictor,
    reference_profile: dict[str, object],
    features: dict[str, object],
) -> tuple[float, int, ExplainabilityResult]:
    """Return probability, class prediction, and local per-feature counterfactual effects."""
    patient_profile: dict[str, object] = {
        feature: features.get(feature) for feature in FEATURES
    }

    probability = _positive_class_probability(predictor, patient_profile)
    binary_prediction = int(probability >= 0.5)
    baseline_probability = _positive_class_probability(predictor, reference_profile)

    feature_effects: list[FeatureEffect] = []
    for feature in FEATURES:
        counterfactual_profile = dict(patient_profile)
        counterfactual_profile[feature] = reference_profile.get(feature)

        counterfactual_probability = _positive_class_probability(
            predictor,
            counterfactual_profile,
        )
        effect = probability - counterfactual_probability

        direction: ExplanationDirection = "neutral"
        if effect > 1e-9:
            direction = "increase"
        elif effect < -1e-9:
            direction = "decrease"

        feature_effects.append(
            {
                "feature": feature,
                "effect": round(effect, 4),
                "direction": direction,
                "patient_value": _to_serialized_value(patient_profile.get(feature)),
                "reference_value": _to_serialized_value(reference_profile.get(feature)),
            },
        )

    feature_effects.sort(key=lambda item: abs(item["effect"]), reverse=True)

    explanation: ExplainabilityResult = {
        "method": "counterfactual_single_feature_delta",
        "baseline_probability": round(baseline_probability, 3),
        "feature_effects": feature_effects,
    }

    return round(probability, 3), binary_prediction, explanation
