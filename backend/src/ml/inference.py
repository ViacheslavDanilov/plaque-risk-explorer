import os
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor
from pandas.api.types import is_numeric_dtype

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

_LABEL = "adverse_outcome"
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
_INT_REFERENCE_FEATURES = {"age", "angina_functional_class"}


def _serialize(value: object) -> str | float | int | bool | None:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

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


def _extract_positive_proba(probabilities: pd.DataFrame | pd.Series) -> list[float]:
    if isinstance(probabilities, pd.Series):
        return [float(v) for v in probabilities]
    if 1 in probabilities.columns:
        return [float(v) for v in probabilities[1]]
    if "1" in probabilities.columns:
        return [float(v) for v in probabilities["1"]]
    if len(probabilities.columns) < 2:
        raise RuntimeError("predict_proba did not return a positive-class column.")
    return [float(v) for v in probabilities.iloc[:, 1]]


def _batch_predict_proba(
    predictor: TabularPredictor,
    rows: list[dict[str, object]],
) -> list[float]:
    frame = pd.DataFrame(rows, columns=FEATURES)
    probabilities = predictor.predict_proba(frame)
    return _extract_positive_proba(probabilities)


def _build_reference_profile(baseline_df: pd.DataFrame) -> dict[str, object]:
    reference: dict[str, object] = {}
    for feature in FEATURES:
        fallback = _DEFAULT_REFERENCE_PROFILE[feature]
        if feature not in baseline_df.columns:
            reference[feature] = fallback
            continue

        values = baseline_df[feature].dropna()
        if values.empty:
            reference[feature] = fallback
            continue

        if is_numeric_dtype(values):
            median = float(values.median())
            if feature in _INT_REFERENCE_FEATURES:
                reference[feature] = int(round(median))
            elif feature == "ffr":
                reference[feature] = round(median, 2)
            else:
                reference[feature] = round(median, 3)
            continue

        mode = values.mode(dropna=True)
        reference[feature] = mode.iloc[0] if not mode.empty else values.iloc[0]

    return reference


def load_predictor(model_dir: Path) -> tuple[TabularPredictor, dict[str, object]]:
    model_name = os.getenv("MODEL_NAME", "tabpfn_adasyn")
    target_dir = model_dir / model_name
    predictor = TabularPredictor.load(str(target_dir))

    baseline_csv = model_dir.parent / "data" / "features.csv"
    baseline_df = (
        pd.read_csv(baseline_csv)
        if baseline_csv.exists()
        else pd.DataFrame([_DEFAULT_REFERENCE_PROFILE])
    )
    return predictor, _build_reference_profile(baseline_df)


def predict(
    predictor: TabularPredictor,
    reference_profile: dict[str, object],
    features: dict[str, object],
) -> tuple[float, int, dict[str, object]]:
    patient_profile = {feature: features.get(feature) for feature in FEATURES}

    # Build all rows in one batch: patient, baseline, + 15 counterfactuals
    rows: list[dict[str, object]] = [patient_profile, dict(reference_profile)]
    for feature in FEATURES:
        counterfactual = dict(patient_profile)
        counterfactual[feature] = reference_profile.get(feature)
        rows.append(counterfactual)

    # Single predict_proba call for all 17 rows
    all_proba = _batch_predict_proba(predictor, rows)

    probability = all_proba[0]
    baseline_probability = all_proba[1]
    counterfactual_probas = all_proba[2:]

    scored_effects: list[tuple[float, dict[str, object]]] = []
    for i, feature in enumerate(FEATURES):
        effect = round(probability - counterfactual_probas[i], 4)

        direction = "neutral"
        if effect > 0:
            direction = "increase"
        elif effect < 0:
            direction = "decrease"

        scored_effects.append(
            (
                abs(effect),
                {
                    "feature": feature,
                    "effect": effect,
                    "direction": direction,
                    "patient_value": _serialize(patient_profile.get(feature)),
                    "reference_value": _serialize(reference_profile.get(feature)),
                },
            ),
        )

    feature_effects = [
        effect
        for _, effect in sorted(scored_effects, key=lambda item: item[0], reverse=True)
    ]
    explanation = {
        "method": "counterfactual_single_feature_delta",
        "baseline_probability": round(baseline_probability, 3),
        "feature_effects": feature_effects,
    }
    return round(probability, 3), int(probability >= 0.5), explanation
