from __future__ import annotations

import json
import logging
import os
from typing import Any, Literal, TypedDict
from urllib import error, request

RiskTier = Literal["low", "moderate", "high"]
SummarySource = Literal["gemini", "fallback"]
SerializedValue = str | float | int | bool | None


class FeatureEffectRow(TypedDict):
    feature: str
    effect: float
    patient_value: SerializedValue
    reference_value: SerializedValue


class ExecutiveSummaryData(TypedDict):
    headline: str
    clinical_summary: str
    risk_drivers: list[str]
    protective_signals: list[str]
    care_focus: list[str]


class ExecutiveSummaryPayload(ExecutiveSummaryData):
    source: SummarySource


LOGGER = logging.getLogger(__name__)

_GEMINI_API_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
_DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"

_FEATURE_LABELS = {
    "gender": "Gender",
    "age": "Age",
    "angina_functional_class": "Angina Class",
    "post_infarction_cardiosclerosis": "Post-MI Cardiosclerosis",
    "multifocal_atherosclerosis": "Multifocal Atherosclerosis",
    "diabetes_mellitus": "Diabetes Mellitus",
    "hypertension": "Hypertension",
    "cholesterol_level": "Cholesterol",
    "bmi": "BMI",
    "lvef_percent": "LVEF",
    "syntax_score": "SYNTAX Score",
    "ffr": "FFR",
    "plaque_volume_percent": "Plaque Volume",
    "lumen_area": "Lumen Area",
    "unstable_plaque": "Unstable Plaque",
    "other_factors": "Other Factors",
}


def _safe_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _humanize_feature(feature: str) -> str:
    return _FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


def _format_value(value: object) -> str:
    if value is None:
        return "missing"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.2f}"
    return str(value)


def _format_effect(effect: float) -> str:
    sign = "+" if effect >= 0 else "-"
    return f"{sign}{abs(effect) * 100:.1f}%"


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("No JSON object found in Gemini response.")
    return json.loads(cleaned[start : end + 1])


def _extract_feature_effects(explanation: dict[str, object]) -> list[FeatureEffectRow]:
    raw_effects = explanation.get("feature_effects")
    if not isinstance(raw_effects, list):
        return []

    normalized_effects: list[FeatureEffectRow] = []
    for item in raw_effects:
        if not isinstance(item, dict):
            continue
        effect = item.get("effect")
        if isinstance(effect, bool) or not isinstance(effect, (float, int)):
            continue
        feature = item.get("feature")
        if not isinstance(feature, str) or not feature:
            continue
        normalized_effects.append(
            {
                "feature": feature,
                "effect": float(effect),
                "patient_value": item.get("patient_value"),
                "reference_value": item.get("reference_value"),
            },
        )

    normalized_effects.sort(
        key=lambda row: abs(row["effect"]),
        reverse=True,
    )
    return normalized_effects


def _top_effects(
    feature_effects: list[FeatureEffectRow],
    *,
    direction: Literal["increase", "decrease"],
    limit: int = 3,
) -> list[FeatureEffectRow]:
    if direction == "increase":
        selected = [row for row in feature_effects if row["effect"] > 0.0001]
    else:
        selected = [row for row in feature_effects if row["effect"] < -0.0001]
    return selected[:limit]


def _fallback_driver_lines(
    effects: list[FeatureEffectRow],
    *,
    rising_risk: bool,
) -> list[str]:
    lines: list[str] = []
    for row in effects:
        feature = _humanize_feature(row["feature"])
        effect = row["effect"]
        patient_value = _format_value(row["patient_value"])
        reference_value = _format_value(row["reference_value"])
        trend = "raises" if rising_risk else "reduces"
        lines.append(
            f"{feature} ({patient_value}, baseline {reference_value}) {trend} risk by "
            f"{_format_effect(effect)}.",
        )
    return lines


def _follow_up_line(risk_tier: RiskTier) -> str:
    if risk_tier == "high":
        return "Arrange close cardiology follow-up in 2-4 weeks with interval reassessment."
    if risk_tier == "moderate":
        return "Plan structured follow-up in 4-8 weeks and reassess risk trajectory."
    return "Continue routine surveillance and reinforce symptom-triggered early review."


def _feature_specific_focus(
    feature: str,
    patient_features: dict[str, object],
) -> str | None:
    if feature == "cholesterol_level":
        value = patient_features.get("cholesterol_level")
        if isinstance(value, (int, float)):
            return (
                f"Reassess lipid-lowering intensity and dietary adherence "
                f"(cholesterol {float(value):.2f} mmol/L)."
            )
        return "Reassess lipid-lowering intensity and dietary adherence."
    if feature == "lvef_percent":
        value = patient_features.get("lvef_percent")
        if isinstance(value, (int, float)):
            return (
                f"Review ventricular function strategy and optimize therapy "
                f"(LVEF {float(value):.1f}%)."
            )
        return "Review ventricular function strategy and optimize therapy."
    if feature == "syntax_score":
        return "Discuss coronary complexity findings in multidisciplinary review."
    if feature == "ffr":
        return "Reevaluate ischemic burden and whether additional coronary assessment is needed."
    if feature == "plaque_volume_percent" or feature == "unstable_plaque":
        return "Prioritize plaque-stabilizing management and close ischemic symptom monitoring."
    if feature == "lumen_area":
        return "Monitor for progressive luminal compromise and reassess imaging if symptoms change."
    if feature == "diabetes_mellitus":
        return "Intensify glycemic risk-factor control with coordinated cardiometabolic follow-up."
    if feature == "hypertension":
        return "Tighten blood-pressure control with home BP trend review."
    if feature == "multifocal_atherosclerosis":
        return "Address systemic atherosclerotic burden with comprehensive secondary prevention."
    if feature == "post_infarction_cardiosclerosis":
        return "Review post-infarction remodeling management and adherence to cardioprotective therapy."
    if feature == "age":
        return "Individualize follow-up cadence for age-associated event risk."
    if feature == "angina_functional_class":
        return (
            "Track symptom burden and consider escalation if functional class worsens."
        )
    if feature == "bmi":
        return "Set a weight-management plan to lower long-term cardiometabolic risk."
    return None


def _fallback_care_focus(
    risk_tier: RiskTier,
    top_risk_drivers: list[FeatureEffectRow],
    patient_features: dict[str, object],
) -> list[str]:
    items: list[str] = []
    for row in top_risk_drivers:
        action = _feature_specific_focus(row["feature"], patient_features)
        if action and action not in items:
            items.append(action)
        if len(items) >= 3:
            return items

    base_lines = [
        _follow_up_line(risk_tier),
        "Optimize adherence to guideline-directed preventive therapy and lifestyle changes.",
        "Educate on warning symptoms requiring urgent clinical reassessment.",
    ]
    for line in base_lines:
        if line not in items:
            items.append(line)
        if len(items) >= 3:
            break
    return items


def _fallback_summary(
    probability: float,
    risk_tier: RiskTier,
    baseline_probability: float,
    top_risk_drivers: list[FeatureEffectRow],
    top_protective_signals: list[FeatureEffectRow],
    patient_features: dict[str, object],
) -> ExecutiveSummaryData:
    probability_pct = probability * 100
    baseline_pct = baseline_probability * 100
    delta = probability - baseline_probability
    if delta > 0.03:
        relation = "above"
    elif delta < -0.03:
        relation = "below"
    else:
        relation = "near"

    headline = (
        f"{risk_tier.title()} estimated adverse-outcome risk ({probability_pct:.0f}%)."
    )
    clinical_summary = (
        f"The model estimates a {probability_pct:.1f}% probability of adverse cardiovascular "
        f"outcomes, which is {relation} the cohort baseline of {baseline_pct:.1f}%. "
        f"Interpret this output with clinical judgment and follow-up context."
    )

    risk_drivers = _fallback_driver_lines(top_risk_drivers, rising_risk=True)
    if not risk_drivers:
        risk_drivers = [
            "No single feature produced a dominant upward risk shift in this profile.",
        ]
    risk_drivers = _normalize_summary_list(
        risk_drivers,
        [
            "Risk appears distributed across multiple smaller feature effects.",
            "Review lower-ranked feature effects for additional contributing context.",
        ],
    )

    protective_signals = _fallback_driver_lines(
        top_protective_signals,
        rising_risk=False,
    )
    if not protective_signals:
        protective_signals = [
            "No strong protective feature effect was detected versus baseline.",
        ]
    protective_signals = _normalize_summary_list(
        protective_signals,
        [
            "Protective effects are limited and do not offset major risk drivers.",
            "Potential protective signals should be interpreted with the full clinical picture.",
        ],
    )

    care_focus = _fallback_care_focus(risk_tier, top_risk_drivers, patient_features)
    care_focus = _normalize_summary_list(
        care_focus,
        [
            "Continue risk-factor surveillance with coordinated outpatient follow-up.",
            "Align plan updates with symptom changes and interval testing.",
        ],
    )
    return {
        "headline": headline,
        "clinical_summary": clinical_summary,
        "risk_drivers": risk_drivers,
        "protective_signals": protective_signals,
        "care_focus": care_focus,
    }


def _normalize_summary_list(
    value: object,
    fallback: list[str],
    *,
    target_size: int = 3,
) -> list[str]:
    normalized: list[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned and cleaned not in normalized:
                    normalized.append(cleaned)
    for fallback_item in fallback:
        if len(normalized) >= target_size:
            break
        if fallback_item not in normalized:
            normalized.append(fallback_item)
    return normalized[:target_size]


def _normalize_summary(
    raw_summary: dict[str, Any],
    fallback: ExecutiveSummaryData,
) -> ExecutiveSummaryData:
    headline = raw_summary.get("headline")
    clinical_summary = raw_summary.get("clinical_summary")

    normalized_headline = headline.strip() if isinstance(headline, str) else ""
    normalized_clinical = (
        clinical_summary.strip() if isinstance(clinical_summary, str) else ""
    )

    if not normalized_headline:
        normalized_headline = str(fallback["headline"])
    if not normalized_clinical:
        normalized_clinical = str(fallback["clinical_summary"])

    return {
        "headline": normalized_headline,
        "clinical_summary": normalized_clinical,
        "risk_drivers": _normalize_summary_list(
            raw_summary.get("risk_drivers"),
            fallback["risk_drivers"],
        ),
        "protective_signals": _normalize_summary_list(
            raw_summary.get("protective_signals"),
            fallback["protective_signals"],
        ),
        "care_focus": _normalize_summary_list(
            raw_summary.get("care_focus"),
            fallback["care_focus"],
        ),
    }


def _build_prompt(
    patient_features: dict[str, object],
    probability: float,
    risk_tier: RiskTier,
    baseline_probability: float,
    top_risk_drivers: list[FeatureEffectRow],
    top_protective_signals: list[FeatureEffectRow],
) -> str:
    feature_lines = [
        f"- {_humanize_feature(feature)}: {_format_value(value)}"
        for feature, value in patient_features.items()
    ]
    driver_lines = [
        (
            f"- {_humanize_feature(row['feature'])}: {_format_effect(row['effect'])} "
            f"(patient {_format_value(row['patient_value'])}, "
            f"baseline {_format_value(row['reference_value'])})"
        )
        for row in top_risk_drivers
    ]
    protective_lines = [
        (
            f"- {_humanize_feature(row['feature'])}: {_format_effect(row['effect'])} "
            f"(patient {_format_value(row['patient_value'])}, "
            f"baseline {_format_value(row['reference_value'])})"
        )
        for row in top_protective_signals
    ]

    if not driver_lines:
        driver_lines = ["- No dominant risk-increasing features detected."]
    if not protective_lines:
        protective_lines = ["- No dominant protective features detected."]

    return "\n".join(
        [
            "You are a cardiology decision-support assistant.",
            "Generate a patient-specific executive summary from this model output.",
            "",
            "Return JSON only with exactly these keys:",
            "headline (string), clinical_summary (string), risk_drivers (array of 3 strings),",
            "protective_signals (array of 3 strings), care_focus (array of 3 strings).",
            "",
            "Constraints:",
            "- Keep statements concise, clinically professional, and patient-communicable.",
            "- Use only the data provided below.",
            "- Do not mention being an AI model.",
            "- Do not add markdown.",
            "",
            "Patient profile:",
            *feature_lines,
            "",
            f"Predicted adverse outcome probability: {probability * 100:.1f}%",
            f"Risk tier: {risk_tier}",
            f"Cohort baseline probability: {baseline_probability * 100:.1f}%",
            "",
            "Top risk-increasing effects:",
            *driver_lines,
            "",
            "Top risk-reducing effects:",
            *protective_lines,
        ],
    )


def _gemini_generate_json(
    *,
    prompt: str,
    api_key: str,
    model: str,
    temperature: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    url = _GEMINI_API_URL_TEMPLATE.format(model=model)
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
        },
    }
    req = request.Request(
        f"{url}?key={api_key}",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    with request.urlopen(req, timeout=timeout_seconds) as response:
        body = response.read().decode("utf-8")

    parsed = json.loads(body)
    candidates = parsed.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("Gemini response did not include candidates.")

    first_candidate = candidates[0]
    if not isinstance(first_candidate, dict):
        raise RuntimeError("Unexpected Gemini candidate payload.")

    content = first_candidate.get("content")
    if not isinstance(content, dict):
        raise RuntimeError("Gemini response did not include content.")

    parts = content.get("parts")
    if not isinstance(parts, list):
        raise RuntimeError("Gemini response did not include content parts.")

    text_chunks: list[str] = []
    for part in parts:
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str):
                text_chunks.append(text)
    if not text_chunks:
        raise RuntimeError("Gemini response did not include text output.")

    return _extract_json_object("\n".join(text_chunks))


def generate_executive_summary(
    *,
    patient_features: dict[str, object],
    probability: float,
    risk_tier: RiskTier,
    explanation: dict[str, object],
) -> ExecutiveSummaryPayload:
    baseline_raw = explanation.get("baseline_probability")
    baseline_probability = (
        float(baseline_raw)
        if isinstance(baseline_raw, (float, int)) and not isinstance(baseline_raw, bool)
        else probability
    )

    feature_effects = _extract_feature_effects(explanation)
    top_risk_drivers = _top_effects(feature_effects, direction="increase", limit=3)
    top_protective_signals = _top_effects(
        feature_effects,
        direction="decrease",
        limit=3,
    )

    fallback = _fallback_summary(
        probability,
        risk_tier,
        baseline_probability,
        top_risk_drivers,
        top_protective_signals,
        patient_features,
    )

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return {**fallback, "source": "fallback"}

    model = (
        os.getenv("GEMINI_MODEL", _DEFAULT_GEMINI_MODEL).strip()
        or _DEFAULT_GEMINI_MODEL
    )
    temperature = _safe_float(os.getenv("GEMINI_TEMPERATURE", "0.2"), 0.2)
    timeout_seconds = _safe_float(os.getenv("GEMINI_TIMEOUT_SECONDS", "30"), 30.0)

    prompt = _build_prompt(
        patient_features,
        probability,
        risk_tier,
        baseline_probability,
        top_risk_drivers,
        top_protective_signals,
    )

    try:
        generated = _gemini_generate_json(
            prompt=prompt,
            api_key=api_key,
            model=model,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )
        return {**_normalize_summary(generated, fallback), "source": "gemini"}
    except (
        error.HTTPError,
        error.URLError,
        TimeoutError,
        RuntimeError,
        json.JSONDecodeError,
    ) as exc:
        LOGGER.warning("Gemini summary generation failed: %s", exc)
    except Exception as exc:  # pragma: no cover - defensive fallback
        LOGGER.exception("Unexpected Gemini summary generation failure: %s", exc)

    return {**fallback, "source": "fallback"}
