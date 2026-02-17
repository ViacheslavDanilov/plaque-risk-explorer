import os
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Plaque Risk Explorer",
    description="Association of Clinical Factors and Plaque Morphology with Adverse Cardiovascular Outcomes",
    version="0.1.0",
)

default_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
cors_origins_env = os.getenv("CORS_ORIGINS", "")
env_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(dict.fromkeys(default_origins + env_origins)),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    age: int = Field(62, ge=30, le=95)
    gender: Literal["female", "male"] = "male"
    angina_functional_class: Literal[0, 1, 2, 3] = 2
    post_infarction_cardiosclerosis: bool = False
    multifocal_atherosclerosis: bool = False
    diabetes_mellitus: bool = False
    hypertension: bool = True
    cholesterol_level: float = Field(5.2, ge=2.0, le=12.0)
    bmi: float = Field(28.0, ge=15.0, le=60.0)
    lvef_percent: float = Field(51.0, ge=20.0, le=80.0)
    ffr: float | None = Field(default=0.83, ge=0.4, le=1.0)
    syntax_score: float = Field(18.0, ge=0.0, le=60.0)


class BinaryTargetPrediction(BaseModel):
    probability: float
    prediction: int
    risk_tier: Literal["low", "moderate", "high"]


class PredictionResponse(BaseModel):
    unstable_plaque: BinaryTargetPrediction
    adverse_outcome: BinaryTargetPrediction
    plaque_volume_percent: float
    lumen_area: float
    recommendations: list[str]


def _score_mock_risk(payload: PredictionRequest) -> PredictionResponse:
    ffr_for_scoring = payload.ffr if payload.ffr is not None else 0.89

    unstable = 0.08
    unstable += 0.002 * max(payload.age - 55, 0)
    unstable += 0.007 * max(payload.syntax_score - 8, 0)
    unstable += 0.28 * max(0.88 - ffr_for_scoring, 0)
    unstable += 0.004 * max(payload.cholesterol_level - 5.0, 0)
    unstable += 0.003 * max(55 - payload.lvef_percent, 0)

    if payload.diabetes_mellitus:
        unstable += 0.08
    if payload.hypertension:
        unstable += 0.05
    if payload.gender == "male":
        unstable += 0.03
    if payload.angina_functional_class >= 2:
        unstable += 0.07
    if payload.multifocal_atherosclerosis:
        unstable += 0.05

    unstable_probability = round(min(max(unstable, 0.02), 0.96), 3)

    adverse = 0.03
    adverse += 0.33 * unstable_probability
    adverse += 0.004 * max(payload.syntax_score - 12, 0)
    adverse += 0.003 * max(payload.age - 65, 0)
    adverse += 0.002 * max(55 - payload.lvef_percent, 0)
    if payload.post_infarction_cardiosclerosis:
        adverse += 0.08
    if payload.diabetes_mellitus:
        adverse += 0.07
    if payload.multifocal_atherosclerosis:
        adverse += 0.05
    if payload.ffr is None:
        adverse += 0.02

    adverse_probability = round(min(max(adverse, 0.01), 0.95), 3)
    if adverse_probability >= 0.65:
        risk_tier: Literal["low", "moderate", "high"] = "high"
        recommendations = [
            "Discuss close cardiology follow-up within 2-4 weeks.",
            "Prioritize lipid, blood pressure, and glycemic optimization.",
            "Review indications for additional imaging or invasive reassessment.",
        ]
    elif adverse_probability >= 0.35:
        risk_tier = "moderate"
        recommendations = [
            "Schedule structured outpatient follow-up.",
            "Optimize modifiable risk factors and medication adherence.",
            "Repeat clinical reassessment in 6-8 weeks.",
        ]
    else:
        risk_tier = "low"
        recommendations = [
            "Continue preventive therapy and risk-factor control.",
            "Maintain routine follow-up and symptom monitoring.",
            "Escalate evaluation if symptoms worsen.",
        ]

    if unstable_probability >= 0.65:
        unstable_risk_tier = "high"
    elif unstable_probability >= 0.35:
        unstable_risk_tier = "moderate"
    else:
        unstable_risk_tier = "low"

    plaque_volume = 55.0
    plaque_volume += 11.0 * unstable_probability
    plaque_volume += 0.15 * max(payload.syntax_score - 8, 0)
    plaque_volume -= 0.08 * max(ffr_for_scoring - 0.8, 0) * 100
    plaque_volume = round(min(max(plaque_volume, 35.0), 95.0), 1)

    lumen_area = 5.9
    lumen_area -= 2.7 * unstable_probability
    lumen_area -= 0.03 * max(payload.syntax_score - 8, 0)
    lumen_area += 0.9 * max(ffr_for_scoring - 0.85, 0)
    lumen_area = round(min(max(lumen_area, 1.0), 9.0), 2)

    return PredictionResponse(
        unstable_plaque=BinaryTargetPrediction(
            probability=unstable_probability,
            prediction=int(unstable_probability >= 0.5),
            risk_tier=unstable_risk_tier,
        ),
        adverse_outcome=BinaryTargetPrediction(
            probability=adverse_probability,
            prediction=int(adverse_probability >= 0.5),
            risk_tier=risk_tier,
        ),
        plaque_volume_percent=plaque_volume,
        lumen_area=lumen_area,
        recommendations=recommendations,
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_targets(payload: PredictionRequest):
    """Mock endpoint for all current project targets."""
    return _score_mock_risk(payload)
