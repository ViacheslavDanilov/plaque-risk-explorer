from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ml.inference.mock_predictor import predict_unstable_plaque_and_adverse_outcome

app = FastAPI(
    title="Plaque Risk Explorer",
    description="Association of Clinical Factors and Plaque Morphology with Adverse Cardiovascular Outcomes",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    age: int = Field(62, ge=30, le=95)
    sex: Literal["female", "male"] = "male"
    diabetes_mellitus: bool = False
    hypertension: bool = True
    angina_class: Literal["I", "II", "III", "IV"] = "II"
    lvef_percent: float = Field(51.0, ge=20.0, le=80.0)
    cholesterol_mmol_l: float = Field(5.2, ge=2.0, le=12.0)
    ffr: float = Field(0.83, ge=0.4, le=1.0)
    syntax_score: float = Field(18.0, ge=0.0, le=60.0)


class PredictionResponse(BaseModel):
    probability: float
    risk_tier: Literal["low", "moderate", "high"]
    confidence: float
    mock_model_version: str
    recommendations: list[str]


class DemoBinaryPredictionResponse(BaseModel):
    unstable_plaque_probability: float
    unstable_plaque_prediction: int
    adverse_outcome_probability: float
    adverse_outcome_prediction: int
    model_version: str


def _score_mock_risk(payload: PredictionRequest) -> PredictionResponse:
    risk = 0.11
    risk += 0.002 * (payload.age - 50)
    risk += 0.009 * max(payload.syntax_score - 10, 0)
    risk += 0.38 * max(0.85 - payload.ffr, 0)
    risk += 0.006 * max(payload.cholesterol_mmol_l - 5.0, 0)
    risk += 0.003 * max(55 - payload.lvef_percent, 0)

    if payload.diabetes_mellitus:
        risk += 0.11
    if payload.hypertension:
        risk += 0.07
    if payload.sex == "male":
        risk += 0.03
    if payload.angina_class in {"III", "IV"}:
        risk += 0.08

    probability = round(min(max(risk, 0.02), 0.96), 3)
    confidence = round(0.67 + 0.22 * abs(probability - 0.5), 3)

    if probability >= 0.65:
        risk_tier: Literal["low", "moderate", "high"] = "high"
        recommendations = [
            "Discuss immediate cardiology follow-up within 2 weeks.",
            "Prioritize lipid and blood pressure optimization.",
            "Consider advanced imaging and invasive assessment if symptoms persist.",
        ]
    elif probability >= 0.35:
        risk_tier = "moderate"
        recommendations = [
            "Schedule structured outpatient follow-up.",
            "Optimize risk factors and repeat clinical evaluation in 6-8 weeks.",
            "Track symptom progression and functional status.",
        ]
    else:
        risk_tier = "low"
        recommendations = [
            "Continue standard risk-factor control.",
            "Maintain lifestyle and preventive pharmacotherapy.",
            "Reassess if symptom profile changes.",
        ]

    return PredictionResponse(
        probability=probability,
        risk_tier=risk_tier,
        confidence=confidence,
        mock_model_version="mock-v0.1",
        recommendations=recommendations,
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/api/v1/predict/adverse-outcome", response_model=PredictionResponse)
async def predict_adverse_outcome(payload: PredictionRequest):
    """Mock endpoint used by frontend until trained models are available."""
    return _score_mock_risk(payload)


@app.post("/api/v1/predict/demo-binary", response_model=DemoBinaryPredictionResponse)
async def predict_demo_binary(payload: PredictionRequest):
    """Simple demo endpoint that mimics calling two binary models."""
    return predict_unstable_plaque_and_adverse_outcome(payload.model_dump())
