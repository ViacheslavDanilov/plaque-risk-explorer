import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ml.inference import load_predictor, predict
from plaque_risk_explorer.executive_summary import generate_executive_summary

load_dotenv()

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = Path(os.getenv("MODEL_DIR", str(_BACKEND_ROOT / "models")))


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.predictor, app.state.reference_profile = load_predictor(_MODEL_DIR)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model from {_MODEL_DIR}. "
            "Run `python -m pipelines.train` first.",
        ) from exc
    yield


app = FastAPI(
    title="Plaque Risk Explorer",
    description="Association of Clinical Factors and Plaque Morphology with Adverse Cardiovascular Outcomes",
    version="0.1.0",
    lifespan=lifespan,
)

default_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
cors_origins_env = os.getenv("CORS_ORIGINS", "")
env_origins = [
    origin.strip() for origin in cors_origins_env.split(",") if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(dict.fromkeys(default_origins + env_origins)),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    # Clinical features
    gender: Literal["female", "male"] = "male"
    age: int = Field(62, ge=30, le=95)
    angina_functional_class: Literal[0, 1, 2, 3] = 2
    post_infarction_cardiosclerosis: bool = False
    multifocal_atherosclerosis: bool = False
    diabetes_mellitus: bool = False
    hypertension: bool = True
    cholesterol_level: float = Field(5.2, ge=2.0, le=12.0)
    bmi: float = Field(28.0, ge=15.0, le=60.0)
    lvef_percent: float = Field(51.0, ge=20.0, le=95.0)
    syntax_score: float = Field(18.0, ge=0.0, le=60.0)
    ffr: float | None = Field(default=0.83, ge=0.4, le=1.0)
    # Imaging features
    plaque_volume_percent: float = Field(60.0, ge=0.0, le=100.0)
    lumen_area: float = Field(5.0, ge=0.5, le=15.0)
    unstable_plaque: bool = False


class BinaryTargetPrediction(BaseModel):
    probability: float
    prediction: int
    risk_tier: Literal["low", "moderate", "high"]


class ExecutiveSummary(BaseModel):
    headline: str
    clinical_summary: str
    risk_drivers: list[str]
    protective_signals: list[str]
    care_focus: list[str]
    source: Literal["gemini", "fallback"]


class PredictionResponse(BaseModel):
    adverse_outcome: BinaryTargetPrediction
    executive_summary: ExecutiveSummary
    explanation: dict[str, object]


def _risk_tier(probability: float) -> Literal["low", "moderate", "high"]:
    if probability >= 0.65:
        return "high"
    if probability >= 0.35:
        return "moderate"
    return "low"


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_adverse_outcome(payload: PredictionRequest):
    """Predict adverse cardiovascular outcome probability."""
    try:
        probability, prediction, explanation = predict(
            app.state.predictor,
            app.state.reference_profile,
            payload.model_dump(),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    tier = _risk_tier(probability)
    executive_summary = generate_executive_summary(
        patient_features=payload.model_dump(),
        probability=probability,
        risk_tier=tier,
        explanation=explanation,
    )

    return PredictionResponse(
        adverse_outcome=BinaryTargetPrediction(
            probability=probability,
            prediction=prediction,
            risk_tier=tier,
        ),
        executive_summary=executive_summary,
        explanation=explanation,
    )
