<div align="center">

<img src=".assets/logo.png" width="150" alt="Plaque Risk Explorer Logo">

# Plaque Risk Explorer

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/downloads/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.9-3178c6.svg)](https://www.typescriptlang.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-16-black.svg)](https://nextjs.org/)
[![pandas](https://img.shields.io/badge/pandas-3.0-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)

**Identifying predictors of adverse cardiovascular outcomes using AutoGluon, local counterfactual explainability, and LLM-powered clinical summaries.**

🔗 **Live Demo**: https://plaque-risk-explorer.vercel.app/

</div>

## 📋 Overview

This project is a research-focused prototype designed to identify predictors of adverse cardiovascular outcomes using clinical, morphological, and procedural patient data. The system uses AutoGluon for automated model selection with local counterfactual feature-effect explainability, providing both accurate predictions and interpretable insights. An LLM-powered module generates concise executive summaries for individual patient risk assessments.

## 🎯 Problem Statement

Cardiac patients undergoing coronary interventions face risks of adverse outcomes including death, myocardial infarction, stroke, and need for repeat procedures. The goal is to:

1. **Identify Key Predictors**: Determine which clinical and morphological factors are most predictive of adverse outcomes using local feature-effect attribution
2. **Risk Stratification**: Provide individual patient risk probability with interpretable factor contributions
3. **Clinical Decision Support**: Generate LLM-powered executive summaries with risk interpretation and actionable recommendations

## 📁 Project Structure

```
plaque-risk-explorer/
├── backend/                        # 🐍 Python Backend (UV workspace member)
│   ├── src/
│   │   ├── plaque_risk_explorer/   # FastAPI application
│   │   │   └── main.py             # API endpoints
│   │   ├── ml/                     # ML modules
│   │   │   ├── evaluation/         # Metrics & evaluation logic
│   │   │   ├── inference/          # Prediction & explainability inference
│   │   │   ├── preprocessing/      # Feature engineering
│   │   │   └── training/           # Model training
│   │   └── scripts/                # Entry-point scripts
│   │       ├── build.py            # Data preprocessing script
│   │       ├── train.py            # Model training script
│   │       └── evaluate.py         # Model evaluation script
│   ├── data/                       # Datasets
│   │   ├── source.csv              # Full processed dataset
│   │   ├── features.csv            # Modeling dataset (selected features + target)
│   │   └── features.md             # Notes on source vs features changes
│   ├── models/                     # Trained ML model artifacts
│   ├── reports/                    # Generated analysis reports
│   │   ├── eda.md                  # Exploratory data analysis report
│   │   └── model_performance.md    # Model evaluation & explainability report
│   └── pyproject.toml              # Backend dependencies
│
├── frontend/                       # ⚛️ Next.js Frontend
│   ├── src/app/
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   └── globals.css
│   └── package.json
│
├── pyproject.toml                  # UV workspace definition
├── uv.lock                         # Lockfile
├── .pre-commit-config.yaml         # Code quality hooks
└── README.md
```

## 📊 Dataset

- `backend/data/source.csv`: full processed dataset (`56` rows, `36` columns).
- `backend/data/features.csv`: modeling dataset (`56` rows, `16` columns).

### Predictors (`X`) — 15 features

**Clinical (10)**
- `gender`, `age`, `angina_functional_class`
- `post_infarction_cardiosclerosis`, `multifocal_atherosclerosis`
- `diabetes_mellitus`, `hypertension`
- `cholesterol_level`, `bmi`, `lvef_percent`

**Morphological (3)**
- `plaque_volume_percent`, `lumen_area`, `unstable_plaque`

**Procedural (2)**
- `syntax_score`, `ffr`

### Target (`y`)
- `adverse_outcome` — binary composite endpoint (`1` if any of: hospital death, stent thrombosis, MI, stroke/TIA, repeated hospitalization/revascularization, or MI at follow-up). Positive cases: `5` of `56` (8.9%).

## 🛠️ Tech Stack

### Backend
- **Python 3.13+**
- **FastAPI** - High-performance web framework
- **Pydantic** - Data validation
- **pandas** - Data manipulation and analysis
- **AutoGluon** - Automated ML with model selection and ensembling
- **Counterfactual Attribution** - Local per-feature explainability for individual predictions
- **LLM API** - Configurable provider (OpenAI, Anthropic, etc.) for executive summaries

### Frontend
- **Next.js 16** - React framework with App Router
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS 4** - Modern CSS framework

### Development
- **uv** - Extremely fast Python package manager
- **pnpm** - Efficient Node.js package manager
- **pre-commit** - Git hooks for code quality
- **ruff** - High-performance Linter and Formatter

## 🚀 Getting Started

### Prerequisites

- Python 3.13+
- Node.js 20+
- [uv](https://docs.astral.sh/uv/)
- [pnpm](https://pnpm.io/)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ViacheslavDanilov/plaque-risk-explorer.git
    cd plaque-risk-explorer
    ```

2.  **Install Python dependencies:**
    ```bash
    uv sync
    ```

3.  **Install frontend dependencies:**
    ```bash
    cd frontend
    pnpm install
    cd ..
    ```

### Environment Variables

Copy the example and fill in your API key:
```bash
cp backend/.env.example backend/.env
```

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes | — | Google Gemini API key for executive summaries |
| `GEMINI_MODEL` | No | `gemini-3-flash-preview` | Gemini model ID |
| `GEMINI_TEMPERATURE` | No | `0` | Generation temperature |
| `GEMINI_TIMEOUT_SECONDS` | No | `45` | Request timeout in seconds |
| `NEXT_PUBLIC_API_BASE_URL` | No | `http://localhost:8000` | Backend URL for the frontend |

If `GEMINI_API_KEY` is missing or the API call fails, the app falls back to a template-based summary.

### Running the Application

**Backend (FastAPI):**
```bash
uv run uvicorn plaque_risk_explorer.main:app --reload
```
- API: http://localhost:8000
- API docs: http://localhost:8000/docs

**Frontend (Next.js):**
```bash
cd frontend
pnpm dev
```
Frontend will be available at: http://localhost:3000
