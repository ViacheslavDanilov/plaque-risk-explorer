<div align="center">

<img src=".assets/logo.png" width="150" alt="Plaque Risk Explorer Logo">

# Plaque Risk Explorer

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/downloads/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.9-3178c6.svg)](https://www.typescriptlang.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-16-black.svg)](https://nextjs.org/)
[![pandas](https://img.shields.io/badge/pandas-3.0-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)

**Association of clinical factors and plaque morphology with adverse cardiovascular outcomes.**

</div>

## 📋 Overview

This project is a research-focused prototype designed to analyze the relationship between clinical patient profiles and coronary plaque characteristics. By identifying key morphological markers (such as plaque volume and stability) alongside clinical history, the system aims to highlight potential predictors of adverse cardiovascular events.

## 🎯 Problem Statement

The goal is to provide a tool for cardiac risk stratification by analyzing:
1.  **Clinical-Morphological Correlation**: How clinical data (Diabetes, Hypertension, etc.) relates to the physical state of coronary plaques.
2.  **Adverse Outcome Prediction**: Identifying which combinations of clinical and morphological features lead to a higher probability of events like myocardial infarction, repeat revascularization, or hospital death.

## 📁 Project Structure

```
plaque-risk-explorer/
├── backend/                        # 🐍 Python Backend (UV workspace member)
│   ├── src/plaque_risk_explorer/      # FastAPI application
│   │   ├── __init__.py
│   │   └── main.py                 # API endpoints
│   ├── models/                     # Trained ML model artifacts
│   ├── notebooks/                  # Jupyter notebooks (EDA, experiments)
│   ├── scripts/                    # Training & preprocessing scripts
│   ├── data/                       # Datasets
│   │   ├── source.csv              # Full processed dataset
│   │   ├── features.csv            # Modeling dataset (selected features + targets)
│   │   └── features.md             # Notes on source vs features changes
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

- `backend/data/source.csv`: full processed dataset (`57` rows, `36` columns).
- `backend/data/features.csv`: reduced modeling dataset (`57` rows, `18` columns).

### Features Available in `features.csv`
- `gender`
- `age`
- `angina_functional_class`
- `post_infarction_cardiosclerosis`
- `multifocal_atherosclerosis`
- `diabetes_mellitus`
- `copd_asthma`
- `hypertension`
- `cholesterol_level`
- `bmi`
- `lvef_percent`
- `blood_flow_type`
- `syntax_score`
- `ffr`
- `plaque_volume_percent`
- `lumen_area`

### Task-Specific `X` and `y`
Task 1: Clinical data -> plaque morphology
- `X`: `gender`, `age`, `angina_functional_class`, `post_infarction_cardiosclerosis`, `multifocal_atherosclerosis`, `diabetes_mellitus`, `copd_asthma`, `hypertension`, `cholesterol_level`, `bmi`, `lvef_percent`
- `y`:
  - `unstable_plaque` (classification)
  - `plaque_volume_percent` (regression)
  - `lumen_area` (regression)

Task 2: Adverse outcome prediction
- `X`: Task 1 clinical features + `unstable_plaque`, `plaque_volume_percent`, `lumen_area`, `ffr`, `syntax_score`
- `y`:
  - `adverse_outcome` (classification, derived target in `features.csv`)

## 🛠️ Tech Stack

### Backend
- **Python 3.13+**
- **FastAPI** - High-performance web framework
- **Pydantic** - Data validation
- **pandas** - Data manipulation and analysis

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

### Running the Application

**Backend (FastAPI):**
```bash
uv run uvicorn plaque_risk_explorer.main:app --reload
```
API will be available at: http://localhost:8000
API docs at: http://localhost:8000/docs

**Frontend (Next.js):**
```bash
cd frontend
pnpm dev
```
Frontend will be available at: http://localhost:3000
