<div align="center">

<img src=".assets/logo.png" width="150" alt="Plaque Predictors Logo">

# Plaque Predictors: Cardiac Risk Analysis

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
plaque-predictors/
├── backend/                        # 🐍 Python Backend (UV workspace member)
│   ├── src/plaque_predictors/      # FastAPI application
│   │   ├── __init__.py
│   │   └── main.py                 # API endpoints
│   ├── models/                     # Trained ML model artifacts
│   ├── notebooks/                  # Jupyter notebooks (EDA, experiments)
│   ├── scripts/                    # Training & preprocessing scripts
│   ├── data/                       # Datasets
│   │   ├── source.csv              # Processed English dataset (56 records)
│   │   └── source_ru.csv           # Original Russian dataset
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

The `source.csv` dataset contains 56 clinical records with the following key features:

### Clinical Data
| Feature | Description |
|---------|-------------|
| `gender` | Patient gender (male/female) |
| `age` | Patient age |
| `angina_functional_class` | Angina Functional Class (CCS) |
| `post_infarction_cardiosclerosis` | History of post-myocardial infarction |
| `multifocal_atherosclerosis` | Presence of multifocal atherosclerosis |
| `diabetes_mellitus` | Diabetes Mellitus indicator |
| `hypertension` | Hypertension (High Blood Pressure) indicator |
| `cholesterol_level` | Total cholesterol levels |

### Plaque Morphology
| Feature | Description |
|---------|-------------|
| `unstable_plaque` | Plaque Stability (1 = Unstable, 0 = Stable) |
| `plaque_volume_percent` | Plaque Volume percentage |
| `lumen_area` | Vessel Lumen area in mm² |
| `syntax_score` | Complexity score for coronary artery disease |

### Target Variables (Adverse Outcomes)
| Feature | Description |
|---------|-------------|
| `hospital_death` | Death occurring during hospitalization |
| `repeated_revascularization` | Repeat revascularization procedure |
| `myocardial_infarction_followup` | Myocardial Infarction at follow-up |
| `repeated_hospitalization` | Repeated hospitalization |

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
    git clone https://github.com/[your-username]/plaque-predictors.git
    cd plaque-predictors
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
uv run uvicorn plaque_predictors.main:app --reload
```
API will be available at: http://localhost:8000
API docs at: http://localhost:8000/docs

**Frontend (Next.js):**
```bash
cd frontend
pnpm dev
```
Frontend will be available at: http://localhost:3000
