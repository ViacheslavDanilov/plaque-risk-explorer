# Plaque Predictors

Association of Clinical Factors and Plaque Morphology with Adverse Cardiovascular Outcomes.

## Project Structure

```
plaque-predictors/
├── backend/          # FastAPI backend (Python)
│   ├── src/          # Source code
│   ├── data/         # Data files (contains source.csv)
│   ├── models/       # ML models
│   ├── notebooks/    # Jupyter notebooks
│   └── scripts/      # Utility scripts
├── frontend/         # Next.js frontend (TypeScript)
│   ├── src/app/      # App router pages
│   └── public/       # Static assets
└── pyproject.toml    # UV workspace config
```

## Quick Start

### Prerequisites

- Python 3.13+
- Node.js 20+
- [uv](https://docs.astral.sh/uv/)
- [pnpm](https://pnpm.io/)

### Backend

```bash
# Install dependencies
uv sync --dev

# Run development server
uv run uvicorn plaque_predictors.main:app --reload
```

### Frontend

```bash
cd frontend

# Install dependencies
pnpm install

# Run development server
pnpm dev
```

## Development

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run pre-commit on all files
uv run pre-commit run --all-files
```
