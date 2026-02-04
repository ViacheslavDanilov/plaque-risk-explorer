---
description: Create a new fullstack project with FastAPI backend (UV workspace) and Next.js frontend (pnpm, TypeScript, Tailwind)
---

# New Fullstack Project Workflow

This workflow creates a fullstack project matching the fraud-detection structure.

## Prerequisites

- Python 3.13+ installed
- Node.js 20+ installed
- [uv](https://docs.astral.sh/uv/) installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- [pnpm](https://pnpm.io/) installed (`npm install -g pnpm`)

---

## Step 1: Gather Project Information

Ask the user for:
1. **Project name** (e.g., `my-awesome-app`) - used for folder names and package names
2. **Project description** (e.g., "A web app for managing tasks")
3. **Author name and email** (e.g., "John Doe", "john@example.com")
4. **Target directory** (default: current directory)

---

## Step 2: Create Root Project Structure

```
<project-name>/
├── .agent/workflows/          # Copy this workflow here
├── .github/                   # GitHub Actions (optional)
├── .gitignore                 # Combined Python + Node gitignore
├── .pre-commit-config.yaml    # Pre-commit hooks
├── LICENSE                    # MIT or preferred license
├── README.md                  # Project overview
├── backend/                   # FastAPI backend (UV workspace member)
├── frontend/                  # Next.js frontend
└── pyproject.toml             # UV workspace root config
```

---

## Step 3: Initialize Git Repository

// turbo
```bash
git init
```

---

## Step 4: Create Root pyproject.toml (UV Workspace)

Create `pyproject.toml` at root with:

```toml
[tool.uv.workspace]
members = ["backend"]
```

---

## Step 5: Create Backend Structure

// turbo
```bash
mkdir -p backend/src/<project_name_snake_case>
mkdir -p backend/data backend/models backend/notebooks backend/scripts
```

Create `backend/pyproject.toml`:

```toml
[project]
name = "<project-name>"
version = "0.1.0"
description = "<project-description>"
authors = [
    { name = "<author-name>", email = "<author-email>" }
]
requires-python = ">=3.13"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.34.0",
    "pydantic>=2.10.0",
]

[dependency-groups]
dev = [
    "httpx>=0.28.0",
    "pre-commit>=4.5.1",
    "pytest>=8.0.0",
    "ruff>=0.9.0",
    "mypy>=1.14.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/<project_name_snake_case>"]

[tool.mypy]
python_version = "3.13"
exclude = ["venv", ".venv", "tests"]
ignore_missing_imports = true
disallow_untyped_decorators = false
disallow_untyped_calls = false
explicit_package_bases = true

[tool.ruff]
target-version = "py313"
line-length = 88

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # pyflakes
    "I",      # isort
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "UP",     # pyupgrade
]
ignore = [
    "E501",   # line too long (handled by formatter)
]

[tool.ruff.lint.isort]
known-first-party = ["<project_name_snake_case>"]
```

Create `backend/src/<project_name_snake_case>/__init__.py` (empty file).

Create `backend/src/<project_name_snake_case>/main.py`:

```python
from fastapi import FastAPI

app = FastAPI(
    title="<Project Name>",
    description="<project-description>",
    version="0.1.0",
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
```

Create `backend/README.md`:

```markdown
# <Project Name> - Backend

<project-description>

## Setup

```bash
# From project root
uv sync --dev
```

## Run

```bash
uv run uvicorn <project_name_snake_case>.main:app --reload
```

## Test

```bash
uv run pytest
```
```

---

## Step 6: Install Backend Dependencies

// turbo
```bash
uv sync --dev
```

---

## Step 7: Create Frontend with Next.js

// turbo
```bash
pnpm create next-app@latest frontend --typescript --tailwind --eslint --app --src-dir --no-import-alias --skip-install
```

After creation, update `frontend/package.json`:
- Change `"name"` to match project name
- Add scripts for prettier:
  ```json
  "check": "prettier --check .",
  "format": "prettier --write ."
  ```
- Add dev dependency: `"prettier": "^3.8.1"`

---

## Step 8: Install Frontend Dependencies

// turbo
```bash
cd frontend && pnpm install && cd ..
```

---

## Step 9: Create .gitignore

Use a comprehensive gitignore combining Python and Node patterns. Include:
- Python: `__pycache__/`, `.venv/`, `*.pyc`, `.mypy_cache/`, `.ruff_cache/`
- Node: `node_modules/`, `.next/`, `*.log`
- Editors: `.vscode/`, `.idea/`
- Environment: `.env`, `.envrc`

---

## Step 10: Create .pre-commit-config.yaml

```yaml
repos:
- repo: https://github.com/pre-commit/pre-commit-hooks
  rev: v6.0.0
  hooks:
  - id: check-yaml
  - id: check-toml
  - id: check-json
    exclude: ^frontend/node_modules/
  - id: check-ast
  - id: trailing-whitespace
  - id: end-of-file-fixer

- repo: https://github.com/asottile/add-trailing-comma
  rev: v4.0.0
  hooks:
  - id: add-trailing-comma

- repo: https://github.com/macisamuele/language-formatters-pre-commit-hooks
  rev: v2.16.0
  hooks:
  - id: pretty-format-yaml
    args: [--autofix, --indent, '2', --preserve-quotes]

- repo: https://github.com/hadialqattan/pycln
  rev: v2.6.0
  hooks:
  - id: pycln
    args: [--config, backend/pyproject.toml]

- repo: https://github.com/astral-sh/ruff-pre-commit
  rev: 'v0.14.13'
  hooks:
  - id: ruff
    args: [--fix, --config, backend/pyproject.toml]
  - id: ruff-format
    args: [--config, backend/pyproject.toml]

- repo: https://github.com/pre-commit/mirrors-mypy
  rev: v1.19.1
  hooks:
  - id: mypy
    args: [--config-file, backend/pyproject.toml]

exclude: |
  (?x)(
      ^backend/models/|
      ^backend/data/|
      ^frontend/node_modules/
  )
```

---

## Step 11: Install Pre-commit Hooks

// turbo
```bash
uv run pre-commit install
```

---

## Step 12: Create Root README.md

```markdown
# <Project Name>

<project-description>

## Project Structure

```
<project-name>/
├── backend/          # FastAPI backend (Python)
│   ├── src/          # Source code
│   ├── data/         # Data files
│   ├── models/       # ML models
│   ├── notebooks/    # Jupyter notebooks
│   └── scripts/      # Utility scripts
├── frontend/         # Next.js frontend (TypeScript)
│   ├── src/app/      # App router pages
│   └── public/       # Static assets
└── pyproject.toml    # UV workspace config
```

## Quick Start

### Backend

```bash
# Install dependencies
uv sync --dev

# Run development server
uv run uvicorn <project_name_snake_case>.main:app --reload
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
```

---

## Step 13: Initial Commit

```bash
git add .
git commit -m "Initial project setup"
```

---

## Verification

1. Backend health check: `curl http://localhost:8000/health`
2. Frontend running: `http://localhost:3000`
3. Pre-commit works: `uv run pre-commit run --all-files`

---

## Notes

- Replace `<project-name>` with kebab-case name (e.g., `my-app`)
- Replace `<project_name_snake_case>` with snake_case (e.g., `my_app`)
- Replace `<Project Name>` with title case (e.g., `My App`)
