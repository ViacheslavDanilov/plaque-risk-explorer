from fastapi import FastAPI

app = FastAPI(
    title="Plaque Predictors",
    description="Association of Clinical Factors and Plaque Morphology with Adverse Cardiovascular Outcomes",
    version="0.1.0",
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
