# Plaque Risk Explorer - Adverse Outcome Prediction

## Executive Summary

A research prototype to identify predictors of adverse cardiovascular outcomes using clinical, morphological, and procedural patient data. The system uses AutoGluon for automated model selection with SHAP-based explainability to provide both accurate predictions and interpretable insights.

## Problem Statement

Cardiac patients undergoing coronary interventions face risks of adverse outcomes including death, myocardial infarction, stroke, and need for repeat procedures. Identifying which clinical and plaque morphology factors predict these outcomes can improve risk stratification and guide clinical decision-making.

**Previous iteration** had two tasks (plaque morphology prediction + outcome prediction). Based on stakeholder feedback, this is now focused solely on **finding predictors of adverse outcomes**.

## Success Criteria

1. AUC-ROC > 0.70 on leave-one-out cross-validation (given small dataset constraints)
2. Identification of top 5 most predictive features with statistical confidence
3. Working MVP with individual risk prediction and feature importance dashboard
4. Research report suitable for scientific publication

## User Personas

### Primary: Clinical Researcher
- Medical professionals studying cardiovascular risk factors
- Need: Understand which features predict adverse outcomes
- Technical level: Moderate (can interpret SHAP plots, understand AUC)

### Secondary: Cardiologist
- Practicing physicians assessing patient risk
- Need: Quick risk assessment for individual patients
- Technical level: Low (needs simple probability output + key factors)

## Target Variable

**`adverse_outcome`** (binary: 0/1)

Composite endpoint defined as `1` if ANY of the following occurred:
- `hospital_death`
- `stent_thrombosis`
- `hospital_mi`
- `stroke_tia`
- `stroke`
- `repeated_hospitalization`
- `repeated_revascularization`
- `myocardial_infarction_followup`

**Dataset stats**: 56 patients, 5 positive cases (8.9% prevalence)

## Feature Set

All 15 available features used as predictors:

### Clinical Features (10)
| Feature | Type | Description |
|---------|------|-------------|
| `gender` | Binary | Patient sex |
| `age` | Numeric | Age in years |
| `angina_functional_class` | Ordinal | CCS angina class (1-4) |
| `post_infarction_cardiosclerosis` | Binary | Previous MI with scarring |
| `multifocal_atherosclerosis` | Binary | Atherosclerosis in multiple vascular beds |
| `diabetes_mellitus` | Binary | Diabetes diagnosis |
| `hypertension` | Binary | Hypertension diagnosis |
| `cholesterol_level` | Numeric | Total cholesterol |
| `bmi` | Numeric | Body mass index |
| `lvef_percent` | Numeric | Left ventricular ejection fraction |

### Morphological Features (3)
| Feature | Type | Description |
|---------|------|-------------|
| `plaque_volume_percent` | Numeric | Plaque burden as % of vessel volume |
| `lumen_area` | Numeric | Remaining lumen cross-sectional area |
| `unstable_plaque` | Binary | Presence of vulnerable plaque features |

### Procedural Features (2)
| Feature | Type | Description |
|---------|------|-------------|
| `syntax_score` | Numeric | Coronary lesion complexity score |
| `ffr` | Numeric | Fractional flow reserve measurement |

## Functional Requirements

### Must Have (P0)

#### Backend
- [ ] AutoGluon model training pipeline with leave-one-out CV
- [ ] SHAP value computation for all models
- [ ] Feature importance ranking (permutation + SHAP)
- [ ] Model persistence (save/load trained models)
- [ ] Prediction API endpoint for individual patients
- [ ] LLM integration for executive summary generation (configurable provider)

#### Frontend
- [ ] Patient risk input form (all 15 features)
- [ ] Risk probability display with confidence
- [ ] Top contributing factors for individual predictions
- [ ] Feature importance dashboard (SHAP summary plot)
- [ ] LLM executive summary display (risk interpretation, factors, recommendations)

#### Analysis
- [ ] Jupyter notebook with full analysis pipeline
- [ ] Metrics computation: AUC-ROC, Sensitivity, Specificity, Balanced Accuracy, F1, Precision, Recall, AUC-PR, Brier Score
- [ ] Research report (scientific article format)

### Should Have (P1)
- [ ] Model comparison table (multiple AutoGluon models)
- [ ] Confidence intervals via bootstrap
- [ ] ROC curve visualization
- [ ] SHAP dependence plots for top features

### Nice to Have (P2)
- [ ] Patient cohort analysis
- [ ] Calibration plot
- [ ] Interactive SHAP waterfall for predictions

## LLM Executive Summary Module

### Overview
An LLM-powered module generates a human-readable executive summary for each patient risk prediction, combining the model output with SHAP-based explanations into actionable clinical insights.

### Trigger
- Generated **per individual prediction** when a patient risk assessment is requested

### Content
Each summary includes:
1. **Risk Interpretation**: What the probability means in clinical terms
2. **Key Contributing Factors**: Plain-language explanation of SHAP-identified drivers
3. **Comparison to Average**: How this patient compares to the cohort baseline
4. **Recommendations**: Suggested clinical actions based on risk level and factors

### Tone & Audience
- **Professional yet accessible**: Uses medical terminology appropriate for clinical professionals
- **Concise and actionable**: Clear insights for busy physicians
- **Patient-communicable**: Language suitable for explaining to patients when needed

### Technical Requirements

#### LLM Provider
- **Configurable** via environment variables
- API-based models (OpenAI, Anthropic, etc.)
- Provider to be selected during implementation

#### API Design
```
POST /predict
  → Returns: prediction + SHAP values + LLM summary

POST /generate-summary (internal)
  → Input: prediction result + SHAP contributions + patient features
  → Output: structured executive summary
```

#### Prompt Structure
```
You are a clinical decision support assistant. Given a patient's cardiovascular
risk assessment, provide a concise executive summary.

Patient Profile: {features}
Risk Probability: {probability}%
Risk Category: {low/medium/high}

Top Contributing Factors (SHAP):
{factor_list}

Cohort Comparison:
- Average risk: {cohort_avg}%
- This patient is {above/below} average

Generate a summary with:
1. Risk interpretation (2-3 sentences)
2. Key factors explanation (bullet points)
3. Clinical recommendations (actionable, evidence-based)

Use professional medical terminology but ensure the language is clear enough
to communicate to patients if needed.
```

#### Output Format
```json
{
  "risk_probability": 0.73,
  "risk_category": "high",
  "top_factors": [...],
  "executive_summary": {
    "risk_interpretation": "This patient has a significantly elevated risk (73%) of adverse cardiovascular outcomes...",
    "key_factors": [
      "Presence of unstable plaque substantially increases risk",
      "Diabetes mellitus contributes to elevated cardiovascular risk",
      "Preserved ejection fraction (55%) provides some protective effect"
    ],
    "comparison": "Risk is approximately 2x higher than the cohort average (35%)",
    "recommendations": [
      "Consider aggressive lipid-lowering therapy",
      "Close follow-up within 3 months recommended",
      "Patient education on warning signs of acute coronary syndrome"
    ]
  }
}
```

### Configuration
```env
# LLM Provider Configuration
LLM_PROVIDER=openai  # or anthropic, azure, etc.
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4      # or claude-3-sonnet, etc.
LLM_TEMPERATURE=0.3  # Low for consistent clinical output
```

### P0 Requirements for LLM Module
- [ ] LLM service abstraction (provider-agnostic)
- [ ] Prompt template for executive summary
- [ ] Integration with /predict endpoint
- [ ] Error handling (fallback if LLM unavailable)
- [ ] Response caching (same input → same summary)

## Technical Architecture

### Modeling Pipeline

```
features.csv → AutoGluon TabularPredictor → SHAP Explainer → Predictions + Explanations
     ↓                    ↓                       ↓
  56 rows           LOO-CV (56 folds)      Feature importance
  15 features       Multiple models         SHAP values
  1 target          Best ensemble           Summary plots
```

### AutoGluon Configuration
- `presets`: "best_quality" or "high_quality" (given small data)
- `eval_metric`: "roc_auc"
- `problem_type`: "binary"
- Cross-validation: Leave-one-out (manual loop with 56 train/test splits)

### System Components

```
┌────────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js)                       │
│  ┌─────────────────┐    ┌────────────────────────────────┐    │
│  │ Risk Input Form │    │ Feature Importance Dashboard   │    │
│  │ + LLM Summary   │    │                                │    │
│  └────────┬────────┘    └───────────────┬────────────────┘    │
└───────────┼─────────────────────────────┼──────────────────────┘
            │                             │
            ▼                             ▼
┌────────────────────────────────────────────────────────────────┐
│                     Backend (FastAPI)                          │
│  ┌─────────────────┐    ┌────────────────────────────────┐    │
│  │ /predict        │    │ /feature-importance            │    │
│  │ POST patient    │    │ GET global SHAP summary        │    │
│  │ + LLM summary   │    │                                │    │
│  └────────┬────────┘    └───────────────┬────────────────┘    │
│           │                             │                      │
│           ▼                             │                      │
│  ┌─────────────────┐                    │                      │
│  │ LLM Service     │◄───────────────────┘                      │
│  │ (configurable)  │                                           │
│  └─────────────────┘                                           │
└───────────┼─────────────────────────────┼──────────────────────┘
            │                             │
            ▼                             ▼
┌────────────────────────────────────────────────────────────────┐
│                     Model Artifacts                            │
│  ┌─────────────────┐    ┌────────────────────────────────┐    │
│  │ AutoGluon Model │    │ SHAP Values (precomputed)      │    │
│  │ (persisted)     │    │ Feature Importance             │    │
│  └─────────────────┘    └────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────────────┐
│                     External Services                          │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ LLM API (OpenAI / Anthropic / configurable)             │  │
│  └─────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### Data Model

**Input (prediction request)**:
```json
{
  "gender": 1,
  "age": 65,
  "angina_functional_class": 2,
  "post_infarction_cardiosclerosis": 1,
  "multifocal_atherosclerosis": 0,
  "diabetes_mellitus": 1,
  "hypertension": 1,
  "cholesterol_level": 5.2,
  "bmi": 28.5,
  "lvef_percent": 55,
  "syntax_score": 22,
  "ffr": 0.78,
  "plaque_volume_percent": 45.2,
  "lumen_area": 3.8,
  "unstable_plaque": 1
}
```

**Output (prediction response)**:
```json
{
  "risk_probability": 0.73,
  "risk_category": "high",
  "top_factors": [
    {"feature": "unstable_plaque", "contribution": 0.18, "direction": "increases risk"},
    {"feature": "diabetes_mellitus", "contribution": 0.12, "direction": "increases risk"},
    {"feature": "lvef_percent", "contribution": -0.08, "direction": "decreases risk"}
  ],
  "executive_summary": {
    "risk_interpretation": "This patient has a significantly elevated risk (73%) of adverse cardiovascular outcomes within the follow-up period. This places them in the high-risk category requiring close monitoring.",
    "key_factors": [
      "Presence of unstable plaque is the primary driver of elevated risk",
      "Diabetes mellitus significantly contributes to cardiovascular risk",
      "Despite risk factors, preserved ejection fraction (55%) provides some protective effect"
    ],
    "comparison": "Risk is approximately 2x higher than the cohort average of 35%",
    "recommendations": [
      "Consider aggressive lipid-lowering therapy optimization",
      "Schedule follow-up within 3 months for reassessment",
      "Educate patient on warning signs of acute coronary syndrome"
    ]
  }
}
```

## Evaluation Metrics

### Primary
- **AUC-ROC**: Area under Receiver Operating Characteristic curve

### Secondary
| Metric | Purpose |
|--------|---------|
| Sensitivity (Recall) | % of adverse outcomes correctly identified |
| Specificity | % of non-events correctly identified |
| Balanced Accuracy | Average of sensitivity and specificity |
| Precision (PPV) | % of predicted positives that are true positives |
| F1 Score | Harmonic mean of precision and recall |
| AUC-PR | Area under Precision-Recall curve (better for imbalanced data) |
| Brier Score | Calibration - how close probabilities are to actual outcomes |

### Validation Strategy
- **Leave-one-out cross-validation** (56 folds)
- Each patient used exactly once as test set
- Metrics aggregated across all folds

## Deliverables

### 1. MVP Prototype
- Working web application with risk prediction
- Feature importance visualization
- Deployed and accessible

### 2. Research Report
Scientific article format including:
- Introduction / Background
- Methods (data, model, validation)
- Results (metrics, feature importance, SHAP analysis)
- Discussion
- Conclusion

## Out of Scope

- **Task 1 (plaque morphology prediction)**: Removed per stakeholder direction
- **Individual outcome analysis**: Using composite endpoint only
- **Real-time model retraining**: Static model for MVP
- **Multi-center validation**: Single dataset only
- **Time-to-event analysis**: Binary outcome, not survival analysis

## Open Questions for Implementation

1. Should feature importance be computed on full dataset or averaged across LOO folds?
2. Threshold for risk categories (low/medium/high) - clinical input needed?
3. Report format preference: PDF, Markdown, or Jupyter notebook export?

## Appendix: Key Changes from Previous Iteration

| Aspect | Before | After |
|--------|--------|-------|
| Tasks | 2 tasks (plaque prediction + outcome) | 1 task (outcome only) |
| Targets | unstable_plaque, plaque_volume_percent, lumen_area, adverse_outcome | adverse_outcome only |
| Model | Not specified | AutoGluon with SHAP |
| Validation | Not specified | Leave-one-out CV |
| Metrics | Not specified | AUC-ROC (primary) + 7 secondary |
| Deliverables | Not specified | MVP + scientific report |
| LLM Integration | Not present | Executive summary per prediction (configurable API provider) |