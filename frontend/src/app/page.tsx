"use client";

import { useState } from "react";

type RiskTier = "low" | "moderate" | "high";

type PredictionRequest = {
  gender: "female" | "male";
  age: number;
  angina_functional_class: 0 | 1 | 2 | 3;
  post_infarction_cardiosclerosis: boolean;
  multifocal_atherosclerosis: boolean;
  diabetes_mellitus: boolean;
  hypertension: boolean;
  cholesterol_level: number;
  bmi: number;
  lvef_percent: number;
  syntax_score: number;
  ffr: number | null;
  plaque_volume_percent: number;
  lumen_area: number;
  unstable_plaque: boolean;
};

type BinaryTargetPrediction = {
  probability: number;
  prediction: number;
  risk_tier: RiskTier;
};

type SummarySource = "gemini" | "fallback";

type ExecutiveSummary = {
  headline: string;
  clinical_summary: string;
  risk_drivers: string[];
  protective_signals: string[];
  care_focus: string[];
  source: SummarySource;
};

type SerializedValue = string | number | boolean | null;

type FeatureEffect = {
  feature: string;
  effect: number;
  direction: "increase" | "decrease" | "neutral";
  patient_value: SerializedValue;
  reference_value: SerializedValue;
};

type Explainability = {
  method: "counterfactual_single_feature_delta";
  baseline_probability: number;
  feature_effects: FeatureEffect[];
};

type PredictionResponse = {
  adverse_outcome: BinaryTargetPrediction;
  executive_summary: ExecutiveSummary;
  explanation: Explainability;
};

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ??
  "http://localhost:8000";

const initialForm: PredictionRequest = {
  gender: "male",
  age: 62,
  angina_functional_class: 2,
  post_infarction_cardiosclerosis: false,
  multifocal_atherosclerosis: false,
  diabetes_mellitus: false,
  hypertension: true,
  cholesterol_level: 5.2,
  bmi: 28,
  lvef_percent: 51,
  syntax_score: 18,
  ffr: 0.83,
  plaque_volume_percent: 60.0,
  lumen_area: 5.0,
  unstable_plaque: false,
};

const COMORBIDITIES: { key: keyof PredictionRequest; label: string }[] = [
  { key: "diabetes_mellitus",            label: "Diabetes Mellitus" },
  { key: "hypertension",                 label: "Hypertension" },
  { key: "post_infarction_cardiosclerosis", label: "Post-MI Cardiosclerosis" },
  { key: "multifocal_atherosclerosis",   label: "Multifocal Atherosclerosis" },
];

const humanizeFeature = (feature: string): string => {
  if (feature === "lvef_percent") return "LVEF";
  if (feature === "cholesterol_level") return "Cholesterol";

  return feature
    .split("_")
    .map((word) => {
      if (word === "ffr" || word === "bmi" || word === "lvef" || word === "syntax") {
        return word.toUpperCase();
      }
      return `${word.charAt(0).toUpperCase()}${word.slice(1)}`;
    })
    .join(" ");
};

const formatFeatureValue = (value: SerializedValue): string => {
  if (value === null) return "missing";
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "number") {
    return Number.isInteger(value) ? `${value}` : value.toFixed(2);
  }
  return value;
};

const formatEffect = (effect: number): string =>
  `${effect >= 0 ? "+" : "-"}${Math.abs(effect * 100).toFixed(1)}%`;

const formatPercent = (value: number): string => `${Math.round(value * 100)}%`;

function InfoTooltip({ label, text }: { label: string; text: string }) {
  return (
    <button type="button" className="effects-info" aria-label={label}>
      <svg
        className="effects-info-icon"
        viewBox="0 0 20 20"
        fill="currentColor"
        aria-hidden="true"
      >
        <path
          fillRule="evenodd"
          d="M10 2a8 8 0 100 16 8 8 0 000-16zm.75 4.75a.75.75 0 10-1.5 0v.5a.75.75 0 001.5 0v-.5zm0 3.5a.75.75 0 00-1.5 0v3a.75.75 0 001.5 0v-3z"
          clipRule="evenodd"
        />
      </svg>
      <span className="effects-tooltip">{text}</span>
    </button>
  );
}

export default function Home() {
  const [form, setForm] = useState<PredictionRequest>(initialForm);
  const [ffrInput, setFfrInput] = useState<string>("0.83");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [lastSubmitted, setLastSubmitted] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const updateField = <K extends keyof PredictionRequest>(
    key: K,
    value: PredictionRequest[K],
  ) => setForm((prev) => ({ ...prev, [key]: value }));

  const handleSubmit = async (event: React.SyntheticEvent<HTMLFormElement>) => {
    event.preventDefault();
    const formJson = JSON.stringify(form);
    if (result && formJson === lastSubmitted) return;
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!response.ok) {
        if (response.status === 422) {
          const payload = (await response.json().catch(() => null)) as
            | { detail?: Array<{ loc?: string[]; msg?: string }> | string }
            | null;
          if (Array.isArray(payload?.detail) && payload.detail.length > 0) {
            const first = payload.detail[0];
            const field = first.loc?.[first.loc.length - 1];
            const prefix = field ? `${humanizeFeature(field)}: ` : "";
            throw new Error(`${prefix}${first.msg ?? "invalid value"}`);
          }
          if (typeof payload?.detail === "string") {
            throw new Error(payload.detail);
          }
          throw new Error("Invalid input values.");
        }
        throw new Error("Prediction request failed.");
      }
      setResult((await response.json()) as PredictionResponse);
      setLastSubmitted(formJson);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction request failed.");
    } finally {
      setIsLoading(false);
    }
  };

  const tier = result?.adverse_outcome.risk_tier;
  const gaugeStyle = result
    ? ({
        "--gauge-deg": `${Math.round(result.adverse_outcome.probability * 360)}deg`,
        "--gauge-color":
          tier === "high" ? "var(--high)" :
          tier === "moderate" ? "var(--moderate)" :
          "var(--low)",
      } as React.CSSProperties)
    : undefined;
  const waterfall = (() => {
    if (!result) return null;

    const baseline = result.explanation.baseline_probability;
    const target = result.adverse_outcome.probability;
    const selected = [...result.explanation.feature_effects.slice(0, 6)];
    const selectedTotal = selected.reduce((sum, effect) => sum + effect.effect, 0);
    const residual = target - baseline - selectedTotal;

    if (Math.abs(residual) > 0.0001) {
      selected.push({
        feature: "other_factors",
        effect: residual,
        direction: residual > 0 ? "increase" : residual < 0 ? "decrease" : "neutral",
        patient_value: null,
        reference_value: null,
      });
    }

    const deltaToFinal = target - baseline;
    const maxAbs = Math.max(
      Math.abs(deltaToFinal),
      ...selected.map((segment) => Math.abs(segment.effect)),
      0.001,
    );
    const axisHalfWidth = 40;
    const baselinePos = 50;

    return {
      baseline,
      target,
      baselinePos,
      segments: selected.map((segment, index) => {
        const width = Math.max((Math.abs(segment.effect) / maxAbs) * axisHalfWidth, 1.2);
        const left = segment.effect >= 0 ? baselinePos : baselinePos - width;
        return { ...segment, key: `${segment.feature}-${index}`, left, width };
      }),
    };
  })();

  return (
    <main className="page-shell">
      <header className="site-header">
        <span className="header-tag">Cardiology · Risk Analysis</span>
        <h1>Plaque Risk<br />Explorer</h1>
        <p className="header-sub">
          Estimate the probability of adverse cardiovascular outcomes using patient
          profile, cardiac function, comorbidities, and imaging findings. Review a
          feature-effect waterfall to see which factors increase or decrease the
          predicted risk.
        </p>
      </header>

      <div className="content-grid">
        {/* ── Form ── */}
        <form className="form-panel" onSubmit={handleSubmit}>

          <div className="form-section">
            <div className="section-label">
              <span className="section-label-main">
                Patient Profile
                <InfoTooltip
                  label="About Patient Profile"
                  text="Patient Profile includes basic personal and body measures, such as age, sex, and BMI. These values provide the model with overall background risk context."
                />
              </span>
            </div>
            <div className="form-grid">
              <label className="field">
                <span className="field-label">Gender</span>
                <select
                  value={form.gender}
                  onChange={(e) => updateField("gender", e.currentTarget.value as "female" | "male")}
                >
                  <option value="female">Female</option>
                  <option value="male">Male</option>
                </select>
              </label>
              <label className="field">
                <span className="field-label">Age</span>
                <input
                  type="number" min={30} max={95}
                  value={form.age}
                  onChange={(e) => updateField("age", Number(e.currentTarget.value))}
                />
              </label>
              <label className="field">
                <span className="field-label">BMI</span>
                <input
                  type="number" min={15} max={60} step="0.1"
                  value={form.bmi}
                  onChange={(e) => updateField("bmi", Number(e.currentTarget.value))}
                />
              </label>
            </div>
          </div>

          <div className="form-section">
            <div className="section-label">
              <span className="section-label-main">
                Cardiac Function
                <InfoTooltip
                  label="About Cardiac Function"
                  text="Cardiac Function covers heart-related clinical measurements, including LVEF, SYNTAX score, cholesterol, and FFR. These describe how the heart is functioning and the severity of disease."
                />
              </span>
            </div>
            <div className="form-grid">
              <label className="field">
                <span className="field-label">Angina Class</span>
                <select
                  value={form.angina_functional_class}
                  onChange={(e) =>
                    updateField("angina_functional_class", Number(e.currentTarget.value) as 0 | 1 | 2 | 3)
                  }
                >
                  <option value={0}>0</option>
                  <option value={1}>I</option>
                  <option value={2}>II</option>
                  <option value={3}>III</option>
                </select>
              </label>
              <label className="field">
                <span className="field-label">LVEF (%)</span>
                <input
                  type="number" min={20} max={95} step="0.1"
                  value={form.lvef_percent}
                  onChange={(e) => updateField("lvef_percent", Number(e.currentTarget.value))}
                />
              </label>
              <label className="field">
                <span className="field-label">SYNTAX Score</span>
                <input
                  type="number" min={0} max={60} step="0.1"
                  value={form.syntax_score}
                  onChange={(e) => updateField("syntax_score", Number(e.currentTarget.value))}
                />
              </label>
              <label className="field">
                <span className="field-label">Cholesterol (mmol/L)</span>
                <input
                  type="number" min={2} max={12} step="0.01"
                  value={form.cholesterol_level}
                  onChange={(e) => updateField("cholesterol_level", Number(e.currentTarget.value))}
                />
              </label>
              <label className="field">
                <span className="field-label">FFR</span>
                <input
                  type="number" step="0.01" min={0.4} max={1.0}
                  value={ffrInput}
                  onChange={(e) => {
                    const v = e.currentTarget.value;
                    setFfrInput(v);
                    updateField("ffr", v === "" ? null : Number(v));
                  }}
                />
              </label>
            </div>
          </div>

          <div className="form-section">
            <div className="section-label">
              <span className="section-label-main">
                Comorbidities
                <InfoTooltip
                  label="About Comorbidities"
                  text="Comorbidities are other health conditions, such as diabetes or hypertension, that can increase or decrease cardiovascular risk."
                />
              </span>
            </div>
            <div className="toggle-grid">
              {COMORBIDITIES.map(({ key, label }) => (
                <label key={key} className="toggle-label">
                  <input
                    type="checkbox"
                    checked={form[key] as boolean}
                    onChange={(e) => updateField(key, e.currentTarget.checked)}
                  />
                  <span className="toggle-dot" />
                  {label}
                </label>
              ))}
            </div>
          </div>

          <div className="form-section">
            <div className="section-label">
              <span className="section-label-main">
                Imaging
                <InfoTooltip
                  label="About Imaging"
                  text="Imaging includes plaque and vessel measurements from scans, for example plaque volume, lumen area, and plaque stability. These reflect structural disease features."
                />
              </span>
            </div>
            <div className="form-grid">
              <label className="field">
                <span className="field-label">Plaque Volume (%)</span>
                <input
                  type="number" step="0.1" min={0} max={100}
                  value={form.plaque_volume_percent}
                  onChange={(e) => updateField("plaque_volume_percent", Number(e.currentTarget.value))}
                />
              </label>
              <label className="field">
                <span className="field-label">Lumen Area (mm²)</span>
                <input
                  type="number" step="0.01" min={0.5} max={15}
                  value={form.lumen_area}
                  onChange={(e) => updateField("lumen_area", Number(e.currentTarget.value))}
                />
              </label>
            </div>
            <div className="toggle-grid" style={{ gridTemplateColumns: "1fr", marginTop: "12px" }}>
              <label className="toggle-label">
                <input
                  type="checkbox"
                  checked={form.unstable_plaque}
                  onChange={(e) => updateField("unstable_plaque", e.currentTarget.checked)}
                />
                <span className="toggle-dot" />
                Unstable Plaque
              </label>
            </div>
          </div>

          <button className="run-btn" type="submit" disabled={isLoading}>
            {isLoading ? (
              <><span className="spinner" />Analyzing</>
            ) : (
              "Run Analysis"
            )}
          </button>
        </form>

        {/* ── Result ── */}
        <div className="result-panel">
          <div className="result-heading">
            <span className="result-heading-main">
              Adverse Outcome
              <InfoTooltip
                label="About Adverse Outcome"
                text="Adverse Outcome is the predicted chance of a serious cardiovascular event. The model combines all entered factors into one risk estimate."
              />
            </span>
          </div>

          {result ? (
            <>
              <div className="gauge-wrap">
                <div className="gauge-ring" style={gaugeStyle}>
                  <div className="gauge-inner">
                    <div className="gauge-percent">
                      {Math.round(result.adverse_outcome.probability * 100)}%
                    </div>
                    <div className={`gauge-tier tier-${tier}`}>
                      {tier}
                    </div>
                  </div>
                </div>
              </div>

              <section className="explanation-block">
                <div className="explanation-head">
                  <span className="explanation-head-main">
                    Feature Effects
                    <InfoTooltip
                      label="How to read Feature Effects"
                      text="Baseline is the model's starting risk for a typical patient profile, built from reference values in the training data (for example, median numeric values). Each bar shows how one feature moves risk up (red) or down (green) from that baseline. Example: baseline 22% with -2.7% gives about 19.3%."
                    />
                  </span>
                </div>

                {waterfall && (
                  <>
                    <div className="waterfall-summary">
                      <span>Baseline {formatPercent(waterfall.baseline)}</span>
                      <span>Final {formatPercent(waterfall.target)}</span>
                    </div>

                    <ul className="waterfall-list">
                      {waterfall.segments.map((segment) => (
                        <li key={segment.key} className="waterfall-row">
                          <div className="waterfall-meta">
                            <span className="waterfall-name">
                              {humanizeFeature(segment.feature)}
                            </span>
                            <span
                              className={`waterfall-delta waterfall-delta-${segment.direction}`}
                            >
                              {formatEffect(segment.effect)}
                            </span>
                          </div>

                          <div className="waterfall-track">
                            <span
                              className="waterfall-marker waterfall-marker-base"
                              style={{ left: `${waterfall.baselinePos}%` }}
                            />
                            <span
                              className={`waterfall-bar waterfall-${segment.direction}`}
                              style={{
                                left: `${segment.left}%`,
                                width: `${segment.width}%`,
                              }}
                            />
                          </div>

                          {segment.feature !== "other_factors" && (
                            <div className="waterfall-values">
                              patient {formatFeatureValue(segment.patient_value)} ·
                              baseline {formatFeatureValue(segment.reference_value)}
                            </div>
                          )}
                        </li>
                      ))}
                    </ul>
                  </>
                )}

              </section>
            </>
          ) : (
            <div className="result-empty">
              <div className="pulse-ring" />
              <span>Awaiting input</span>
            </div>
          )}
        </div>
      </div>

      {result && (
        <section className="summary-panel">
          <div className="summary-card-head">
            <span className="summary-card-title">
              Executive Summary
              <InfoTooltip
                label="About Executive Summary"
                text="A structured summary of the patient's risk profile, key risk drivers, protective factors, and suggested care focus areas."
              />
            </span>
            <span className={`summary-source summary-source-${result.executive_summary.source}`}>
              {result.executive_summary.source === "gemini" ? "Gemini" : "Fallback"}
            </span>
          </div>

          <p className="summary-headline">{result.executive_summary.headline}</p>
          <p className="summary-text">{result.executive_summary.clinical_summary}</p>

          <div className="summary-columns">
            <div className="summary-group">
              <h3>Risk Drivers</h3>
              <ul>
                {result.executive_summary.risk_drivers.map((item, index) => (
                  <li key={`${item}-${index}`}>{item}</li>
                ))}
              </ul>
            </div>

            <div className="summary-group">
              <h3>Protective Signals</h3>
              <ul>
                {result.executive_summary.protective_signals.map((item, index) => (
                  <li key={`${item}-${index}`}>{item}</li>
                ))}
              </ul>
            </div>

            <div className="summary-group">
              <h3>Care Focus</h3>
              <ul>
                {result.executive_summary.care_focus.map((item, index) => (
                  <li key={`${item}-${index}`}>{item}</li>
                ))}
              </ul>
            </div>
          </div>
        </section>
      )}

      {error && <div className="error-bar">{error}</div>}
    </main>
  );
}
