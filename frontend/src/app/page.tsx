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
  recommendations: string[];
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

const humanizeFeature = (feature: string): string =>
  feature
    .split("_")
    .map((word) => `${word.charAt(0).toUpperCase()}${word.slice(1)}`)
    .join(" ");

const formatFeatureValue = (value: SerializedValue): string => {
  if (value === null) return "missing";
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "number") {
    return Number.isInteger(value) ? `${value}` : value.toFixed(2);
  }
  return value;
};

const formatEffect = (effect: number): string =>
  `${effect >= 0 ? "+" : "-"}${Math.abs(effect * 100).toFixed(1)} pp`;

export default function Home() {
  const [form, setForm] = useState<PredictionRequest>(initialForm);
  const [ffrInput, setFfrInput] = useState<string>("0.83");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const updateField = <K extends keyof PredictionRequest>(
    key: K,
    value: PredictionRequest[K],
  ) => setForm((prev) => ({ ...prev, [key]: value }));

  const handleSubmit = async (event: React.SyntheticEvent<HTMLFormElement>) => {
    event.preventDefault();
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!response.ok) throw new Error();
      setResult((await response.json()) as PredictionResponse);
    } catch {
      setError("Prediction request failed. Verify backend is running.");
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
  const topEffects = result?.explanation.feature_effects.slice(0, 8) ?? [];

  return (
    <main className="page-shell">
      <header className="site-header">
        <span className="header-tag">Cardiology · Risk Analysis</span>
        <h1>Plaque Risk<br />Explorer</h1>
        <p className="header-sub">
          Model-driven adverse outcome assessment from clinical and imaging parameters.
        </p>
      </header>

      <div className="content-grid">
        {/* ── Form ── */}
        <form className="form-panel" onSubmit={handleSubmit}>

          <div className="form-section">
            <div className="section-label">Patient Profile</div>
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
                  type="number" step="0.1"
                  value={form.bmi}
                  onChange={(e) => updateField("bmi", Number(e.currentTarget.value))}
                />
              </label>
            </div>
          </div>

          <div className="form-section">
            <div className="section-label">Cardiac Function</div>
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
                  type="number" step="0.1"
                  value={form.lvef_percent}
                  onChange={(e) => updateField("lvef_percent", Number(e.currentTarget.value))}
                />
              </label>
              <label className="field">
                <span className="field-label">SYNTAX Score</span>
                <input
                  type="number" step="0.1"
                  value={form.syntax_score}
                  onChange={(e) => updateField("syntax_score", Number(e.currentTarget.value))}
                />
              </label>
              <label className="field">
                <span className="field-label">Cholesterol (mmol/L)</span>
                <input
                  type="number" step="0.01"
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
            <div className="section-label">Comorbidities</div>
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
            <div className="section-label">Imaging</div>
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
          <div className="result-heading">Adverse Outcome</div>

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

              <ul className={`recs recs-${tier}`}>
                {result.recommendations.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>

              <section className="explanation-block">
                <div className="explanation-head">
                  <span>Feature Effects</span>
                  <span>
                    Baseline {Math.round(result.explanation.baseline_probability * 100)}%
                  </span>
                </div>

                <ul className="effects-list">
                  {topEffects.map((effect) => (
                    <li key={effect.feature} className="effect-item">
                      <div className="effect-line">
                        <span className="effect-name">
                          {humanizeFeature(effect.feature)}
                        </span>
                        <span
                          className={`effect-delta effect-${effect.direction}`}
                          title="Change in risk if this feature is replaced with baseline."
                        >
                          {formatEffect(effect.effect)}
                        </span>
                      </div>
                      <div className="effect-context">
                        patient {formatFeatureValue(effect.patient_value)} ·
                        baseline {formatFeatureValue(effect.reference_value)}
                      </div>
                    </li>
                  ))}
                </ul>

                <p className="effects-note">
                  Delta in predicted risk when one feature is swapped with baseline.
                </p>
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

      {error && <div className="error-bar">{error}</div>}
    </main>
  );
}
