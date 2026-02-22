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

type PredictionResponse = {
  adverse_outcome: BinaryTargetPrediction;
  recommendations: string[];
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

export default function Home() {
  const [form, setForm] = useState<PredictionRequest>(initialForm);
  const [ffrInput, setFfrInput] = useState<string>("0.83");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const updateField = <K extends keyof PredictionRequest>(
    key: K,
    value: PredictionRequest[K],
  ) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });

      if (!response.ok) {
        throw new Error("Prediction request failed.");
      }

      const payload = (await response.json()) as PredictionResponse;
      setResult(payload);
    } catch {
      setError("Prediction request failed. Verify backend is running.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="page-shell">
      <section className="hero-card">
        <p className="eyebrow">Cardiology</p>
        <h1>Plaque Risk Explorer</h1>
        <p className="hero-copy">
          Enter clinical and imaging data to estimate adverse cardiovascular outcome risk.
        </p>
      </section>

      <section className="content-grid">
        <form className="panel" onSubmit={handleSubmit}>

          <h2>Patient Profile</h2>
          <div className="form-grid">
            <label>
              Gender
              <select
                value={form.gender}
                onChange={(event) =>
                  updateField("gender", event.currentTarget.value as "female" | "male")
                }
              >
                <option value="female">Female</option>
                <option value="male">Male</option>
              </select>
            </label>
            <label>
              Age
              <input
                type="number"
                value={form.age}
                min={30}
                max={95}
                onChange={(event) =>
                  updateField("age", Number(event.currentTarget.value))
                }
              />
            </label>
            <label>
              BMI
              <input
                type="number"
                step="0.1"
                value={form.bmi}
                onChange={(event) =>
                  updateField("bmi", Number(event.currentTarget.value))
                }
              />
            </label>
          </div>

          <h2>Cardiac Function</h2>
          <div className="form-grid">
            <label>
              Angina Functional Class
              <select
                value={form.angina_functional_class}
                onChange={(event) =>
                  updateField(
                    "angina_functional_class",
                    Number(event.currentTarget.value) as 0 | 1 | 2 | 3,
                  )
                }
              >
                <option value={0}>0</option>
                <option value={1}>1</option>
                <option value={2}>2</option>
                <option value={3}>3</option>
              </select>
            </label>
            <label>
              LVEF (%)
              <input
                type="number"
                step="0.1"
                value={form.lvef_percent}
                onChange={(event) =>
                  updateField("lvef_percent", Number(event.currentTarget.value))
                }
              />
            </label>
            <label>
              SYNTAX Score
              <input
                type="number"
                step="0.1"
                value={form.syntax_score}
                onChange={(event) =>
                  updateField("syntax_score", Number(event.currentTarget.value))
                }
              />
            </label>
            <label>
              Cholesterol (mmol/L)
              <input
                type="number"
                step="0.01"
                value={form.cholesterol_level}
                onChange={(event) =>
                  updateField("cholesterol_level", Number(event.currentTarget.value))
                }
              />
            </label>
            <label>
              FFR
              <input
                type="number"
                step="0.01"
                min={0.4}
                max={1.0}
                value={ffrInput}
                onChange={(event) => {
                  const value = event.currentTarget.value;
                  setFfrInput(value);
                  updateField("ffr", value === "" ? null : Number(value));
                }}
              />
            </label>
          </div>

          <h2>Comorbidities</h2>
          <div className="switch-row">
            <label className="switch">
              <input
                type="checkbox"
                checked={form.diabetes_mellitus}
                onChange={(event) =>
                  updateField("diabetes_mellitus", event.currentTarget.checked)
                }
              />
              Diabetes Mellitus
            </label>
            <label className="switch">
              <input
                type="checkbox"
                checked={form.hypertension}
                onChange={(event) =>
                  updateField("hypertension", event.currentTarget.checked)
                }
              />
              Hypertension
            </label>
            <label className="switch">
              <input
                type="checkbox"
                checked={form.post_infarction_cardiosclerosis}
                onChange={(event) =>
                  updateField("post_infarction_cardiosclerosis", event.currentTarget.checked)
                }
              />
              Post-Infarction Cardiosclerosis
            </label>
            <label className="switch">
              <input
                type="checkbox"
                checked={form.multifocal_atherosclerosis}
                onChange={(event) =>
                  updateField("multifocal_atherosclerosis", event.currentTarget.checked)
                }
              />
              Multifocal Atherosclerosis
            </label>
          </div>

          <h2>Imaging</h2>
          <div className="form-grid">
            <label>
              Plaque Volume (%)
              <input
                type="number"
                step="0.1"
                min={0}
                max={100}
                value={form.plaque_volume_percent}
                onChange={(event) =>
                  updateField("plaque_volume_percent", Number(event.currentTarget.value))
                }
              />
            </label>
            <label>
              Lumen Area (mm²)
              <input
                type="number"
                step="0.01"
                min={0.5}
                max={15}
                value={form.lumen_area}
                onChange={(event) =>
                  updateField("lumen_area", Number(event.currentTarget.value))
                }
              />
            </label>
          </div>
          <div className="switch-row">
            <label className="switch">
              <input
                type="checkbox"
                checked={form.unstable_plaque}
                onChange={(event) =>
                  updateField("unstable_plaque", event.currentTarget.checked)
                }
              />
              Unstable Plaque
            </label>
          </div>

          <button type="submit" className="btn-primary" disabled={isLoading}>
            {isLoading ? "Calculating..." : "Run Prediction"}
          </button>
        </form>

        <div className="panel result-panel">
          <h2>Adverse Outcome Prediction</h2>
          {result ? (
            <>
              <div className="target-cards-grid">
                <article className="target-card">
                  <p className={`risk-badge risk-${result.adverse_outcome.risk_tier}`}>
                    ADVERSE OUTCOME: {result.adverse_outcome.risk_tier.toUpperCase()}
                  </p>
                  <p className="target-percent">
                    {Math.round(result.adverse_outcome.probability * 100)}%
                  </p>
                </article>
              </div>

              <ul className="recommendations">
                {result.recommendations.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </>
          ) : (
            <p className="placeholder">
              Submit the form to estimate adverse cardiovascular outcome risk.
            </p>
          )}
        </div>
      </section>

      {error ? <p className="error">{error}</p> : null}
    </main>
  );
}
