"use client";

import { FormEvent, useEffect, useState } from "react";

type RiskTier = "low" | "moderate" | "high";

type PredictionRequest = {
  age: number;
  sex: "female" | "male";
  diabetes_mellitus: boolean;
  hypertension: boolean;
  angina_class: "I" | "II" | "III" | "IV";
  lvef_percent: number;
  cholesterol_mmol_l: number;
  ffr: number;
  syntax_score: number;
};

type PredictionResponse = {
  probability: number;
  risk_tier: RiskTier;
  confidence: number;
  mock_model_version: string;
  recommendations: string[];
};

type CohortSummary = {
  patients_analyzed: number;
  high_risk_share: number;
  mean_syntax_score: number;
  mean_ffr: number;
};

type RecentCase = {
  case_id: string;
  age: number;
  risk_tier: RiskTier;
  probability: number;
};

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ??
  "http://localhost:8000";

const initialForm: PredictionRequest = {
  age: 62,
  sex: "male",
  diabetes_mellitus: false,
  hypertension: true,
  angina_class: "II",
  lvef_percent: 51,
  cholesterol_mmol_l: 5.2,
  ffr: 0.83,
  syntax_score: 18,
};

export default function Home() {
  const [form, setForm] = useState<PredictionRequest>(initialForm);
  const [summary, setSummary] = useState<CohortSummary | null>(null);
  const [recentCases, setRecentCases] = useState<RecentCase[]>([]);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isBootstrapping, setIsBootstrapping] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        setError(null);
        const [summaryResponse, casesResponse] = await Promise.all([
          fetch(`${API_BASE}/api/v1/demo/cohort-summary`),
          fetch(`${API_BASE}/api/v1/demo/recent-cases`),
        ]);

        if (!summaryResponse.ok || !casesResponse.ok) {
          throw new Error("Failed to load demo data from backend.");
        }

        const summaryData = (await summaryResponse.json()) as CohortSummary;
        const casesData = (await casesResponse.json()) as { items: RecentCase[] };

        setSummary(summaryData);
        setRecentCases(casesData.items);
      } catch {
        setError("Backend unavailable. Start FastAPI at http://localhost:8000.");
      } finally {
        setIsBootstrapping(false);
      }
    };

    void loadData();
  }, []);

  const updateField = <K extends keyof PredictionRequest>(
    key: K,
    value: PredictionRequest[K],
  ) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/api/v1/predict/adverse-outcome`, {
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
        <p className="eyebrow">Cardiology Demo</p>
        <h1>Plaque Risk Explorer</h1>
        <p className="hero-copy">
          This interface is wired to mock FastAPI endpoints and can be directly
          reused when trained models are available.
        </p>
      </section>

      <section className="metrics-grid">
        <article className="metric">
          <span>Patients analyzed</span>
          <strong>{summary?.patients_analyzed ?? "..."}</strong>
        </article>
        <article className="metric">
          <span>High-risk share</span>
          <strong>
            {summary ? `${Math.round(summary.high_risk_share * 100)}%` : "..."}
          </strong>
        </article>
        <article className="metric">
          <span>Mean SYNTAX</span>
          <strong>{summary?.mean_syntax_score ?? "..."}</strong>
        </article>
        <article className="metric">
          <span>Mean FFR</span>
          <strong>{summary?.mean_ffr ?? "..."}</strong>
        </article>
      </section>

      <section className="content-grid">
        <form className="panel" onSubmit={handleSubmit}>
          <h2>Clinical Input</h2>
          <div className="form-grid">
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
              Sex
              <select
                value={form.sex}
                onChange={(event) =>
                  updateField("sex", event.currentTarget.value as "female" | "male")
                }
              >
                <option value="female">Female</option>
                <option value="male">Male</option>
              </select>
            </label>

            <label>
              Angina Class
              <select
                value={form.angina_class}
                onChange={(event) =>
                  updateField(
                    "angina_class",
                    event.currentTarget.value as "I" | "II" | "III" | "IV",
                  )
                }
              >
                <option value="I">I</option>
                <option value="II">II</option>
                <option value="III">III</option>
                <option value="IV">IV</option>
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
              Cholesterol (mmol/L)
              <input
                type="number"
                step="0.1"
                value={form.cholesterol_mmol_l}
                onChange={(event) =>
                  updateField("cholesterol_mmol_l", Number(event.currentTarget.value))
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
                value={form.ffr}
                onChange={(event) =>
                  updateField("ffr", Number(event.currentTarget.value))
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
          </div>

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
          </div>

          <button type="submit" className="btn-primary" disabled={isLoading}>
            {isLoading ? "Calculating..." : "Run Demo Prediction"}
          </button>
        </form>

        <div className="panel result-panel">
          <h2>Prediction Output</h2>
          {result ? (
            <>
              <p className={`risk-badge risk-${result.risk_tier}`}>
                {result.risk_tier.toUpperCase()} RISK
              </p>
              <p className="probability">{Math.round(result.probability * 100)}%</p>
              <p className="meta">
                Confidence: {Math.round(result.confidence * 100)}% | Model:{" "}
                {result.mock_model_version}
              </p>
              <ul className="recommendations">
                {result.recommendations.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </>
          ) : (
            <p className="placeholder">
              Submit the form to preview a mock adverse outcome score.
            </p>
          )}
        </div>
      </section>

      <section className="panel table-panel">
        <h2>Recent Mock Cases</h2>
        {isBootstrapping ? (
          <p className="placeholder">Loading demo data...</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Case ID</th>
                <th>Age</th>
                <th>Risk Tier</th>
                <th>Probability</th>
              </tr>
            </thead>
            <tbody>
              {recentCases.map((item) => (
                <tr key={item.case_id}>
                  <td>{item.case_id}</td>
                  <td>{item.age}</td>
                  <td className={`tier-${item.risk_tier}`}>{item.risk_tier}</td>
                  <td>{Math.round(item.probability * 100)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      {error ? <p className="error">{error}</p> : null}
    </main>
  );
}
