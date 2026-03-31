import { useState } from "react";
import {
  SYMPTOMS_BY_CATEGORY,
  DURATIONS,
  SEVERITIES,
} from "./symptoms";
import { FrequencySlider } from "./FrequencySlider";
import "./App.css";

const API_BASE = import.meta.env.DEV ? "/api" : "http://127.0.0.1:8002";

function App() {
  const [selectedSymptoms, setSelectedSymptoms] = useState([]);
  const [age, setAge] = useState(25);
  const [frequency, setFrequency] = useState("Rarely");
  const [gender, setGender] = useState("male");
  const [duration, setDuration] = useState(DURATIONS[0]);
  const [severity, setSeverity] = useState(SEVERITIES[0]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const toggleSymptom = (sym) => {
    setSelectedSymptoms((prev) =>
      prev.includes(sym) ? prev.filter((s) => s !== sym) : [...prev, sym]
    );
  };

  const buildConversation = () => {
    const syms = selectedSymptoms.join(", ");
    const dur = duration.label;
    const sev = severity.toLowerCase();
    return `I have ${syms}. Duration: ${dur}. Severity: ${sev}.`;
  };

  const handleAnalyze = async () => {
    if (selectedSymptoms.length === 0) {
      setError("Please select at least one symptom.");
      return;
    }
    setError(null);
    setResult(null);
    setLoading(true);

    try {
      const conversation = buildConversation();
      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          conversation,
          demographics: { age, gender },
        }),
      });

      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Failed to analyze. Is the server running on port 8002?");
    } finally {
      setLoading(false);
    }
  };

  const triage = result?.triage_recommendation;
  const riskScore = result?.risk_score ?? 0;
  const confidence = result?.confidence ?? 0;
  const isOTC = triage === "OTC Drug";
  const isDoctor = triage === "Doctor Consultation";

  return (
    <div className="app">
      <nav className="topnav">
        <div className="nav-left">
          <div className="nav-logo">⚕️</div>
          <div>
            <div className="nav-title">AI Triage</div>
            <div className="nav-sub">RAG-Powered Symptom Assessment</div>
          </div>
        </div>
        <div className="nav-badges">
          <span className="badge badge-green">RAG</span>
          <span className="badge badge-blue">SYNAPSE</span>
        </div>
      </nav>

      <main className="main">
        <div className="grid">
          {/* Left: Symptoms */}
          <section className="card">
            <div className="card-header">
              <span className="card-icon">🩺</span>
              <h2>Select Symptoms</h2>
              <p className="card-sub">Model-optimized list</p>
            </div>

            {Object.entries(SYMPTOMS_BY_CATEGORY).map(([category, symptoms]) => (
              <div key={category} className="symptom-group">
                <h3>{category}</h3>
                <div className="symptom-grid">
                  {symptoms.map((sym) => (
                    <button
                      key={sym}
                      type="button"
                      className={`symptom-chip ${selectedSymptoms.includes(sym) ? "selected" : ""}`}
                      onClick={() => toggleSymptom(sym)}
                    >
                      <span className="dot" />
                      {sym}
                    </button>
                  ))}
                </div>
              </div>
            ))}

            <div className="form-row">
              <label>
                <span>Duration</span>
                <select
                  value={duration.label}
                  onChange={(e) =>
                    setDuration(DURATIONS.find((d) => d.label === e.target.value))
                  }
                >
                  {DURATIONS.map((d) => (
                    <option key={d.label} value={d.label}>
                      {d.label}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span>Severity</span>
                <select
                  value={severity}
                  onChange={(e) => setSeverity(e.target.value)}
                >
                  {SEVERITIES.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <div className="form-row full">
              <label>
                <span>Frequency</span>
                <FrequencySlider value={frequency} onChange={setFrequency} />
              </label>
            </div>

            <div className="form-row">
              <label>
                <span>Age</span>
                <input
                  type="number"
                  min={1}
                  max={120}
                  value={age}
                  onChange={(e) => setAge(Number(e.target.value) || 25)}
                />
              </label>
              <label>
                <span>Gender</span>
                <select
                  value={gender}
                  onChange={(e) => setGender(e.target.value)}
                >
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                </select>
              </label>
            </div>

            {error && <div className="error">{error}</div>}

            <button
              className="btn-analyze"
              onClick={handleAnalyze}
              disabled={loading || selectedSymptoms.length === 0}
            >
              {loading ? "Analyzing…" : "Analyze"}
            </button>

            {selectedSymptoms.length > 0 && (
              <p className="selected-count">
                {selectedSymptoms.length} symptom(s) selected
              </p>
            )}
          </section>

          {/* Right: Result */}
          <section className="card result-card">
            <div className="card-header">
              <span className="card-icon">📋</span>
              <h2>Result</h2>
            </div>

            {!result && !loading && (
              <div className="placeholder">
                Select symptoms and click <strong>Analyze</strong> to get triage
                recommendation.
              </div>
            )}

            {loading && (
              <div className="loading">
                <div className="spinner" />
                <p>Running AI pipeline…</p>
              </div>
            )}

            {result && (
              <div className="result-content">
                <div
                  className={`result-hero ${
                    isDoctor ? "high" : isOTC ? "low" : "medium"
                  }`}
                >
                  <div className="rh-badge">
                    {isDoctor ? "🔴" : isOTC ? "🟢" : "🟡"}{" "}
                    {triage || "Unknown"}
                  </div>
                  <div className="rh-action">
                    {isOTC
                      ? "Over-the-counter medication may help. Monitor symptoms."
                      : isDoctor
                        ? "Consult a doctor for proper evaluation."
                        : "Seek medical advice if symptoms persist."}
                  </div>
                  <div className="rh-meta">
                    <span>Risk: {(riskScore * 100).toFixed(0)}%</span>
                    <span>Confidence: {(confidence * 100).toFixed(0)}%</span>
                    <span>Severity: {result.severity || "—"}</span>
                  </div>
                </div>

                {result.possible_conditions?.length > 0 && (
                  <div className="conditions">
                    <h4>Possible conditions</h4>
                    <ul>
                      {result.possible_conditions.map((c, i) => (
                        <li key={i}>{c}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {result.llm_clarification?.clarifying_questions?.length > 0 && (
                  <div className="clarification">
                    <h4>Follow-up questions</h4>
                    <ul>
                      {result.llm_clarification.clarifying_questions.map(
                        (q, i) => (
                          <li key={i}>{q}</li>
                        )
                      )}
                    </ul>
                  </div>
                )}

                <div className="disclaimer">
                  ⚠️ This is for informational purposes only. It does not replace
                  professional medical advice. In emergencies, call emergency
                  services.
                </div>
              </div>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}

export default App;
