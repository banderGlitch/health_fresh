import { useState, useRef, useEffect } from "react";
import {
  SYMPTOMS_BY_CATEGORY,
  ALLOWED_SYMPTOMS_FLAT,
} from "./symptoms";
import "./App.css";

function apiBase() {
  if (import.meta.env.DEV) return "/api";
  return import.meta.env.VITE_API_URL || "http://127.0.0.1:8001";
}

const API_BASE = apiBase();

let _msgId = 0;
const nid = () => `m-${++_msgId}-${Date.now()}`;

const PLACEHOLDER_REASONING =
  "Collecting more details before full risk scoring.";

function isPlaceholderReasoning(text) {
  const t = String(text || "").trim();
  return !t || t === PLACEHOLDER_REASONING;
}

/** One assistant bubble per API turn — structured cards like the reference UI. */
function buildAssistantMessages(result) {
  return [{ id: nid(), role: "assistant", kind: "turn", result }];
}

function App() {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [selectedSymptoms, setSelectedSymptoms] = useState([]);
  const [age, setAge] = useState(25);
  const [gender, setGender] = useState("male");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [messages, setMessages] = useState([]);
  const [error, setError] = useState(null);
  const [composerText, setComposerText] = useState("");
  const [followUpLoading, setFollowUpLoading] = useState(false);
  const [followUpError, setFollowUpError] = useState(null);
  const threadRef = useRef(null);

  const scrollToBottom = () => {
    const el = threadRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading, followUpLoading]);

  const toggleSymptom = (sym) => {
    setSelectedSymptoms((prev) =>
      prev.includes(sym) ? prev.filter((s) => s !== sym) : [...prev, sym]
    );
  };

  /** Symptoms only — duration & severity are collected via assistant follow-up questions. */
  const buildConversation = () => {
    const syms = selectedSymptoms.join(", ");
    return `I have ${syms}.`;
  };

  const resetChat = () => {
    setMessages([]);
    setResult(null);
    setComposerText("");
    setFollowUpError(null);
    setError(null);
    setSelectedSymptoms([]);
  };

  const runAnalyze = async (conversation) => {
    setError(null);
    setLoading(true);
    try {
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
      setMessages([
        { id: nid(), role: "user", text: conversation },
        ...buildAssistantMessages(data),
      ]);
      setComposerText("");
      setDrawerOpen(false);
    } catch (err) {
      setError(
        err.message ||
          "Failed to reach the API. Check that uvicorn is running and VITE_API_URL matches its port."
      );
    } finally {
      setLoading(false);
    }
  };

  /** First turn: symptoms from chips only; optional notes. No duration/severity in the payload. */
  const buildFirstTurnConversation = () => {
    let conv = buildConversation();
    const notes = composerText.trim();
    if (notes) {
      conv += ` Patient notes: ${notes}`;
    }
    return conv;
  };

  const handleStartConversation = async () => {
    setFollowUpError(null);
    if (selectedSymptoms.length === 0) {
      setError(
        "Choose one or more symptoms from the list in the chat (or in Symptoms & details)."
      );
      return;
    }
    const conversation = buildFirstTurnConversation();
    resetChat();
    await runAnalyze(conversation);
  };

  /** Panel: send structured line from chips (+ optional composer notes). */
  const handlePanelSend = async () => {
    setFollowUpError(null);
    if (selectedSymptoms.length === 0) {
      setError("Select at least one symptom in the panel.");
      return;
    }
    const conversation = buildFirstTurnConversation();
    resetChat();
    await runAnalyze(conversation);
  };

  const applyChipsToComposer = () => {
    if (selectedSymptoms.length === 0) {
      setError("Select symptoms first.");
      return;
    }
    setComposerText(buildConversation());
    setError(null);
  };

  const canAnswerFollowUp =
    Boolean(result?.session_id) &&
    (result?.llm_clarification?.clarifying_questions?.length ?? 0) > 0 &&
    result?.conversation_status !== "completed";

  const busy = loading || followUpLoading;

  const handleComposerSend = async () => {
    const text = composerText.trim();
    if (busy) return;

    if (result?.session_id && canAnswerFollowUp) {
      setFollowUpError(null);
      setFollowUpLoading(true);
      try {
        const res = await fetch(`${API_BASE}/analyze/continue`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: result.session_id,
            answers: text,
          }),
        });
        if (!res.ok) {
          if (res.status === 404) {
            throw new Error("Session expired. Start a new chat.");
          }
          throw new Error(`API error: ${res.status}`);
        }
        const data = await res.json();
        setResult(data);
        setMessages((prev) => [
          ...prev,
          { id: nid(), role: "user", text },
          ...buildAssistantMessages(data),
        ]);
        setComposerText("");
      } catch (err) {
        setFollowUpError(err.message || "Could not send.");
      } finally {
        setFollowUpLoading(false);
      }
      return;
    }

    await handleStartConversation();
  };

  const composerEnabled =
    !busy &&
    (canAnswerFollowUp ||
      (!result?.session_id && messages.length === 0));

  const formatSymptomLine = (s) => {
    const bits = [s?.name].filter(Boolean);
    if (s?.duration) bits.push(`Duration: ${s.duration}`);
    if (s?.severity) bits.push(`Severity: ${s.severity}`);
    return bits.join(" · ");
  };

  const triageTone = (triage) => {
    if (triage === "Emergency" || triage === "Doctor Consultation") return "high";
    if (triage === "OTC Drug") return "low";
    return "medium";
  };

  const triageLead = (triage) => {
    if (triage === "OTC Drug")
      return "Over-the-counter options may be appropriate; keep monitoring how you feel.";
    if (triage === "Doctor Consultation")
      return "In-person or urgent medical evaluation is the recommended next step.";
    if (triage === "Emergency")
      return "Serious symptoms may be present — seek emergency care if appropriate.";
    return "Continue to share details so we can refine this assessment.";
  };

  const renderAssistantTurn = (msg) => {
    const r = msg.result;
    const reasoning = r?.llm_clarification?.reasoning_summary;
    const hasRichReasoning = !isPlaceholderReasoning(reasoning);
    const analysisText = hasRichReasoning
      ? String(reasoning).trim()
      : "We're reviewing what you shared and may ask follow-up questions to narrow things down.";

    const triage = r?.triage_recommendation;
    const riskScore = r?.risk_score;
    const confidence = r?.confidence;
    const hasNumericRisk =
      riskScore != null && !Number.isNaN(Number(riskScore));
    const hasNumericConf =
      confidence != null && !Number.isNaN(Number(confidence));
    const collectionOnly = Boolean(r?.collection_only);
    const conditions = r?.possible_conditions || [];
    const qs = r?.llm_clarification?.clarifying_questions || [];
    const hasRecorded =
      (r?.symptoms?.length ?? 0) > 0 || (r?.negated?.length ?? 0) > 0;
    const tone = triageTone(triage);

    return (
      <div key={msg.id} className="chat-row assistant">
        <div className="chat-avatar" aria-hidden>
          ⚕️
        </div>
        <div className="assistant-turn">
          <div className="at-block at-analysis">
            <p className="at-analysis-text">{analysisText}</p>
            {r?.help_message && (
              <p className="at-help-banner">{r.help_message}</p>
            )}
          </div>

          {hasRecorded && (
            <section className="at-card at-card-recorded" aria-label="What we recorded">
              <div className="at-card-head">
                <span className="at-card-icon" aria-hidden>
                  📋
                </span>
                <h3 className="at-card-title">What we recorded</h3>
              </div>
              {r.symptoms?.length > 0 && (
                <ul className="at-list">
                  {r.symptoms.map((s, i) => (
                    <li key={i}>{formatSymptomLine(s)}</li>
                  ))}
                </ul>
              )}
              {r.negated?.length > 0 && (
                <p className="at-negated">
                  <strong>Not reported:</strong> {r.negated.join(", ")}
                </p>
              )}
            </section>
          )}

          {conditions.length > 0 && (
            <section className="at-card at-card-conditions" aria-label="Possible conditions">
              <div className="at-card-head">
                <span className="at-card-icon" aria-hidden>
                  🔎
                </span>
                <h3 className="at-card-title">Possible conditions</h3>
              </div>
              <p className="at-card-lead">
                Based on current information (not a diagnosis):
              </p>
              <ul className="at-list at-conditions-list">
                {conditions.map((c, i) => (
                  <li key={i}>{c}</li>
                ))}
              </ul>
            </section>
          )}

          {!collectionOnly && triage && (
            <section
              className={`at-card at-card-outcome triage-${tone}`}
              aria-label="Recommended next step"
            >
              <div className="at-card-head">
                <span className="at-card-icon" aria-hidden>
                  ⚕️
                </span>
                <h3 className="at-card-title">Outcome &amp; next step</h3>
              </div>
              <div className="outcome-badge">
                {triage === "Doctor Consultation"
                  ? "🔴"
                  : triage === "OTC Drug"
                    ? "🟢"
                    : "🟡"}{" "}
                <strong>{triage}</strong>
              </div>
              <p className="at-outcome-lead">{triageLead(triage)}</p>
              <div className="at-meta-row">
                {hasNumericRisk && (
                  <span>Risk score: {(Number(riskScore) * 100).toFixed(0)}%</span>
                )}
                {hasNumericConf && (
                  <span>
                    Model confidence: {(Number(confidence) * 100).toFixed(0)}%
                  </span>
                )}
                {r?.severity && <span>Severity: {r.severity}</span>}
              </div>
            </section>
          )}

          {collectionOnly && (
            <section className="at-card at-card-pending" aria-label="Status">
              <div className="at-card-head">
                <span className="at-card-icon" aria-hidden>
                  ⏳
                </span>
                <h3 className="at-card-title">Still collecting details</h3>
              </div>
              <p className="at-pending-text">
                Full risk scoring runs once we have enough structured symptoms
                and answers. Share more below if asked.
              </p>
            </section>
          )}

          {qs.length > 0 && (
            <section className="at-card at-card-questions" aria-label="Questions for you">
              <div className="at-card-head">
                <span className="at-card-icon at-icon-question" aria-hidden>
                  ❓
                </span>
                <h3 className="at-card-title">Questions for you</h3>
              </div>
              {qs.map((q, i) => (
                <p key={i} className="at-question-line">
                  {q}
                </p>
              ))}
            </section>
          )}

          <p className="at-disclaimer">
            Informational only — not medical advice or a diagnosis. For
            emergencies, call emergency services.
          </p>
        </div>
      </div>
    );
  };

  const renderMessage = (msg) => {
    if (msg.role === "user") {
      return (
        <div key={msg.id} className="chat-row user">
          <div className="chat-bubble user-bubble">
            <p className="chat-text">{msg.text}</p>
          </div>
        </div>
      );
    }

    if (msg.kind === "turn") {
      return renderAssistantTurn(msg);
    }

    return null;
  };

  const isFirstTurn = !result?.session_id && messages.length === 0;
  const sendDisabled =
    busy ||
    (canAnswerFollowUp
      ? !composerText.trim()
      : isFirstTurn
        ? selectedSymptoms.length === 0
        : false);

  return (
    <div className="app chatbot-app">
      <header className="chatbot-nav">
        <div className="chatbot-nav-brand">
          <div className="nav-logo">⚕️</div>
          <div>
            <div className="nav-title">AI Triage</div>
            <div className="nav-sub">Chat assistant</div>
          </div>
        </div>
        <div className="chatbot-nav-actions">
          <button
            type="button"
            className="btn-nav"
            onClick={() => setDrawerOpen(true)}
          >
            Symptoms &amp; details
          </button>
          <button type="button" className="btn-nav btn-nav-ghost" onClick={resetChat}>
            New chat
          </button>
        </div>
      </header>

      <main className="chatbot-main">
        <section className="chat-panel card">
          <div className="chat-panel-head">
            <div className="chat-bot-identity">
              <span className="chat-bot-avatar" aria-hidden>
                ⚕️
              </span>
              <div>
                <h1 className="chat-bot-name">Triage assistant</h1>
                <p className="chat-bot-status">
                  {result?.conversation_round != null && result?.max_rounds != null
                    ? `Round ${result.conversation_round} of ${result.max_rounds}`
                    : "Online — pick symptoms; the assistant will ask for timing & severity"}
                </p>
              </div>
            </div>
          </div>

          <div className="chat-thread" ref={threadRef}>
            {messages.length === 0 && !loading && (
              <>
                <div className="chat-row assistant">
                  <div className="chat-avatar">⚕️</div>
                  <div className="chat-bubble assistant-bubble chat-intro">
                    <p className="chat-text">
                      Hi — I’m your triage assistant.{" "}
                      <strong>Tap the symptoms that apply</strong> (only these
                      options are supported). Set age in{" "}
                      <strong>Symptoms &amp; details</strong> if you like, then{" "}
                      <strong>Send</strong>. I’ll ask how long it’s been and how
                      bad it feels before we finish triage.
                    </p>
                  </div>
                </div>
                <div className="chat-symptom-picker" aria-label="Choose symptoms">
                  <p className="picker-label">Symptoms you can select</p>
                  <div className="symptom-pill-grid">
                    {ALLOWED_SYMPTOMS_FLAT.map((sym) => (
                      <button
                        key={sym}
                        type="button"
                        className={`symptom-pill ${selectedSymptoms.includes(sym) ? "selected" : ""}`}
                        onClick={() => toggleSymptom(sym)}
                      >
                        {sym}
                      </button>
                    ))}
                  </div>
                </div>
              </>
            )}

            {messages.map((m) => renderMessage(m))}

            {loading && (
              <div className="chat-row assistant typing-row">
                <div className="chat-avatar">⚕️</div>
                <div className="typing-dots" aria-live="polite">
                  <span />
                  <span />
                  <span />
                </div>
              </div>
            )}

            {followUpLoading && (
              <div className="chat-row assistant typing-row">
                <div className="chat-avatar">⚕️</div>
                <div className="typing-dots" aria-live="polite">
                  <span />
                  <span />
                  <span />
                </div>
              </div>
            )}
          </div>

          {(error || followUpError) && (
            <div className="composer-banner-error">
              {error || followUpError}
            </div>
          )}

          <div className="chat-composer chat-composer-sticky">
            <div className="composer-inner">
              <textarea
                className="composer-input"
                rows={2}
                placeholder={
                  canAnswerFollowUp
                    ? "Reply to the assistant…"
                    : result && !canAnswerFollowUp
                      ? "This turn is done — New chat to start over, or wait for a follow-up."
                      : "Optional notes (timing, context) — symptoms must be chosen above…"
                }
                value={composerText}
                onChange={(e) => setComposerText(e.target.value)}
                disabled={!composerEnabled}
                onKeyDown={(e) => {
                  if (
                    e.key === "Enter" &&
                    !e.shiftKey &&
                    composerEnabled &&
                    !sendDisabled
                  ) {
                    e.preventDefault();
                    handleComposerSend();
                  }
                }}
              />
              <button
                type="button"
                className="composer-send"
                onClick={handleComposerSend}
                disabled={sendDisabled}
              >
                {loading ? "…" : followUpLoading ? "…" : "Send"}
              </button>
            </div>
            <p className="composer-hint">
              {canAnswerFollowUp
                ? "Enter to send · Shift+Enter for new line"
                : messages.length === 0
                  ? "Select symptoms, then Send. Don’t type duration/severity here — the assistant will ask."
                  : !canAnswerFollowUp && messages.length > 0
                    ? "Start a new chat to describe different symptoms."
                    : ""}
            </p>
          </div>
        </section>
      </main>

      <div
        className={`drawer-backdrop ${drawerOpen ? "open" : ""}`}
        aria-hidden={!drawerOpen}
        onClick={() => setDrawerOpen(false)}
      />

      <aside className={`symptom-drawer card ${drawerOpen ? "open" : ""}`}>
        <div className="drawer-head">
          <div className="drawer-head-text">
            <h2>Symptoms &amp; details</h2>
            <p className="drawer-sub">
              Same symptom list as the chat. Duration and severity are not filled
              here — the assistant asks you in chat. Age and gender help the
              model.
            </p>
          </div>
          <button
            type="button"
            className="drawer-close"
            onClick={() => setDrawerOpen(false)}
            aria-label="Close"
          >
            ×
          </button>
        </div>

        <div className="drawer-scroll">
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
              <select value={gender} onChange={(e) => setGender(e.target.value)}>
                <option value="male">Male</option>
                <option value="female">Female</option>
              </select>
            </label>
          </div>

          {error && <div className="error">{error}</div>}

          <div className="drawer-actions">
            <button
              type="button"
              className="btn-secondary"
              onClick={applyChipsToComposer}
              disabled={selectedSymptoms.length === 0}
            >
              Paste symptoms into notes
            </button>
            <button
              type="button"
              className="btn-analyze"
              onClick={handlePanelSend}
              disabled={loading || selectedSymptoms.length === 0}
            >
              {loading ? "Sending…" : "Send from panel"}
            </button>
          </div>
          {selectedSymptoms.length > 0 && (
            <p className="selected-count">
              {selectedSymptoms.length} symptom(s) selected
            </p>
          )}
        </div>
      </aside>
    </div>
  );
}

export default App;
