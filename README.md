# AI-Analyzer — Medical Triage Pipeline

**By BanderSnatch** — An intelligent medical research and triage pipeline: extract symptoms from natural language, map to SNOMED CT, score risk (including RAG over clinical cases), and generate LLM-powered clarification questions. Includes an optional **React (Vite)** chat-style triage UI.

### Contributors

| Name |
|------|
| Nernay Kumar |
| Avinash |
| Divyani |
| Navaneeth Bulayi |
| Manthan Soni |

> **Disclaimer:** This software is for research and educational use. It is **not** a substitute for professional medical advice, diagnosis, or treatment.

---

## Table of contents

1. [What it does](#what-it-does)
2. [Architecture](#architecture)
3. [Repository layout](#repository-layout)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Configuration (environment variables)](#configuration-environment-variables)
7. [Run the backend API](#run-the-backend-api)
8. [RAG triage (optional but recommended)](#rag-triage-optional-but-recommended)
9. [Run the web frontend](#run-the-web-frontend)
10. [API reference](#api-reference)
11. [ML NER (Phase 1)](#ml-ner-phase-1)
12. [Troubleshooting](#troubleshooting)
13. [Further documentation](#further-documentation)

---

## What it does

| Capability | Description |
|------------|-------------|
| **Symptom extraction** | DistilBERT token classification + phrase/RAG-style span enrichment; rules for duration, severity, negation |
| **Ontology** | Maps symptoms toward **SNOMED CT** codes where possible |
| **Features** | Builds numeric/categorical features (counts, red flags, syndromes, demographics) |
| **Risk / triage** | **RAG retrieval** over the SYNAPSE-style case corpus (embeddings + majority vote), with fallbacks |
| **LLM layer** | Clarifying questions, merge of follow-up answers (Groq / OpenAI / Gemini via config) |
| **Sessions** | **MongoDB** when configured; otherwise **in-memory** sessions for multi-turn chat |

Objectives:

1. Extract symptoms precisely from free text  
2. Normalize toward medical ontology (SNOMED)  
3. Capture context (duration, severity, associations)  
4. Infer possible conditions (heuristic + RAG context)  
5. Estimate risk, triage band, and uncertainty  

---

## Architecture

This project is a **layered client–server system**: a **FastAPI** backend runs the clinical NLP and triage pipeline; an optional **React + Vite** frontend provides a chat-style UI. Sessions may be stored in **MongoDB** or **in memory**.

### System context

```
┌─────────────────┐     HTTP/JSON      ┌──────────────────────────────────────────┐
│  Web browser    │  ───────────────►  │  FastAPI (`api/main.py`)                │
│  (React + Vite) │  ◄───────────────  │  • /analyze, /analyze/continue, /health  │
└─────────────────┘                    └──────────────────┬───────────────────────┘
                                                        │
                                                        ▼
                                           ┌────────────────────────┐
                                           │  AIAnalyzerPipeline    │
                                           │  (`src/pipeline.py`)   │
                                           └────────────────────────┘
```

### Processing pipeline (backend)

End-to-end flow for a single analysis request:

| Phase | Component (`src/…`) | Role |
|------|------------------------|------|
| **1 — Extraction** | `extraction/` (`MLNERExtractor`, lexicon, optional `symptom_rag`) | DistilBERT NER spans, phrase/RAG spans, rules for duration, severity, negation |
| **2 — Ontology** | `ontology/` | Map symptoms to **SNOMED CT** where possible |
| **3 — Features** | `features/` | Numeric/categorical features (syndromes, red flags, demographics, severity cues) |
| **4 — Risk / triage** | `risk_model/` + `rag_triage/` | **RAG** over embedded SYNAPSE-like cases (majority vote), with **SYNAPSE** / finetuned / heuristic fallbacks |
| **5 — LLM reasoning** | `llm_reasoning/` | Clarifying questions, merge of follow-up answers (Groq / OpenAI / Gemini via env) |

**Session handling:** `api/main.py` persists turn state via `src/storage` (MongoDB) or in-process memory for local use.

**Frontend (optional):** In development, the Vite dev server proxies `/api/*` to the backend (`frontend/vite.config.js` + `VITE_API_URL`). The UI calls `POST /analyze` and `POST /analyze/continue` for multi-turn triage.

### Linear flow (same pipeline as code)

```
Patient text
    → Phase 1: ML NER + rules (+ optional SYNAPSE phrase RAG)
    → Phase 2: Ontology (SNOMED mapping)
    → Phase 3: Feature builder
    → Phase 4: Risk / triage (RAG over SYNAPSE index or other predictors)
    → Phase 5: LLM clarification & structured response
    → JSON profile + session state
```

**Entry point in code:** `api/main.py` → `src/pipeline.py` → `AIAnalyzerPipeline.run()`.

---

## Repository layout

```
api/              FastAPI application (routes, CORS, sessions)
src/
  extraction/     NER, lexicon, symptom RAG helpers
  ontology/       SNOMED mapping
  features/       Feature vectors for risk model
  risk_model/     Triage predictors + integration with RAG
  llm_reasoning/  LLM prompts (clarify, merge, etc.)
  storage/        Optional MongoDB session store
rag_triage/       RAG index build + SYNAPSE retrieval
frontend/         React + Vite UI (proxies /api → backend)
scripts/          NER training, tests, utilities
data/             Training data, SYNAPSE CSV (see below)
models/           Trained NER weights (typically gitignored, large)
docs/             Additional design / structure notes
```

Full tree: [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md).

**Remote reference:** [github.com/banderGlitch/healthcare](https://github.com/banderGlitch/healthcare)

---

## Prerequisites

- **Python 3.10+** (3.13 used in some dev setups; match your venv)
- **Node.js 18+** and **npm** (for `frontend/`)
- **pip** packages from `requirements.txt` (and ML stack — see [ML NER](#ml-ner-phase-1))
- Optional: **MongoDB Atlas** or local Mongo for persistent sessions
- Optional: **Groq / OpenAI / Google** API keys for LLM features
- Optional: **Hugging Face** access for downloading sentence-transformers models (RAG index / first inference)

---

## Installation

From the **project root** (repository root folder):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install **extra** dependencies for training or full ML stack as needed:

```powershell
pip install -r requirements-ner.txt   # NER training pipeline
```

**Frontend:**

```powershell
cd frontend
npm install
```

---

## Configuration (environment variables)

Create a **`.env`** file in the **project root** (same folder as `api/`). Copy from `.env.example`:

| Variable | Purpose |
|----------|---------|
| `GROQ_API_KEY` | Primary LLM path (Groq OpenAI-compatible API) |
| `OPENAI_API_KEY` | Alternative LLM backend |
| Google GenAI | Used when configured in LLM module (see `src/llm_reasoning`) |
| `mongodb_uri` or `MONGODB_URI` | Persistent chat sessions; if missing or invalid, API uses **in-memory** sessions |

**Never commit real API keys.** `.gitignore` should exclude `.env`.

### Frontend proxy target

Create **`frontend/.env`** (see `frontend/.env.example`):

```env
VITE_API_URL=http://127.0.0.1:8002
```

**Critical:** The port and host **must match** the uvicorn process you start (`--port` and `--host`). If they differ, the dev server will proxy to the wrong place → **502 Bad Gateway** (`ECONNREFUSED`) or **500** from a stale process.

---

## Run the backend API

Always run commands from the **repository root** so `api.main` and `src` resolve correctly.

### Option A — helper script (Windows, port **8002**)

```powershell
.\run_server.ps1
```

or `run_server.bat`. The script frees port **8002** then starts:

`python -m uvicorn api.main:app --host 127.0.0.1 --port 8002`

Health check: [http://127.0.0.1:8002/health](http://127.0.0.1:8002/health)

### Option B — manual uvicorn (any port)

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8005 --reload
```

- **`--reload`**: auto-reload on code changes (development only)  
- Open **API docs**: [http://127.0.0.1:PORT/docs](http://127.0.0.1:8002/docs) (root `/` redirects to `/docs`)

Then set `frontend/.env` `VITE_API_URL` to the **same** port.

---

## RAG triage (optional but recommended)

The risk layer can use **retrieval-augmented triage** over embedded SYNAPSE-like cases.

```powershell
cd rag_triage
pip install -r requirements.txt
python scripts/build_index.py
```

Details: [rag_triage/README.md](rag_triage/README.md).  
Default data path: `data/SYNAPSE_An Expert Annotated Dataset of Patient symptoms and Demographics.csv` (configure via `rag_triage` config / env if you move the file).

Symptom phrase expansion may also read the same CSV via `src/extraction/symptom_rag.py` (graceful if file missing).

---

## Run the web frontend

1. Start the **API** first (see above).  
2. Ensure **`frontend/.env`** → `VITE_API_URL` matches the API port.  
3. Start Vite:

```powershell
cd frontend
npm run dev
```

- Default Vite URL: **http://localhost:5173**  
- If 5173 is busy, Vite picks **5174**, **5175**, etc. — the UI URL changes; the API URL in `.env` does **not** (only the backend port matters for the proxy).

In **development**, the app calls paths like `/api/analyze`; Vite **proxies** `/api` → `VITE_API_URL` (see `frontend/vite.config.js`).

For **production** builds, the browser calls `VITE_API_URL` directly; configure CORS on the API if the origin differs.

---

## API reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Redirects to `/docs` |
| `GET` | `/health` | Liveness: `status`, `llm_configured`, `ner_mode`, `session_store`, `version` |
| `POST` | `/analyze` | Full or collection-phase pipeline; creates a **session** |
| `POST` | `/analyze/continue` | Continue with `session_id` + patient `answers`; merges narrative and re-runs |
| `POST` | `/extract` | Phase 1 only: NER extraction JSON |
| `DELETE` | `/session/{session_id}` | Clear server-side session |

### Example: analyze

```bash
curl -X POST http://127.0.0.1:8002/analyze ^
  -H "Content-Type: application/json" ^
  -d "{\"conversation\": \"I have had fever for 3 days and a mild headache.\", \"demographics\": {\"age\": 30, \"gender\": \"male\"}}"
```

### Example: extract only

```bash
curl -X POST http://127.0.0.1:8002/extract ^
  -H "Content-Type: application/json" ^
  -d "{\"conversation\": \"I have had fever for 3 days.\"}"
```

---

## ML NER (Phase 1)

Phase 1 uses a **fine-tuned DistilBERT** for token classification (e.g. B-SYMPTOM / I-SYMPTOM), plus rules for negation and structured fields.

- Trained weights live under **`models/`** (often **gitignored**, ~large).  
- To train: see `scripts/train_ner.py`, `scripts/prepare_ner_data.py`, and `scripts/README_NER.md` if present.

Without local weights, behavior depends on fallbacks implemented in `MLNERExtractor` (may download base models from Hugging Face on first run).

---

## Troubleshooting

| Symptom | Likely cause | What to do |
|--------|----------------|------------|
| **502** / `ECONNREFUSED 127.0.0.1:PORT` in Vite terminal | Nothing listening on `VITE_API_URL` | Start uvicorn; match **exact** port in `frontend/.env`; restart `npm run dev` after editing `.env` |
| **500** on `/api/analyze` | Backend exception | Check uvicorn logs for traceback; verify RAG index / env; ensure code is up to date |
| **CORS** errors in browser | Frontend origin not allowed | Add your dev origin (e.g. `http://localhost:5174`) to `allow_origins` in `api/main.py` if previewing on a non-default port |
| **WinError 10013** / bind denied | Port reserved or permissions | Choose another `--port` or free the port (admin / Hyper-V exclusions on Windows) |
| Slow first request | Model downloads (DistilBERT, sentence-transformers) | Wait once; optional HF token for rate limits |
| MongoDB errors in logs | Invalid URI or network | Fix `mongodb_uri` or ignore — API falls back to **memory** sessions |

---

## Further documentation

- [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) — Directory layout (see `docs/` in your checkout)  
- [rag_triage/README.md](rag_triage/README.md) — RAG index build and usage  
- [frontend/README.md](frontend/README.md) — UI-specific notes  
- [scripts/README.md](scripts/README.md) — Script index  
- [CHANGELOG.md](CHANGELOG.md) — Version history  

---

## License / attribution

Refer to repository settings and upstream references (e.g. project inspiration cited in `requirements.txt` comments).
