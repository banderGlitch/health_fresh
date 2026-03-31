# AI Triage Frontend

React frontend for the RAG-powered symptom triage system. Uses only **model-optimized symptoms** from the SYNAPSE dataset.

## Run

1. **Start the API** (from project root):
   ```bat
   run_server.bat
   ```

2. **Start the frontend**:
   ```bat
   cd frontend
   npm install
   npm run dev
   ```

3. Open http://localhost:5173

## Features

- Curated symptom list (model-good cases only)
- Duration, severity, age, gender
- Calls `POST /analyze` API
- OTC vs Doctor triage result
- Possible conditions & follow-up questions

## API

The frontend proxies `/api` to `http://127.0.0.1:8002` in development.
