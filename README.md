# Project MAYA

Project MAYA is a production-ready skeleton for a realtime AI podcast guest named Maya. The build includes a FastAPI backend, React TypeScript producer console, Realtime session configuration, Render deployment files, and smoke tests.

## Structure

- `backend/` - FastAPI application and infrastructure.
- `frontend/` - React, TypeScript, and Vite application.
- `brain/` - Domain contracts for future conversation intelligence.
- `docs/` - Architecture notes.
- `tests/` - Smoke tests.

## Backend

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd ..
uvicorn backend.main:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/maya/session-config
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

## Tests

```bash
pip install -r backend/requirements.txt httpx pytest
pytest
```

## Deployment

Render deploys Project Maya as a single Docker web service from `render.yaml`. The Docker image builds the React frontend and serves the compiled app from FastAPI in production.

### Render Steps

1. Push this repository to GitHub.
2. In Render, create a new Blueprint and select this repository's `render.yaml`.
3. Add `OPENAI_API_KEY` as a Render environment variable. Do not commit it to the repository.
4. Add `OPENAI_SAFETY_IDENTIFIER` as a stable, privacy-preserving Render environment variable.
5. Deploy the Blueprint.
6. Open the Render URL and allow microphone access when the browser asks.

Render sets `ENVIRONMENT=production` in `render.yaml`. Local development can configure CORS with `CORS_ORIGINS`, for example:

```bash
CORS_ORIGINS='["http://localhost:5173"]'
```

Production smoke test:

```bash
curl https://YOUR-RENDER-SERVICE.onrender.com/health
```

Expected response:

```json
{"status":"ok"}
```
