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

## Local Midlifing Knowledge Ingestion

The Midlifing ingestion pipeline is manual and local. It is for Maya's internal background and live retrieval only, not public transcript publishing. Do not run it on Render and do not add background jobs for it.

Set the official RSS feed URL locally:

```bash
export MIDLIFING_RSS_URL="https://example.com/official-midlifing-rss.xml"
```

Discover and select episodes:

```bash
python scripts/fetch_midlifing_episodes.py --episode 1 --episode 2 --max-count 2
```

Optional local audio download, cached outside git:

```bash
python scripts/fetch_midlifing_episodes.py --episode 1 --download --max-count 1
```

Transcribe local audio with OpenAI from your machine:

```bash
export OPENAI_API_KEY="..."
python scripts/transcribe_midlifing_episodes.py .cache/midlifing/audio/example.mp3
```

The transcription script defaults to `gpt-4o-mini-transcribe`. Override it when needed:

```bash
python scripts/transcribe_midlifing_episodes.py --model gpt-4o-transcribe .cache/midlifing/audio/example.mp3
```

Build local summaries and searchable chunks:

```bash
python scripts/build_midlifing_index.py
```

Export the private deployment runtime index after local indexing:

```bash
python scripts/export_midlifing_runtime_index.py
```

This writes `brain/knowledge/midlifing/runtime_index.json`, which is ignored by Git.
Upload that file to private persistent storage for deployment, for example a private
Render disk at `/var/data/maya/runtime_index.json`, or set `MAYA_KNOWLEDGE_INDEX_PATH`
to another private path. Do not commit raw audio, full transcripts, or the runtime
index to the repository.

Transcripts can contain errors and must not be treated as verbatim source material. Maya receives only concise retrieved context during conversation and must not recite transcripts, claim exact recall unless supported, or reveal internal notes.

## Deployment

Render deploys Project Maya as a single Docker web service from `render.yaml`. The Docker image builds the React frontend and serves the compiled app from FastAPI in production.

### Render Steps

1. Push this repository to GitHub.
2. In Render, create a new Blueprint and select this repository's `render.yaml`.
3. Add `OPENAI_API_KEY` as a Render environment variable. Do not commit it to the repository.
4. Add `OPENAI_SAFETY_IDENTIFIER` as a stable, privacy-preserving Render environment variable.
5. Upload `runtime_index.json` to a private Render disk at `/var/data/maya/runtime_index.json`, or set `MAYA_KNOWLEDGE_INDEX_PATH` to its private location.
6. Deploy the Blueprint.
7. Open the Render URL and allow microphone access when the browser asks.

If no runtime index is present, Maya still starts and the backend logs a warning; live Midlifing retrieval simply returns no background context.

### One-Time Runtime Index Upload

For a private one-use upload to a running backend, set an admin token in Render:

```bash
MAYA_ADMIN_UPLOAD_TOKEN="a-long-random-one-time-token"
```

Then upload the local derived runtime index:

```bash
curl -X POST "https://YOUR-RENDER-SERVICE.onrender.com/maya/admin/upload-runtime-index" \
  -H "X-Maya-Admin-Token: $MAYA_ADMIN_UPLOAD_TOKEN" \
  -H "Content-Type: application/json" \
  --data-binary @brain/knowledge/midlifing/runtime_index.json
```

The backend accepts one JSON runtime index up to 2 MB, validates that it contains no raw transcript paths, audio paths, transcript fields, or debug score fields, writes it atomically to `/var/data/maya/runtime_index.json`, and reloads retrieval immediately. Remove or rotate `MAYA_ADMIN_UPLOAD_TOKEN` after the upload.

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
