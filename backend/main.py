"""Top-level FastAPI endpoints for Project MAYA."""

import json
import os
from pathlib import Path
import tempfile
from typing import Any
from urllib.error import HTTPError

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.openai_sessions import (
    MissingOpenAIAPIKeyError,
    MissingOpenAISafetyIdentifierError,
    OpenAISessionClient,
)
from backend.realtime_session import build_session_config
from brain import CharacterEngine
from brain.midlifing_retrieval import MidlifingKnowledgeRetriever
from brain.runtime_knowledge_store import load_runtime_knowledge_store


DEFAULT_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
RUNTIME_INDEX_UPLOAD_PATH = Path("/var/data/maya/runtime_index.json")
MAX_RUNTIME_INDEX_UPLOAD_BYTES = 2 * 1024 * 1024
FORBIDDEN_RUNTIME_INDEX_KEYS = {
    "audio_path",
    "debug_score",
    "debug_scores",
    "local_audio_path",
    "raw_transcript",
    "score_components",
    "source_audio",
    "source_transcript_path",
    "transcript",
    "transcript_json_path",
    "transcript_text_path",
}
FORBIDDEN_RUNTIME_INDEX_VALUE_MARKERS = (
    ".mp3",
    ".wav",
    ".m4a",
    "transcripts/",
    "audio/",
)


def create_app(frontend_dist: Path | None = None) -> FastAPI:
    """Create the Project MAYA API without opening realtime network connections."""
    app = FastAPI(title="Project MAYA")
    configured_frontend_dist = frontend_dist or DEFAULT_FRONTEND_DIST
    runtime_store = load_runtime_knowledge_store()
    retriever = MidlifingKnowledgeRetriever(runtime_store=runtime_store)
    runtime_index_uploaded = False
    configure_cors(app)

    @app.get("/health")
    def health() -> dict[str, str]:
        """Return service health."""
        return {"status": "ok"}

    @app.get("/maya/session-config")
    def maya_session_config() -> dict[str, Any]:
        """Return the local Realtime session config for Maya."""
        maya = CharacterEngine().create_maya()
        return build_session_config(maya)

    @app.post("/maya/ephemeral-session")
    def maya_ephemeral_session() -> dict[str, Any]:
        """Create an ephemeral OpenAI Realtime session for browser clients."""
        maya = CharacterEngine().create_maya()
        session_config = build_session_config(maya)
        try:
            return OpenAISessionClient().create_ephemeral_session(session_config)
        except (MissingOpenAIAPIKeyError, MissingOpenAISafetyIdentifierError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except HTTPError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"OpenAI rejected ephemeral session creation: {exc.code}",
            ) from exc

    @app.get("/maya/knowledge-status")
    def maya_knowledge_status() -> dict[str, int | bool | str]:
        """Return local Midlifing knowledge artifact counts for producer UI."""
        return runtime_store.status()

    @app.post("/maya/retrieve-context")
    def maya_retrieve_context(payload: dict[str, Any]) -> dict[str, Any]:
        """Return bounded Midlifing context for the current host turn."""
        utterance = str(payload.get("utterance", "")).strip()
        rolling_context = tuple(
            str(item).strip()
            for item in payload.get("rolling_context", [])
            if isinstance(item, str) and item.strip()
        )[-4:]
        context = retriever.retrieve(utterance, rolling_context)
        context_block = render_retrieved_context_block(context)
        return {
            "knowledge_loaded": runtime_store.knowledge_loaded,
            "context_block": context_block,
            "summary_count": 1 if context.episode_summary else 0,
            "chunk_count": len(context.chunks),
        }

    @app.post("/maya/admin/upload-runtime-index")
    async def maya_admin_upload_runtime_index(
        request: Request,
        x_maya_admin_token: str | None = Header(default=None),
    ) -> dict[str, int]:
        """One-use private runtime index upload endpoint for deployment setup."""
        nonlocal runtime_store, retriever, runtime_index_uploaded
        expected_token = os.environ.get("MAYA_ADMIN_UPLOAD_TOKEN")
        if not expected_token or x_maya_admin_token != expected_token:
            raise HTTPException(status_code=401, detail="Unauthorized")
        if runtime_index_uploaded:
            raise HTTPException(status_code=409, detail="Runtime index already uploaded")

        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_RUNTIME_INDEX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Runtime index upload is too large")
        body = await request.body()
        if len(body) > MAX_RUNTIME_INDEX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Runtime index upload is too large")
        try:
            runtime_index = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail="Invalid runtime index JSON") from exc

        counts = validate_runtime_index_payload(runtime_index)
        write_runtime_index_atomically(runtime_index, RUNTIME_INDEX_UPLOAD_PATH)
        runtime_store = load_runtime_knowledge_store(RUNTIME_INDEX_UPLOAD_PATH)
        retriever = MidlifingKnowledgeRetriever(runtime_store=runtime_store)
        runtime_index_uploaded = True
        return {
            "episode_count": counts["episode_count"],
            "summary_count": counts["summary_count"],
            "chunk_count": counts["chunk_count"],
            "file_size": RUNTIME_INDEX_UPLOAD_PATH.stat().st_size,
        }

    mount_frontend(app, configured_frontend_dist)
    return app


def configure_cors(app: FastAPI) -> None:
    """Configure CORS for local development while keeping production same-origin."""
    allowed_origins = configured_cors_origins()
    if not allowed_origins:
        return

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )


def configured_cors_origins() -> list[str]:
    """Return configured local development CORS origins."""
    configured_origins = os.environ.get("CORS_ORIGINS")
    if configured_origins:
        try:
            parsed_origins = json.loads(configured_origins)
        except json.JSONDecodeError:
            return [origin.strip() for origin in configured_origins.split(",") if origin.strip()]
        if isinstance(parsed_origins, list):
            return [origin for origin in parsed_origins if isinstance(origin, str)]

    if os.environ.get("ENVIRONMENT", "development").lower() == "development":
        return ["http://localhost:5173", "http://127.0.0.1:5173"]

    return []


def mount_frontend(app: FastAPI, frontend_dist: Path) -> None:
    """Serve the built React app when frontend assets are available."""
    index_path = frontend_dist / "index.html"
    assets_path = frontend_dist / "assets"

    if assets_path.exists():
        app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend_app(path: str) -> FileResponse:
        """Serve React index.html for non-API routes."""
        if path == "health" or path.startswith("maya/") or path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Frontend build not found")
        return FileResponse(index_path)


def render_retrieved_context_block(context: Any) -> str:
    """Render bounded context for a Realtime session.update."""
    lines = [
        "Relevant Midlifing background — use only if natural.",
        "Do not recite this wording. Do not mention private notes.",
    ]
    if context.episode_summary is None and not context.chunks:
        lines.append("No relevant Midlifing background for this turn.")
        return "\n".join(lines)

    if context.episode_summary is not None:
        lines.extend(
            (
                "",
                "Episode summary:",
                f"- {context.episode_summary.title}: {bounded_text(context.episode_summary.summary, 420)}",
            )
        )
    if context.chunks:
        lines.extend(("", "Relevant details:"))
        for chunk in context.chunks[:3]:
            lines.append(f"- {chunk.title}: {bounded_text(chunk.text, 300)}")
    return "\n".join(lines)


def bounded_text(value: str, limit: int) -> str:
    """Return compact single-line text."""
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def validate_runtime_index_payload(payload: Any) -> dict[str, int]:
    """Validate a compact runtime index and return safe counts."""
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Runtime index must be a JSON object")
    assert_runtime_index_has_no_forbidden_fields(payload)
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise HTTPException(status_code=400, detail="Runtime index must contain episodes")

    summary_count = 0
    chunk_count = 0
    for episode in episodes:
        if not isinstance(episode, dict):
            raise HTTPException(status_code=400, detail="Runtime index episode is invalid")
        if not isinstance(episode.get("summary"), dict):
            raise HTTPException(status_code=400, detail="Runtime index episode summary is missing")
        chunks = episode.get("chunks")
        if not isinstance(chunks, list):
            raise HTTPException(status_code=400, detail="Runtime index episode chunks are missing")
        summary_count += 1
        chunk_count += len([chunk for chunk in chunks if isinstance(chunk, dict)])
    if chunk_count == 0:
        raise HTTPException(status_code=400, detail="Runtime index must contain chunks")
    return {
        "episode_count": len(episodes),
        "summary_count": summary_count,
        "chunk_count": chunk_count,
    }


def assert_runtime_index_has_no_forbidden_fields(value: Any) -> None:
    """Reject raw transcript, audio, and debug fields in uploaded runtime knowledge."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_RUNTIME_INDEX_KEYS:
                raise HTTPException(status_code=400, detail=f"Forbidden runtime index field: {key}")
            assert_runtime_index_has_no_forbidden_fields(item)
    elif isinstance(value, list):
        for item in value:
            assert_runtime_index_has_no_forbidden_fields(item)
    elif isinstance(value, str):
        lowered = value.lower()
        if any(marker in lowered for marker in FORBIDDEN_RUNTIME_INDEX_VALUE_MARKERS):
            raise HTTPException(status_code=400, detail="Forbidden runtime index value")


def write_runtime_index_atomically(payload: dict[str, Any], target_path: Path) -> None:
    """Write runtime index JSON using an atomic replace in the target directory."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=target_path.parent,
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, target_path)


app = create_app()
