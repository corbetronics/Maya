"""Top-level FastAPI endpoints for Project MAYA."""

import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

from fastapi import FastAPI, HTTPException
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


def create_app(frontend_dist: Path | None = None) -> FastAPI:
    """Create the Project MAYA API without opening realtime network connections."""
    app = FastAPI(title="Project MAYA")
    configured_frontend_dist = frontend_dist or DEFAULT_FRONTEND_DIST
    runtime_store = load_runtime_knowledge_store()
    retriever = MidlifingKnowledgeRetriever(runtime_store=runtime_store)
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


app = create_app()
