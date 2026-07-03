"""Read-only runtime Midlifing knowledge store."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)
DEFAULT_RENDER_RUNTIME_INDEX_PATH = Path("/var/data/maya/runtime_index.json")
DEFAULT_LOCAL_RUNTIME_INDEX_PATH = Path(__file__).parent / "knowledge" / "midlifing" / "runtime_index.json"


@dataclass(frozen=True, slots=True)
class RuntimeKnowledgeStore:
    """Read-only runtime knowledge loaded from a compact exported JSON file."""

    path: Path | None
    episodes: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    source: str = "none"

    @property
    def knowledge_loaded(self) -> bool:
        """Return whether runtime knowledge is available."""
        return bool(self.episodes)

    @property
    def summaries(self) -> tuple[dict[str, Any], ...]:
        """Return summary records in retriever-compatible shape."""
        return tuple(
            dict(episode.get("summary", {}), episode_id=episode.get("episode_id"))
            for episode in self.episodes
            if isinstance(episode.get("summary"), dict)
        )

    @property
    def chunks(self) -> tuple[dict[str, Any], ...]:
        """Return chunk records in retriever-compatible shape."""
        chunks: list[dict[str, Any]] = []
        for episode in self.episodes:
            for chunk in episode.get("chunks", []):
                if isinstance(chunk, dict):
                    chunks.append(chunk)
        return tuple(chunks)

    @property
    def tags_by_episode_id(self) -> dict[str, tuple[str, ...]]:
        """Return retrieval tags keyed by episode id."""
        return {
            str(episode.get("episode_id", "")): tuple(
                str(tag).lower() for tag in episode.get("retrieval_tags", []) if tag
            )
            for episode in self.episodes
            if episode.get("episode_id")
        }

    def status(self) -> dict[str, int | bool | str]:
        """Return producer-safe runtime knowledge counts."""
        return {
            "knowledge_loaded": self.knowledge_loaded,
            "indexed_episodes": len(self.episodes),
            "summaries": len(self.summaries),
            "retrieval_chunks": len(self.chunks),
            "source": self.source,
        }


def load_runtime_knowledge_store(path: Path | None = None) -> RuntimeKnowledgeStore:
    """Load runtime knowledge from configured or deployed paths, safely."""
    resolved_path = path or configured_runtime_index_path()
    if resolved_path is None or not resolved_path.exists():
        LOGGER.warning("Midlifing runtime index is not available; Maya will run without it.")
        return RuntimeKnowledgeStore(path=resolved_path)

    loaded = json.loads(resolved_path.read_text(encoding="utf-8"))
    episodes = tuple(
        episode for episode in loaded.get("episodes", []) if isinstance(episode, dict)
    )
    return RuntimeKnowledgeStore(path=resolved_path, episodes=episodes, source="runtime_index")


def configured_runtime_index_path() -> Path | None:
    """Return the best configured runtime index path."""
    env_path = os.environ.get("MAYA_KNOWLEDGE_INDEX_PATH")
    if env_path:
        return Path(env_path)
    if DEFAULT_RENDER_RUNTIME_INDEX_PATH.exists():
        return DEFAULT_RENDER_RUNTIME_INDEX_PATH
    return DEFAULT_LOCAL_RUNTIME_INDEX_PATH
