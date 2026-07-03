"""Runtime Midlifing knowledge delivery tests."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import create_app
from brain.runtime_knowledge_store import load_runtime_knowledge_store
from scripts.export_midlifing_runtime_index import export_runtime_index


def test_runtime_export_excludes_raw_paths_and_full_transcripts(tmp_path: Path) -> None:
    """Confirm runtime export contains only derived deployment knowledge."""
    knowledge_dir = write_runtime_fixture(tmp_path)
    output_path = knowledge_dir / "runtime_index.json"

    result = export_runtime_index(knowledge_dir, output_path)

    exported_text = output_path.read_text(encoding="utf-8")
    exported = json.loads(exported_text)
    assert result["episode_count"] == 1
    assert result["chunk_count"] == 1
    assert "transcripts/" not in exported_text
    assert ".mp3" not in exported_text
    assert "source_transcript_path" not in exported_text
    assert "raw transcript sentence 1 raw transcript sentence 2 raw transcript sentence 3" not in exported_text
    assert exported["episodes"][0]["summary"]["summary"] == "A compact derived summary."


def test_runtime_store_loads_expected_counts(tmp_path: Path) -> None:
    """Confirm runtime store exposes read-only counts and retriever records."""
    knowledge_dir = write_runtime_fixture(tmp_path)
    output_path = knowledge_dir / "runtime_index.json"
    export_runtime_index(knowledge_dir, output_path)

    store = load_runtime_knowledge_store(output_path)

    assert store.status() == {
        "knowledge_loaded": True,
        "indexed_episodes": 1,
        "summaries": 1,
        "retrieval_chunks": 1,
        "source": "runtime_index",
    }
    assert len(store.summaries) == 1
    assert len(store.chunks) == 1


def test_retrieval_endpoint_returns_bounded_context(tmp_path: Path, monkeypatch) -> None:
    """Confirm backend returns one summary and at most three chunks without debug data."""
    knowledge_dir = write_runtime_fixture(tmp_path)
    output_path = knowledge_dir / "runtime_index.json"
    export_runtime_index(knowledge_dir, output_path)
    monkeypatch.setenv("MAYA_KNOWLEDGE_INDEX_PATH", str(output_path))
    client = TestClient(create_app())

    response = client.post(
        "/maya/retrieve-context",
        json={"utterance": "technology nostalgia", "rolling_context": ["walkman memory"]},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["knowledge_loaded"] is True
    assert payload["summary_count"] <= 1
    assert payload["chunk_count"] <= 3
    assert "Relevant Midlifing background" in payload["context_block"]
    assert "score_components" not in payload["context_block"]
    assert "source_transcript_path" not in payload["context_block"]


def test_missing_runtime_store_fails_safely(tmp_path: Path) -> None:
    """Confirm absent runtime index keeps Maya usable."""
    store = load_runtime_knowledge_store(tmp_path / "missing.json")

    assert store.status() == {
        "knowledge_loaded": False,
        "indexed_episodes": 0,
        "summaries": 0,
        "retrieval_chunks": 0,
        "source": "none",
    }


def test_browser_context_update_guards_are_present() -> None:
    """Confirm browser-side retrieval is debounced and skips unchanged context."""
    source = Path("frontend/src/sessionBehaviour.ts").read_text(encoding="utf-8")

    assert "shouldSendContextUpdate(lastContextBlock, contextBlock)" in source
    assert "RETRIEVAL_DEBOUNCE_MS" in source
    assert "if (mayaIsResponding" in source
    assert '"session.update"' in source
    assert "summary_count" in source
    assert "chunk_count" in source


def test_browser_context_update_is_bounded_to_one_summary_and_three_chunks() -> None:
    """Confirm frontend consumes bounded backend payload rather than raw knowledge."""
    source = Path("frontend/src/sessionBehaviour.ts").read_text(encoding="utf-8")

    assert "context_block" in source
    assert "score_components" not in source
    assert "source_transcript_path" not in source


def write_runtime_fixture(tmp_path: Path) -> Path:
    """Create a minimal derived knowledge fixture."""
    knowledge_dir = tmp_path / "midlifing"
    summaries_dir = knowledge_dir / "summaries"
    chunks_dir = knowledge_dir / "chunks"
    summaries_dir.mkdir(parents=True)
    chunks_dir.mkdir()
    episode_id = "episode-289-walkman"
    transcript_path = knowledge_dir / "transcripts" / f"{episode_id}.txt"
    manifest = {
        "episodes": [
            {
                "episode_number": 289,
                "title": "289: Nothing Has Rocked My World Like the Walkman",
                "publication_date": "2026-06-17",
                "retrieval_tags": ["technology_nostalgia", "memory", "travel"],
                "indexed": True,
                "summary_path": str(summaries_dir / f"{episode_id}.json"),
                "chunks_path": str(chunks_dir / f"{episode_id}.json"),
                "transcript_text_path": str(transcript_path),
                "local_audio_path": ".cache/midlifing/audio/episode-289.mp3",
            }
        ]
    }
    (knowledge_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (summaries_dir / f"{episode_id}.json").write_text(
        json.dumps(
            {
                "episode_id": episode_id,
                "episode_number": 289,
                "title": "289: Nothing Has Rocked My World Like the Walkman",
                "publication_date": "2026-06-17",
                "source_transcript_path": str(transcript_path),
                "summary": "A compact derived summary.",
                "topics": ["walkman", "memory"],
                "people_mentioned": ["Simon", "Lee"],
                "simon_observations": ["Simon remembers old mobile phones."],
                "lee_observations": ["Lee talks about deleting email apps."],
                "recurring_jokes_or_callbacks": [],
                "questions_maya_could_ask": ["What changed around memory?"],
            }
        ),
        encoding="utf-8",
    )
    (chunks_dir / f"{episode_id}.json").write_text(
        json.dumps(
            [
                {
                    "episode_id": episode_id,
                    "episode_number": 289,
                    "title": "289: Nothing Has Rocked My World Like the Walkman",
                    "publication_date": "2026-06-17",
                    "source_transcript_path": str(transcript_path),
                    "chunk_id": f"{episode_id}-0",
                    "text": "Walkman memories and old mobile phones shaped the conversation.",
                    "relevant_topics": ["walkman", "memory"],
                    "start_seconds": None,
                    "end_seconds": None,
                }
            ]
        ),
        encoding="utf-8",
    )
    return knowledge_dir
