"""Export derived Midlifing runtime knowledge without raw transcripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_KNOWLEDGE_DIR = Path("brain/knowledge/midlifing")
DEFAULT_OUTPUT_PATH = DEFAULT_KNOWLEDGE_DIR / "runtime_index.json"
EXCLUDED_KEYS = {
    "audio_size_bytes",
    "chunks_path",
    "description",
    "enclosure_url",
    "local_audio_path",
    "source_audio",
    "source_transcript_path",
    "summary_path",
    "transcript_json_path",
    "transcript_text_path",
}


def export_runtime_index(
    knowledge_dir: Path = DEFAULT_KNOWLEDGE_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """Write a compact runtime index from local derived knowledge artifacts."""
    manifest = json.loads((knowledge_dir / "manifest.json").read_text(encoding="utf-8"))
    summaries_dir = knowledge_dir / "summaries"
    chunks_dir = knowledge_dir / "chunks"
    episodes: list[dict[str, Any]] = []
    chunk_count = 0

    for episode in manifest.get("episodes", []):
        if not isinstance(episode, dict) or episode.get("indexed") is not True:
            continue
        episode_id = Path(str(episode.get("summary_path", ""))).stem
        if not episode_id:
            episode_id = Path(str(episode.get("transcript_text_path", ""))).stem
        summary_path = summaries_dir / f"{episode_id}.json"
        chunks_path = chunks_dir / f"{episode_id}.json"
        if not summary_path.exists() or not chunks_path.exists():
            continue

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        safe_chunks = [runtime_chunk(chunk) for chunk in chunks if isinstance(chunk, dict)]
        episodes.append(
            {
                "episode_id": episode_id,
                "episode_number": episode.get("episode_number"),
                "title": episode.get("title"),
                "publication_date": episode.get("publication_date"),
                "retrieval_tags": list(episode.get("retrieval_tags", [])),
                "summary": runtime_summary(summary),
                "chunks": safe_chunks,
            }
        )
        chunk_count += len(safe_chunks)

    output = {
        "schema_version": 1,
        "source": "derived_midlifing_runtime_index",
        "episodes": episodes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    size_bytes = output_path.stat().st_size
    return {
        "episode_count": len(episodes),
        "chunk_count": chunk_count,
        "output_path": str(output_path),
        "output_size_bytes": size_bytes,
    }


def runtime_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Return only derived summary fields needed at runtime."""
    return {
        "episode_id": summary.get("episode_id"),
        "episode_number": summary.get("episode_number"),
        "title": summary.get("title"),
        "publication_date": summary.get("publication_date"),
        "summary": summary.get("summary"),
        "topics": list(summary.get("topics", [])),
        "people_mentioned": list(summary.get("people_mentioned", [])),
        "simon_observations": list(summary.get("simon_observations", [])),
        "lee_observations": list(summary.get("lee_observations", [])),
        "recurring_jokes_or_callbacks": list(summary.get("recurring_jokes_or_callbacks", [])),
        "questions_maya_could_ask": list(summary.get("questions_maya_could_ask", [])),
    }


def runtime_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    """Return only bounded derived chunk data needed at runtime."""
    return {
        "episode_id": chunk.get("episode_id"),
        "episode_number": chunk.get("episode_number"),
        "title": chunk.get("title"),
        "publication_date": chunk.get("publication_date"),
        "chunk_id": chunk.get("chunk_id"),
        "text": chunk.get("text"),
        "relevant_topics": list(chunk.get("relevant_topics", [])),
        "start_seconds": chunk.get("start_seconds"),
        "end_seconds": chunk.get("end_seconds"),
    }


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge-dir", type=Path, default=DEFAULT_KNOWLEDGE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    print(json.dumps(export_runtime_index(args.knowledge_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
