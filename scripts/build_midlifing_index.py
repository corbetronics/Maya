"""Build local Midlifing summaries and searchable chunks from transcripts."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Any


DEFAULT_KNOWLEDGE_DIR = Path("brain/knowledge/midlifing")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z']+")


class TextExtractor(HTMLParser):
    """Small HTML-to-text extractor for manifest descriptions."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        """Collect visible text nodes."""
        text = data.strip()
        if text:
            self.parts.append(text)

    def text(self) -> str:
        """Return normalised text."""
        return " ".join(self.parts)


def build_index(knowledge_dir: Path = DEFAULT_KNOWLEDGE_DIR) -> dict[str, Any]:
    """Generate summaries and chunks from manifest-approved local transcripts."""
    manifest_path = knowledge_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    summaries_dir = knowledge_dir / "summaries"
    chunks_dir = knowledge_dir / "chunks"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    indexed_episode_count = 0
    total_chunk_count = 0

    for episode in transcribed_episodes(manifest):
        transcript_path = transcript_path_for_episode(knowledge_dir, episode)
        if transcript_path is None or not transcript_path.exists():
            failures.append(f"Episode {episode.get('episode_number')}: transcript not found")
            continue
        episode_id = transcript_path.stem
        transcript_text = transcript_path.read_text(encoding="utf-8")
        metadata = dict(episode, episode_id=episode_id)
        summary = summarise_transcript(episode_id, transcript_text, metadata)
        chunks = chunk_transcript(episode_id, transcript_text, metadata)
        summary_path = summaries_dir / f"{episode_id}.json"
        chunks_path = chunks_dir / f"{episode_id}.json"
        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        chunks_path.write_text(
            json.dumps(chunks, indent=2) + "\n",
            encoding="utf-8",
        )
        episode.update(
            {
                "indexed": True,
                "summary_path": str(summary_path),
                "chunks_path": str(chunks_path),
                "indexed_at": utc_now_iso(),
            }
        )
        indexed_episode_count += 1
        total_chunk_count += len(chunks)

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "indexed_episode_count": indexed_episode_count,
        "summary_count": count_files(summaries_dir, "*.json"),
        "total_retrieval_chunk_count": total_chunk_count,
        "failures": failures,
    }


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load the Midlifing manifest."""
    if not manifest_path.exists():
        return {"episodes": []}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def transcribed_episodes(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Return manifest entries that are ready for indexing."""
    return [
        episode
        for episode in manifest.get("episodes", [])
        if isinstance(episode, dict) and episode.get("transcribed") is True
    ]


def transcript_path_for_episode(knowledge_dir: Path, episode: dict[str, Any]) -> Path | None:
    """Return an episode transcript path from manifest metadata."""
    configured_path = episode.get("transcript_text_path")
    if isinstance(configured_path, str) and configured_path:
        return Path(configured_path)
    local_audio_path = episode.get("local_audio_path")
    if isinstance(local_audio_path, str) and local_audio_path:
        return knowledge_dir / "transcripts" / f"{Path(local_audio_path).stem}.txt"
    return None


def metadata_for_episode(knowledge_dir: Path, episode_id: str) -> dict[str, Any]:
    """Return manifest metadata for an episode when available."""
    manifest_path = knowledge_dir / "manifest.json"
    if not manifest_path.exists():
        return {"episode_id": episode_id, "title": episode_id}
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    for episode in loaded.get("episodes", []):
        candidates = {
            str(episode.get("episode_number", "")),
            slug(str(episode.get("title", ""))),
            Path(str(episode.get("transcript_text_path", ""))).stem,
        }
        if episode_id in candidates:
            return dict(episode, episode_id=episode_id)
    return {"episode_id": episode_id, "title": episode_id}


def summarise_transcript(
    episode_id: str,
    transcript_text: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Create a concise local heuristic summary artifact."""
    sentences = [sentence.strip() for sentence in SENTENCE_PATTERN.split(transcript_text) if sentence.strip()]
    description_text = clean_description_text(html_to_text(str(metadata.get("description", ""))))
    description_sentences = [
        sentence.strip()
        for sentence in SENTENCE_PATTERN.split(description_text)
        if sentence.strip() and not sentence.strip().lower().startswith("send us fan mail")
    ]
    summary_text = " ".join(description_sentences[:3] or sentences[:3])[:650]
    topics = top_terms(transcript_text, limit=8)
    return {
        "episode_id": episode_id,
        "episode_number": metadata.get("episode_number"),
        "title": str(metadata.get("title", episode_id)),
        "publication_date": metadata.get("publication_date"),
        "source_transcript_path": metadata.get("transcript_text_path"),
        "summary": summary_text,
        "topics": topics,
        "people_mentioned": mentioned_people(transcript_text),
        "simon_observations": observations_for_name(transcript_text, "Simon"),
        "lee_observations": observations_for_name(transcript_text, "Lee"),
        "recurring_jokes_or_callbacks": callback_candidates(transcript_text),
        "questions_maya_could_ask": questions_from_topics(topics),
    }


def chunk_transcript(
    episode_id: str,
    transcript_text: str,
    metadata: dict[str, Any],
    words_per_chunk: int = 90,
) -> list[dict[str, Any]]:
    """Chunk transcript text into small searchable passages."""
    words = transcript_text.split()
    chunks: list[dict[str, Any]] = []
    for index in range(0, len(words), words_per_chunk):
        text = " ".join(words[index : index + words_per_chunk])
        if not text:
            continue
        chunk_number = len(chunks)
        chunks.append(
            {
                "episode_id": episode_id,
                "episode_number": metadata.get("episode_number"),
                "title": str(metadata.get("title", episode_id)),
                "publication_date": metadata.get("publication_date"),
                "source_transcript_path": metadata.get("transcript_text_path"),
                "chunk_id": f"{episode_id}-{chunk_number}",
                "text": text,
                "relevant_topics": relevant_topics(text, metadata),
                "start_seconds": None,
                "end_seconds": None,
            }
        )
    return chunks


def top_terms(text: str, limit: int) -> list[str]:
    """Return simple frequent topic terms."""
    stop_words = {
        "about",
        "also",
        "because",
        "been",
        "come",
        "does",
        "doing",
        "don't",
        "from",
        "going",
        "have",
        "into",
        "it's",
        "just",
        "know",
        "like",
        "mean",
        "more",
        "much",
        "okay",
        "really",
        "right",
        "some",
        "something",
        "that",
        "that's",
        "them",
        "then",
        "there",
        "thing",
        "things",
        "they",
        "think",
        "this",
        "very",
        "want",
        "what",
        "when",
        "which",
        "with",
        "would",
        "yeah",
        "yes",
        "your",
        "were",
    }
    counts: dict[str, int] = {}
    for word in WORD_PATTERN.findall(text.lower()):
        if len(word) < 4 or word in stop_words:
            continue
        counts[word] = counts.get(word, 0) + 1
    return [word for word, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def mentioned_people(text: str) -> list[str]:
    """Return capitalised name-like words mentioned in a transcript."""
    names = sorted(set(re.findall(r"\b[A-Z][a-z]{2,}\b", text)))
    excluded = {"Good", "Hello", "Welcome", "This", "That", "Because", "There", "Exactly"}
    return [name for name in names if name not in excluded][:20]


def observations_for_name(text: str, name: str) -> list[str]:
    """Return short sentences mentioning a host name."""
    observations: list[str] = []
    for sentence in SENTENCE_PATTERN.split(text):
        stripped = sentence.strip()
        if name not in stripped:
            continue
        if len(stripped.split()) > 55:
            stripped = " ".join(stripped.split()[:55]) + "..."
        observations.append(stripped)
    return observations[:5]


def callback_candidates(text: str) -> list[str]:
    """Return simple callback candidates from laughter/joke mentions."""
    return [
        sentence.strip()
        for sentence in SENTENCE_PATTERN.split(text)
        if any(signal in sentence.lower() for signal in ("laugh", "joke", "callback", "again", "as usual"))
    ][:5]


def questions_from_topics(topics: list[str]) -> list[str]:
    """Generate natural question seeds from local topics."""
    return [f"What changed for you around {topic}?" for topic in topics[:5]]


def relevant_topics(chunk_text: str, metadata: dict[str, Any]) -> list[str]:
    """Return compact topic labels relevant to one retrieval chunk."""
    chunk_terms = set(top_terms(chunk_text, limit=8))
    episode_topics = [str(topic) for topic in metadata.get("topics", []) if topic]
    if not episode_topics:
        episode_topics = top_terms(chunk_text, limit=5)
    selected = [topic for topic in episode_topics if topic.lower() in chunk_terms]
    return selected[:5] or top_terms(chunk_text, limit=3)


def html_to_text(value: str) -> str:
    """Convert manifest HTML descriptions into plain text."""
    parser = TextExtractor()
    parser.feed(value)
    return parser.text()


def clean_description_text(value: str) -> str:
    """Remove feed boilerplate from manifest descriptions."""
    cut_markers = (
        "Mentioned",
        "Get in touch",
        "Related links",
        "Related Links",
        "The Midlifing logo",
        "---",
    )
    cleaned = value
    for marker in cut_markers:
        marker_index = cleaned.find(marker)
        if marker_index >= 0:
            cleaned = cleaned[:marker_index]
    return cleaned.strip()


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 form."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def count_files(path: Path, pattern: str) -> int:
    """Count files matching a local artifact pattern."""
    if not path.exists():
        return 0
    return sum(1 for item in path.glob(pattern) if item.is_file())


def slug(value: str) -> str:
    """Create a simple slug for matching generated files to manifest entries."""
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def main() -> None:
    """CLI entrypoint for local index generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge-dir", type=Path, default=DEFAULT_KNOWLEDGE_DIR)
    args = parser.parse_args()
    result = build_index(args.knowledge_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
