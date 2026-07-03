"""Local retrieval over curated Midlifing knowledge artifacts."""

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any

from brain.runtime_knowledge_store import RuntimeKnowledgeStore


DEFAULT_MIDLIFING_DIR = Path(__file__).parent / "knowledge" / "midlifing"
WORD_PATTERN = re.compile(r"[a-z0-9']+")

# Retrieval is intentionally deterministic and local-only. The weights below blend
# chunk-level lexical matches with episode-level metadata so broad producer queries
# do not depend on transcript wording alone.
CHUNK_EXACT_MATCH_WEIGHT = 3.0
CHUNK_EXPANDED_MATCH_WEIGHT = 1.5
CHUNK_TOPIC_MATCH_WEIGHT = 4.0
TITLE_MATCH_WEIGHT = 5.0
SUMMARY_MATCH_WEIGHT = 2.5
EPISODE_TAG_MATCH_WEIGHT = 7.0
LOW_INFORMATION_PENALTY = 2.0
REPEATED_EPISODE_CHUNK_PENALTY = 1.25
SUMMARY_SELECTED_CHUNK_BOOST = 5.0
MAX_CHUNKS_PER_EPISODE = 2
MIN_CHUNK_SCORE = 2.0

LOW_INFORMATION_TERMS = {"job", "work", "class", "technology", "help"}
RETRIEVAL_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "for",
    "from",
    "get",
    "got",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "they",
    "this",
    "to",
    "we",
    "with",
    "you",
}
QUERY_ALIASES = {
    "technology nostalgia": (
        "walkman cassette mobile phones post restante email analogue memory old technology"
    ),
    "academia and work": (
        "university higher education professor promotion panel teaching academic career "
        "employment expertise"
    ),
    "dance and ageing": "performance dancer body bodies visibility rehearsal movement midlife",
    "anxiety in the body": (
        "panic cortisol intrusive thoughts breath heart rate stress reassurance"
    ),
    "masculinity and class": (
        "inherited masculinity aggression social class privilege protection father men"
    ),
    "friendship and asking for help": (
        "support intimacy vulnerability practical help receiving help"
    ),
    "travel and belonging": "home distance place Italy holiday migration return",
    "simon and lee's conversational style": (
        "two friends podcast pleasures absurdities imperfections banter conversational style"
    ),
}


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One relevant local transcript chunk for Maya."""

    episode_id: str
    title: str
    text: str
    start_seconds: float | None = None
    end_seconds: float | None = None
    score: float = 0.0
    selection_reason: str = ""
    score_components: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievedEpisodeSummary:
    """One concise local episode summary for Maya."""

    episode_id: str
    title: str
    summary: str
    topics: tuple[str, ...] = field(default_factory=tuple)
    selection_reason: str = ""
    score_components: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ContextForMaya:
    """Concise retrieved Midlifing context for prompt composition."""

    chunks: tuple[RetrievedChunk, ...] = field(default_factory=tuple)
    episode_summary: RetrievedEpisodeSummary | None = None


@dataclass(frozen=True, slots=True)
class MidlifingKnowledgeRetriever:
    """Retrieves small local context from generated Midlifing chunks and summaries."""

    knowledge_dir: Path = DEFAULT_MIDLIFING_DIR
    max_chunks: int = 3
    runtime_store: RuntimeKnowledgeStore | None = None

    def retrieve(
        self,
        live_topic: str,
        recent_host_utterances: tuple[str, ...] = (),
    ) -> ContextForMaya:
        """Return at most three relevant chunks and one episode summary."""
        query = build_query(" ".join((live_topic, *recent_host_utterances)))
        if not query.exact_terms:
            return ContextForMaya()

        summary_by_episode_id = {
            str(summary.get("episode_id", "")): summary for summary in self._load_summaries()
        }
        tag_by_episode_id = self._load_episode_tags()
        ranked_chunks = sorted(
            (
                chunk
                for chunk in (
                    self._score_chunk(raw_chunk, query, summary_by_episode_id, tag_by_episode_id)
                    for raw_chunk in self._load_chunks()
                )
                if chunk.score >= MIN_CHUNK_SCORE
            ),
            key=lambda chunk: (-chunk.score, chunk.episode_id, chunk.start_seconds or 0.0),
        )
        selected_chunks = tuple(diversify_chunks(ranked_chunks, self.max_chunks))
        summary = self._best_summary(selected_chunks, query, tag_by_episode_id)
        return ContextForMaya(chunks=selected_chunks, episode_summary=summary)

    def _load_chunks(self) -> list[dict[str, Any]]:
        """Load generated chunk JSON files."""
        if self.runtime_store is not None:
            return list(self.runtime_store.chunks)
        chunks_dir = self.knowledge_dir / "chunks"
        if not chunks_dir.exists():
            return []
        chunks: list[dict[str, Any]] = []
        for path in sorted(chunks_dir.glob("*.json")):
            with path.open(encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, list):
                chunks.extend(item for item in loaded if isinstance(item, dict))
            elif isinstance(loaded, dict):
                chunks.append(loaded)
        return chunks

    def _load_summaries(self) -> list[dict[str, Any]]:
        """Load generated summary JSON files."""
        if self.runtime_store is not None:
            return list(self.runtime_store.summaries)
        summaries_dir = self.knowledge_dir / "summaries"
        if not summaries_dir.exists():
            return []
        summaries: list[dict[str, Any]] = []
        for path in sorted(summaries_dir.glob("*.json")):
            with path.open(encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                summaries.append(loaded)
        return summaries

    def _load_episode_tags(self) -> dict[str, tuple[str, ...]]:
        """Load manually curated episode retrieval tags from the manifest."""
        if self.runtime_store is not None:
            return self.runtime_store.tags_by_episode_id
        manifest_path = self.knowledge_dir / "manifest.json"
        if not manifest_path.exists():
            return {}
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        tags_by_id: dict[str, tuple[str, ...]] = {}
        for episode in loaded.get("episodes", []):
            if not isinstance(episode, dict):
                continue
            episode_id = Path(str(episode.get("transcript_text_path", ""))).stem
            tags = tuple(str(tag).lower() for tag in episode.get("retrieval_tags", []) if tag)
            if episode_id:
                tags_by_id[episode_id] = tags
        return tags_by_id

    def _score_chunk(
        self,
        chunk: dict[str, Any],
        query: "RetrievalQuery",
        summary_by_episode_id: dict[str, dict[str, Any]],
        tag_by_episode_id: dict[str, tuple[str, ...]],
    ) -> RetrievedChunk:
        """Convert a raw chunk dict into a scored retrieved chunk."""
        text = str(chunk.get("text", ""))
        episode_id = str(chunk.get("episode_id", ""))
        title = str(chunk.get("title", ""))
        summary = summary_by_episode_id.get(episode_id, {})
        tags = tag_by_episode_id.get(episode_id, ())
        chunk_terms = tokenise(text)
        topic_terms = tokenise(" ".join(str(topic) for topic in chunk.get("relevant_topics", []) if topic))
        title_terms = tokenise(title)
        summary_terms = tokenise(
            " ".join(
                (
                    str(summary.get("summary", "")),
                    " ".join(str(topic) for topic in summary.get("topics", []) if topic),
                )
            )
        )
        tag_terms = tokenise(" ".join(tags).replace("_", " "))
        components = {
            "chunk_exact": CHUNK_EXACT_MATCH_WEIGHT * len(query.exact_terms & chunk_terms),
            "chunk_expanded": CHUNK_EXPANDED_MATCH_WEIGHT * len(query.expanded_terms & chunk_terms),
            "topic": CHUNK_TOPIC_MATCH_WEIGHT * len(query.all_terms & topic_terms),
            "title": TITLE_MATCH_WEIGHT * len(query.all_terms & title_terms),
            "summary": SUMMARY_MATCH_WEIGHT * len(query.all_terms & summary_terms),
            "episode_tag": EPISODE_TAG_MATCH_WEIGHT * len(query.all_terms & tag_terms),
            "low_information_penalty": low_information_penalty(query, chunk_terms | title_terms | tag_terms),
        }
        score = sum(components.values())
        return RetrievedChunk(
            episode_id=episode_id,
            title=title,
            text=text,
            start_seconds=optional_float(chunk.get("start_seconds")),
            end_seconds=optional_float(chunk.get("end_seconds")),
            score=score,
            selection_reason=selection_reason(components),
            score_components=components,
        )

    def _best_summary(
        self,
        chunks: tuple[RetrievedChunk, ...],
        query: "RetrievalQuery",
        tag_by_episode_id: dict[str, tuple[str, ...]],
    ) -> RetrievedEpisodeSummary | None:
        """Return the best summary for selected chunks or query terms."""
        summaries = self._load_summaries()
        if not summaries:
            return None

        selected_episode_ids = {chunk.episode_id for chunk in chunks}
        ranked: list[tuple[float, dict[str, Any], dict[str, float]]] = []
        for summary in summaries:
            summary_episode_id = str(summary.get("episode_id", ""))
            tags = tag_by_episode_id.get(summary_episode_id, ())
            text = " ".join(
                (
                    str(summary.get("title", "")),
                    str(summary.get("summary", "")),
                    " ".join(str(topic) for topic in summary.get("topics", []) if topic),
                )
            )
            title_terms = tokenise(str(summary.get("title", "")))
            summary_terms = tokenise(text)
            tag_terms = tokenise(" ".join(tags).replace("_", " "))
            components = {
                "title": TITLE_MATCH_WEIGHT * len(query.all_terms & title_terms),
                "summary": SUMMARY_MATCH_WEIGHT * len(query.all_terms & summary_terms),
                "episode_tag": EPISODE_TAG_MATCH_WEIGHT * len(query.all_terms & tag_terms),
                "selected_chunk": SUMMARY_SELECTED_CHUNK_BOOST
                if summary_episode_id in selected_episode_ids
                else 0.0,
                "low_information_penalty": low_information_penalty(
                    query,
                    title_terms | summary_terms | tag_terms,
                ),
            }
            score = sum(components.values())
            ranked.append((score, summary, components))

        ranked.sort(key=lambda item: (-item[0], str(item[1].get("episode_id", ""))))
        if ranked[0][0] <= 0:
            return None

        summary = ranked[0][1]
        components = ranked[0][2]
        return RetrievedEpisodeSummary(
            episode_id=str(summary.get("episode_id", "")),
            title=str(summary.get("title", "")),
            summary=str(summary.get("summary", "")),
            topics=tuple(str(topic) for topic in summary.get("topics", []) if topic),
            selection_reason=selection_reason(components),
            score_components=components,
        )


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    """Tokenised query with curated producer-facing expansion terms."""

    exact_terms: set[str]
    expanded_terms: set[str]

    @property
    def all_terms(self) -> set[str]:
        """Return exact and expanded terms."""
        return self.exact_terms | self.expanded_terms


def build_query(text: str) -> RetrievalQuery:
    """Tokenise a query and apply a small curated alias map."""
    normalised = text.lower()
    exact_terms = tokenise(normalised)
    expanded_text_parts = [
        aliases for phrase, aliases in QUERY_ALIASES.items() if phrase in normalised
    ]
    return RetrievalQuery(
        exact_terms=exact_terms,
        expanded_terms=tokenise(" ".join(expanded_text_parts)),
    )


def low_information_penalty(query: RetrievalQuery, matched_terms: set[str]) -> float:
    """Penalise vague matches when no expanded support terms are present."""
    low_matches = query.exact_terms & LOW_INFORMATION_TERMS & matched_terms
    supporting_matches = query.expanded_terms & matched_terms
    if not low_matches or supporting_matches:
        return 0.0
    return -LOW_INFORMATION_PENALTY * len(low_matches)


def selection_reason(components: dict[str, float]) -> str:
    """Summarise positive score components for debug views."""
    positive = [
        name
        for name, value in sorted(components.items(), key=lambda item: (-item[1], item[0]))
        if value > 0
    ]
    return ", ".join(positive[:4]) or "low lexical relevance"


def diversify_chunks(chunks: list[RetrievedChunk], max_chunks: int) -> list[RetrievedChunk]:
    """Select chunks with a cap per episode and a small repeat penalty."""
    selected: list[RetrievedChunk] = []
    selected_counts: dict[str, int] = {}
    candidates = list(chunks)
    while candidates and len(selected) < max_chunks:
        rescored = sorted(
            (
                (
                    chunk.score
                    - REPEATED_EPISODE_CHUNK_PENALTY * selected_counts.get(chunk.episode_id, 0),
                    chunk,
                )
                for chunk in candidates
                if selected_counts.get(chunk.episode_id, 0) < MAX_CHUNKS_PER_EPISODE
            ),
            key=lambda item: (-item[0], item[1].episode_id, item[1].start_seconds or 0.0),
        )
        if not rescored:
            break
        _score, chosen = rescored[0]
        selected.append(chosen)
        selected_counts[chosen.episode_id] = selected_counts.get(chosen.episode_id, 0) + 1
        candidates = [chunk for chunk in candidates if chunk is not chosen]
    return selected


def tokenise(text: str) -> set[str]:
    """Return simple lowercase search terms for local retrieval."""
    return {
        match.group(0)
        for match in WORD_PATTERN.finditer(text.lower())
        if match.group(0) not in RETRIEVAL_STOP_WORDS
    }


def optional_float(value: object) -> float | None:
    """Convert a JSON value to float when possible."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
