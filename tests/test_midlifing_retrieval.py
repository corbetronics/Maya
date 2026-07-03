"""Midlifing retrieval tests."""

import json
from collections import Counter

from brain import CharacterEngine, PromptComposer
from brain.midlifing_retrieval import (
    ContextForMaya,
    MidlifingKnowledgeRetriever,
    RetrievedChunk,
    RetrievedEpisodeSummary,
)


def test_retriever_returns_no_more_than_three_chunks(tmp_path) -> None:
    """Confirm retrieval caps chunk context at three passages."""
    chunks_dir = tmp_path / "chunks"
    summaries_dir = tmp_path / "summaries"
    chunks_dir.mkdir()
    summaries_dir.mkdir()
    chunks = [
        {
            "episode_id": "ep1",
            "title": "Episode One",
            "text": f"music memory change passage {index}",
            "start_seconds": index * 10,
        }
        for index in range(6)
    ]
    (chunks_dir / "ep1.json").write_text(json.dumps(chunks), encoding="utf-8")
    (summaries_dir / "ep1.json").write_text(
        json.dumps(
            {
                "episode_id": "ep1",
                "title": "Episode One",
                "summary": "A concise summary about music and memory.",
                "topics": ["music", "memory"],
            }
        ),
        encoding="utf-8",
    )

    context = MidlifingKnowledgeRetriever(knowledge_dir=tmp_path).retrieve(
        "music memory",
        ("That music memory thing was important.",),
    )

    assert len(context.chunks) <= 3
    assert len(context.chunks) == 2
    assert context.episode_summary is not None
    assert context.episode_summary.title == "Episode One"


def test_technology_nostalgia_selects_walkman_summary() -> None:
    """Confirm query expansion finds the Walkman nostalgia episode."""
    context = MidlifingKnowledgeRetriever().retrieve("technology nostalgia")

    assert context.episode_summary is not None
    assert context.episode_summary.title == "289: Nothing Has Rocked My World Like the Walkman"


def test_academia_and_work_returns_strong_academic_episode() -> None:
    """Confirm academia/work retrieval reaches the strongest local episodes."""
    context = MidlifingKnowledgeRetriever().retrieve("academia and work")

    returned_titles = {chunk.title for chunk in context.chunks}
    assert returned_titles & {
        "291: Maya Got A Job In Cyber",
        "285: The Thing That Sounds Like It Knows What It's Doing",
        "253: The coat has ceased to fit me",
    }


def test_masculinity_and_class_ranks_episode_287_first() -> None:
    """Confirm class is treated as social class for this curated query."""
    context = MidlifingKnowledgeRetriever().retrieve("masculinity and class")

    assert context.chunks
    assert context.chunks[0].title == "287: What Would I Need Protecting From?"
    assert context.episode_summary is not None
    assert context.episode_summary.title == "287: What Would I Need Protecting From?"


def test_maya_job_in_cyber_ranks_episode_291_first() -> None:
    """Confirm title and summary weighting rank the Maya/cyber episode first."""
    context = MidlifingKnowledgeRetriever().retrieve("Maya got a job in cyber")

    assert context.chunks
    assert context.chunks[0].title == "291: Maya Got A Job In Cyber"
    assert context.episode_summary is not None
    assert context.episode_summary.title == "291: Maya Got A Job In Cyber"


def test_retrieval_limits_chunks_per_episode() -> None:
    """Confirm diversity prevents one episode from crowding out all chunks."""
    context = MidlifingKnowledgeRetriever().retrieve("technology nostalgia")

    counts = Counter(chunk.episode_id for chunk in context.chunks)
    assert len(context.chunks) <= 3
    assert all(count <= 2 for count in counts.values())


def test_retrieval_debug_fields_are_available() -> None:
    """Confirm producer/debug score explanations are present but separate."""
    context = MidlifingKnowledgeRetriever().retrieve("AI and expertise")

    assert context.chunks
    assert context.chunks[0].selection_reason
    assert context.chunks[0].score_components
    assert context.episode_summary is not None
    assert context.episode_summary.selection_reason
    assert context.episode_summary.score_components


def test_prompt_uses_retrieved_context_without_full_transcript() -> None:
    """Confirm prompt includes concise retrieved context, not a wholesale transcript."""
    full_transcript = " ".join(f"raw transcript sentence {index}" for index in range(80))
    retrieved_text = "raw transcript sentence 1 raw transcript sentence 2"
    context = ContextForMaya(
        chunks=(
            RetrievedChunk(
                episode_id="ep1",
                title="Episode One",
                text=retrieved_text,
                score=3,
                selection_reason="debug-only reason",
                score_components={"debug": 3.0},
            ),
        ),
        episode_summary=RetrievedEpisodeSummary(
            episode_id="ep1",
            title="Episode One",
        summary="A concise summary about memory.",
        topics=("memory",),
        selection_reason="debug-only summary reason",
        score_components={"debug": 1.0},
        ),
    )
    character = CharacterEngine().create_maya()

    bundle = PromptComposer().compose(character, retrieved_context=context)

    assert "Relevant Midlifing background" in bundle.system_prompt
    assert retrieved_text in bundle.system_prompt
    assert "A concise summary about memory." in bundle.system_prompt
    assert full_transcript not in bundle.system_prompt
    assert "Maya should never recite transcript language" in bundle.system_prompt
    assert "debug-only" not in bundle.system_prompt
