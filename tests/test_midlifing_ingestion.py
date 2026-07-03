"""Midlifing ingestion script tests."""

from io import BytesIO
from datetime import date
import json
from pathlib import Path
from urllib.error import HTTPError

from scripts.fetch_midlifing_episodes import parse_rss, select_episodes
from scripts.build_midlifing_index import build_index
from scripts.transcribe_midlifing_episodes import (
    MAX_UPLOAD_BYTES,
    TranscriptJob,
    filter_jobs_by_episode,
    format_http_error,
    has_valid_transcript,
    jobs_from_manifest,
    needs_duration_splitting,
    needs_splitting,
    print_dry_run,
    run_transcription_batch,
    transcription_form_fields,
)


RSS_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<rss xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd" version="2.0">
  <channel>
    <item>
      <title>Episode 12: Midlife Music</title>
      <link>https://midlifing.example/12</link>
      <pubDate>Tue, 02 Jan 2024 10:00:00 GMT</pubDate>
      <description>Music and memory.</description>
      <itunes:episode>12</itunes:episode>
      <itunes:duration>00:45:00</itunes:duration>
      <enclosure url="https://cdn.example/12.mp3" type="audio/mpeg" />
    </item>
    <item>
      <title>Episode 13: Work</title>
      <link>https://midlifing.example/13</link>
      <pubDate>Tue, 09 Jan 2024 10:00:00 GMT</pubDate>
      <description>Work and change.</description>
      <itunes:episode>13</itunes:episode>
      <itunes:duration>00:44:00</itunes:duration>
      <enclosure url="https://cdn.example/13.mp3" type="audio/mpeg" />
    </item>
  </channel>
</rss>
"""


def test_parse_rss_extracts_episode_metadata() -> None:
    """Confirm RSS metadata and enclosure URLs are parsed."""
    episodes = parse_rss(RSS_FIXTURE)

    assert len(episodes) == 2
    assert episodes[0].episode_number == 12
    assert episodes[0].title == "Episode 12: Midlife Music"
    assert episodes[0].publication_date == "2024-01-02"
    assert episodes[0].duration == "00:45:00"
    assert episodes[0].description == "Music and memory."
    assert episodes[0].enclosure_url == "https://cdn.example/12.mp3"
    assert episodes[0].source_url == "https://midlifing.example/12"


def test_select_episodes_by_number_date_and_manifest(tmp_path) -> None:
    """Confirm explicit, date, and curated manifest selection works."""
    episodes = parse_rss(RSS_FIXTURE)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"episodes": [{"episode_number": 13, "selected": True}]}),
        encoding="utf-8",
    )

    by_number = select_episodes(episodes, episode_numbers={12})
    by_date = select_episodes(episodes, start_date=date(2024, 1, 8), end_date=date(2024, 1, 10))
    by_manifest = select_episodes(episodes, curated_manifest_path=manifest_path)

    assert [episode.episode_number for episode in by_number] == [12]
    assert [episode.episode_number for episode in by_date] == [13]
    assert [episode.episode_number for episode in by_manifest] == [13]
    assert by_manifest[0].selected is True


def test_transcription_jobs_read_local_audio_paths_from_manifest(tmp_path) -> None:
    """Confirm transcription jobs come from selected manifest audio paths."""
    audio_path = tmp_path / "episode-12.mp3"
    audio_path.write_bytes(b"audio")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "episode_number": 12,
                        "title": "Episode 12",
                        "duration": "00:45:00",
                        "selected": True,
                        "local_audio_path": str(audio_path),
                    },
                    {
                        "episode_number": 13,
                        "title": "Episode 13",
                        "duration": "120",
                        "selected": False,
                        "local_audio_path": str(tmp_path / "episode-13.mp3"),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    jobs = jobs_from_manifest(manifest_path)

    assert len(jobs) == 1
    assert jobs[0].episode_number == 12
    assert jobs[0].duration_seconds == 2700
    assert jobs[0].audio_path == audio_path


def test_filter_jobs_by_episode_keeps_requested_episode(tmp_path) -> None:
    """Confirm one-episode mode narrows manifest jobs."""
    jobs = [
        TranscriptJob(12, "Episode 12", 120, tmp_path / "episode-12.mp3"),
        TranscriptJob(13, "Episode 13", 120, tmp_path / "episode-13.mp3"),
    ]

    filtered = filter_jobs_by_episode(jobs, 13)

    assert filtered == [jobs[1]]


def test_valid_transcript_requires_json_and_text_outputs(tmp_path) -> None:
    """Confirm reruns skip only complete transcript pairs."""
    output_dir = tmp_path / "transcripts"
    output_dir.mkdir()
    (output_dir / "episode-12.json").write_text(json.dumps({"text": "hello"}), encoding="utf-8")

    assert has_valid_transcript("episode-12", output_dir) is False

    (output_dir / "episode-12.txt").write_text("hello", encoding="utf-8")

    assert has_valid_transcript("episode-12", output_dir) is True


def test_dry_run_reports_size_and_split_command(tmp_path, capsys) -> None:
    """Confirm dry-run reports oversized audio without calling OpenAI."""
    audio_path = tmp_path / "episode-12.mp3"
    audio_path.write_bytes(b"0" * (MAX_UPLOAD_BYTES + 1))
    job = TranscriptJob(12, "Episode 12", 120, audio_path)

    print_dry_run([job])

    output = capsys.readouterr().out
    assert "Episode 12:" in output
    assert "split_required=true" in output
    assert "ffmpeg command:" in output
    assert str(audio_path) in output


def test_needs_splitting_uses_24_mb_limit(tmp_path) -> None:
    """Confirm files over the OpenAI upload guardrail are split first."""
    audio_path = tmp_path / "episode-12.mp3"
    audio_path.write_bytes(b"0" * (MAX_UPLOAD_BYTES + 1))

    assert needs_splitting(audio_path) is True


def test_gpt_4o_models_split_long_audio_by_duration(tmp_path) -> None:
    """Confirm long 4o transcription inputs are chunked before upload."""
    audio_path = tmp_path / "episode-291.mp3"
    audio_path.write_bytes(b"audio")
    job = TranscriptJob(291, "Episode 291", 1494, audio_path)

    assert needs_duration_splitting(job, "gpt-4o-mini-transcribe") is True
    assert needs_duration_splitting(job, "gpt-4o-transcribe") is True
    assert needs_duration_splitting(job, "whisper-1") is False


def test_http_error_diagnostic_includes_body_without_api_key(tmp_path) -> None:
    """Confirm OpenAI HTTP errors include useful context without credentials."""
    audio_path = tmp_path / "episode-12.mp3"
    audio_path.write_bytes(b"audio")
    job = TranscriptJob(12, "Episode 12", 120, audio_path)
    exc = HTTPError(
        url="https://api.openai.com/v1/audio/transcriptions",
        code=400,
        msg="Bad Request",
        hdrs={},
        fp=BytesIO(b'{"error":{"message":"Invalid file format"}}'),
    )

    message = format_http_error(job, exc)

    assert "episode=12" in message
    assert "filename=episode-12.mp3" in message
    assert "http_status=400" in message
    assert "Invalid file format" in message
    assert "OPENAI_API_KEY" not in message


def test_gpt_4o_mini_transcribe_uses_json_response_format() -> None:
    """Confirm gpt-4o-mini-transcribe avoids verbose-only options."""
    fields = transcription_form_fields("gpt-4o-mini-transcribe")

    assert ("response_format", "json") in fields
    assert ("response_format", "verbose_json") not in fields
    assert not any(name == "timestamp_granularities[]" for name, _value in fields)


def test_gpt_4o_transcribe_uses_json_response_format() -> None:
    """Confirm gpt-4o-transcribe avoids verbose-only options."""
    fields = transcription_form_fields("gpt-4o-transcribe")

    assert ("response_format", "json") in fields
    assert ("response_format", "verbose_json") not in fields
    assert not any(name == "timestamp_granularities[]" for name, _value in fields)


def test_whisper_transcribe_keeps_verbose_json_response_format() -> None:
    """Confirm whisper-1 can still request verbose segment metadata."""
    fields = transcription_form_fields("whisper-1")

    assert ("response_format", "verbose_json") in fields
    assert ("timestamp_granularities[]", "segment") in fields
    assert ("response_format", "json") not in fields


def test_build_index_reads_only_transcribed_manifest_entries(tmp_path) -> None:
    """Confirm indexing writes compact artifacts only for completed transcripts."""
    knowledge_dir = tmp_path / "midlifing"
    transcripts_dir = knowledge_dir / "transcripts"
    transcripts_dir.mkdir(parents=True)
    transcript_path = transcripts_dir / "episode-12-midlife-music.txt"
    transcript_path.write_text(
        "Simon talks about music and memory. Lee remembers a cassette player. "
        "They joke about old headphones and ask what changed about listening.",
        encoding="utf-8",
    )
    skipped_path = transcripts_dir / "episode-13-work.txt"
    skipped_path.write_text("This should not be indexed.", encoding="utf-8")
    manifest_path = knowledge_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "episode_number": 12,
                        "title": "Episode 12: Midlife Music",
                        "publication_date": "2024-01-02",
                        "description": "<p>Simon and Lee talk about music and memory.</p>",
                        "selected": True,
                        "transcribed": True,
                        "transcript_text_path": str(transcript_path),
                    },
                    {
                        "episode_number": 13,
                        "title": "Episode 13: Work",
                        "publication_date": "2024-01-09",
                        "selected": True,
                        "transcribed": False,
                        "transcript_text_path": str(skipped_path),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = build_index(knowledge_dir)

    summary_path = knowledge_dir / "summaries" / "episode-12-midlife-music.json"
    chunks_path = knowledge_dir / "chunks" / "episode-12-midlife-music.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    indexed_episode = manifest["episodes"][0]
    skipped_episode = manifest["episodes"][1]
    assert result["indexed_episode_count"] == 1
    assert result["summary_count"] == 1
    assert result["failures"] == []
    assert summary["summary"] == "Simon and Lee talk about music and memory."
    assert summary["topics"]
    assert summary["simon_observations"]
    assert summary["lee_observations"]
    assert chunks[0]["episode_number"] == 12
    assert chunks[0]["title"] == "Episode 12: Midlife Music"
    assert chunks[0]["publication_date"] == "2024-01-02"
    assert chunks[0]["source_transcript_path"] == str(transcript_path)
    assert chunks[0]["text"]
    assert chunks[0]["relevant_topics"]
    assert indexed_episode["indexed"] is True
    assert indexed_episode["summary_path"] == str(summary_path)
    assert indexed_episode["chunks_path"] == str(chunks_path)
    assert indexed_episode["indexed_at"].endswith("Z")
    assert "indexed" not in skipped_episode


def test_successful_transcription_updates_manifest(tmp_path, monkeypatch) -> None:
    """Confirm successful transcriptions mark manifest metadata."""
    audio_path = tmp_path / "episode-12.mp3"
    audio_path.write_bytes(b"audio")
    output_dir = tmp_path / "transcripts"
    manifest_path = write_transcription_manifest(tmp_path, audio_path)
    job = TranscriptJob(12, "Episode 12", 120, audio_path)

    monkeypatch.setattr(
        "scripts.transcribe_midlifing_episodes.transcribe_job",
        lambda _job, _model: {"text": "hello world", "chunks": [{}, {}]},
    )

    successes, failures, skipped = run_transcription_batch(
        [job],
        "gpt-4o-mini-transcribe",
        output_dir,
        manifest_path,
    )

    episode = manifest_episode(manifest_path)
    assert successes == [job]
    assert failures == []
    assert skipped == 0
    assert episode["transcribed"] is True
    assert episode["transcript_text_path"] == str(output_dir / "episode-12.txt")
    assert episode["transcript_json_path"] == str(output_dir / "episode-12.json")
    assert episode["transcription_model"] == "gpt-4o-mini-transcribe"
    assert episode["transcription_chunk_count"] == 2
    assert episode["transcribed_at"].endswith("Z")
    assert "transcription_error" not in episode


def test_skipped_existing_transcript_backfills_manifest(tmp_path, monkeypatch) -> None:
    """Confirm valid existing outputs backfill absent manifest metadata."""
    audio_path = tmp_path / "episode-12.mp3"
    audio_path.write_bytes(b"audio")
    output_dir = tmp_path / "transcripts"
    output_dir.mkdir()
    (output_dir / "episode-12.json").write_text(
        json.dumps({"text": "hello world", "chunks": [{}, {}]}),
        encoding="utf-8",
    )
    (output_dir / "episode-12.txt").write_text("hello world", encoding="utf-8")
    manifest_path = write_transcription_manifest(
        tmp_path,
        audio_path,
        {"transcribed_at": "2024-01-01T00:00:00Z"},
    )
    job = TranscriptJob(12, "Episode 12", 120, audio_path)

    def fail_if_called(_job, _model):
        raise AssertionError("skipped transcript should not call OpenAI")

    monkeypatch.setattr("scripts.transcribe_midlifing_episodes.transcribe_job", fail_if_called)

    successes, failures, skipped = run_transcription_batch(
        [job],
        "gpt-4o-mini-transcribe",
        output_dir,
        manifest_path,
    )

    episode = manifest_episode(manifest_path)
    assert successes == [job]
    assert failures == []
    assert skipped == 1
    assert episode["transcribed"] is True
    assert episode["transcript_text_path"] == str(output_dir / "episode-12.txt")
    assert episode["transcript_json_path"] == str(output_dir / "episode-12.json")
    assert episode["transcription_model"] == "gpt-4o-mini-transcribe"
    assert episode["transcription_chunk_count"] == 2
    assert episode["transcribed_at"] == "2024-01-01T00:00:00Z"


def test_failed_transcription_preserves_success_metadata_and_records_safe_error(
    tmp_path,
    monkeypatch,
) -> None:
    """Confirm failures keep prior success fields and store redacted error context."""
    audio_path = tmp_path / "episode-12.mp3"
    audio_path.write_bytes(b"audio")
    output_dir = tmp_path / "transcripts"
    manifest_path = write_transcription_manifest(
        tmp_path,
        audio_path,
        {
            "transcribed": True,
            "transcript_text_path": "existing.txt",
            "transcript_json_path": "existing.json",
            "transcription_model": "gpt-4o-mini-transcribe",
            "transcription_chunk_count": 1,
            "transcribed_at": "2024-01-01T00:00:00Z",
        },
    )
    job = TranscriptJob(12, "Episode 12", 120, audio_path)
    monkeypatch.setenv("OPENAI_API_KEY", "secret-key")

    def fail_transcription(_job, _model):
        raise RuntimeError("request failed for secret-key")

    monkeypatch.setattr("scripts.transcribe_midlifing_episodes.transcribe_job", fail_transcription)

    successes, failures, skipped = run_transcription_batch(
        [job],
        "gpt-4o-mini-transcribe",
        output_dir,
        manifest_path,
    )

    episode = manifest_episode(manifest_path)
    assert successes == []
    assert failures == [(job, "request failed for [redacted]")]
    assert skipped == 0
    assert episode["transcribed"] is True
    assert episode["transcript_text_path"] == "existing.txt"
    assert episode["transcript_json_path"] == "existing.json"
    assert episode["transcription_model"] == "gpt-4o-mini-transcribe"
    assert episode["transcription_chunk_count"] == 1
    assert episode["transcribed_at"] == "2024-01-01T00:00:00Z"
    assert episode["transcription_error"] == "request failed for [redacted]"


def write_transcription_manifest(
    tmp_path,
    audio_path,
    extra_fields: dict[str, object] | None = None,
) -> Path:
    """Write a minimal transcription manifest fixture."""
    manifest_path = tmp_path / "manifest.json"
    episode = {
        "episode_number": 12,
        "title": "Episode 12",
        "duration": "120",
        "selected": True,
        "local_audio_path": str(audio_path),
    }
    episode.update(extra_fields or {})
    manifest_path.write_text(json.dumps({"episodes": [episode]}), encoding="utf-8")
    return manifest_path


def manifest_episode(manifest_path: Path) -> dict[str, object]:
    """Return the first manifest episode from a fixture."""
    return json.loads(manifest_path.read_text(encoding="utf-8"))["episodes"][0]
