"""Transcribe locally cached Midlifing audio with OpenAI from the command line."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from dataclasses import dataclass
import json
import mimetypes
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
from urllib import error, request

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None


DEFAULT_MODEL = "gpt-4o-mini-transcribe"
DEFAULT_MANIFEST_PATH = Path("brain/knowledge/midlifing/manifest.json")
DEFAULT_OUTPUT_DIR = Path("brain/knowledge/midlifing/transcripts")
TRANSCRIPTIONS_URL = "https://api.openai.com/v1/audio/transcriptions"
MAX_UPLOAD_BYTES = 24 * 1024 * 1024
DEFAULT_CHUNK_SECONDS = 20 * 60
BYTES_PER_MB = 1024 * 1024
JSON_ONLY_TRANSCRIPTION_MODELS = {"gpt-4o-mini-transcribe", "gpt-4o-transcribe"}


@dataclass(frozen=True)
class TranscriptJob:
    """One manifest-backed audio transcription job."""

    episode_number: int | None
    title: str
    duration_seconds: int
    audio_path: Path


@dataclass(frozen=True)
class AudioPart:
    """One uploadable audio file, with a transcript timestamp offset."""

    path: Path
    order: int
    offset_seconds: int


def load_local_env() -> None:
    """Load local env files without overriding exported shell values."""
    for env_path in (Path(".env.local"), Path(".env")):
        if load_dotenv is not None:
            load_dotenv(env_path, override=False)
        else:
            load_simple_env_file(env_path)


def load_simple_env_file(env_path: Path) -> None:
    """Load basic KEY=value env files when python-dotenv is unavailable."""
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        name = name.strip()
        if not name or name in os.environ:
            continue
        os.environ[name] = value.strip().strip("\"'")


def file_size_mb(path: Path) -> float:
    """Return a file size in MiB."""
    return path.stat().st_size / BYTES_PER_MB


def needs_splitting(audio_path: Path) -> bool:
    """Return whether an audio file is too large for a direct upload."""
    return audio_path.stat().st_size > MAX_UPLOAD_BYTES


def chunk_output_dir(audio_path: Path) -> Path:
    """Return the deterministic chunk cache directory for an audio file."""
    return audio_path.parent / "chunks" / audio_path.stem


def ffmpeg_split_command(
    audio_path: Path,
    output_dir: Path | None = None,
    chunk_seconds: int = DEFAULT_CHUNK_SECONDS,
) -> list[str]:
    """Build the ffmpeg command used to split audio into sequential chunks."""
    target_dir = output_dir or chunk_output_dir(audio_path)
    output_pattern = target_dir / f"{audio_path.stem}.part-%03d{audio_path.suffix}"
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-reset_timestamps",
        "1",
        "-c",
        "copy",
        str(output_pattern),
    ]


def shell_join(command: list[str]) -> str:
    """Return a safely displayable shell command."""
    return " ".join(shlex_quote(part) for part in command)


def shlex_quote(value: str) -> str:
    """Quote a shell token without importing shlex on older project Pythons."""
    if value and all(character.isalnum() or character in "@%_+=:,./-" for character in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def split_audio_file(audio_path: Path, chunk_seconds: int = DEFAULT_CHUNK_SECONDS) -> list[AudioPart]:
    """Split oversized audio into uploadable sequential chunks."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            f"{audio_path} needs splitting before upload, but ffmpeg is not available. "
            f"File size: {file_size_mb(audio_path):.2f} MB."
        )

    output_dir = chunk_output_dir(audio_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_chunk in output_dir.glob(f"{audio_path.stem}.part-*{audio_path.suffix}"):
        stale_chunk.unlink()

    command = ffmpeg_split_command(audio_path, output_dir, chunk_seconds)
    command[0] = ffmpeg_path
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    chunk_paths = sorted(output_dir.glob(f"{audio_path.stem}.part-*{audio_path.suffix}"))
    if not chunk_paths:
        raise RuntimeError(f"ffmpeg did not create chunks for {audio_path}.")

    oversized_chunks = [path for path in chunk_paths if needs_splitting(path)]
    if oversized_chunks:
        chunk_list = ", ".join(f"{path.name} ({file_size_mb(path):.2f} MB)" for path in oversized_chunks)
        raise RuntimeError(f"Split chunks still exceed 24 MB: {chunk_list}")

    return [
        AudioPart(path=path, order=index + 1, offset_seconds=index * chunk_seconds)
        for index, path in enumerate(chunk_paths)
    ]


def audio_parts_for_upload(job: TranscriptJob, model: str = DEFAULT_MODEL) -> list[AudioPart]:
    """Return one or more safe upload parts for a transcription job."""
    if not needs_splitting(job.audio_path) and not needs_duration_splitting(job, model):
        return [AudioPart(path=job.audio_path, order=1, offset_seconds=0)]
    return split_audio_file(job.audio_path)


def needs_duration_splitting(job: TranscriptJob, model: str) -> bool:
    """Return whether a model should receive this episode in shorter chunks."""
    return model in JSON_ONLY_TRANSCRIPTION_MODELS and job.duration_seconds > DEFAULT_CHUNK_SECONDS


def transcribe_audio_file(audio_path: Path, model: str = DEFAULT_MODEL) -> dict[str, object]:
    """Transcribe one audio file using OPENAI_API_KEY."""
    load_local_env()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for local transcription.")

    boundary = f"----maya-{uuid.uuid4().hex}"
    body = multipart_body(audio_path, model, boundary)
    transcription_request = request.Request(
        TRANSCRIPTIONS_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    with request.urlopen(transcription_request) as response:
        return json.loads(response.read().decode("utf-8"))


def transcribe_job(job: TranscriptJob, model: str) -> dict[str, object]:
    """Transcribe one job, splitting and merging chunks when needed."""
    parts = audio_parts_for_upload(job, model)
    transcripts = [transcribe_audio_file(part.path, model) for part in parts]
    if len(transcripts) == 1:
        return transcripts[0]
    return merge_transcripts(job, parts, transcripts)


def merge_transcripts(
    job: TranscriptJob,
    parts: list[AudioPart],
    transcripts: list[dict[str, object]],
) -> dict[str, object]:
    """Merge chunk transcripts into one episode-level transcript JSON."""
    merged_segments: list[dict[str, object]] = []
    text_parts: list[str] = []
    for part, transcript in zip(parts, transcripts):
        text = str(transcript.get("text", "")).strip()
        if text:
            text_parts.append(text)
        for segment in transcript.get("segments", []):
            if not isinstance(segment, dict):
                continue
            merged_segment = dict(segment)
            for timestamp_key in ("start", "end"):
                value = merged_segment.get(timestamp_key)
                if isinstance(value, (int, float)):
                    merged_segment[timestamp_key] = value + part.offset_seconds
            merged_segment["chunk_order"] = part.order
            merged_segment["chunk_file"] = part.path.name
            merged_segments.append(merged_segment)

    return {
        "text": "\n".join(text_parts),
        "segments": merged_segments,
        "episode_number": job.episode_number,
        "title": job.title,
        "source_audio": str(job.audio_path),
        "chunks": [
            {
                "order": part.order,
                "file": str(part.path),
                "offset_seconds": part.offset_seconds,
                "size_mb": round(file_size_mb(part.path), 2),
            }
            for part in parts
        ],
    }


def multipart_body(audio_path: Path, model: str, boundary: str) -> bytes:
    """Build multipart form data for OpenAI transcription."""
    content_type = mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"
    file_bytes = audio_path.read_bytes()
    parts = [form_field(boundary, name, value) for name, value in transcription_form_fields(model)]
    parts.extend(
        [
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{audio_path.name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
            file_bytes,
            f"\r\n--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    return b"".join(parts)


def transcription_form_fields(model: str) -> list[tuple[str, str]]:
    """Return model-compatible OpenAI audio transcription form fields."""
    fields = [("model", model)]
    if model in JSON_ONLY_TRANSCRIPTION_MODELS:
        fields.append(("response_format", "json"))
    else:
        fields.extend(
            [
                ("response_format", "verbose_json"),
                ("timestamp_granularities[]", "segment"),
            ]
        )
    return fields


def form_field(boundary: str, name: str, value: str) -> bytes:
    """Build one multipart text field."""
    return (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
        f"{value}\r\n"
    ).encode("utf-8")


def write_transcript_outputs(
    episode_id: str,
    transcript: dict[str, object],
    output_dir: Path,
) -> None:
    """Write timestamped JSON and plain text transcript files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{episode_id}.json"
    text_path = output_dir / f"{episode_id}.txt"
    json_path.write_text(json.dumps(transcript, indent=2) + "\n", encoding="utf-8")
    text_path.write_text(str(transcript.get("text", "")), encoding="utf-8")


def update_manifest_success(
    manifest_path: Path,
    job: TranscriptJob,
    output_dir: Path,
    model: str,
    chunk_count: int,
    backfill_only: bool = False,
) -> None:
    """Mark a manifest episode as successfully transcribed."""
    update_manifest_entry(
        manifest_path,
        job,
        {
            "transcribed": True,
            "transcript_text_path": str(transcript_paths(job.audio_path.stem, output_dir)[1]),
            "transcript_json_path": str(transcript_paths(job.audio_path.stem, output_dir)[0]),
            "transcription_model": model,
            "transcription_chunk_count": chunk_count,
            "transcribed_at": utc_now_iso(),
        },
        remove_keys={"transcription_error"},
        backfill_only=backfill_only,
    )


def update_manifest_failure(manifest_path: Path, job: TranscriptJob, error_summary: str) -> None:
    """Record a safe transcription failure without clearing success metadata."""
    update_manifest_entry(
        manifest_path,
        job,
        {"transcription_error": safe_error_summary(error_summary)},
        remove_keys=set(),
    )


def update_manifest_entry(
    manifest_path: Path,
    job: TranscriptJob,
    updates: dict[str, object],
    remove_keys: set[str],
    backfill_only: bool = False,
) -> None:
    """Update the matching manifest entry for a transcription job."""
    if job.episode_number is None or not manifest_path.exists():
        return
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed = False
    for episode in loaded.get("episodes", []):
        if episode.get("episode_number") != job.episode_number:
            continue
        for key in remove_keys:
            changed = episode.pop(key, None) is not None or changed
        for key, value in updates.items():
            if backfill_only and key in episode:
                continue
            if episode.get(key) != value:
                episode[key] = value
                changed = True
        break
    if changed:
        manifest_path.write_text(json.dumps(loaded, indent=2) + "\n", encoding="utf-8")


def safe_error_summary(message: str) -> str:
    """Return a bounded, non-secret transcription error summary."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        message = message.replace(api_key, "[redacted]")
    return message.strip()[:1000]


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 form."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def transcript_paths(episode_id: str, output_dir: Path) -> tuple[Path, Path]:
    """Return the JSON and text output paths for an episode."""
    return output_dir / f"{episode_id}.json", output_dir / f"{episode_id}.txt"


def has_valid_transcript(episode_id: str, output_dir: Path) -> bool:
    """Return whether both transcript outputs exist and contain usable content."""
    json_path, text_path = transcript_paths(episode_id, output_dir)
    if not json_path.exists() or not text_path.exists():
        return False
    try:
        loaded = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(str(loaded.get("text", "")).strip()) and bool(text_path.read_text(encoding="utf-8").strip())


def jobs_from_manifest(manifest_path: Path) -> list[TranscriptJob]:
    """Read selected local audio paths directly from the Midlifing manifest."""
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs: list[TranscriptJob] = []
    for episode in loaded.get("episodes", []):
        if not episode.get("selected"):
            continue
        audio_path_value = episode.get("local_audio_path")
        if not audio_path_value:
            continue
        jobs.append(
            TranscriptJob(
                episode_number=episode.get("episode_number"),
                title=str(episode.get("title", "")),
                duration_seconds=parse_duration_seconds(episode.get("duration")),
                audio_path=Path(audio_path_value),
            )
        )
    return jobs


def filter_jobs_by_episode(jobs: list[TranscriptJob], episode_number: int | None) -> list[TranscriptJob]:
    """Return all jobs or one requested episode number."""
    if episode_number is None:
        return jobs
    return [job for job in jobs if job.episode_number == episode_number]


def parse_duration_seconds(value: object) -> int:
    """Parse RSS duration values in seconds or HH:MM:SS form."""
    if isinstance(value, int):
        return value
    text = str(value or "").strip()
    if not text:
        return 0
    if text.isdigit():
        return int(text)
    parts = text.split(":")
    if not all(part.isdigit() for part in parts):
        return 0
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + int(part)
    return seconds


def run_transcription_batch(
    jobs: list[TranscriptJob],
    model: str,
    output_dir: Path,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> tuple[list[TranscriptJob], list[tuple[TranscriptJob, str]], int]:
    """Transcribe jobs, skipping valid existing outputs and continuing on failure."""
    successes: list[TranscriptJob] = []
    failures: list[tuple[TranscriptJob, str]] = []
    skipped = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        episode_id = job.audio_path.stem
        if has_valid_transcript(episode_id, output_dir):
            skipped += 1
            successes.append(job)
            update_manifest_success(
                manifest_path,
                job,
                output_dir,
                model,
                existing_chunk_count(episode_id, output_dir),
                backfill_only=True,
            )
            print(f"Skipping existing transcript for episode {job.episode_number}: {episode_id}")
            continue
        if not job.audio_path.exists():
            message = f"Audio file not found: {job.audio_path}"
            failures.append((job, message))
            update_manifest_failure(manifest_path, job, message)
            continue
        try:
            transcript = transcribe_job(job, model)
            write_transcript_outputs(episode_id, transcript, output_dir)
            successes.append(job)
            update_manifest_success(manifest_path, job, output_dir, model, transcript_chunk_count(transcript))
            print(f"Transcribed episode {job.episode_number}: {episode_id}")
        except error.HTTPError as exc:
            message = safe_error_summary(format_http_error(job, exc))
            failures.append((job, message))
            update_manifest_failure(manifest_path, job, message)
            print(message, file=sys.stderr)
        except (RuntimeError, OSError, error.URLError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
            message = safe_error_summary(str(exc))
            failures.append((job, message))
            update_manifest_failure(manifest_path, job, message)
            print(f"Failed episode {job.episode_number}: {message}", file=sys.stderr)
    return successes, failures, skipped


def transcript_chunk_count(transcript: dict[str, object]) -> int:
    """Return chunk count encoded in a transcript response."""
    chunks = transcript.get("chunks")
    if isinstance(chunks, list) and chunks:
        return len(chunks)
    return 1


def existing_chunk_count(episode_id: str, output_dir: Path) -> int:
    """Return chunk count from an existing transcript JSON when present."""
    json_path, _text_path = transcript_paths(episode_id, output_dir)
    try:
        loaded = json.loads(json_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return 1
    if isinstance(loaded, dict):
        return transcript_chunk_count(loaded)
    return 1


def format_http_error(job: TranscriptJob, exc: error.HTTPError) -> str:
    """Build a safe OpenAI HTTP error diagnostic without printing secrets."""
    response_body = exc.read().decode("utf-8", errors="replace")
    size_label = "missing"
    if job.audio_path.exists():
        size_label = f"{file_size_mb(job.audio_path):.2f} MB"
    return (
        "OpenAI transcription failed: "
        f"episode={job.episode_number or 'manual'} "
        f"filename={job.audio_path.name} "
        f"size={size_label} "
        f"http_status={exc.code} "
        f"response_body={response_body}"
    )


def print_dry_run(jobs: list[TranscriptJob], model: str = DEFAULT_MODEL) -> None:
    """Print transcription upload plan without calling OpenAI."""
    for job in jobs:
        episode_label = job.episode_number or "manual"
        filename = job.audio_path.name
        if not job.audio_path.exists():
            print(f"Episode {episode_label}: {filename} missing")
            continue
        size_mb = file_size_mb(job.audio_path)
        split_required = needs_splitting(job.audio_path) or needs_duration_splitting(job, model)
        print(
            f"Episode {episode_label}: {filename} size={size_mb:.2f} MB "
            f"split_required={str(split_required).lower()}"
        )
        if split_required:
            print(f"ffmpeg command: {shell_join(ffmpeg_split_command(job.audio_path))}")


def main() -> None:
    """CLI entrypoint for local manual transcription."""
    load_local_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, nargs="*")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--model", default=os.environ.get("MIDLIFING_TRANSCRIBE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episode", type=int, help="Transcribe only one manifest episode number.")
    parser.add_argument("--dry-run", action="store_true", help="Report upload plan without OpenAI calls.")
    args = parser.parse_args()

    jobs = [
        TranscriptJob(None, audio_path.stem, 0, audio_path)
        for audio_path in args.audio
    ] if args.audio else jobs_from_manifest(args.manifest)
    jobs = filter_jobs_by_episode(jobs, args.episode)
    if args.episode is not None and not jobs:
        raise SystemExit(f"No selected manifest episode found for --episode {args.episode}.")
    if args.dry_run:
        print_dry_run(jobs, args.model)
        return
    successes, failures, skipped = run_transcription_batch(
        jobs,
        args.model,
        args.output_dir,
        args.manifest,
    )
    print(f"Transcript successes: {len(successes)}")
    print(f"Transcript skipped_existing: {skipped}")
    print(f"Transcript failures: {len(failures)}")
    if failures:
        for job, message in failures:
            print(f"- {job.episode_number or job.audio_path.stem}: {message}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
