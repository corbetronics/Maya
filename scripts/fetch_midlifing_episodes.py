"""Fetch selected Midlifing RSS metadata and optionally download episode audio."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from datetime import date, datetime
import json
import os
from pathlib import Path
import re
import sys
from typing import Iterable
from urllib import error, request
import xml.etree.ElementTree as ET


DEFAULT_MANIFEST_PATH = Path("brain/knowledge/midlifing/manifest.json")
DEFAULT_AUDIO_CACHE_DIR = Path(os.environ.get("MIDLIFING_AUDIO_CACHE", ".cache/midlifing/audio"))
RSS_USER_AGENT = "ProjectMayaLocalIngestion/0.1 (+local metadata discovery)"


@dataclass(frozen=True, slots=True)
class EpisodeMetadata:
    """RSS-derived episode metadata used by the local ingestion pipeline."""

    episode_number: int | None
    title: str
    publication_date: str
    duration: str
    description: str
    enclosure_url: str
    source_url: str
    selected: bool = False


def parse_rss(xml_text: str) -> list[EpisodeMetadata]:
    """Parse Midlifing RSS XML into episode metadata."""
    root = ET.fromstring(xml_text)
    episodes: list[EpisodeMetadata] = []
    for item in root.findall(".//channel/item"):
        title = text_from_child(item, "title")
        source_url = text_from_child(item, "link") or text_from_child(item, "guid")
        enclosure = item.find("enclosure")
        enclosure_url = enclosure.attrib.get("url", "") if enclosure is not None else ""
        episodes.append(
            EpisodeMetadata(
                episode_number=episode_number_from_item(item, title),
                title=title,
                publication_date=normalise_pub_date(text_from_child(item, "pubDate")),
                duration=duration_from_item(item),
                description=text_from_child(item, "description"),
                enclosure_url=enclosure_url,
                source_url=source_url,
            )
        )
    return episodes


def select_episodes(
    episodes: Iterable[EpisodeMetadata],
    episode_numbers: set[int] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    curated_manifest_path: Path | None = None,
    max_count: int | None = None,
) -> list[EpisodeMetadata]:
    """Select episodes by number, date range, or curated manifest."""
    selected_numbers = set(episode_numbers or set())
    if curated_manifest_path:
        selected_numbers.update(numbers_from_curated_manifest(curated_manifest_path))

    selected: list[EpisodeMetadata] = []
    for episode in episodes:
        include = not selected_numbers and start_date is None and end_date is None
        if episode.episode_number is not None and episode.episode_number in selected_numbers:
            include = True
        episode_date = parse_iso_date(episode.publication_date)
        if start_date and episode_date and episode_date >= start_date:
            include = True
        if end_date and episode_date and episode_date <= end_date and (start_date or not selected_numbers):
            include = True
        if start_date and end_date and episode_date:
            include = start_date <= episode_date <= end_date
        if include:
            selected.append(EpisodeMetadata(**{**asdict(episode), "selected": True}))
        if max_count is not None and len(selected) >= max_count:
            break
    return selected


def write_manifest(path: Path, episodes: list[EpisodeMetadata]) -> None:
    """Write selected episode metadata to manifest JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"episodes": [asdict(episode) for episode in episodes]}, indent=2) + "\n",
        encoding="utf-8",
    )


def print_episode_listing(episodes: list[EpisodeMetadata], output=sys.stdout) -> None:
    """Print RSS episode metadata without writing a manifest or downloading audio."""
    for episode in episodes:
        episode_number = episode.episode_number if episode.episode_number is not None else ""
        print(
            "\t".join(
                (
                    str(episode_number),
                    episode.publication_date,
                    episode.duration,
                    episode.title,
                    episode.enclosure_url,
                )
            ),
            file=output,
        )


def download_selected_audio(
    episodes: list[EpisodeMetadata],
    cache_dir: Path = DEFAULT_AUDIO_CACHE_DIR,
    max_download_count: int | None = None,
) -> list[Path]:
    """Download selected episode audio to a local cache outside git."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    failures: list[str] = []
    for episode in episodes:
        if not episode.selected or not episode.enclosure_url:
            continue
        if max_download_count is not None and len(downloaded) >= max_download_count:
            break
        filename = safe_episode_filename(episode)
        target_path = cache_dir / filename
        if not target_path.exists():
            try:
                with request.urlopen(metadata_request(episode.enclosure_url)) as response:
                    target_path.write_bytes(response.read())
            except (OSError, error.URLError) as exc:
                failures.append(f"{episode.episode_number or 'unknown'}: {exc}")
                continue
        if target_path.exists():
            downloaded.append(target_path)
    for failure in failures:
        print(f"Failed to download episode {failure}", file=sys.stderr)
    return downloaded


def annotate_manifest_audio_paths(
    manifest_path: Path,
    episodes: list[EpisodeMetadata],
    cache_dir: Path = DEFAULT_AUDIO_CACHE_DIR,
) -> None:
    """Record local cached audio metadata for selected manifest entries."""
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_by_number = {
        episode.episode_number: episode for episode in episodes if episode.selected and episode.episode_number is not None
    }
    for entry in loaded.get("episodes", []):
        episode_number = entry.get("episode_number")
        episode = selected_by_number.get(episode_number)
        if episode is None:
            continue
        audio_path = cache_dir / safe_episode_filename(episode)
        if audio_path.exists():
            entry["local_audio_path"] = str(audio_path)
            entry["audio_format"] = audio_path.suffix.lstrip(".").lower() or "unknown"
            entry["audio_size_bytes"] = audio_path.stat().st_size
    manifest_path.write_text(json.dumps(loaded, indent=2) + "\n", encoding="utf-8")


def metadata_request(url: str) -> request.Request:
    """Build an HTTP request for local RSS/audio metadata retrieval."""
    return request.Request(url, headers={"User-Agent": RSS_USER_AGENT})


def text_from_child(item: ET.Element, child_name: str) -> str:
    """Return text from a child element, ignoring XML namespaces."""
    for child in item:
        if child.tag.split("}")[-1] == child_name and child.text:
            return child.text.strip()
    return ""


def duration_from_item(item: ET.Element) -> str:
    """Return iTunes duration text when present."""
    for child in item:
        if child.tag.endswith("duration") and child.text:
            return child.text.strip()
    return ""


def episode_number_from_item(item: ET.Element, title: str) -> int | None:
    """Return an explicit iTunes episode number or infer one from title text."""
    for child in item:
        if child.tag.endswith("episode") and child.text:
            try:
                return int(child.text.strip())
            except ValueError:
                return None
    match = re.search(r"\b(?:episode|ep\.?)\s*(\d+)\b", title, re.IGNORECASE)
    return int(match.group(1)) if match else None


def normalise_pub_date(pub_date: str) -> str:
    """Convert RSS pubDate text to YYYY-MM-DD where possible."""
    if not pub_date:
        return ""
    try:
        from email.utils import parsedate_to_datetime

        return parsedate_to_datetime(pub_date).date().isoformat()
    except (TypeError, ValueError, IndexError):
        return pub_date


def parse_iso_date(value: str) -> date | None:
    """Parse an ISO date string if possible."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def numbers_from_curated_manifest(path: Path) -> set[int]:
    """Return selected episode numbers from an existing curated manifest."""
    loaded = json.loads(path.read_text(encoding="utf-8"))
    numbers: set[int] = set()
    for episode in loaded.get("episodes", []):
        if episode.get("selected") and isinstance(episode.get("episode_number"), int):
            numbers.add(episode["episode_number"])
    return numbers


def safe_episode_filename(episode: EpisodeMetadata) -> str:
    """Return a filesystem-safe audio cache filename."""
    stem = f"episode-{episode.episode_number or 'unknown'}-{episode.title}".lower()
    stem = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    extension = Path(episode.enclosure_url.split("?", 1)[0]).suffix or ".mp3"
    return f"{stem}{extension}"


def main() -> None:
    """CLI entrypoint for manual local RSS discovery and optional download."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--episode", type=int, action="append", default=[])
    parser.add_argument("--start-date", type=lambda value: datetime.strptime(value, "%Y-%m-%d").date())
    parser.add_argument("--end-date", type=lambda value: datetime.strptime(value, "%Y-%m-%d").date())
    parser.add_argument("--curated-manifest", type=Path)
    parser.add_argument("--max-count", type=int)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audio-cache", type=Path, default=DEFAULT_AUDIO_CACHE_DIR)
    args = parser.parse_args()

    rss_url = os.environ.get("MIDLIFING_RSS_URL")
    if not rss_url:
        raise SystemExit("MIDLIFING_RSS_URL is required.")

    with request.urlopen(metadata_request(rss_url)) as response:
        rss_xml = response.read().decode("utf-8")
    episodes = parse_rss(rss_xml)
    if args.list:
        print_episode_listing(episodes[: args.max_count] if args.max_count else episodes)
        return

    selected = select_episodes(
        episodes,
        episode_numbers=set(args.episode),
        start_date=args.start_date,
        end_date=args.end_date,
        curated_manifest_path=args.curated_manifest,
        max_count=args.max_count,
    )
    if args.dry_run:
        print_episode_listing(selected)
        return

    write_manifest(args.manifest, selected)
    if args.download:
        download_selected_audio(selected, args.audio_cache, args.max_count)
        annotate_manifest_audio_paths(args.manifest, selected, args.audio_cache)


if __name__ == "__main__":
    main()
