"""Read-only curated knowledge loading for Maya."""

from dataclasses import dataclass
from pathlib import Path


DEFAULT_KNOWLEDGE_DIR = Path(__file__).with_name("knowledge")


@dataclass(frozen=True, slots=True)
class KnowledgeBundle:
    """Raw curated background notes for a live Midlifing conversation."""

    show_context: str
    simon_context: str
    lee_context: str
    current_episode_context: str


@dataclass(frozen=True, slots=True)
class KnowledgeLoader:
    """Loads curated markdown knowledge files without interpretation."""

    knowledge_dir: Path = DEFAULT_KNOWLEDGE_DIR

    def load(self) -> KnowledgeBundle:
        """Load every curated knowledge document verbatim."""
        return KnowledgeBundle(
            show_context=self._read_markdown("midlifing_show.md"),
            simon_context=self._read_markdown("simon.md"),
            lee_context=self._read_markdown("lee.md"),
            current_episode_context=self._read_markdown("current_episode.md"),
        )

    def _read_markdown(self, filename: str) -> str:
        """Read one required markdown file exactly as stored."""
        path = self.knowledge_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Knowledge document not found: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Knowledge path is not a file: {path}")
        return path.read_text(encoding="utf-8")
