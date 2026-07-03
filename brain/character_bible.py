"""Raw Character Bible document loading for Project MAYA."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


DEFAULT_CHARACTER_BIBLE_PATH = Path(__file__).with_name("maya_character_bible_v2.md")


@dataclass(frozen=True, slots=True)
class CharacterBibleDocument:
    """Read-only Character Bible document loaded from disk."""

    path: Path
    content: str
    loaded_at: datetime
    modified_ns: int


@dataclass(slots=True)
class CharacterBibleLoader:
    """Loads and optionally refreshes Maya's Character Bible document."""

    path: Path = DEFAULT_CHARACTER_BIBLE_PATH
    development_mode: bool = False
    _document: CharacterBibleDocument | None = None

    def get_document(self) -> CharacterBibleDocument:
        """Return the current read-only Character Bible document."""
        if self._document is None:
            self._document = self.load()
            return self._document

        if self.development_mode and self.has_changed():
            self._document = self.load()

        return self._document

    def load(self) -> CharacterBibleDocument:
        """Load the raw Character Bible markdown without interpreting its content."""
        if not self.path.exists():
            raise FileNotFoundError(f"Character Bible document not found: {self.path}")
        if not self.path.is_file():
            raise FileNotFoundError(f"Character Bible path is not a file: {self.path}")

        stat = self.path.stat()
        return CharacterBibleDocument(
            path=self.path,
            content=self.path.read_text(encoding="utf-8"),
            loaded_at=datetime.now(UTC),
            modified_ns=stat.st_mtime_ns,
        )

    def has_changed(self) -> bool:
        """Return whether the watched Character Bible file changed on disk."""
        if self._document is None:
            return True
        if not self.path.exists() or not self.path.is_file():
            raise FileNotFoundError(f"Character Bible document not found: {self.path}")
        return self.path.stat().st_mtime_ns != self._document.modified_ns
