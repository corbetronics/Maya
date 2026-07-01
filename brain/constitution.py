"""Raw Constitution document loading for Project MAYA."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


DEFAULT_CONSTITUTION_PATH = Path(__file__).with_name("constitution.md")


@dataclass(frozen=True, slots=True)
class ConstitutionDocument:
    """Read-only raw Constitution document loaded from disk."""

    path: Path
    content: str
    loaded_at: datetime
    modified_ns: int


@dataclass(slots=True)
class ConstitutionLoader:
    """Loads and optionally refreshes Maya's Constitution document."""

    path: Path = DEFAULT_CONSTITUTION_PATH
    development_mode: bool = False
    _document: ConstitutionDocument | None = None

    def get_document(self) -> ConstitutionDocument:
        """Return the current read-only Constitution document."""
        if self._document is None:
            self._document = self.load()
            return self._document

        if self.development_mode and self.has_changed():
            self._document = self.load()

        return self._document

    def load(self) -> ConstitutionDocument:
        """Load the raw Constitution markdown without interpreting its content."""
        if not self.path.exists():
            raise FileNotFoundError(f"Constitution document not found: {self.path}")
        if not self.path.is_file():
            raise FileNotFoundError(f"Constitution path is not a file: {self.path}")

        stat = self.path.stat()
        return ConstitutionDocument(
            path=self.path,
            content=self.path.read_text(encoding="utf-8"),
            loaded_at=datetime.now(UTC),
            modified_ns=stat.st_mtime_ns,
        )

    def has_changed(self) -> bool:
        """Return whether the watched Constitution file changed on disk."""
        if self._document is None:
            return True
        if not self.path.exists() or not self.path.is_file():
            raise FileNotFoundError(f"Constitution document not found: {self.path}")
        return self.path.stat().st_mtime_ns != self._document.modified_ns
