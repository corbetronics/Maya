"""SQLite connection helpers."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from backend.app.core.config import Settings


def initialize_database(settings: Settings) -> None:
    """Create the SQLite database file and apply baseline connection settings."""
    database_path = Path(settings.sqlite_path)
    if database_path.parent != Path("."):
        database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA foreign_keys=ON;")


@contextmanager
def sqlite_connection(settings: Settings) -> Iterator[sqlite3.Connection]:
    """Yield a configured SQLite connection."""
    connection = sqlite3.connect(settings.sqlite_path)
    connection.execute("PRAGMA foreign_keys=ON;")
    try:
        yield connection
    finally:
        connection.close()
