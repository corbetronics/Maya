"""Runtime configuration for Project MAYA."""

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_name: str = "Project MAYA"
    version: str = "0.1.0"
    environment: str = "development"
    api_prefix: str = "/api"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    sqlite_path: str = Field(
        default="maya.sqlite3",
        validation_alias=AliasChoices("SQLITE_PATH", "DATABASE_PATH"),
    )
    openai_api_key: str | None = None
    openai_realtime_model: str = "gpt-realtime"


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
