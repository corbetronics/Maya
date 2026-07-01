"""OpenAI Realtime session creation client."""

from dataclasses import dataclass
import json
import logging
import os
from typing import Any
from urllib import request
from urllib.error import HTTPError


OPENAI_REALTIME_CLIENT_SECRETS_URL = "https://api.openai.com/v1/realtime/client_secrets"
LOCAL_DEVELOPMENT_SAFETY_IDENTIFIER = "project-maya-local-development"
logger = logging.getLogger(__name__)


class MissingOpenAIAPIKeyError(RuntimeError):
    """Raised when an ephemeral OpenAI session is requested without credentials."""


class MissingOpenAISafetyIdentifierError(RuntimeError):
    """Raised when production session creation lacks a configured safety identifier."""


@dataclass(frozen=True, slots=True)
class OpenAISessionClient:
    """Creates browser-safe ephemeral OpenAI Realtime sessions."""

    sessions_url: str = OPENAI_REALTIME_CLIENT_SECRETS_URL
    timeout_seconds: float = 10.0

    def create_ephemeral_session(self, session_config: dict[str, Any]) -> dict[str, Any]:
        """Create an ephemeral Realtime session using OPENAI_API_KEY."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise MissingOpenAIAPIKeyError("OPENAI_API_KEY is required to create a session.")

        body = json.dumps({"session": session_config}).encode("utf-8")
        session_request = request.Request(
            self.sessions_url,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "OpenAI-Safety-Identifier": self._safety_identifier(),
            },
            method="POST",
        )

        try:
            with request.urlopen(session_request, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
                return json.loads(response_body)
        except HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            logger.warning(
                "OpenAI rejected ephemeral session creation: status=%s body=%s",
                exc.code,
                response_body,
            )
            raise

    def _safety_identifier(self) -> str:
        """Return the backend-only safety identifier for OpenAI session creation."""
        configured_identifier = os.environ.get("OPENAI_SAFETY_IDENTIFIER")
        if configured_identifier:
            return configured_identifier

        environment = os.environ.get("ENVIRONMENT", "development").lower()
        if environment == "development":
            return LOCAL_DEVELOPMENT_SAFETY_IDENTIFIER

        raise MissingOpenAISafetyIdentifierError(
            "OPENAI_SAFETY_IDENTIFIER is required outside development."
        )
