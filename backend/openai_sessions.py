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

        payload = self._client_secret_payload(session_config)
        body = json.dumps(payload).encode("utf-8")
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
            logger.info(
                "Creating OpenAI realtime client secret with session keys=%s",
                sorted(payload["session"].keys()),
            )
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

    def _client_secret_payload(self, session_config: dict[str, Any]) -> dict[str, Any]:
        """Build the allowlisted OpenAI client-secret request payload."""
        session: dict[str, Any] = {}

        session_type = session_config.get("type")
        if session_type is not None:
            session["type"] = session_type

        model = session_config.get("model")
        if model is not None:
            session["model"] = model

        instructions = session_config.get("instructions")
        if instructions is not None:
            session["instructions"] = instructions

        voice = self._voice_from_session_config(session_config)
        if voice is not None:
            session["audio"] = {
                "output": {
                    "voice": voice,
                },
            }

        return {"session": session}

    def _voice_from_session_config(self, session_config: dict[str, Any]) -> Any:
        """Return the configured output voice from supported internal shapes."""
        audio = session_config.get("audio")
        if isinstance(audio, dict):
            output = audio.get("output")
            if isinstance(output, dict):
                voice = output.get("voice")
                if voice is not None:
                    return voice

        return session_config.get("voice")

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
