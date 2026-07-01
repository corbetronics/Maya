"""Realtime service configuration objects."""

from pydantic import BaseModel, Field


class RealtimeSessionConfig(BaseModel):
    """Configuration for an OpenAI Realtime session."""

    model: str = Field(description="OpenAI Realtime model identifier.")
    voice: str = Field(default="alloy", description="Synthetic voice identifier.")
    input_audio_format: str = Field(default="pcm16", description="Expected input audio encoding.")
    output_audio_format: str = Field(default="pcm16", description="Produced output audio encoding.")
