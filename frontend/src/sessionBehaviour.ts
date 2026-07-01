export type TranscriptSpeaker = "Host" | "Maya";

export type TranscriptUpdate = {
  speaker: TranscriptSpeaker;
  text: string;
  isPartial: boolean;
};

type SessionBehaviourOptions = {
  onTranscript: (update: TranscriptUpdate) => void;
};

type RealtimeServerEvent = {
  type?: unknown;
  transcript?: unknown;
  delta?: unknown;
};

export function attachSessionBehaviour(
  dataChannel: RTCDataChannel,
  { onTranscript }: SessionBehaviourOptions
): () => void {
  const handleOpen = () => {
    dataChannel.send(
      JSON.stringify({
        type: "session.update",
        session: {
          turn_detection: {
            type: "server_vad",
            threshold: 0.65,
            prefix_padding_ms: 300,
            silence_duration_ms: 750,
            interrupt_response: true,
            create_response: true
          },
          input_audio_transcription: {
            model: "whisper-1"
          }
        }
      })
    );
  };

  const handleMessage = (event: MessageEvent) => {
    const transcriptUpdate = extractTranscriptUpdate(parseMessage(event.data));
    if (transcriptUpdate) {
      onTranscript(transcriptUpdate);
    }
  };

  dataChannel.addEventListener("open", handleOpen);
  dataChannel.addEventListener("message", handleMessage);

  return () => {
    dataChannel.removeEventListener("open", handleOpen);
    dataChannel.removeEventListener("message", handleMessage);
  };
}

export function extractTranscriptUpdate(event: unknown): TranscriptUpdate | null {
  if (!isRealtimeServerEvent(event) || typeof event.type !== "string") {
    return null;
  }

  if (event.type === "conversation.item.input_audio_transcription.completed") {
    return transcriptFromField("Host", event.transcript, false);
  }

  if (event.type === "response.output_audio_transcript.delta") {
    return transcriptFromField("Maya", event.delta, true);
  }

  if (event.type === "response.output_audio_transcript.done") {
    return transcriptFromField("Maya", event.transcript, false);
  }

  return null;
}

function parseMessage(data: unknown): unknown {
  if (typeof data !== "string") {
    return data;
  }

  try {
    return JSON.parse(data) as unknown;
  } catch {
    return data;
  }
}

function isRealtimeServerEvent(event: unknown): event is RealtimeServerEvent {
  return Boolean(event && typeof event === "object" && "type" in event);
}

function transcriptFromField(
  speaker: TranscriptSpeaker,
  value: unknown,
  isPartial: boolean
): TranscriptUpdate | null {
  if (typeof value !== "string" || value.trim().length === 0) {
    return null;
  }
  return {
    speaker,
    text: value,
    isPartial
  };
}
