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

type RetrieveContextResponse = {
  context_block?: unknown;
  summary_count?: unknown;
  chunk_count?: unknown;
};

const MIDLIFING_CONTEXT_START = "Relevant Midlifing background — use only if natural.";
const MIDLIFING_CONTEXT_END = "End relevant Midlifing background.";
const RETRIEVAL_DEBOUNCE_MS = 900;
const HOST_CONTEXT_WINDOW = 4;

export function attachSessionBehaviour(
  dataChannel: RTCDataChannel,
  { onTranscript }: SessionBehaviourOptions
): () => void {
  let baseInstructions = "";
  let lastContextBlock = "";
  let pendingHostUtterance = "";
  let debounceTimer: ReturnType<typeof setTimeout> | null = null;
  let mayaIsResponding = false;
  const recentHostUtterances: string[] = [];

  void loadBaseInstructions().then((instructions) => {
    baseInstructions = instructions;
  });

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
    const parsedEvent = parseMessage(event.data);
    mayaIsResponding = updateMayaResponseState(parsedEvent, mayaIsResponding);
    if (!mayaIsResponding && pendingHostUtterance) {
      scheduleContextRetrieval();
    }

    const transcriptUpdate = extractTranscriptUpdate(parsedEvent);
    if (transcriptUpdate) {
      onTranscript(transcriptUpdate);
      if (transcriptUpdate.speaker === "Host" && !transcriptUpdate.isPartial) {
        recentHostUtterances.push(transcriptUpdate.text);
        recentHostUtterances.splice(0, Math.max(0, recentHostUtterances.length - HOST_CONTEXT_WINDOW));
        pendingHostUtterance = transcriptUpdate.text;
        if (!mayaIsResponding) {
          scheduleContextRetrieval();
        }
      }
    }
  };

  const scheduleContextRetrieval = () => {
    if (debounceTimer) {
      clearTimeout(debounceTimer);
    }
    debounceTimer = setTimeout(() => {
      void retrieveAndSendContextUpdate();
    }, RETRIEVAL_DEBOUNCE_MS);
  };

  const retrieveAndSendContextUpdate = async () => {
    if (mayaIsResponding || dataChannel.readyState !== "open" || !pendingHostUtterance) {
      return;
    }
    const utterance = pendingHostUtterance;
    pendingHostUtterance = "";
    const contextBlock = await retrieveMidlifingContext(utterance, recentHostUtterances);
    if (!shouldSendContextUpdate(lastContextBlock, contextBlock)) {
      return;
    }
    lastContextBlock = contextBlock;
    dataChannel.send(
      JSON.stringify({
        type: "session.update",
        session: {
          instructions: buildInstructionsWithMidlifingContext(baseInstructions, contextBlock)
        }
      })
    );
  };

  dataChannel.addEventListener("open", handleOpen);
  dataChannel.addEventListener("message", handleMessage);

  return () => {
    if (debounceTimer) {
      clearTimeout(debounceTimer);
    }
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

export function shouldSendContextUpdate(previousContext: string, nextContext: string): boolean {
  return normaliseContext(previousContext) !== normaliseContext(nextContext);
}

export function buildInstructionsWithMidlifingContext(
  baseInstructions: string,
  contextBlock: string
): string {
  const safeBase = removeDelimitedMidlifingContext(baseInstructions).trim();
  const safeContext = contextBlock.trim().startsWith(MIDLIFING_CONTEXT_START)
    ? contextBlock.trim()
    : `${MIDLIFING_CONTEXT_START}\n${contextBlock.trim()}`;
  return [
    safeBase,
    `${safeContext}\n${MIDLIFING_CONTEXT_END}`
  ].filter(Boolean).join("\n\n");
}

function removeDelimitedMidlifingContext(instructions: string): string {
  const start = instructions.indexOf(MIDLIFING_CONTEXT_START);
  if (start < 0) {
    return instructions;
  }
  const end = instructions.indexOf(MIDLIFING_CONTEXT_END, start);
  if (end < 0) {
    return instructions.slice(0, start);
  }
  return `${instructions.slice(0, start)}${instructions.slice(end + MIDLIFING_CONTEXT_END.length)}`;
}

function normaliseContext(value: string): string {
  return value.trim().replace(/\s+/g, " ");
}

function updateMayaResponseState(event: unknown, currentState: boolean): boolean {
  if (!isRealtimeServerEvent(event) || typeof event.type !== "string") {
    return currentState;
  }
  if (event.type === "response.created" || event.type === "response.output_item.added") {
    return true;
  }
  if (
    event.type === "response.done" ||
    event.type === "response.cancelled" ||
    event.type === "response.output_audio_transcript.done"
  ) {
    return false;
  }
  return currentState;
}

async function loadBaseInstructions(): Promise<string> {
  try {
    const response = await fetch("/maya/session-config", { headers: { Accept: "application/json" } });
    if (!response.ok) {
      return "";
    }
    const payload = (await response.json()) as { instructions?: unknown };
    return typeof payload.instructions === "string" ? payload.instructions : "";
  } catch {
    return "";
  }
}

async function retrieveMidlifingContext(
  utterance: string,
  rollingContext: string[]
): Promise<string> {
  const response = await fetch("/maya/retrieve-context", {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      utterance,
      rolling_context: rollingContext
    })
  });
  if (!response.ok) {
    return "";
  }
  const payload = (await response.json()) as RetrieveContextResponse;
  return typeof payload.context_block === "string" ? payload.context_block : "";
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
