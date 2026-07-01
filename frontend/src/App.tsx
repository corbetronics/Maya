import { Power, Radio, Unplug } from "lucide-react";
import { useRef, useState } from "react";
import "./App.css";
import {
  connectRealtimeSession,
  type RealtimeConnection,
  type RealtimeEventLogEntry
} from "./realtimeClient";
import type { TranscriptUpdate } from "./sessionBehaviour";

type ConnectionStatus = "idle" | "requesting session" | "connected" | "error";

type EphemeralSessionResponse = {
  client_secret?: {
    value?: string;
  };
};

type TranscriptLine = {
  id: string;
  speaker: TranscriptUpdate["speaker"];
  text: string;
  isPartial: boolean;
};

export function App() {
  const [status, setStatus] = useState<ConnectionStatus>("idle");
  const [clientSecret, setClientSecret] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [realtimeEvents, setRealtimeEvents] = useState<RealtimeEventLogEntry[]>([]);
  const [transcriptLines, setTranscriptLines] = useState<TranscriptLine[]>([]);
  const connectionRef = useRef<RealtimeConnection | null>(null);
  const remoteAudioRef = useRef<HTMLAudioElement | null>(null);

  const addRealtimeEvent = (event: RealtimeEventLogEntry) => {
    setRealtimeEvents((events) => [event, ...events].slice(0, 20));
  };

  const addTranscriptUpdate = (update: TranscriptUpdate) => {
    setTranscriptLines((lines) => mergeTranscriptUpdate(lines, update));
  };

  const connectToMaya = async () => {
    setStatus("requesting session");
    setErrorMessage(null);
    setClientSecret(null);
    connectionRef.current?.disconnect();
    connectionRef.current = null;

    try {
      if (!remoteAudioRef.current) {
        throw new Error("Remote audio element is not ready.");
      }

      const response = await fetch("/maya/ephemeral-session", {
        method: "POST",
        headers: {
          Accept: "application/json"
        }
      });

      if (!response.ok) {
        throw new Error(`Session request failed with status ${response.status}`);
      }

      const payload = (await response.json()) as EphemeralSessionResponse;
      const secret = payload.client_secret?.value;
      if (!secret) {
        throw new Error("Session response did not include a client secret.");
      }

      setClientSecret(secret);
      connectionRef.current = await connectRealtimeSession({
        clientSecret: secret,
        remoteAudioElement: remoteAudioRef.current,
        onEvent: addRealtimeEvent,
        onTranscript: addTranscriptUpdate
      });
      setStatus("connected");
    } catch (error) {
      connectionRef.current?.disconnect();
      connectionRef.current = null;
      setStatus("error");
      setErrorMessage(error instanceof Error ? error.message : "Unable to request a session.");
    }
  };

  const disconnectFromMaya = () => {
    connectionRef.current?.disconnect();
    connectionRef.current = null;
    setClientSecret(null);
    setErrorMessage(null);
    setStatus("idle");
  };

  return (
    <main className="app-shell" aria-label="Project Maya producer console">
      <section className="studio-panel">
        <div className="brand-row">
          <div className="brand-mark" aria-hidden="true">
            <Radio size={26} />
          </div>
          <div>
            <p className="eyebrow">Producer console</p>
            <h1>Project Maya</h1>
          </div>
        </div>

        <div className="session-surface">
          <div className="maya-avatar" aria-hidden="true">
            M
          </div>
          <div>
            <p className="session-label">Guest seat</p>
            <h2>Maya</h2>
            <p className="session-copy">
              Request an ephemeral Realtime session, open WebRTC, and listen for Maya's audio.
            </p>
            <audio ref={remoteAudioRef} className="remote-audio" autoPlay />
          </div>
        </div>

        <div className="control-row" aria-label="Studio controls">
          <button
            type="button"
            className="action-button primary"
            onClick={connectToMaya}
            disabled={status === "requesting session" || status === "connected"}
          >
            <Power size={20} aria-hidden="true" />
            Connect to Maya
          </button>
          <button
            type="button"
            className="action-button"
            onClick={disconnectFromMaya}
            disabled={status === "idle"}
          >
            <Unplug size={20} aria-hidden="true" />
            Disconnect
          </button>
        </div>
      </section>

      <aside className="status-panel" aria-label="System status">
        <h2>Status</h2>
        <div className="status-row">
          <span>Connection</span>
          <strong>{status}</strong>
        </div>
        <div className="status-row">
          <span>Session secret</span>
          <strong>{clientSecret ? "stored" : "none"}</strong>
        </div>
        {errorMessage ? <p className="error-message">{errorMessage}</p> : null}
        <section className="transcript-panel" aria-label="Live transcript">
          <h2>Live transcript</h2>
          {transcriptLines.length === 0 ? (
            <p className="empty-events">No transcript yet.</p>
          ) : (
            <ol className="transcript-list">
              {transcriptLines.map((line) => (
                <li className="transcript-item" key={line.id}>
                  <strong>{line.speaker}</strong>
                  <p>
                    {line.text}
                    {line.isPartial ? <span className="live-marker"> live</span> : null}
                  </p>
                </li>
              ))}
            </ol>
          )}
        </section>
        <section className="events-panel" aria-label="Recent Realtime events">
          <h2>Realtime events</h2>
          {realtimeEvents.length === 0 ? (
            <p className="empty-events">No events yet.</p>
          ) : (
            <ol className="event-list">
              {realtimeEvents.map((event) => (
                <li className="event-item" key={event.id}>
                  <div className="event-heading">
                    <span>{event.type}</span>
                    <time>{formatEventTime(event.receivedAt)}</time>
                  </div>
                  <pre>{formatEventPayload(event.payload)}</pre>
                </li>
              ))}
            </ol>
          )}
        </section>
      </aside>
    </main>
  );
}

function mergeTranscriptUpdate(
  lines: TranscriptLine[],
  update: TranscriptUpdate
): TranscriptLine[] {
  if (update.speaker === "Host") {
    return [
      ...lines,
      {
        id: createTranscriptLineId(),
        speaker: "Host",
        text: update.text,
        isPartial: false
      }
    ];
  }

  const lastLine = lines.at(-1);
  if (update.isPartial) {
    if (lastLine?.speaker === "Maya" && lastLine.isPartial) {
      return [
        ...lines.slice(0, -1),
        {
          ...lastLine,
          text: `${lastLine.text}${update.text}`,
          isPartial: true
        }
      ];
    }
    return [
      ...lines,
      {
        id: createTranscriptLineId(),
        speaker: "Maya",
        text: update.text,
        isPartial: true
      }
    ];
  }

  if (lastLine?.speaker === "Maya" && lastLine.isPartial) {
    return [
      ...lines.slice(0, -1),
      {
        ...lastLine,
        text: update.text,
        isPartial: false
      }
    ];
  }

  return [
    ...lines,
    {
      id: createTranscriptLineId(),
      speaker: "Maya",
      text: update.text,
      isPartial: false
    }
  ];
}

function createTranscriptLineId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function formatEventPayload(payload: unknown): string {
  if (typeof payload === "string") {
    return payload;
  }
  return JSON.stringify(payload, null, 2);
}

function formatEventTime(receivedAt: string): string {
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  }).format(new Date(receivedAt));
}
