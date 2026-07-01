import {
  attachSessionBehaviour,
  type TranscriptUpdate
} from "./sessionBehaviour";

const OPENAI_REALTIME_WEBRTC_URL = "https://api.openai.com/v1/realtime/calls";

export type RealtimeEventLogEntry = {
  id: string;
  direction: "system" | "server";
  type: string;
  payload: unknown;
  receivedAt: string;
};

export type RealtimeConnection = {
  peerConnection: RTCPeerConnection;
  dataChannel: RTCDataChannel;
  localStream: MediaStream;
  remoteStream: MediaStream;
  disconnect: () => void;
};

type ConnectRealtimeOptions = {
  ephemeralKey: string;
  remoteAudioElement: HTMLAudioElement;
  onEvent: (event: RealtimeEventLogEntry) => void;
  onTranscript: (update: TranscriptUpdate) => void;
};

export async function connectRealtimeSession({
  ephemeralKey,
  remoteAudioElement,
  onEvent,
  onTranscript
}: ConnectRealtimeOptions): Promise<RealtimeConnection> {
  const peerConnection = new RTCPeerConnection();
  const localStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const remoteStream = new MediaStream();
  const dataChannel = peerConnection.createDataChannel("oai-events");
  attachDataChannelLogging(dataChannel, onEvent);
  const detachSessionBehaviour = attachSessionBehaviour(dataChannel, { onTranscript });

  peerConnection.ontrack = (event) => {
    for (const track of event.streams[0]?.getAudioTracks() ?? [event.track]) {
      remoteStream.addTrack(track);
    }
    remoteAudioElement.srcObject = remoteStream;
  };

  for (const track of localStream.getAudioTracks()) {
    peerConnection.addTrack(track, localStream);
  }

  try {
    const offer = await peerConnection.createOffer();
    await peerConnection.setLocalDescription(offer);

    if (!offer.sdp) {
      throw new Error("Realtime SDP offer did not include SDP text.");
    }

    const answerSdp = await postSdpOffer(ephemeralKey, offer.sdp);
    await peerConnection.setRemoteDescription({
      type: "answer",
      sdp: answerSdp
    });

    return {
      peerConnection,
      dataChannel,
      localStream,
      remoteStream,
      disconnect: () =>
        disconnectRealtimeSession(
          peerConnection,
          dataChannel,
          localStream,
          remoteAudioElement,
          detachSessionBehaviour
        )
    };
  } catch (error) {
    disconnectRealtimeSession(
      peerConnection,
      dataChannel,
      localStream,
      remoteAudioElement,
      detachSessionBehaviour
    );
    throw error;
  }
}

function attachDataChannelLogging(
  dataChannel: RTCDataChannel,
  onEvent: (event: RealtimeEventLogEntry) => void
): void {
  dataChannel.addEventListener("open", () => {
    onEvent(createLogEntry("system", "data_channel.open", "oai-events"));
  });

  dataChannel.addEventListener("close", () => {
    onEvent(createLogEntry("system", "data_channel.close", "oai-events"));
  });

  dataChannel.addEventListener("error", (event) => {
    onEvent(createLogEntry("system", "data_channel.error", describeDataChannelError(event)));
  });

  dataChannel.addEventListener("message", (event) => {
    const payload = parseDataChannelMessage(event.data);
    onEvent(createLogEntry("server", inferEventType(payload), payload));
  });
}

function createLogEntry(
  direction: RealtimeEventLogEntry["direction"],
  type: string,
  payload: unknown
): RealtimeEventLogEntry {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
    direction,
    type,
    payload,
    receivedAt: new Date().toISOString()
  };
}

function parseDataChannelMessage(data: unknown): unknown {
  if (typeof data !== "string") {
    return data;
  }

  try {
    return JSON.parse(data) as unknown;
  } catch {
    return data;
  }
}

function inferEventType(payload: unknown): string {
  if (payload && typeof payload === "object" && "type" in payload) {
    const maybeType = (payload as { type?: unknown }).type;
    if (typeof maybeType === "string") {
      return maybeType;
    }
  }
  return "message";
}

function describeDataChannelError(event: Event): string {
  if ("error" in event) {
    const maybeError = (event as RTCErrorEvent).error;
    return maybeError.message || maybeError.name;
  }
  return "Data channel error";
}

async function postSdpOffer(ephemeralKey: string, offerSdp: string): Promise<string> {
  const response = await fetch(OPENAI_REALTIME_WEBRTC_URL, {
    method: "POST",
    body: offerSdp,
    headers: {
      Authorization: `Bearer ${ephemeralKey}`,
      "Content-Type": "application/sdp"
    }
  });

  if (!response.ok) {
    throw new Error(`Realtime WebRTC offer failed with status ${response.status}`);
  }

  return response.text();
}

function disconnectRealtimeSession(
  peerConnection: RTCPeerConnection,
  dataChannel: RTCDataChannel,
  localStream: MediaStream,
  remoteAudioElement: HTMLAudioElement,
  detachSessionBehaviour: () => void
): void {
  detachSessionBehaviour();
  if (dataChannel.readyState === "open" || dataChannel.readyState === "connecting") {
    dataChannel.close();
  }
  for (const sender of peerConnection.getSenders()) {
    sender.track?.stop();
  }
  for (const track of localStream.getTracks()) {
    track.stop();
  }
  peerConnection.close();
  remoteAudioElement.srcObject = null;
}
