import websocket
import json
import threading
import pyaudio
import base64
import os

# audio config — must match what ElevenLabs expects
CHUNK = 4096
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # ElevenLabs expects 16kHz PCM


p = pyaudio.PyAudio()
connected = threading.Event()
agent_speaking = threading.Event()



connected = threading.Event()
agent_responded = threading.Event()  # new — gates input after each send

memory = []

def on_open(ws):
    print("Connected! Start chatting.\n")
    ws.send(json.dumps({
        "type": "conversation_initiation_client_data",
        
    }))
    connected.set()
    #agent_responded.set()  # allow first input immediately

def on_message(ws, message):
    msg = json.loads(message)
    msg_type = msg.get("type")

    if msg_type == "conversation_initiation_metadata":
        print("[Ready — speak now]")
        # start mic capture once ready
        threading.Thread(target=send_audio, args=(ws,), daemon=True).start()

    elif msg_type == "audio":
        audio_b64 = msg.get("audio_event", {}).get("audio_base_64")
        if audio_b64:
            play_audio(base64.b64decode(audio_b64))

    elif msg_type == "agent_response":
        print(f"\nAgent: {msg.get('agent_response_event', {}).get('agent_response')}")

    elif msg_type == "user_transcript":
        print(f"You: {msg.get('user_transcription_event', {}).get('user_transcript')}")

    elif msg_type == "ping":
        ws.send(json.dumps({
            "type": "pong",
            "event_id": msg.get("ping_event", {}).get("event_id")
        }))

    elif msg_type == "interruption":
        print("[interrupted]")

def play_audio(audio_bytes):
    """Play decoded audio bytes"""
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=16000,  # match agent output format from initiation metadata
        output=True
    )
    stream.write(audio_bytes)
    stream.stop_stream()
    stream.close()


def send_audio(ws):
    """Capture mic and stream to ElevenLabs"""
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    print("[Mic open]")

    while connected.is_set():
        data = stream.read(CHUNK, exception_on_overflow=False)
        encoded = base64.b64encode(data).decode("utf-8")
        ws.send(json.dumps({
            "user_audio_chunk": encoded
        }))

    stream.stop_stream()
    stream.close()

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print(f"Closed — code: {close_status_code}, msg: {close_msg}")
    connected.clear()
    agent_responded.set()  # unblock input loop so it can exit cleanly

def input_loop(ws):
    connected.wait()

    while connected.is_set():
        try:
            agent_responded.wait()        # wait for agent to finish
            agent_responded.clear()       # reset before prompting

            if not connected.is_set():    # connection dropped while waiting
                break

            user_input = input("You: ")

            if user_input.lower() in ("exit", "quit"):
                ws.close()
                break

            if ws.sock and ws.sock.connected:
                print("sending ", user_input)
                ws.send(json.dumps({
                    "type": "user_message",
                    "text": user_input
                }))
                # agent_responded stays cleared — loop blocks until reply comes in
            else:
                print("[Lost connection]")
                break

        except (EOFError, KeyboardInterrupt):
            ws.close()
            break

def connect_agent_websocket(agent_id: str, api_key: str):
    url = f"wss://api.elevenlabs.io/v1/convai/conversation?agent_id={agent_id}"

    ws = websocket.WebSocketApp(
        url,
        header=[f"xi-api-key: {api_key}"],
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    thread = threading.Thread(target=input_loop, args=(ws,), daemon=True)
    thread.start()

    ws.run_forever()

if __name__ == "__main__":
    id = os.getenv("AGENT_ID")
    key = os.getenv("ELEVENLABS_API_KEY")
    connect_agent_websocket(id, key)