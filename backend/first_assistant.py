import os
import json
import io
import re
import base64
from typing import Optional, List, Dict, Any, Tuple, Literal
from dataclasses import dataclass, field
from threading import Lock
from time import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI, APIError
from dotenv import load_dotenv
from elevenlabs import ElevenLabs

# -------------------------
# Env + app setup
# -------------------------
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is missing in .env")
if not os.getenv("ELEVENLABS_API_KEY"):
    raise ValueError("ELEVENLABS_API_KEY is missing in .env")

app = FastAPI(title="Mentorra Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ok for dev; lock down for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# -------------------------
# Models
# -------------------------
class FounderProfile(BaseModel):
    industry: Optional[str] = None
    stage: Optional[str] = None
    key_challenges: List[str] = Field(default_factory=list)


class UnifiedAssistRequest(BaseModel):
    session_id: Optional[str] = None
    mode: Literal["text", "voice"] = "text"

    # text path
    user_message: Optional[str] = None

    # voice path: send base64 audio from browser MediaRecorder chunks or a whole clip
    # examples: audio/webm, audio/wav, audio/mp4, audio/mpeg
    audio_base64: Optional[str] = None
    audio_mime_type: Optional[str] = "audio/webm"
    audio_filename: Optional[str] = None

    founder_profile: Optional[FounderProfile] = None
    active_mentor_track: Optional[str] = None
    memory_context: Optional[str] = ""
    set_preferred_mentor: Optional[bool] = False

    # optional UI helpers
    selected_agents: List[str] = Field(default_factory=list)
    voice_id: Optional[str] = "JBFqnCBsd6RMkjVDRZzb"
    tts_model_id: Optional[str] = "eleven_turbo_v2_5"
    tts_output_format: Optional[str] = "mp3_44100_128"


class MentorResponse(BaseModel):
    mentor_track: str
    switched_track: bool
    reply: str
    clarifying_question: Optional[str] = None
    suggested_agents: List[str] = Field(default_factory=list)
    memory_update: str = ""
    preferred_mentor: Optional[str] = None
    session_id: Optional[str] = None

    # unified additions
    mode: Literal["text", "voice"] = "text"
    transcript: Optional[str] = None
    audio_base64: Optional[str] = None
    audio_mime_type: Optional[str] = None


# -------------------------
# Mentor roster
# -------------------------
AGENTS = [
    {"id": "vincent_forge", "name": "Vincent Forge"},
    {"id": "katerina_catalyst", "name": "Katerina Catalyst"},
    {"id": "sophia_architect", "name": "Sophia Architect"},
    {"id": "adrian_insight", "name": "Adrian Insight"},
]

AGENT_IDS = {a["id"] for a in AGENTS}
AGENT_NAMES = {a["name"] for a in AGENTS}
NAME_TO_ID = {a["name"].strip().lower(): a["id"] for a in AGENTS}
ID_TO_NAME = {a["id"]: a["name"] for a in AGENTS}

ALIASES_TO_ID = {
    "vincent": "vincent_forge",
    "forge": "vincent_forge",
    "vincent forge": "vincent_forge",
    "katerina": "katerina_catalyst",
    "catalyst": "katerina_catalyst",
    "katerina catalyst": "katerina_catalyst",
    "sophia": "sophia_architect",
    "architect": "sophia_architect",
    "sophia architect": "sophia_architect",
    "adrian": "adrian_insight",
    "insight": "adrian_insight",
    "adrian insight": "adrian_insight",
}


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def coerce_track_to_id(track: Optional[str]) -> Optional[str]:
    t = _norm(track)
    if not t:
        return None
    if t in AGENT_IDS:
        return t
    if t in NAME_TO_ID:
        return NAME_TO_ID[t]
    if t in ALIASES_TO_ID:
        return ALIASES_TO_ID[t]
    t2 = re.sub(r"[^a-z0-9_ ]+", "", t).strip()
    if t2 in AGENT_IDS:
        return t2
    if t2 in NAME_TO_ID:
        return NAME_TO_ID[t2]
    if t2 in ALIASES_TO_ID:
        return ALIASES_TO_ID[t2]
    return None


def normalize_suggested_agents(val: Any) -> List[str]:
    if not val:
        return []

    out: List[str] = []
    if isinstance(val, list):
        for item in val:
            if isinstance(item, str):
                mid = coerce_track_to_id(item)
                if mid:
                    out.append(mid)
            elif isinstance(item, dict) and item:
                k = next(iter(item.keys()))
                mid = coerce_track_to_id(str(k))
                if mid:
                    out.append(mid)
    elif isinstance(val, dict):
        for k in val.keys():
            mid = coerce_track_to_id(str(k))
            if mid:
                out.append(mid)

    seen = set()
    deduped: List[str] = []
    for x in out:
        if x in AGENT_IDS and x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped


def should_restrict_agents(suggested: List[str]) -> List[str]:
    if not suggested:
        return []
    if len(set(suggested)) >= len(AGENTS):
        return []
    return suggested


def extract_explicit_mentor_from_user_text(user_message: str) -> Optional[str]:
    msg = _norm(user_message)
    if not msg:
        return None

    mentioned_id: Optional[str] = None
    for alias, mid in ALIASES_TO_ID.items():
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, msg):
            mentioned_id = mid
            break

    if not mentioned_id:
        return None

    explicit_phrases = [
        "switch", "change mentor", "different mentor", "use", "talk to",
        "can i get", "give me", "i want", "route me to"
    ]
    if any(p in msg for p in explicit_phrases):
        return mentioned_id

    if msg.strip() in ALIASES_TO_ID:
        return mentioned_id

    return None


def should_switch_mentor(
    active_id: Optional[str],
    routed_id: Optional[str],
    suggested_ids: List[str],
    user_message: str,
) -> Tuple[bool, Optional[str]]:
    if not active_id:
        return True, None

    forced = extract_explicit_mentor_from_user_text(user_message)
    if forced:
        return True, forced

    if not routed_id or routed_id == active_id:
        return False, None

    if not suggested_ids:
        return False, None

    top = suggested_ids[0]
    if top == routed_id:
        return True, None

    return False, None


# -------------------------
# In-memory session store
# -------------------------
@dataclass
class SessionState:
    preferred_mentor_id: Optional[str] = None
    current_mentor_id: Optional[str] = None
    accept_count: Dict[str, int] = field(default_factory=dict)
    switch_count: int = 0
    last_seen_ts: float = field(default_factory=lambda: time())
    memory_context: str = ""


SESSION_STORE: Dict[str, SessionState] = {}
SESSION_LOCK = Lock()


def get_session(session_id: str) -> SessionState:
    sid = (session_id or "").strip() or "default"
    with SESSION_LOCK:
        st = SESSION_STORE.get(sid)
        if not st:
            st = SessionState()
            SESSION_STORE[sid] = st
        st.last_seen_ts = time()
        return st


# -------------------------
# Router prompt
# -------------------------
SYSTEM_PROMPT_TEMPLATE = """
You are the Mentorra Routing Agent. You act as the brain behind a founder's mentorship experience.

You have access to the following roster of Elite Mentors. You must analyze the user's input to determine which of these specific mentors is best suited to help them based on their current problem, stage, and emotional state.

### MENTOR ROSTER
1. Vincent Forge — The Impossible Builder
2. Katerina Catalyst — The Scrappy Disruptor
3. Sophia Architect — The Experience Designer
4. Adrian Insight — The Startup Sage

### INSTRUCTIONS
You will receive:
- user_message: a single string from the founder
- founder_profile: JSON summary of what we know so far
- active_mentor: current mentor track name if already selected
- memory_context: previous context of the conversation

Primary goals:
1) Analyze which mentor fits best and decide whether switching is necessary.
2) Select a single best mentor track (use the mentor's name, e.g., \"Vincent Forge\").
3) Reply AS the selected mentor (voice/style matching).
4) Memory: update \"memory_update\" compactly.

### IMPORTANT: suggested_agents behavior for the UI
- If you have a strong preference, set \"suggested_agents\" to a list of mentor IDs in order, like:
  [\"vincent_forge\",\"sophia_architect\"]
- If you do NOT have a strong preference (unclear, general, or user is just exploring), set:
  \"suggested_agents\": []

Valid mentor IDs:
- vincent_forge
- katerina_catalyst
- sophia_architect
- adrian_insight

### OUTPUT FORMAT (JSON ONLY)
Return valid JSON matching this schema exactly:
{
  \"mentor_track\": \"string\",
  \"switched_track\": boolean,
  \"reply\": \"string\",
  \"clarifying_question\": \"string or null\",
  \"suggested_agents\": [\"string\", \"...\"],
  \"memory_update\": \"string\"
}
"""


# -------------------------
# Helpers
# -------------------------
def decode_audio_to_bytes(audio_base64: str) -> bytes:
    try:
        if "," in audio_base64 and audio_base64.strip().startswith("data:"):
            audio_base64 = audio_base64.split(",", 1)[1]
        return base64.b64decode(audio_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio payload: {exc}")


def guess_audio_filename(mime_type: Optional[str], filename: Optional[str]) -> str:
    if filename:
        return filename
    mapping = {
        "audio/webm": "audio.webm",
        "audio/wav": "audio.wav",
        "audio/x-wav": "audio.wav",
        "audio/mpeg": "audio.mp3",
        "audio/mp3": "audio.mp3",
        "audio/mp4": "audio.mp4",
        "audio/aac": "audio.aac",
        "audio/ogg": "audio.ogg",
    }
    return mapping.get((mime_type or "").lower(), "audio.webm")


def transcribe_audio_bytes_with_openai(audio_bytes: bytes, filename: str) -> str:
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename
    transcript = openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
    )
    return transcript.text


def synthesize_text_to_base64_audio(text: str, voice_id: str, model_id: str, output_format: str) -> Tuple[str, str]:
    audio_stream = elevenlabs_client.text_to_speech.convert(
        voice_id=voice_id,
        model_id=model_id,
        text=text,
        output_format=output_format,
    )
    audio_bytes = b"".join(chunk for chunk in audio_stream if chunk)

    mime_type = "audio/mpeg"
    if output_format.startswith("mp3"):
        mime_type = "audio/mpeg"
    elif output_format.startswith("pcm"):
        mime_type = "audio/wav"

    return base64.b64encode(audio_bytes).decode("utf-8"), mime_type


def run_router(request: UnifiedAssistRequest, effective_user_message: str) -> Dict[str, Any]:
    sid = (request.session_id or "").strip() or "default"
    st = get_session(sid)

    active_id = (
        coerce_track_to_id(request.active_mentor_track)
        or st.preferred_mentor_id
        or st.current_mentor_id
    )

    if request.memory_context:
        st.memory_context = request.memory_context

    user_input_context = f"""
INPUT DATA:
- User Message: "{effective_user_message}"
- Active Mentor Track: {request.active_mentor_track if request.active_mentor_track else (ID_TO_NAME.get(active_id) if active_id else "None")}
- Founder Profile: {request.founder_profile.model_dump_json() if request.founder_profile else "Unknown"}
- Memory Context: {request.memory_context or st.memory_context}
""".strip()

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
                {"role": "user", "content": user_input_context},
            ],
            temperature=0.7,
        )
    except APIError:
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
                {"role": "user", "content": user_input_context},
            ],
            temperature=0.7,
        )

    data = json.loads(completion.choices[0].message.content)
    data.setdefault("mentor_track", ID_TO_NAME.get(active_id, "Adrian Insight"))
    data.setdefault("switched_track", False)
    data.setdefault("reply", "")
    data.setdefault("clarifying_question", None)
    data.setdefault("memory_update", "")
    data.setdefault("suggested_agents", [])

    suggested_ids = normalize_suggested_agents(data.get("suggested_agents"))
    suggested_ids = should_restrict_agents(suggested_ids)
    data["suggested_agents"] = suggested_ids

    routed_id = coerce_track_to_id(data.get("mentor_track"))
    do_switch, forced_target_id = should_switch_mentor(
        active_id=active_id,
        routed_id=routed_id,
        suggested_ids=suggested_ids,
        user_message=effective_user_message,
    )

    if forced_target_id:
        final_id = forced_target_id
        switched = active_id is not None and final_id != active_id
    else:
        if not active_id:
            final_id = routed_id or "adrian_insight"
            switched = False
        elif do_switch:
            final_id = routed_id or active_id
            switched = final_id != active_id
        else:
            final_id = active_id
            switched = False

    st.current_mentor_id = final_id
    st.accept_count[final_id] = st.accept_count.get(final_id, 0) + 1
    if switched:
        st.switch_count += 1

    if request.set_preferred_mentor:
        st.preferred_mentor_id = final_id

    data["mentor_track"] = ID_TO_NAME.get(final_id, "Adrian Insight")
    data["switched_track"] = bool(switched)

    if active_id and not switched:
        data["suggested_agents"] = []

    data["preferred_mentor"] = ID_TO_NAME.get(st.preferred_mentor_id) if st.preferred_mentor_id else None
    data["session_id"] = sid
    return data


# -------------------------
# Health
# -------------------------
@app.get("/health")
def health():
    return {"ok": True}


# -------------------------
# Unified endpoint
# -------------------------
@app.post("/api/assist", response_model=MentorResponse)
async def assist(request: UnifiedAssistRequest):
    try:
        transcript: Optional[str] = None
        effective_user_message = (request.user_message or "").strip()

        if request.mode == "voice":
            if not request.audio_base64:
                raise HTTPException(status_code=400, detail="audio_base64 is required when mode='voice'.")

            audio_bytes = decode_audio_to_bytes(request.audio_base64)
            filename = guess_audio_filename(request.audio_mime_type, request.audio_filename)
            transcript = transcribe_audio_bytes_with_openai(audio_bytes, filename)
            effective_user_message = transcript.strip()

        if not effective_user_message:
            raise HTTPException(status_code=400, detail="user_message is required for text mode, or provide transcribable audio for voice mode.")

        data = run_router(request, effective_user_message)

        audio_base64 = None
        audio_mime_type = None
        if request.mode == "voice" and data.get("reply"):
            audio_base64, audio_mime_type = synthesize_text_to_base64_audio(
                text=data["reply"],
                voice_id=request.voice_id or "JBFqnCBsd6RMkjVDRZzb",
                model_id=request.tts_model_id or "eleven_turbo_v2_5",
                output_format=request.tts_output_format or "mp3_44100_128",
            )

        return MentorResponse(
            **data,
            mode=request.mode,
            transcript=transcript,
            audio_base64=audio_base64,
            audio_mime_type=audio_mime_type,
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Unified Assist Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# Optional backward-compatible endpoints
# -------------------------
@app.post("/api/mentor-assist", response_model=MentorResponse)
async def mentor_assist_compat(request: UnifiedAssistRequest):
    request.mode = "text"
    return await assist(request)


@app.post("/api/voice/speak")
async def text_to_speech_stream(text: str, voice_id: str = "JBFqnCBsd6RMkjVDRZzb", model_id: str = "eleven_turbo_v2_5", output_format: str = "mp3_44100_128"):
    try:
        audio_stream = elevenlabs_client.text_to_speech.convert(
            voice_id=voice_id,
            model_id=model_id,
            text=text,
            output_format=output_format,
        )

        def iterfile():
            for chunk in audio_stream:
                if chunk:
                    yield chunk

        return StreamingResponse(iterfile(), media_type="audio/mpeg")
    except Exception as e:
        print(f"TTS Error: {e}")
        raise HTTPException(status_code=500, detail=f"Text-to-speech failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
