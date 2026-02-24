import os
import json
import io
import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from threading import Lock
from time import time

from fastapi import FastAPI, HTTPException, UploadFile, File
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

class UserRequest(BaseModel):
    # NEW: session id to persist state in-memory for testing
    session_id: Optional[str] = None

    user_message: str
    founder_profile: Optional[FounderProfile] = None
    active_mentor_track: Optional[str] = None  # user-selected / preferred / current mentor (string)
    memory_context: Optional[str] = ""

    # NEW: if true, lock the returned mentor as the preferred long-run mentor
    set_preferred_mentor: Optional[bool] = False

class MentorResponse(BaseModel):
    mentor_track: str                 # always mentor NAME
    switched_track: bool
    reply: str
    clarifying_question: Optional[str] = None
    suggested_agents: List[str] = Field(default_factory=list)  # list of IDs
    memory_update: str = ""

    # NEW: debug / useful for UI testing
    preferred_mentor: Optional[str] = None   # mentor NAME if stored
    session_id: Optional[str] = None         # echo back

class TTSRequest(BaseModel):
    text: str
    voice_id: Optional[str] = "JBFqnCBsd6RMkjVDRZzb"
    model_id: Optional[str] = "eleven_monolingual_v1"
    output_format: Optional[str] = "mp3_44100_128"

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

# helpful aliases (lowercase keys)
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
    """
    Accepts mentor track as either:
      - exact name ("Sophia Architect")
      - id ("sophia_architect")
      - alias ("sophia", "architect")
    Returns:
      - mentor id or None
    """
    t = _norm(track)
    if not t:
        return None
    if t in AGENT_IDS:
        return t
    if t in NAME_TO_ID:
        return NAME_TO_ID[t]
    if t in ALIASES_TO_ID:
        return ALIASES_TO_ID[t]
    # try fuzzy: remove punctuation
    t2 = re.sub(r"[^a-z0-9_ ]+", "", t).strip()
    if t2 in AGENT_IDS:
        return t2
    if t2 in NAME_TO_ID:
        return NAME_TO_ID[t2]
    if t2 in ALIASES_TO_ID:
        return ALIASES_TO_ID[t2]
    return None

def coerce_track_to_name(track: Optional[str], fallback_name: str = "Adrian Insight") -> str:
    """
    Ensures response mentor_track is always a mentor NAME.
    """
    mid = coerce_track_to_id(track)
    if mid and mid in ID_TO_NAME:
        return ID_TO_NAME[mid]
    # If already matches a known name
    t = (track or "").strip()
    if t in AGENT_NAMES:
        return t
    return fallback_name

def normalize_suggested_agents(val: Any) -> List[str]:
    """
    Accepts:
      - ["vincent_forge", "Sophia Architect", ...]
      - [{"Vincent Forge": 0.95}, {"Sophia Architect": 0.87}]  (legacy)
      - {"vincent_forge":0.9, ...}
    Returns:
      - ["vincent_forge", "sophia_architect", ...] (deduped)
    """
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
    elif isinstance(val, dict) and val:
        for k in val.keys():
            mid = coerce_track_to_id(str(k))
            if mid:
                out.append(mid)

    # dedupe preserve order
    seen = set()
    deduped: List[str] = []
    for x in out:
        if x in AGENT_IDS and x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped

def should_restrict_agents(suggested: List[str]) -> List[str]:
    """
    If router isn't confident / is basically suggesting everyone, return [] so the UI shows all.
    """
    if not suggested:
        return []
    if len(set(suggested)) >= len(AGENTS):
        return []
    return suggested

def extract_explicit_mentor_from_user_text(user_message: str) -> Optional[str]:
    """
    Detect explicit user intention to use/switch to a specific mentor.
    Examples:
      "switch to sophia"
      "talk to adrian"
      "use vincent forge"
      "can i get katerina?"
    Returns:
      mentor id or None
    """
    msg = _norm(user_message)
    if not msg:
        return None

    mentioned_id: Optional[str] = None

    # Look for full names / ids / common aliases as whole words
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
    user_message: str
) -> Tuple[bool, Optional[str]]:
    """
    Returns:
      (should_switch, forced_target_id_if_any)

    Rules:
      1) If no active mentor -> allow router to pick (switch = True)
      2) If user explicitly asks for a mentor -> force switch to that mentor
      3) If router isn't confident (suggested empty) -> don't switch
      4) If router is confident and top suggestion matches routed mentor -> allow switch
      Otherwise keep active.
    """
    if not active_id:
        return True, None  # new/exploring: allow router

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
# In-memory session store (for testing)
# -------------------------
@dataclass
class SessionState:
    preferred_mentor_id: Optional[str] = None   # long-run default
    current_mentor_id: Optional[str] = None     # last used mentor
    accept_count: Dict[str, int] = field(default_factory=dict)
    switch_count: int = 0
    last_seen_ts: float = field(default_factory=lambda: time())
    memory_context: str = ""  # optional: store memory for testing

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
2) Select a single best mentor track (use the mentor's name, e.g., "Vincent Forge").
3) Reply AS the selected mentor (voice/style matching).
4) Memory: update "memory_update" compactly.

### IMPORTANT: suggested_agents behavior for the UI
- If you have a strong preference, set "suggested_agents" to a list of mentor IDs in order, like:
  ["vincent_forge","sophia_architect"]
- If you do NOT have a strong preference (unclear, general, or user is just exploring), set:
  "suggested_agents": []
This lets the UI show all mentors by default.

Valid mentor IDs:
- vincent_forge
- katerina_catalyst
- sophia_architect
- adrian_insight

### OUTPUT FORMAT (JSON ONLY)

Return valid JSON matching this schema exactly:
{
  "mentor_track": "string",
  "switched_track": boolean,
  "reply": "string",
  "clarifying_question": "string or null",
  "suggested_agents": ["string", "..."],
  "memory_update": "string"
}
"""

# -------------------------
# Health
# -------------------------
@app.get("/health")
def health():
    return {"ok": True}

# -------------------------
# Mentorra main route
# -------------------------
@app.post("/api/mentor-assist", response_model=MentorResponse)
async def mentor_assist(request: UserRequest):
    try:
        # Load session state (in-memory)
        sid = (request.session_id or "").strip() or "default"
        st = get_session(sid)

        # Determine active mentor preference priority:
        # 1) explicit request.active_mentor_track (from UI)
        # 2) session.preferred_mentor_id (locked-in long-run mentor)
        # 3) session.current_mentor_id (last used)
        active_id = (
            coerce_track_to_id(request.active_mentor_track)
            or st.preferred_mentor_id
            or st.current_mentor_id
        )

        # optional: keep session memory_context updated (for testing)
        if request.memory_context:
            st.memory_context = request.memory_context

        user_input_context = f"""
INPUT DATA:
- User Message: "{request.user_message}"
- Active Mentor Track: {request.active_mentor_track if request.active_mentor_track else (ID_TO_NAME.get(active_id) if active_id else "None")}
- Founder Profile: {request.founder_profile.model_dump_json() if request.founder_profile else "Unknown"}
- Memory Context: {request.memory_context or st.memory_context}
""".strip()

        # Call router model
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

        raw_content = completion.choices[0].message.content
        data = json.loads(raw_content)

        # -------------------------
        # Defaults + normalization
        # -------------------------
        data.setdefault("mentor_track", ID_TO_NAME.get(active_id, "Adrian Insight"))
        data.setdefault("switched_track", False)
        data.setdefault("reply", "")
        data.setdefault("clarifying_question", None)
        data.setdefault("memory_update", "")
        data.setdefault("suggested_agents", [])

        # Normalize suggested_agents to list[IDs]
        suggested_ids = normalize_suggested_agents(data.get("suggested_agents"))
        suggested_ids = should_restrict_agents(suggested_ids)  # [] if not confident / all
        data["suggested_agents"] = suggested_ids

        # Normalize routed mentor track -> ID
        routed_id = coerce_track_to_id(data.get("mentor_track"))

        # Decide switching policy (sticky mentor)
        do_switch, forced_target_id = should_switch_mentor(
            active_id=active_id,
            routed_id=routed_id,
            suggested_ids=suggested_ids,
            user_message=request.user_message,
        )

        final_id: Optional[str] = None
        switched = False

        if forced_target_id:
            # explicit user request overrides everything
            final_id = forced_target_id
            switched = (active_id is not None and final_id != active_id)
        else:
            if not active_id:
                # no current preference -> accept router choice (fallback to Adrian)
                final_id = routed_id or "adrian_insight"
                switched = False
            else:
                if do_switch:
                    final_id = routed_id or active_id
                    switched = (final_id != active_id)
                else:
                    final_id = active_id
                    switched = False

        # Persist session state
        st.current_mentor_id = final_id
        st.accept_count[final_id] = st.accept_count.get(final_id, 0) + 1
        if switched:
            st.switch_count += 1

        # If user wants to lock this mentor in for the long run
        if request.set_preferred_mentor:
            st.preferred_mentor_id = final_id

        # Set response mentor_track as NAME (always)
        data["mentor_track"] = ID_TO_NAME.get(final_id, "Adrian Insight")
        data["switched_track"] = bool(switched)

        # If we didn't switch and we're sticking with active mentor,
        # avoid restricting the UI based on a non-switching router suggestion.
        if active_id and not switched:
            data["suggested_agents"] = []

        # Return session debug info
        data["preferred_mentor"] = ID_TO_NAME.get(st.preferred_mentor_id) if st.preferred_mentor_id else None
        data["session_id"] = sid

        return MentorResponse(**data)

    except Exception as e:
        print(f"Mentor Assist Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# Voice endpoints
# -------------------------
@app.post("/api/voice/speak")
async def text_to_speech(request: TTSRequest):
    try:
        audio_stream = elevenlabs_client.text_to_speech.convert(
            voice_id=request.voice_id,
            model_id=request.model_id,
            text=request.text,
            output_format=request.output_format,
        )

        def iterfile():
            for chunk in audio_stream:
                if chunk:
                    yield chunk

        return StreamingResponse(iterfile(), media_type="audio/mpeg")

    except Exception as e:
        print(f"TTS Error: {e}")
        raise HTTPException(status_code=500, detail=f"Text-to-speech failed: {str(e)}")

@app.post("/api/voice/transcribe")
async def speech_to_text(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = file.filename or "audio.webm"

        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )

        return {"text": transcript.text}

    except Exception as e:
        print(f"STT Error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)