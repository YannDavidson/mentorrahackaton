import os
import json
import io
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI, APIError
from dotenv import load_dotenv

# ✅ Updated import (NO .client)
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
    user_message: str
    founder_profile: Optional[FounderProfile] = None
    active_mentor_track: Optional[str] = None
    memory_context: Optional[str] = ""

class MentorResponse(BaseModel):
    mentor_track: str
    switched_track: bool
    reply: str
    clarifying_question: Optional[str] = None
    # ✅ HTML expects list[str] (ids or name fragments)
    suggested_agents: List[str] = Field(default_factory=list)
    memory_update: str = ""

class TTSRequest(BaseModel):
    text: str
    voice_id: Optional[str] = "JBFqnCBsd6RMkjVDRZzb"
    model_id: Optional[str] = "eleven_monolingual_v1"
    output_format: Optional[str] = "mp3_44100_128"

# -------------------------
# Router config
# -------------------------
AGENTS = [
    {"id": "vincent_forge", "name": "Vincent Forge"},
    {"id": "katerina_catalyst", "name": "Katerina Catalyst"},
    {"id": "sophia_architect", "name": "Sophia Architect"},
    {"id": "adrian_insight", "name": "Adrian Insight"},
]

AGENT_ID_MAP = {
    "vincent forge": "vincent_forge",
    "katerina catalyst": "katerina_catalyst",
    "sophia architect": "sophia_architect",
    "adrian insight": "adrian_insight",
}

def normalize_suggested_agents(val: Any) -> List[str]:
    """
    Accepts:
      - ["vincent_forge", "Sophia Architect", ...]
      - [{"Vincent Forge": 0.95}, {"Sophia Architect": 0.87}]  (legacy)
    Returns:
      - ["vincent_forge", "sophia_architect", ...] (deduped)
    """
    if not val:
        return []

    out: List[str] = []

    if isinstance(val, list):
        for item in val:
            if isinstance(item, str):
                s = item.strip().lower()
                # allow passing id directly
                if s in {a["id"] for a in AGENTS}:
                    out.append(s)
                else:
                    out.append(AGENT_ID_MAP.get(s, item.strip()))
            elif isinstance(item, dict) and item:
                # legacy dict form
                k = next(iter(item.keys()))
                s = str(k).strip().lower()
                out.append(AGENT_ID_MAP.get(s, str(k).strip()))
    elif isinstance(val, dict) and val:
        # sometimes model might output dict {"vincent_forge":0.9,...}
        for k in val.keys():
            s = str(k).strip().lower()
            out.append(AGENT_ID_MAP.get(s, str(k).strip()))

    # dedupe preserve order
    seen = set()
    deduped = []
    for x in out:
        # normalize ids to lowercase snake for consistency if matches
        lx = x.strip().lower()
        if lx in {a["id"] for a in AGENTS}:
            x = lx
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped

def should_restrict_agents(suggested: List[str], selected_track: str) -> List[str]:
    """
    If router is not confident / not really suggesting, return [] so the UI shows all.
    We implement a simple heuristic:
      - If list is empty -> no restriction
      - If list contains all 4 agents -> treat as no restriction (UI shows all)
      - If top suggestion does not match selected track -> still allow restriction (fine)
    """
    if not suggested:
        return []
    # If router returns all 4, treat as "no specific suggestions"
    ids = [s for s in suggested if s in {a["id"] for a in AGENTS}]
    if len(set(ids)) >= 4:
        return []
    return suggested

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
# Health (optional but useful)
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
        user_input_context = f"""
INPUT DATA:
- User Message: "{request.user_message}"
- Active Mentor Track: {request.active_mentor_track if request.active_mentor_track else "None"}
- Founder Profile: {request.founder_profile.model_dump_json() if request.founder_profile else "Unknown"}
- Memory Context: {request.memory_context}
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

        raw_content = completion.choices[0].message.content
        data = json.loads(raw_content)

        # -------------------------
        # Normalize + defaults
        # -------------------------
        data.setdefault("mentor_track", request.active_mentor_track or "Adrian Insight")
        data.setdefault("switched_track", False)
        data.setdefault("reply", "")
        data.setdefault("clarifying_question", None)
        data.setdefault("memory_update", "")

        # normalize suggested_agents to list[str]
        suggested = normalize_suggested_agents(data.get("suggested_agents"))
        # optionally restrict only if it's a real suggestion
        suggested = should_restrict_agents(suggested, str(data.get("mentor_track", "")))

        data["suggested_agents"] = suggested

        # Return validated response
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
