import os
import json
import io
from typing import Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# ✅ Updated import (NO .client)
from elevenlabs import ElevenLabs

# Load environment variables
load_dotenv()

# Verify API Keys exist
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is missing in .env")
if not os.getenv("ELEVENLABS_API_KEY"):
    raise ValueError("ELEVENLABS_API_KEY is missing in .env")

app = FastAPI(title="Mentorra Backend")

# Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# --- Data Models ---

class FounderProfile(BaseModel):
    industry: Optional[str] = None
    stage: Optional[str] = None
    key_challenges: Optional[List[str]] = []

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
    next_actions: List[str]
    memory_update: str

class TTSRequest(BaseModel):
    text: str
    voice_id: Optional[str] = "JBFqnCBsd6RMkjVDRZzb"  # Default ElevenLabs voice
    model_id: Optional[str] = "eleven_monolingual_v1"
    output_format: Optional[str] = "mp3_44100_128"

# --- Toolhouse Agent Logic ---

SYSTEM_PROMPT_TEMPLATE = """
You are the Mentorra Routing Agent. You act as the brain behind a founder's mentorship experience.

You will receive:
- role: The current agent role (Router & Mentor Persona)
- user_message: A single string from the founder
- founder_profile: JSON summary of what we know so far
- active_mentor: Current mentor track id if already selected
- memory_context: Previous context of the conversation

Primary goals:
1) Classify the user’s message into exactly ONE mentor track (e.g., "Product", "Sales", "Fundraising", "Leadership", "Growth").
2) Decide whether we should switch mentors or stay on the current one. Prefer stability unless the user’s intent clearly changed.
3) Reply as the selected mentor in a concise, supportive, operator style (no fluff).
4) Ask at most ONE clarifying question, only if necessary to proceed.
5) Provide 2–5 next actions that the founder can do immediately (this week).
6) Update "memory_update" compactly so the next call stays consistent.

Output must be valid JSON matching this schema:
{
  "mentor_track": "string",
  "switched_track": boolean,
  "reply": "string",
  "clarifying_question": "string or null",
  "next_actions": ["action1", "action2"],
  "memory_update": "string summary of new facts"
}
"""

@app.post("/api/mentor-assist", response_model=MentorResponse)
async def mentor_assist(request: UserRequest):
    try:
        user_input_context = f"""
        INPUT DATA:
        - User Message: "{request.user_message}"
        - Active Mentor Track: {request.active_mentor_track if request.active_mentor_track else "None"}
        - Founder Profile: {request.founder_profile.model_dump_json() if request.founder_profile else "Unknown"}
        - Memory Context: {request.memory_context}
        """

        completion = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
                {"role": "user", "content": user_input_context},
            ],
            temperature=0.7,
        )

        raw_content = completion.choices[0].message.content
        data = json.loads(raw_content)
        return MentorResponse(**data)

    except Exception as e:
        print(f"Mentor Assist Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Voice Mode Endpoints ---

@app.post("/api/voice/speak")
async def text_to_speech(request: TTSRequest):
    """
    Converts text into audio using ElevenLabs.
    Streams MP3 audio back to the client.
    """
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
    """
    Transcribes an uploaded audio file into text using OpenAI Whisper.
    """
    try:
        audio_bytes = await file.read()

        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = file.filename or "audio.mp3"

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
