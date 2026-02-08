import logging

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import Response

from backend.models.schemas import TranscriptionResponse, SynthesisRequest
from backend.services.voice_service import VoiceService
from backend.services.cost_tracker import CostTracker
from backend.dependencies import get_voice_service, get_cost_tracker

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/voice", tags=["voice"])


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    voice_service: VoiceService = Depends(get_voice_service),
    cost_tracker: CostTracker = Depends(get_cost_tracker),
):
    """Transcribe audio file to text using Whisper."""
    try:
        audio_bytes = await file.read()
        result = voice_service.transcribe(audio_bytes, file.filename or "audio.wav")

        duration_minutes = result.get("audio_duration_seconds", 0) / 60.0
        cost = cost_tracker.calculate_stt_cost(duration_minutes)
        await cost_tracker.log_cost(
            model_id="whisper-1",
            operation="stt",
            audio_minutes=duration_minutes,
            cost_usd=cost,
        )

        return TranscriptionResponse(**result)
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/synthesize")
async def synthesize_speech(
    request: SynthesisRequest,
    voice_service: VoiceService = Depends(get_voice_service),
    cost_tracker: CostTracker = Depends(get_cost_tracker),
):
    """Convert text to speech. Returns audio bytes."""
    try:
        audio_bytes = voice_service.synthesize(request.text, request.voice)

        cost = cost_tracker.calculate_tts_cost(len(request.text))
        await cost_tracker.log_cost(
            model_id="tts-1",
            operation="tts",
            tts_characters=len(request.text),
            cost_usd=cost,
        )

        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=speech.mp3"},
        )
    except Exception as e:
        logger.exception("TTS failed")
        raise HTTPException(status_code=500, detail=str(e))
