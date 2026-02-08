import io
import logging

from openai import OpenAI

from backend.config import Settings

logger = logging.getLogger(__name__)


class VoiceService:
    def __init__(self, settings: Settings):
        self._client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
        self._stt_model = settings.voice_config.get("stt_model", "whisper-1")
        self._tts_model = settings.voice_config.get("tts_model", "tts-1")
        self._tts_voice = settings.voice_config.get("tts_voice", "nova")
        self._tts_format = settings.voice_config.get("tts_response_format", "mp3")

    def transcribe(self, audio_bytes: bytes, filename: str = "recording.wav") -> dict:
        """Transcribe audio bytes using OpenAI Whisper."""
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = filename

        transcript = self._client.audio.transcriptions.create(
            model=self._stt_model,
            file=audio_file,
            response_format="verbose_json",
        )

        return {
            "text": transcript.text,
            "audio_duration_seconds": getattr(transcript, "duration", 0.0),
        }

    def synthesize(self, text: str, voice: str | None = None) -> bytes:
        """Convert text to speech using OpenAI TTS. Returns audio bytes."""
        voice = voice or self._tts_voice

        response = self._client.audio.speech.create(
            model=self._tts_model,
            voice=voice,
            input=text,
            response_format=self._tts_format,
        )

        return response.content
