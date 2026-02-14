import json
import os
from typing import Optional

import httpx
from httpx_sse import connect_sse

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")


class APIClient:
    def __init__(self, base_url: str = BACKEND_URL):
        self.base_url = base_url

    # --- Streaming chat (synchronous for Streamlit) ---

    def stream_chat(
        self,
        message: str,
        model_id: str,
        conversation_id: Optional[str] = None,
        use_rag: bool = False,
        temperature: float = 0.7,
    ):
        """Synchronous SSE streaming for use in Streamlit."""
        payload = {
            "message": message,
            "model_id": model_id,
            "conversation_id": conversation_id,
            "use_rag": use_rag,
            "temperature": temperature,
        }
        with httpx.Client(timeout=httpx.Timeout(300.0)) as client:
            with connect_sse(
                client, "POST", f"{self.base_url}/chat/completions",
                json=payload,
            ) as event_source:
                for sse in event_source.iter_sse():
                    try:
                        data = json.loads(sse.data) if sse.data else {}
                    except json.JSONDecodeError:
                        data = {"raw": sse.data}
                    yield {
                        "event": sse.event,
                        "data": data,
                    }

    # --- REST calls ---

    def list_conversations(self) -> list[dict]:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(f"{self.base_url}/conversations")
            r.raise_for_status()
            return r.json()["conversations"]

    def create_conversation(self, model_id: str, title: str = "New Conversation") -> dict:
        with httpx.Client(timeout=10.0) as client:
            r = client.post(
                f"{self.base_url}/conversations",
                json={"model_id": model_id, "title": title},
            )
            r.raise_for_status()
            return r.json()

    def get_conversation(self, conversation_id: str) -> dict:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(f"{self.base_url}/conversations/{conversation_id}")
            r.raise_for_status()
            return r.json()

    def get_messages(self, conversation_id: str) -> list[dict]:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(f"{self.base_url}/conversations/{conversation_id}/messages")
            r.raise_for_status()
            return r.json()

    def delete_conversation(self, conversation_id: str):
        with httpx.Client(timeout=10.0) as client:
            r = client.delete(f"{self.base_url}/conversations/{conversation_id}")
            r.raise_for_status()

    def update_system_prompt(self, conversation_id: str, system_prompt: str):
        with httpx.Client(timeout=10.0) as client:
            r = client.put(
                f"{self.base_url}/conversations/{conversation_id}/system-prompt",
                json={"system_prompt": system_prompt},
            )
            r.raise_for_status()

    def upload_document(
        self, file_bytes: bytes, filename: str, conversation_id: str | None = None
    ) -> dict:
        with httpx.Client(timeout=120.0) as client:
            files = {"file": (filename, file_bytes)}
            data = {}
            if conversation_id:
                data["conversation_id"] = conversation_id
            r = client.post(
                f"{self.base_url}/documents/upload", files=files, data=data
            )
            r.raise_for_status()
            return r.json()

    def list_documents(self, conversation_id: str | None = None) -> list[dict]:
        with httpx.Client(timeout=10.0) as client:
            params = {}
            if conversation_id:
                params["conversation_id"] = conversation_id
            r = client.get(f"{self.base_url}/documents", params=params)
            r.raise_for_status()
            return r.json()["documents"]

    def transcribe_audio(
        self, audio_bytes: bytes, filename: str = "recording.wav"
    ) -> dict:
        with httpx.Client(timeout=60.0) as client:
            files = {"file": (filename, audio_bytes)}
            r = client.post(f"{self.base_url}/voice/transcribe", files=files)
            r.raise_for_status()
            return r.json()

    def synthesize_speech(self, text: str, voice: str = "nova") -> bytes:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(
                f"{self.base_url}/voice/synthesize",
                json={"text": text, "voice": voice},
            )
            r.raise_for_status()
            return r.content

    def get_cost_summary(self, conversation_id: str | None = None) -> dict:
        with httpx.Client(timeout=10.0) as client:
            params = {}
            if conversation_id:
                params["conversation_id"] = conversation_id
            r = client.get(f"{self.base_url}/costs/summary", params=params)
            r.raise_for_status()
            return r.json()

    def log_analytics_event(self, event_type: str, event_data: dict | None = None):
        """Fire-and-forget analytics event."""
        try:
            with httpx.Client(timeout=5.0) as client:
                client.post(
                    f"{self.base_url}/analytics/event",
                    json={
                        "event_type": event_type,
                        "event_data": event_data or {},
                    },
                )
        except Exception:
            pass  # Non-critical — never block the UI

    def get_analytics_summary(self, days: int = 30) -> dict:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(
                f"{self.base_url}/analytics/summary",
                params={"days": days},
            )
            r.raise_for_status()
            return r.json()

    def get_suggestions(self) -> list[str]:
        """Fetch dynamic suggested prompts (3s timeout, fallback on failure)."""
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.get(f"{self.base_url}/suggestions")
                r.raise_for_status()
                return r.json().get("suggestions", [])
        except Exception:
            return []

    def health_check(self) -> dict:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(f"{self.base_url}/health")
            r.raise_for_status()
            return r.json()
