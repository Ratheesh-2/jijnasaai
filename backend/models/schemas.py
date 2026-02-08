from pydantic import BaseModel, Field
from typing import Optional


# --- Chat ---
class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str = Field(..., min_length=1, max_length=50000)
    model_id: str = "gpt-4o"
    use_rag: bool = False
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    model_id: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    used_docs: bool = False
    created_at: str


# --- Conversations ---
class ConversationCreate(BaseModel):
    model_id: str = "gpt-4o"
    title: str = "New Conversation"
    system_prompt: str = ""


class ConversationResponse(BaseModel):
    id: str
    title: str
    model_id: str
    system_prompt: str = ""
    created_at: str
    updated_at: str
    total_cost_usd: float = 0.0
    message_count: int = 0


class ConversationListResponse(BaseModel):
    conversations: list[ConversationResponse]


# --- Documents ---
class DocumentUploadResponse(BaseModel):
    id: str
    filename: str
    chunk_count: int
    file_size: int = 0


class DocumentListResponse(BaseModel):
    documents: list[dict]


# --- Voice ---
class TranscriptionResponse(BaseModel):
    text: str
    audio_duration_seconds: float = 0.0


class SynthesisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096)
    voice: str = "nova"


# --- Health ---
class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "0.1.0"
    document_count: int = 0
    conversation_count: int = 0


# --- Cost ---
class CostSummaryResponse(BaseModel):
    conversation_id: Optional[str] = None
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    breakdown: list[dict] = []
