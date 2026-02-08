from dataclasses import dataclass
from typing import Optional


@dataclass
class Conversation:
    id: str
    title: str
    model_id: str
    system_prompt: str
    created_at: str
    updated_at: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0


@dataclass
class Message:
    id: str
    conversation_id: str
    role: str
    content: str
    model_id: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    used_docs: int = 0
    created_at: str = ""


@dataclass
class Document:
    id: str
    filename: str
    file_type: str
    file_size: int = 0
    chunk_count: int = 0
    conversation_id: Optional[str] = None
    uploaded_at: str = ""
