from abc import ABC, abstractmethod
from typing import AsyncGenerator
from dataclasses import dataclass, field


@dataclass
class StreamChunk:
    """Normalized token chunk from any LLM provider."""
    text: str = ""
    is_final: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str | None = None
    citations: list[dict] = field(default_factory=list)  # Web search sources


class BaseLLMProvider(ABC):
    @abstractmethod
    async def stream_chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Yield StreamChunk objects as tokens arrive."""
        ...

    @abstractmethod
    def get_provider_name(self) -> str:
        ...
