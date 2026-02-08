import logging
from typing import AsyncGenerator

from openai import AsyncOpenAI

from backend.services.providers.base import BaseLLMProvider, StreamChunk

logger = logging.getLogger(__name__)


class PerplexityProvider(BaseLLMProvider):
    """Perplexity Sonar models via OpenAI-compatible API."""

    def __init__(self, api_key: str):
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai",
        )

    def get_provider_name(self) -> str:
        return "perplexity"

    async def stream_chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[StreamChunk, None]:
        stream = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        usage_found = False
        async for chunk in stream:
            logger.debug("Perplexity chunk: %s", chunk)
            if chunk.usage:
                usage_found = True
                yield StreamChunk(
                    is_final=True,
                    input_tokens=chunk.usage.prompt_tokens or 0,
                    output_tokens=chunk.usage.completion_tokens or 0,
                )
            elif chunk.choices and chunk.choices[0].delta.content is not None:
                yield StreamChunk(text=chunk.choices[0].delta.content)
            elif chunk.choices and chunk.choices[0].finish_reason:
                yield StreamChunk(finish_reason=chunk.choices[0].finish_reason)
        # Fallback: if Perplexity didn't provide usage via streaming,
        # emit a final chunk so downstream code always sees is_final=True
        if not usage_found:
            yield StreamChunk(is_final=True, input_tokens=0, output_tokens=0)
