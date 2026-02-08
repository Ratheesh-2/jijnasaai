from typing import AsyncGenerator
from openai import AsyncOpenAI

from backend.services.providers.base import BaseLLMProvider, StreamChunk


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str):
        self._client = AsyncOpenAI(api_key=api_key)

    def get_provider_name(self) -> str:
        return "openai"

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
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            if chunk.usage:
                yield StreamChunk(
                    is_final=True,
                    input_tokens=chunk.usage.prompt_tokens,
                    output_tokens=chunk.usage.completion_tokens,
                )
            elif chunk.choices and chunk.choices[0].delta.content is not None:
                yield StreamChunk(text=chunk.choices[0].delta.content)
            elif chunk.choices and chunk.choices[0].finish_reason:
                yield StreamChunk(finish_reason=chunk.choices[0].finish_reason)
