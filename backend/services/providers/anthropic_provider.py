from typing import AsyncGenerator
from anthropic import AsyncAnthropic

from backend.services.providers.base import BaseLLMProvider, StreamChunk


class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key: str):
        self._client = AsyncAnthropic(api_key=api_key)

    def get_provider_name(self) -> str:
        return "anthropic"

    async def stream_chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> AsyncGenerator[StreamChunk, None]:
        # Anthropic requires system message extracted from the messages list
        system_msg = ""
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg = (system_msg + "\n" + m["content"]).strip()
            else:
                chat_messages.append({"role": m["role"], "content": m["content"]})

        # Ensure messages alternate user/assistant (Anthropic requirement)
        # If first message isn't from user, this will cause an API error,
        # but that's an upstream issue to handle in the chat router.

        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": chat_messages,
        }
        if system_msg:
            kwargs["system"] = system_msg

        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield StreamChunk(text=text)

            final = await stream.get_final_message()
            yield StreamChunk(
                is_final=True,
                input_tokens=final.usage.input_tokens,
                output_tokens=final.usage.output_tokens,
            )
