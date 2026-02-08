import asyncio
import logging
from typing import AsyncGenerator

from openai import AsyncOpenAI

from backend.services.providers.base import BaseLLMProvider, StreamChunk

logger = logging.getLogger(__name__)

# Timeout for the entire Perplexity streaming response (seconds).
# Perplexity can sometimes hang silently -- this prevents infinite waits.
_STREAM_TIMEOUT = 120.0


class PerplexityProvider(BaseLLMProvider):
    """Perplexity Sonar models via OpenAI-compatible API."""

    def __init__(self, api_key: str):
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai",
            timeout=_STREAM_TIMEOUT,
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
        citations = []
        usage_found = False
        got_any_text = False

        try:
            stream = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                ),
                timeout=30.0,  # Timeout for initial connection
            )

            async for chunk in stream:
                logger.debug("Perplexity chunk: %s", chunk)

                # Extract citations from the chunk's raw data.
                # Perplexity returns citations as a top-level field in the
                # response JSON alongside choices/usage.  The OpenAI SDK
                # doesn't model this field, so we access it via the raw
                # model_extra / __dict__ on the chunk object.
                raw_citations = (
                    getattr(chunk, "citations", None)
                    or (chunk.model_extra or {}).get("citations")
                )
                if raw_citations and isinstance(raw_citations, list):
                    for i, url in enumerate(raw_citations):
                        if isinstance(url, str) and url not in [c["url"] for c in citations]:
                            citations.append({
                                "url": url,
                                "title": f"Source {i + 1}",
                                "source": "perplexity",
                            })

                if chunk.usage:
                    usage_found = True
                    yield StreamChunk(
                        is_final=True,
                        input_tokens=chunk.usage.prompt_tokens or 0,
                        output_tokens=chunk.usage.completion_tokens or 0,
                        citations=citations,
                    )
                elif chunk.choices and chunk.choices[0].delta.content is not None:
                    got_any_text = True
                    yield StreamChunk(text=chunk.choices[0].delta.content)
                elif chunk.choices and chunk.choices[0].finish_reason:
                    yield StreamChunk(finish_reason=chunk.choices[0].finish_reason)

        except asyncio.TimeoutError:
            logger.error(
                "Perplexity stream timed out for model %s after %.0fs",
                model, _STREAM_TIMEOUT,
            )
            if got_any_text:
                yield StreamChunk(text="\n\n*[Response timed out]*")
            else:
                yield StreamChunk(text="*Sorry, the request timed out. Please try again.*")
            yield StreamChunk(is_final=True, input_tokens=0, output_tokens=0, citations=citations)
            return
        except Exception as e:
            logger.error("Perplexity streaming error for model %s: %s", model, e, exc_info=True)
            if not got_any_text:
                yield StreamChunk(text=f"*Error from Perplexity: {e}*")
            yield StreamChunk(is_final=True, input_tokens=0, output_tokens=0, citations=citations)
            return

        # Fallback: if Perplexity didn't provide usage via streaming,
        # emit a final chunk so downstream code always sees is_final=True
        if not usage_found:
            logger.warning("Perplexity did not return usage data for model %s", model)
            yield StreamChunk(is_final=True, input_tokens=0, output_tokens=0, citations=citations)
