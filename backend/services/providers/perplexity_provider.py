import asyncio
import logging
from typing import AsyncGenerator

from openai import AsyncOpenAI

from backend.services.providers.base import BaseLLMProvider, StreamChunk

logger = logging.getLogger(__name__)

# Timeout for the Perplexity API call (seconds).
# Perplexity does web search before answering, so allow generous time.
_REQUEST_TIMEOUT = 60.0


class PerplexityProvider(BaseLLMProvider):
    """Perplexity Sonar models via OpenAI-compatible API.

    Uses non-streaming (stream=False) as the primary mode because
    Perplexity's streaming responses deviate from the OpenAI spec:
    - They return chat.completion objects (not chat.completion.chunk)
    - delta.content is often empty while message.content has the text
    - Citations are only reliably present in non-streaming responses

    See: https://github.com/BerriAI/litellm/issues/8455
    See: https://github.com/BerriAI/litellm/issues/13777
    """

    def __init__(self, api_key: str):
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai",
            timeout=_REQUEST_TIMEOUT,
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
        """Call Perplexity API (non-streaming) and yield the result.

        Non-streaming is used for reliability:
        - text is always in response.choices[0].message.content
        - citations are always in response.citations / model_extra
        - no delta vs message ambiguity
        - no stream-hang risk

        The tradeoff is no token-by-token animation, but Perplexity
        already takes a few seconds for web search, so the UX difference
        is minimal.
        """
        citations: list[dict] = []

        try:
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                ),
                timeout=_REQUEST_TIMEOUT,
            )

            logger.info(
                "Perplexity response received: model=%s choices=%d",
                model, len(response.choices) if response.choices else 0,
            )

            # --- Extract text ---
            text = ""
            if response.choices and len(response.choices) > 0:
                msg = response.choices[0].message
                if msg and msg.content:
                    text = msg.content

            if not text:
                logger.warning(
                    "Perplexity returned empty text for model %s. "
                    "Response object: %s",
                    model, response,
                )

            # --- Extract citations ---
            raw_citations = (
                getattr(response, "citations", None)
                or (getattr(response, "model_extra", None) or {}).get("citations")
            )
            if raw_citations and isinstance(raw_citations, list):
                for i, item in enumerate(raw_citations):
                    if isinstance(item, str):
                        url = item
                        if url and url not in [c["url"] for c in citations]:
                            citations.append({
                                "url": url,
                                "title": f"Source {i + 1}",
                                "source": "perplexity",
                            })
                    elif isinstance(item, dict):
                        url = item.get("url", "")
                        title = item.get("title", f"Source {i + 1}")
                        if url and url not in [c["url"] for c in citations]:
                            citations.append({
                                "url": url,
                                "title": title,
                                "source": "perplexity",
                            })

            logger.info(
                "Perplexity result: model=%s text_len=%d citations=%d",
                model, len(text), len(citations),
            )

            # --- Extract usage ---
            input_tokens = 0
            output_tokens = 0
            if response.usage:
                input_tokens = response.usage.prompt_tokens or 0
                output_tokens = response.usage.completion_tokens or 0

            # --- Yield results ---
            # Yield the full text as a single chunk (no streaming animation)
            if text:
                yield StreamChunk(text=text)
            else:
                yield StreamChunk(
                    text="*No response received from Perplexity. Please try again.*"
                )

            yield StreamChunk(
                is_final=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                citations=citations,
            )

        except asyncio.TimeoutError:
            logger.error(
                "Perplexity request timed out for model %s after %.0fs",
                model, _REQUEST_TIMEOUT,
            )
            yield StreamChunk(
                text="*Request to Perplexity timed out. Please try again.*"
            )
            yield StreamChunk(
                is_final=True,
                input_tokens=0,
                output_tokens=0,
                citations=citations,
            )

        except Exception as e:
            logger.error(
                "Perplexity API error for model %s: %s",
                model, e, exc_info=True,
            )
            yield StreamChunk(text=f"*Error from Perplexity: {e}*")
            yield StreamChunk(
                is_final=True,
                input_tokens=0,
                output_tokens=0,
                citations=citations,
            )
