import asyncio
import logging
from typing import AsyncGenerator
from google import genai
from google.genai import types

from backend.services.providers.base import BaseLLMProvider, StreamChunk

logger = logging.getLogger(__name__)


class GeminiProvider(BaseLLMProvider):
    def __init__(self, api_key: str):
        self._client = genai.Client(api_key=api_key)

    def get_provider_name(self) -> str:
        return "google"

    async def stream_chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> AsyncGenerator[StreamChunk, None]:
        # Convert OpenAI-format messages to Gemini contents format
        contents = []
        system_instruction = None
        for m in messages:
            if m["role"] == "system":
                if system_instruction:
                    system_instruction += "\n" + m["content"]
                else:
                    system_instruction = m["content"]
            elif m["role"] == "user":
                contents.append(
                    types.Content(role="user", parts=[types.Part(text=m["content"])])
                )
            elif m["role"] == "assistant":
                contents.append(
                    types.Content(role="model", parts=[types.Part(text=m["content"])])
                )

        # Enable Google Search grounding so Gemini can access real-time info
        grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            tools=[grounding_tool],
        )
        if system_instruction:
            config.system_instruction = system_instruction

        # google-genai generate_content_stream is synchronous --
        # run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()

        def _sync_stream():
            chunks = []
            try:
                for chunk in self._client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=config,
                ):
                    chunks.append(chunk)
            except Exception as e:
                logger.error("Gemini streaming error: %s", e, exc_info=True)
                raise
            return chunks

        chunks = await loop.run_in_executor(None, _sync_stream)

        total_input = 0
        total_output = 0
        grounding_citations = []

        for chunk in chunks:
            if chunk.text:
                yield StreamChunk(text=chunk.text)
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                total_input = getattr(chunk.usage_metadata, "prompt_token_count", 0) or 0
                total_output = getattr(chunk.usage_metadata, "candidates_token_count", 0) or 0

            # Extract grounding metadata (search sources) from candidates
            if hasattr(chunk, "candidates") and chunk.candidates:
                for candidate in chunk.candidates:
                    gm = getattr(candidate, "grounding_metadata", None)
                    if gm:
                        # Extract grounding chunks (the actual web sources)
                        g_chunks = getattr(gm, "grounding_chunks", None)
                        if g_chunks:
                            for gc in g_chunks:
                                web = getattr(gc, "web", None)
                                if web:
                                    citation = {
                                        "url": getattr(web, "uri", "") or "",
                                        "title": getattr(web, "title", "") or "",
                                        "source": "google_search",
                                    }
                                    # Deduplicate by URL
                                    if citation["url"] and citation not in grounding_citations:
                                        grounding_citations.append(citation)

                        # Log search queries used
                        queries = getattr(gm, "web_search_queries", None)
                        if queries:
                            logger.info("Gemini grounding search queries: %s", queries)

        yield StreamChunk(
            is_final=True,
            input_tokens=total_input,
            output_tokens=total_output,
            citations=grounding_citations,
        )
