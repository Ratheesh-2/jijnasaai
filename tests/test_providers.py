"""Tests for LLM provider changes: StreamChunk citations, provider behaviors."""

import asyncio
from dataclasses import fields
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services.providers.base import StreamChunk


class TestStreamChunk:
    """Test the StreamChunk dataclass has the expected fields."""

    def test_default_citations_empty(self):
        chunk = StreamChunk(text="hello")
        assert chunk.citations == []
        assert chunk.text == "hello"
        assert chunk.is_final is False

    def test_citations_field_exists(self):
        field_names = [f.name for f in fields(StreamChunk)]
        assert "citations" in field_names

    def test_citations_not_shared_between_instances(self):
        """Ensure citations list is not shared across instances (mutable default)."""
        chunk1 = StreamChunk()
        chunk2 = StreamChunk()
        chunk1.citations.append({"url": "http://test.com"})
        assert len(chunk2.citations) == 0  # Should NOT be affected

    def test_final_chunk_with_citations(self):
        citations = [
            {"url": "http://example.com", "title": "Example", "source": "perplexity"},
            {"url": "http://google.com", "title": "Google", "source": "google_search"},
        ]
        chunk = StreamChunk(
            is_final=True,
            input_tokens=100,
            output_tokens=200,
            citations=citations,
        )
        assert chunk.is_final is True
        assert chunk.input_tokens == 100
        assert chunk.output_tokens == 200
        assert len(chunk.citations) == 2
        assert chunk.citations[0]["source"] == "perplexity"
        assert chunk.citations[1]["source"] == "google_search"

    def test_text_chunk_no_citations(self):
        """Regular text chunks should have empty citations."""
        chunk = StreamChunk(text="Hello world")
        assert chunk.text == "Hello world"
        assert chunk.citations == []
        assert chunk.is_final is False


class TestPerplexityProviderConfig:
    """Test Perplexity provider configuration."""

    def test_provider_has_timeout(self):
        """Verify the provider has a timeout configured."""
        from backend.services.providers.perplexity_provider import _REQUEST_TIMEOUT
        assert _REQUEST_TIMEOUT > 0
        assert _REQUEST_TIMEOUT <= 300  # Reasonable upper bound

    def test_provider_name(self):
        """Test provider name without needing an API key."""
        from backend.services.providers.perplexity_provider import PerplexityProvider
        provider = PerplexityProvider(api_key="fake-key")
        assert provider.get_provider_name() == "perplexity"


class TestGeminiProviderConfig:
    """Test Gemini provider configuration."""

    def test_provider_name(self):
        """Test provider name without needing an API key."""
        from backend.services.providers.gemini_provider import GeminiProvider
        provider = GeminiProvider(api_key="fake-key")
        assert provider.get_provider_name() == "google"


# ---------------------------------------------------------------------------
# Mocked Perplexity provider tests
# ---------------------------------------------------------------------------

def _make_mock_response(
    content: str = "Hello from Perplexity",
    citations: list | None = None,
    prompt_tokens: int = 50,
    completion_tokens: int = 100,
    choices_empty: bool = False,
):
    """Build a mock ChatCompletion response mimicking Perplexity non-streaming."""
    # Mock usage
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    # Mock message inside choice
    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    # Mock response
    response = MagicMock()
    response.choices = [] if choices_empty else [choice]
    response.usage = usage

    # Citations can come via attribute or model_extra
    if citations is not None:
        response.citations = citations
        response.model_extra = {"citations": citations}
    else:
        response.citations = None
        response.model_extra = {}

    return response


def _make_provider_with_mock(mock_response=None, side_effect=None):
    """Create a PerplexityProvider with a mocked AsyncOpenAI client."""
    from backend.services.providers.perplexity_provider import PerplexityProvider

    provider = PerplexityProvider(api_key="fake-key")

    mock_create = AsyncMock()
    if side_effect:
        mock_create.side_effect = side_effect
    else:
        mock_create.return_value = mock_response

    provider._client = MagicMock()
    provider._client.chat = MagicMock()
    provider._client.chat.completions = MagicMock()
    provider._client.chat.completions.create = mock_create

    return provider


SAMPLE_MESSAGES = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is the weather?"},
]


@pytest.mark.asyncio
class TestPerplexityProviderHappyPath:
    """Test the non-streaming Perplexity provider with mocked responses."""

    async def test_text_and_citations(self):
        """Happy path: response has text + string-URL citations."""
        mock_resp = _make_mock_response(
            content="Tesla stock is $250 today.",
            citations=["https://finance.yahoo.com/tsla", "https://reuters.com/tsla"],
            prompt_tokens=30,
            completion_tokens=15,
        )
        provider = _make_provider_with_mock(mock_response=mock_resp)

        chunks = []
        async for chunk in provider.stream_chat(
            messages=SAMPLE_MESSAGES,
            model="sonar-pro",
        ):
            chunks.append(chunk)

        # Should have 2 chunks: text + final
        assert len(chunks) == 2

        text_chunk = chunks[0]
        assert text_chunk.text == "Tesla stock is $250 today."
        assert text_chunk.is_final is False

        final_chunk = chunks[1]
        assert final_chunk.is_final is True
        assert final_chunk.input_tokens == 30
        assert final_chunk.output_tokens == 15
        assert len(final_chunk.citations) == 2
        assert final_chunk.citations[0]["url"] == "https://finance.yahoo.com/tsla"
        assert final_chunk.citations[0]["source"] == "perplexity"
        assert final_chunk.citations[1]["url"] == "https://reuters.com/tsla"

    async def test_text_without_citations(self):
        """Response has text but no citations (some queries don't trigger search)."""
        mock_resp = _make_mock_response(
            content="The square root of 144 is 12.",
            citations=None,
            prompt_tokens=20,
            completion_tokens=10,
        )
        provider = _make_provider_with_mock(mock_response=mock_resp)

        chunks = []
        async for chunk in provider.stream_chat(
            messages=SAMPLE_MESSAGES,
            model="sonar",
        ):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].text == "The square root of 144 is 12."
        assert chunks[1].is_final is True
        assert chunks[1].citations == []
        assert chunks[1].input_tokens == 20
        assert chunks[1].output_tokens == 10

    async def test_dict_citations(self):
        """Citations can also come as dicts with url and title."""
        mock_resp = _make_mock_response(
            content="AI news today.",
            citations=[
                {"url": "https://example.com/ai", "title": "AI Article"},
                {"url": "https://example.com/ml", "title": "ML Article"},
            ],
        )
        provider = _make_provider_with_mock(mock_response=mock_resp)

        chunks = []
        async for chunk in provider.stream_chat(
            messages=SAMPLE_MESSAGES,
            model="sonar-pro",
        ):
            chunks.append(chunk)

        final = chunks[1]
        assert len(final.citations) == 2
        assert final.citations[0]["title"] == "AI Article"
        assert final.citations[1]["title"] == "ML Article"

    async def test_duplicate_citation_urls_deduplicated(self):
        """Duplicate URLs should be deduplicated."""
        mock_resp = _make_mock_response(
            content="Some answer.",
            citations=[
                "https://example.com/page",
                "https://example.com/page",  # duplicate
                "https://other.com/page",
            ],
        )
        provider = _make_provider_with_mock(mock_response=mock_resp)

        chunks = []
        async for chunk in provider.stream_chat(
            messages=SAMPLE_MESSAGES,
            model="sonar-pro",
        ):
            chunks.append(chunk)

        final = chunks[1]
        assert len(final.citations) == 2  # Deduplicated from 3 to 2


@pytest.mark.asyncio
class TestPerplexityProviderErrors:
    """Test error handling in the Perplexity provider."""

    async def test_empty_choices(self):
        """Empty choices array should yield a fallback message."""
        mock_resp = _make_mock_response(choices_empty=True)
        provider = _make_provider_with_mock(mock_response=mock_resp)

        chunks = []
        async for chunk in provider.stream_chat(
            messages=SAMPLE_MESSAGES,
            model="sonar-pro",
        ):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert "No response received" in chunks[0].text
        assert chunks[1].is_final is True

    async def test_none_message_content(self):
        """message.content is None — should yield fallback message."""
        mock_resp = _make_mock_response(content=None)
        # Force content to be None (the helper sets it via MagicMock)
        mock_resp.choices[0].message.content = None
        provider = _make_provider_with_mock(mock_response=mock_resp)

        chunks = []
        async for chunk in provider.stream_chat(
            messages=SAMPLE_MESSAGES,
            model="sonar-pro",
        ):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert "No response received" in chunks[0].text
        assert chunks[1].is_final is True

    async def test_timeout_error(self):
        """asyncio.TimeoutError should yield a timeout message, not raise."""
        provider = _make_provider_with_mock(
            side_effect=asyncio.TimeoutError("timed out")
        )

        chunks = []
        async for chunk in provider.stream_chat(
            messages=SAMPLE_MESSAGES,
            model="sonar-pro",
        ):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert "timed out" in chunks[0].text.lower()
        assert chunks[1].is_final is True
        assert chunks[1].input_tokens == 0
        assert chunks[1].output_tokens == 0

    async def test_api_exception(self):
        """Generic API exception should yield an error message, not raise."""
        provider = _make_provider_with_mock(
            side_effect=RuntimeError("API key invalid")
        )

        chunks = []
        async for chunk in provider.stream_chat(
            messages=SAMPLE_MESSAGES,
            model="sonar-pro",
        ):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert "API key invalid" in chunks[0].text
        assert chunks[1].is_final is True
        assert chunks[1].input_tokens == 0

    async def test_empty_string_content(self):
        """message.content is empty string — should yield fallback."""
        mock_resp = _make_mock_response(content="")
        mock_resp.choices[0].message.content = ""
        provider = _make_provider_with_mock(mock_response=mock_resp)

        chunks = []
        async for chunk in provider.stream_chat(
            messages=SAMPLE_MESSAGES,
            model="sonar-pro",
        ):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert "No response received" in chunks[0].text
        assert chunks[1].is_final is True
