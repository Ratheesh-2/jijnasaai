"""Tests for LLM provider changes: StreamChunk citations, provider behaviors."""

import pytest
from dataclasses import fields

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
        from backend.services.providers.perplexity_provider import _STREAM_TIMEOUT
        assert _STREAM_TIMEOUT > 0
        assert _STREAM_TIMEOUT <= 300  # Reasonable upper bound

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
