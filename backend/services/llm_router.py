import logging
from typing import AsyncGenerator

from backend.config import Settings
from backend.services.providers.base import BaseLLMProvider, StreamChunk
from backend.services.providers.openai_provider import OpenAIProvider
from backend.services.providers.anthropic_provider import AnthropicProvider
from backend.services.providers.gemini_provider import GeminiProvider
from backend.services.providers.perplexity_provider import PerplexityProvider

logger = logging.getLogger(__name__)


class LLMRouter:
    """Routes chat requests to the appropriate LLM provider based on model_id."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._providers: dict[str, BaseLLMProvider] = {}
        self._model_provider_map: dict[str, str] = {}

        # Build provider instances for each configured API key
        openai_key = settings.openai_api_key.get_secret_value()
        if openai_key:
            self._providers["openai"] = OpenAIProvider(openai_key)

        anthropic_key = settings.anthropic_api_key.get_secret_value()
        if anthropic_key:
            self._providers["anthropic"] = AnthropicProvider(anthropic_key)

        google_key = settings.google_api_key.get_secret_value()
        if google_key:
            self._providers["google"] = GeminiProvider(google_key)

        perplexity_key = settings.perplexity_api_key.get_secret_value()
        if perplexity_key:
            self._providers["perplexity"] = PerplexityProvider(perplexity_key)

        # Map model IDs to providers from YAML config
        for model_cfg in settings.models_config:
            self._model_provider_map[model_cfg["id"]] = model_cfg["provider"]

    def _get_provider(self, model_id: str) -> BaseLLMProvider:
        provider_name = self._model_provider_map.get(model_id)
        if not provider_name:
            raise ValueError(f"Unknown model: {model_id}")
        provider = self._providers.get(provider_name)
        if not provider:
            raise ValueError(
                f"Provider '{provider_name}' not configured. "
                f"Set the API key in .env for this provider."
            )
        return provider

    def get_provider_name(self, model_id: str) -> str:
        return self._model_provider_map.get(model_id, "unknown")

    def get_available_models(self) -> list[dict]:
        """Return models whose providers have API keys configured."""
        available = []
        for model_cfg in self._settings.models_config:
            provider_name = model_cfg["provider"]
            if provider_name in self._providers:
                available.append(model_cfg)
        return available

    async def stream_chat(
        self,
        messages: list[dict],
        model_id: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[StreamChunk, None]:
        provider = self._get_provider(model_id)
        logger.info(f"Routing to {provider.get_provider_name()} for model {model_id}")
        async for chunk in provider.stream_chat(
            messages=messages,
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield chunk
