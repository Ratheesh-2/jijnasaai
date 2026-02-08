import pytest
from backend.services.llm_router import LLMRouter


class TestLLMRouter:
    def test_get_provider_name(self, test_settings):
        router = LLMRouter(test_settings)
        assert router.get_provider_name("gpt-4o") == "openai"
        assert router.get_provider_name("gpt-4o-mini") == "openai"
        assert router.get_provider_name("claude-sonnet-4-5-20250929") == "anthropic"
        assert router.get_provider_name("claude-haiku-4-5-20251001") == "anthropic"
        assert router.get_provider_name("gemini-2.0-flash") == "google"
        assert router.get_provider_name("gemini-1.5-pro") == "google"
        assert router.get_provider_name("sonar-pro") == "perplexity"
        assert router.get_provider_name("sonar") == "perplexity"
        assert router.get_provider_name("sonar-reasoning-pro") == "perplexity"

    def test_unknown_model_raises(self, test_settings):
        router = LLMRouter(test_settings)
        with pytest.raises(ValueError, match="Unknown model"):
            router._get_provider("nonexistent-model")

    def test_get_available_models(self, test_settings):
        router = LLMRouter(test_settings)
        models = router.get_available_models()
        # All providers have fake keys, so all models should be available
        assert len(models) == 9
        model_ids = [m["id"] for m in models]
        assert "gpt-4o" in model_ids
        assert "claude-sonnet-4-5-20250929" in model_ids
        assert "gemini-2.0-flash" in model_ids
        assert "sonar-pro" in model_ids
