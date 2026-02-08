import logging
from backend.config import Settings
from backend.database import get_db

logger = logging.getLogger(__name__)


class CostTracker:
    def __init__(self, settings: Settings):
        self._pricing = settings.pricing_config

    def _get_model_pricing(self, model_id: str) -> dict:
        """Look up pricing for a model across all providers."""
        for provider, models in self._pricing.items():
            if model_id in models:
                return models[model_id]
        return {}

    def calculate_chat_cost(
        self, model_id: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate cost in USD for a chat completion (per 1M tokens)."""
        pricing = self._get_model_pricing(model_id)
        if not pricing:
            return 0.0
        input_cost_per_m = pricing.get("input", 0.0)
        output_cost_per_m = pricing.get("output", 0.0)
        cost = (
            (input_tokens / 1_000_000) * input_cost_per_m
            + (output_tokens / 1_000_000) * output_cost_per_m
        )
        return round(cost, 8)

    def calculate_embedding_cost(self, token_count: int) -> float:
        pricing = self._get_model_pricing("text-embedding-3-small")
        input_cost_per_m = pricing.get("input", 0.02)
        return round((token_count / 1_000_000) * input_cost_per_m, 8)

    def calculate_stt_cost(self, audio_minutes: float) -> float:
        pricing = self._get_model_pricing("whisper-1")
        per_minute = pricing.get("per_minute", 0.006)
        return round(audio_minutes * per_minute, 8)

    def calculate_tts_cost(self, character_count: int) -> float:
        pricing = self._get_model_pricing("tts-1")
        per_m_chars = pricing.get("per_million_chars", 15.0)
        return round((character_count / 1_000_000) * per_m_chars, 8)

    async def log_cost(
        self,
        model_id: str,
        operation: str,
        conversation_id: str | None = None,
        message_id: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        audio_minutes: float = 0.0,
        tts_characters: int = 0,
        cost_usd: float = 0.0,
    ):
        """Persist a cost record to the cost_log table."""
        async with get_db() as db:
            await db.execute(
                """INSERT INTO cost_log
                   (conversation_id, message_id, model_id, operation,
                    input_tokens, output_tokens, audio_minutes,
                    tts_characters, cost_usd)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (conversation_id, message_id, model_id, operation,
                 input_tokens, output_tokens, audio_minutes,
                 tts_characters, cost_usd),
            )
            await db.commit()

    async def get_cost_summary(self, conversation_id: str | None = None) -> dict:
        """Get cost summary, optionally filtered by conversation."""
        async with get_db() as db:
            if conversation_id:
                cursor = await db.execute(
                    """SELECT
                        COALESCE(SUM(cost_usd), 0) as total_cost_usd,
                        COALESCE(SUM(input_tokens), 0) as total_input_tokens,
                        COALESCE(SUM(output_tokens), 0) as total_output_tokens
                       FROM cost_log WHERE conversation_id = ?""",
                    (conversation_id,),
                )
            else:
                cursor = await db.execute(
                    """SELECT
                        COALESCE(SUM(cost_usd), 0) as total_cost_usd,
                        COALESCE(SUM(input_tokens), 0) as total_input_tokens,
                        COALESCE(SUM(output_tokens), 0) as total_output_tokens
                       FROM cost_log"""
                )
            row = await cursor.fetchone()

            # Breakdown by operation
            if conversation_id:
                breakdown_cursor = await db.execute(
                    """SELECT operation, model_id,
                              COALESCE(SUM(cost_usd), 0) as cost,
                              COALESCE(SUM(input_tokens), 0) as input_tokens,
                              COALESCE(SUM(output_tokens), 0) as output_tokens
                       FROM cost_log WHERE conversation_id = ?
                       GROUP BY operation, model_id""",
                    (conversation_id,),
                )
            else:
                breakdown_cursor = await db.execute(
                    """SELECT operation, model_id,
                              COALESCE(SUM(cost_usd), 0) as cost,
                              COALESCE(SUM(input_tokens), 0) as input_tokens,
                              COALESCE(SUM(output_tokens), 0) as output_tokens
                       FROM cost_log
                       GROUP BY operation, model_id"""
                )
            breakdown_rows = await breakdown_cursor.fetchall()

            return {
                "conversation_id": conversation_id,
                "total_cost_usd": row["total_cost_usd"] if row else 0.0,
                "total_input_tokens": row["total_input_tokens"] if row else 0,
                "total_output_tokens": row["total_output_tokens"] if row else 0,
                "breakdown": [dict(r) for r in breakdown_rows],
            }
