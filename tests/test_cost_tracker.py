import pytest
from backend.services.cost_tracker import CostTracker
from backend.services.conversation_service import ConversationService


@pytest.fixture
def cost_tracker(test_settings):
    return CostTracker(test_settings)


class TestCostCalculations:
    def test_chat_cost_gpt4o(self, cost_tracker):
        cost = cost_tracker.calculate_chat_cost("gpt-4o", 1000, 500)
        # 1000/1M * 2.50 + 500/1M * 10.00 = 0.0025 + 0.005 = 0.0075
        assert abs(cost - 0.0075) < 0.0001

    def test_chat_cost_gpt4o_mini(self, cost_tracker):
        cost = cost_tracker.calculate_chat_cost("gpt-4o-mini", 10000, 5000)
        # 10000/1M * 0.15 + 5000/1M * 0.60 = 0.0015 + 0.003 = 0.0045
        assert abs(cost - 0.0045) < 0.0001

    def test_chat_cost_claude_sonnet(self, cost_tracker):
        cost = cost_tracker.calculate_chat_cost("claude-sonnet-4-5-20250929", 1000, 500)
        # 1000/1M * 3.00 + 500/1M * 15.00 = 0.003 + 0.0075 = 0.0105
        assert abs(cost - 0.0105) < 0.0001

    def test_chat_cost_gemini_flash(self, cost_tracker):
        cost = cost_tracker.calculate_chat_cost("gemini-2.0-flash", 10000, 5000)
        # 10000/1M * 0.10 + 5000/1M * 0.40 = 0.001 + 0.002 = 0.003
        assert abs(cost - 0.003) < 0.0001

    def test_chat_cost_unknown_model(self, cost_tracker):
        cost = cost_tracker.calculate_chat_cost("unknown-model", 1000, 500)
        assert cost == 0.0

    def test_stt_cost(self, cost_tracker):
        cost = cost_tracker.calculate_stt_cost(1.0)
        assert abs(cost - 0.006) < 0.001

    def test_tts_cost(self, cost_tracker):
        cost = cost_tracker.calculate_tts_cost(1000)
        # 1000/1M * 15.00 = 0.015
        assert abs(cost - 0.015) < 0.001

    def test_embedding_cost(self, cost_tracker):
        cost = cost_tracker.calculate_embedding_cost(10000)
        # 10000/1M * 0.02 = 0.0002
        assert abs(cost - 0.0002) < 0.0001


class TestCostLogging:
    @pytest.mark.asyncio
    async def test_log_and_summarize(self, initialized_db, test_settings):
        tracker = CostTracker(test_settings)
        conv_service = ConversationService()

        # Create a real conversation to satisfy FK constraint
        conv = await conv_service.create_conversation("gpt-4o", "Test")
        conv_id = conv["id"]

        await tracker.log_cost(
            model_id="gpt-4o",
            operation="chat",
            conversation_id=conv_id,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
        )
        await tracker.log_cost(
            model_id="gpt-4o",
            operation="chat",
            conversation_id=conv_id,
            input_tokens=200,
            output_tokens=100,
            cost_usd=0.02,
        )

        summary = await tracker.get_cost_summary(conv_id)
        assert abs(summary["total_cost_usd"] - 0.03) < 0.001
        assert summary["total_input_tokens"] == 300
        assert summary["total_output_tokens"] == 150

    @pytest.mark.asyncio
    async def test_global_summary(self, initialized_db, test_settings):
        tracker = CostTracker(test_settings)
        conv_service = ConversationService()

        conv_a = await conv_service.create_conversation("gpt-4o", "A")
        conv_b = await conv_service.create_conversation("gpt-4o", "B")

        await tracker.log_cost(
            model_id="gpt-4o", operation="chat",
            conversation_id=conv_a["id"], cost_usd=0.01
        )
        await tracker.log_cost(
            model_id="gpt-4o", operation="chat",
            conversation_id=conv_b["id"], cost_usd=0.02
        )

        summary = await tracker.get_cost_summary()
        assert abs(summary["total_cost_usd"] - 0.03) < 0.001
