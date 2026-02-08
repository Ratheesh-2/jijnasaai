import pytest
from backend.services.conversation_service import ConversationService


class TestConversationService:
    @pytest.mark.asyncio
    async def test_create_and_list(self, initialized_db):
        service = ConversationService()
        result = await service.create_conversation("gpt-4o", "Test Chat")
        assert result["id"]
        assert result["title"] == "Test Chat"

        convos = await service.list_conversations()
        assert len(convos) == 1
        assert convos[0]["title"] == "Test Chat"

    @pytest.mark.asyncio
    async def test_add_and_get_messages(self, initialized_db):
        service = ConversationService()
        conv = await service.create_conversation("gpt-4o")
        conv_id = conv["id"]

        await service.add_message(conv_id, "user", "Hello")
        await service.add_message(
            conv_id, "assistant", "Hi there!",
            model_id="gpt-4o", input_tokens=10, output_tokens=5, cost_usd=0.001
        )

        messages = await service.get_conversation_messages(conv_id)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["model_id"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_message_count(self, initialized_db):
        service = ConversationService()
        conv = await service.create_conversation("gpt-4o")
        conv_id = conv["id"]

        assert await service.get_message_count(conv_id) == 0
        await service.add_message(conv_id, "user", "Hello")
        assert await service.get_message_count(conv_id) == 1

    @pytest.mark.asyncio
    async def test_update_title(self, initialized_db):
        service = ConversationService()
        conv = await service.create_conversation("gpt-4o", "Original")
        await service.update_conversation_title(conv["id"], "Updated Title")

        updated = await service.get_conversation(conv["id"])
        assert updated["title"] == "Updated Title"

    @pytest.mark.asyncio
    async def test_update_system_prompt(self, initialized_db):
        service = ConversationService()
        conv = await service.create_conversation("gpt-4o")
        await service.update_system_prompt(conv["id"], "Be concise.")

        updated = await service.get_conversation(conv["id"])
        assert updated["system_prompt"] == "Be concise."

    @pytest.mark.asyncio
    async def test_delete_conversation(self, initialized_db):
        service = ConversationService()
        conv = await service.create_conversation("gpt-4o")
        await service.add_message(conv["id"], "user", "Hello")

        await service.delete_conversation(conv["id"])
        convos = await service.list_conversations()
        assert len(convos) == 0

    @pytest.mark.asyncio
    async def test_cost_accumulation(self, initialized_db):
        service = ConversationService()
        conv = await service.create_conversation("gpt-4o")
        conv_id = conv["id"]

        await service.add_message(
            conv_id, "assistant", "Response 1",
            input_tokens=100, output_tokens=50, cost_usd=0.01
        )
        await service.add_message(
            conv_id, "assistant", "Response 2",
            input_tokens=200, output_tokens=100, cost_usd=0.02
        )

        updated = await service.get_conversation(conv_id)
        assert updated["total_input_tokens"] == 300
        assert updated["total_output_tokens"] == 150
        assert abs(updated["total_cost_usd"] - 0.03) < 0.001
