import uuid
from backend.database import get_db


class ConversationService:

    async def create_conversation(
        self, model_id: str, title: str = "New Conversation", system_prompt: str = ""
    ) -> dict:
        conv_id = str(uuid.uuid4())
        async with get_db() as db:
            await db.execute(
                "INSERT INTO conversations (id, title, model_id, system_prompt) VALUES (?, ?, ?, ?)",
                (conv_id, title, model_id, system_prompt),
            )
            await db.commit()
            cursor = await db.execute(
                "SELECT * FROM conversations WHERE id = ?", (conv_id,)
            )
            row = await cursor.fetchone()
            return dict(row)

    async def list_conversations(self) -> list[dict]:
        async with get_db() as db:
            cursor = await db.execute(
                """SELECT c.*, COUNT(m.id) as message_count
                   FROM conversations c
                   LEFT JOIN messages m ON m.conversation_id = c.id
                   GROUP BY c.id
                   ORDER BY c.updated_at DESC"""
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_conversation(self, conversation_id: str) -> dict | None:
        async with get_db() as db:
            cursor = await db.execute(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            )
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def get_conversation_messages(self, conversation_id: str) -> list[dict]:
        async with get_db() as db:
            cursor = await db.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
                (conversation_id,),
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        model_id: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        used_docs: bool = False,
    ) -> str:
        msg_id = str(uuid.uuid4())
        async with get_db() as db:
            await db.execute(
                """INSERT INTO messages
                   (id, conversation_id, role, content, model_id,
                    input_tokens, output_tokens, cost_usd, used_docs)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (msg_id, conversation_id, role, content, model_id,
                 input_tokens, output_tokens, cost_usd, 1 if used_docs else 0),
            )
            await db.execute(
                """UPDATE conversations SET
                   updated_at = datetime('now'),
                   total_input_tokens = total_input_tokens + ?,
                   total_output_tokens = total_output_tokens + ?,
                   total_cost_usd = total_cost_usd + ?
                   WHERE id = ?""",
                (input_tokens, output_tokens, cost_usd, conversation_id),
            )
            await db.commit()
        return msg_id

    async def update_conversation_title(self, conversation_id: str, title: str):
        async with get_db() as db:
            await db.execute(
                "UPDATE conversations SET title = ?, updated_at = datetime('now') WHERE id = ?",
                (title, conversation_id),
            )
            await db.commit()

    async def update_system_prompt(self, conversation_id: str, system_prompt: str):
        async with get_db() as db:
            await db.execute(
                "UPDATE conversations SET system_prompt = ?, updated_at = datetime('now') WHERE id = ?",
                (system_prompt, conversation_id),
            )
            await db.commit()

    async def delete_conversation(self, conversation_id: str):
        async with get_db() as db:
            await db.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            await db.execute("DELETE FROM cost_log WHERE conversation_id = ?", (conversation_id,))
            await db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            await db.commit()

    async def get_message_count(self, conversation_id: str) -> int:
        async with get_db() as db:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            )
            row = await cursor.fetchone()
            return row[0] if row else 0
