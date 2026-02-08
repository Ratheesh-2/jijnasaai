import json
import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import Optional

from backend.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/summary")
async def get_analytics_summary(
    days: int = Query(default=30, ge=1, le=365),
):
    """Return aggregate analytics for the admin dashboard.

    Metrics returned:
    - Conversations created per day
    - Messages sent per day
    - Model popularity (% of messages per model)
    - RAG usage (% of messages using documents)
    - Total daily API spend
    - Distinct active days (proxy for retention)
    - Top models by cost
    """
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    async with get_db() as db:
        # --- Totals ---
        row = await (await db.execute(
            "SELECT COUNT(*) FROM conversations WHERE created_at >= ?", (cutoff,)
        )).fetchone()
        total_conversations = row[0]

        row = await (await db.execute(
            "SELECT COUNT(*) FROM messages WHERE created_at >= ? AND role != 'system'",
            (cutoff,),
        )).fetchone()
        total_messages = row[0]

        row = await (await db.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM cost_log WHERE created_at >= ?",
            (cutoff,),
        )).fetchone()
        total_cost = row[0]

        row = await (await db.execute(
            "SELECT COUNT(*) FROM documents WHERE uploaded_at >= ?", (cutoff,)
        )).fetchone()
        total_documents = row[0]

        # --- Conversations per day ---
        rows = await (await db.execute(
            "SELECT date(created_at) AS day, COUNT(*) AS count "
            "FROM conversations WHERE created_at >= ? "
            "GROUP BY day ORDER BY day",
            (cutoff,),
        )).fetchall()
        conversations_per_day = [{"date": r[0], "count": r[1]} for r in rows]

        # --- Messages per day ---
        rows = await (await db.execute(
            "SELECT date(created_at) AS day, COUNT(*) AS count "
            "FROM messages WHERE created_at >= ? AND role != 'system' "
            "GROUP BY day ORDER BY day",
            (cutoff,),
        )).fetchall()
        messages_per_day = [{"date": r[0], "count": r[1]} for r in rows]

        # --- Model popularity (by message count) ---
        rows = await (await db.execute(
            "SELECT model_id, COUNT(*) AS count "
            "FROM cost_log WHERE created_at >= ? AND operation = 'chat' "
            "GROUP BY model_id ORDER BY count DESC",
            (cutoff,),
        )).fetchall()
        model_usage = [{"model_id": r[0], "count": r[1]} for r in rows]

        # --- Model cost breakdown ---
        rows = await (await db.execute(
            "SELECT model_id, "
            "       COALESCE(SUM(cost_usd), 0) AS total_cost, "
            "       COALESCE(SUM(input_tokens), 0) AS total_input, "
            "       COALESCE(SUM(output_tokens), 0) AS total_output, "
            "       COUNT(*) AS call_count "
            "FROM cost_log WHERE created_at >= ? "
            "GROUP BY model_id ORDER BY total_cost DESC",
            (cutoff,),
        )).fetchall()
        model_costs = [
            {
                "model_id": r[0],
                "total_cost": r[1],
                "total_input_tokens": r[2],
                "total_output_tokens": r[3],
                "call_count": r[4],
            }
            for r in rows
        ]

        # --- Daily spend ---
        rows = await (await db.execute(
            "SELECT date(created_at) AS day, COALESCE(SUM(cost_usd), 0) AS cost "
            "FROM cost_log WHERE created_at >= ? "
            "GROUP BY day ORDER BY day",
            (cutoff,),
        )).fetchall()
        daily_spend = [{"date": r[0], "cost": r[1]} for r in rows]

        # --- RAG usage ---
        row = await (await db.execute(
            "SELECT COUNT(*) FROM messages "
            "WHERE created_at >= ? AND used_docs = 1",
            (cutoff,),
        )).fetchone()
        rag_message_count = row[0]

        # --- Distinct active days ---
        row = await (await db.execute(
            "SELECT COUNT(DISTINCT date(created_at)) "
            "FROM messages WHERE created_at >= ?",
            (cutoff,),
        )).fetchone()
        active_days = row[0]

        # --- Operations breakdown ---
        rows = await (await db.execute(
            "SELECT operation, COUNT(*) AS count, COALESCE(SUM(cost_usd), 0) AS cost "
            "FROM cost_log WHERE created_at >= ? "
            "GROUP BY operation ORDER BY cost DESC",
            (cutoff,),
        )).fetchall()
        operations = [
            {"operation": r[0], "count": r[1], "cost": r[2]} for r in rows
        ]

        # --- Feature events (comparison mode, etc.) ---
        rows = await (await db.execute(
            "SELECT event_type, COUNT(*) AS count "
            "FROM analytics_events WHERE created_at >= ? "
            "GROUP BY event_type ORDER BY count DESC",
            (cutoff,),
        )).fetchall()
        feature_events = [{"event_type": r[0], "count": r[1]} for r in rows]

    return {
        "period_days": days,
        "cutoff_date": cutoff,
        "totals": {
            "conversations": total_conversations,
            "messages": total_messages,
            "cost_usd": total_cost,
            "documents_uploaded": total_documents,
            "rag_messages": rag_message_count,
            "active_days": active_days,
        },
        "conversations_per_day": conversations_per_day,
        "messages_per_day": messages_per_day,
        "daily_spend": daily_spend,
        "model_usage": model_usage,
        "model_costs": model_costs,
        "operations": operations,
        "feature_events": feature_events,
    }


class AnalyticsEvent(BaseModel):
    event_type: str
    event_data: dict = {}


@router.post("/event")
async def log_analytics_event(event: AnalyticsEvent):
    """Log a feature usage event (e.g. comparison_mode, rag_query)."""
    async with get_db() as db:
        await db.execute(
            "INSERT INTO analytics_events (event_type, event_data) VALUES (?, ?)",
            (event.event_type, json.dumps(event.event_data)),
        )
        await db.commit()
    return {"status": "ok"}
