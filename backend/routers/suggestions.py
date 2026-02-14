"""Endpoint to generate dynamic suggested prompts based on user's chat history."""

import asyncio
import json
import logging

from fastapi import APIRouter, Depends

from backend.dependencies import get_conversation_service, get_llm_router
from backend.services.conversation_service import ConversationService
from backend.services.llm_router import LLMRouter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/suggestions", tags=["suggestions"])

# The cheapest/fastest model we use for generating suggestions.
_SUGGESTION_MODEL = "gpt-4o-mini"
_SUGGESTION_TIMEOUT = 3.0  # hard timeout — landing page must never hang
_NUM_SUGGESTIONS = 6
_NUM_CONVERSATIONS = 5  # look at the last N conversations

_SYSTEM_PROMPT = (
    "You generate short, engaging suggested questions for an AI chat app. "
    "Given the user's recent conversation topics, produce exactly {n} diverse "
    "follow-up questions they might want to explore next. "
    "Mix their past interests with fresh angles. Keep each question under 60 characters. "
    "Return ONLY a JSON array of {n} strings, no markdown, no explanation."
)

_FALLBACK_PROMPTS = [
    "What are the biggest tech stories this week?",
    "Write a Python function to merge two sorted lists",
    "Summarize my uploaded PDF document",
    "Compare the latest iPhone vs Samsung Galaxy",
    "Help me write a professional email",
    "Explain quantum computing in simple terms",
]


@router.get("")
async def get_suggestions(
    conv_service: ConversationService = Depends(get_conversation_service),
    llm_router: LLMRouter = Depends(get_llm_router),
):
    """Return 6 suggested prompts, personalised if the user has chat history."""
    try:
        conversations = await conv_service.list_conversations()
    except Exception:
        logger.debug("Could not fetch conversations, returning fallback prompts")
        return {"suggestions": _FALLBACK_PROMPTS, "source": "fallback"}

    # Need at least some history to personalise
    recent = conversations[:_NUM_CONVERSATIONS]
    if len(recent) < 2:
        return {"suggestions": _FALLBACK_PROMPTS, "source": "fallback"}

    # Build a compact summary of recent topics for the LLM
    topic_lines = []
    for c in recent:
        title = (c.get("title") or "Untitled")[:80]
        model = c.get("model_id", "")
        topic_lines.append(f"- {title} (model: {model})")
    topics_text = "\n".join(topic_lines)

    messages = [
        {
            "role": "system",
            "content": _SYSTEM_PROMPT.format(n=_NUM_SUGGESTIONS),
        },
        {
            "role": "user",
            "content": (
                f"My recent conversations:\n{topics_text}\n\n"
                f"Generate {_NUM_SUGGESTIONS} suggested questions."
            ),
        },
    ]

    try:
        # Collect the full text from the streaming provider within the timeout
        text_parts: list[str] = []

        async def _generate():
            async for chunk in llm_router.stream_chat(
                messages=messages,
                model_id=_SUGGESTION_MODEL,
                temperature=0.9,
                max_tokens=300,
            ):
                if chunk.text:
                    text_parts.append(chunk.text)

        await asyncio.wait_for(_generate(), timeout=_SUGGESTION_TIMEOUT)

        raw = "".join(text_parts).strip()
        # Strip markdown fences if the model wraps the JSON
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        suggestions = json.loads(raw)

        if (
            isinstance(suggestions, list)
            and len(suggestions) >= _NUM_SUGGESTIONS
            and all(isinstance(s, str) for s in suggestions)
        ):
            return {
                "suggestions": suggestions[:_NUM_SUGGESTIONS],
                "source": "llm",
            }

        logger.warning("LLM returned unexpected format: %s", raw[:200])
        return {"suggestions": _FALLBACK_PROMPTS, "source": "fallback"}

    except asyncio.TimeoutError:
        logger.info("Suggestions LLM call timed out after %.1fs", _SUGGESTION_TIMEOUT)
        return {"suggestions": _FALLBACK_PROMPTS, "source": "fallback"}
    except Exception as e:
        logger.warning("Suggestions generation failed: %s", e)
        return {"suggestions": _FALLBACK_PROMPTS, "source": "fallback"}
