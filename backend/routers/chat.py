import json
import logging

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from backend.models.schemas import ChatRequest
from backend.services.llm_router import LLMRouter
from backend.services.rag_engine import RAGEngine
from backend.services.cost_tracker import CostTracker
from backend.services.conversation_service import ConversationService
from backend.config import get_settings
from backend.database import get_db
from backend.dependencies import (
    get_llm_router, get_rag_engine,
    get_cost_tracker, get_conversation_service,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, accurate, and concise AI assistant. "
    "When provided with context from documents, base your answers on that context "
    "and cite the source documents. If you are unsure, say so."
)

RAG_SYSTEM_PROMPT = (
    "You are an assistant answering questions using ONLY the following documents as context. "
    "If the answer is not found in the documents, say so clearly. "
    "Cite the source document and chunk when referencing information.\n\n"
    "--- DOCUMENT CONTEXT ---\n{context}\n--- END CONTEXT ---"
)


@router.post("/completions")
async def chat_completions(
    request: ChatRequest,
    llm_router: LLMRouter = Depends(get_llm_router),
    rag_engine: RAGEngine = Depends(get_rag_engine),
    cost_tracker: CostTracker = Depends(get_cost_tracker),
    conv_service: ConversationService = Depends(get_conversation_service),
):
    """Stream chat completion via SSE."""

    async def event_generator():
        try:
            # Check daily spend cap
            settings = get_settings()
            if settings.max_daily_spend_usd > 0:
                async with get_db() as db:
                    cursor = await db.execute(
                        "SELECT COALESCE(SUM(cost_usd), 0) FROM cost_log "
                        "WHERE created_at >= date('now')"
                    )
                    row = await cursor.fetchone()
                    today_spend = row[0] if row else 0.0
                if today_spend >= settings.max_daily_spend_usd:
                    yield {
                        "event": "error",
                        "data": json.dumps({
                            "error": f"Daily budget of ${settings.max_daily_spend_usd:.2f} "
                                     f"reached (${today_spend:.2f} spent today). "
                                     f"Try again tomorrow."
                        }),
                    }
                    return

            # Create or load conversation
            conversation_id = request.conversation_id
            is_new = False
            if not conversation_id:
                result = await conv_service.create_conversation(request.model_id)
                conversation_id = result["id"]
                is_new = True
                yield {
                    "event": "conversation",
                    "data": json.dumps({"conversation_id": conversation_id}),
                }

            # Get conversation for system prompt
            conv = await conv_service.get_conversation(conversation_id)
            custom_system_prompt = conv.get("system_prompt", "") if conv else ""

            # Save user message
            await conv_service.add_message(
                conversation_id, "user", request.message, used_docs=request.use_rag
            )

            # Build messages list from conversation history
            system_prompt = custom_system_prompt or DEFAULT_SYSTEM_PROMPT
            messages = []

            # If RAG is enabled, retrieve context and augment system prompt
            sources = []
            if request.use_rag:
                context, sources = await rag_engine.retrieve_context(
                    request.message, conversation_id
                )
                if context:
                    system_prompt = RAG_SYSTEM_PROMPT.format(context=context)
                if sources:
                    yield {
                        "event": "sources",
                        "data": json.dumps(sources),
                    }

            messages.append({"role": "system", "content": system_prompt})

            # Add conversation history
            history = await conv_service.get_conversation_messages(conversation_id)
            for msg in history:
                if msg["role"] in ("user", "assistant"):
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"],
                    })

            # Get max_tokens from model config
            model_cfg = _get_model_config(llm_router, request.model_id)
            max_tokens = model_cfg.get("max_tokens", 4096) if model_cfg else 4096

            # Stream LLM response
            full_response = ""
            input_tokens = 0
            output_tokens = 0
            web_citations = []

            async for chunk in llm_router.stream_chat(
                messages=messages,
                model_id=request.model_id,
                temperature=request.temperature,
                max_tokens=max_tokens,
            ):
                if chunk.text:
                    full_response += chunk.text
                    yield {
                        "event": "token",
                        "data": json.dumps({"text": chunk.text}),
                    }
                if chunk.citations:
                    web_citations.extend(chunk.citations)
                if chunk.is_final:
                    input_tokens = chunk.input_tokens
                    output_tokens = chunk.output_tokens

            # Emit web search sources if any provider returned citations
            if web_citations:
                # Deduplicate by URL
                seen_urls = set()
                unique_citations = []
                for c in web_citations:
                    url = c.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        unique_citations.append(c)
                yield {
                    "event": "web_sources",
                    "data": json.dumps(unique_citations),
                }

            # Calculate cost
            cost = cost_tracker.calculate_chat_cost(
                request.model_id, input_tokens, output_tokens
            )

            # Save assistant message
            msg_id = await conv_service.add_message(
                conversation_id, "assistant", full_response,
                model_id=request.model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                used_docs=request.use_rag and bool(sources),
            )

            # Log cost
            await cost_tracker.log_cost(
                conversation_id=conversation_id,
                message_id=msg_id,
                model_id=request.model_id,
                operation="chat",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
            )

            # Auto-title new conversations after first exchange
            if is_new or (await conv_service.get_message_count(conversation_id)) <= 2:
                try:
                    await _auto_title(
                        conv_service, llm_router, conversation_id,
                        request.message, request.model_id
                    )
                except Exception:
                    pass  # Non-critical

            # Send usage summary
            yield {
                "event": "usage",
                "data": json.dumps({
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": cost,
                    "model_id": request.model_id,
                    "conversation_id": conversation_id,
                }),
            }

            yield {"event": "done", "data": json.dumps({"status": "complete"})}

        except Exception as e:
            logger.exception("Chat streaming failed")
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}),
            }

    return EventSourceResponse(event_generator())


def _get_model_config(llm_router: LLMRouter, model_id: str) -> dict | None:
    """Look up model config from settings."""
    for m in llm_router._settings.models_config:
        if m["id"] == model_id:
            return m
    return None


async def _auto_title(
    conv_service: ConversationService,
    llm_router: LLMRouter,
    conversation_id: str,
    user_message: str,
    model_id: str,
):
    """Generate a short title for the conversation."""
    messages = [
        {
            "role": "system",
            "content": (
                "Generate a short title (max 6 words) for a conversation that starts "
                "with the following message. Reply with ONLY the title, no quotes or punctuation."
            ),
        },
        {"role": "user", "content": user_message[:500]},
    ]
    title_parts = []
    async for chunk in llm_router.stream_chat(
        messages=messages, model_id=model_id, max_tokens=20, temperature=0.3
    ):
        if chunk.text:
            title_parts.append(chunk.text)
    title = "".join(title_parts).strip()[:50]
    if title:
        await conv_service.update_conversation_title(conversation_id, title)
