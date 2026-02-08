from fastapi import APIRouter, Depends, HTTPException
from backend.models.schemas import (
    ConversationCreate,
    ConversationResponse,
    ConversationListResponse,
)
from backend.services.conversation_service import ConversationService
from backend.dependencies import get_conversation_service

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    service: ConversationService = Depends(get_conversation_service),
):
    convos = await service.list_conversations()
    return ConversationListResponse(conversations=convos)


@router.post("")
async def create_conversation(
    body: ConversationCreate,
    service: ConversationService = Depends(get_conversation_service),
):
    result = await service.create_conversation(
        model_id=body.model_id,
        title=body.title,
        system_prompt=body.system_prompt,
    )
    return result


@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    service: ConversationService = Depends(get_conversation_service),
):
    conv = await service.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@router.get("/{conversation_id}/messages")
async def get_messages(
    conversation_id: str,
    service: ConversationService = Depends(get_conversation_service),
):
    return await service.get_conversation_messages(conversation_id)


@router.put("/{conversation_id}/system-prompt")
async def update_system_prompt(
    conversation_id: str,
    body: dict,
    service: ConversationService = Depends(get_conversation_service),
):
    await service.update_system_prompt(conversation_id, body.get("system_prompt", ""))
    return {"status": "updated"}


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    service: ConversationService = Depends(get_conversation_service),
):
    await service.delete_conversation(conversation_id)
    return {"status": "deleted"}
