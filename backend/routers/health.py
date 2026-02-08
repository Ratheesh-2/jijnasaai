from fastapi import APIRouter
from backend.models.schemas import HealthResponse
from backend.database import get_db

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    async with get_db() as db:
        docs = await db.execute("SELECT COUNT(*) FROM documents")
        doc_count = (await docs.fetchone())[0]
        convos = await db.execute("SELECT COUNT(*) FROM conversations")
        conv_count = (await convos.fetchone())[0]
    return HealthResponse(document_count=doc_count, conversation_count=conv_count)
