import logging

from fastapi import APIRouter
from backend.models.schemas import HealthResponse
from backend.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Return service health.  If the DB isn't ready yet (e.g. during
    container startup before lifespan runs), return a 200 with
    status="starting" so Railway/Docker healthchecks don't fail."""
    try:
        async with get_db() as db:
            docs = await db.execute("SELECT COUNT(*) FROM documents")
            doc_count = (await docs.fetchone())[0]
            convos = await db.execute("SELECT COUNT(*) FROM conversations")
            conv_count = (await convos.fetchone())[0]
        return HealthResponse(
            status="healthy",
            document_count=doc_count,
            conversation_count=conv_count,
        )
    except Exception as exc:
        logger.warning("Health check: DB not ready yet (%s)", exc)
        return HealthResponse(status="starting")
