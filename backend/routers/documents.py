import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException

from backend.models.schemas import DocumentUploadResponse, DocumentListResponse
from backend.services.rag_engine import RAGEngine
from backend.dependencies import get_rag_engine
from backend.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
UPLOAD_DIR = Path("data/uploads")


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    conversation_id: str = Form(default=None),
    rag_engine: RAGEngine = Depends(get_rag_engine),
):
    """Upload and ingest a document into the vector store."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=ext, dir=str(UPLOAD_DIR)
        ) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        result = await rag_engine.ingest_document(
            tmp_path, file.filename, len(content), conversation_id
        )
        return DocumentUploadResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Ingestion failed for {file.filename}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


@router.get("", response_model=DocumentListResponse)
async def list_documents(conversation_id: str = None):
    async with get_db() as db:
        if conversation_id:
            cursor = await db.execute(
                "SELECT * FROM documents WHERE conversation_id = ? ORDER BY uploaded_at DESC",
                (conversation_id,),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM documents ORDER BY uploaded_at DESC"
            )
        rows = await cursor.fetchall()
        return DocumentListResponse(documents=[dict(r) for r in rows])
