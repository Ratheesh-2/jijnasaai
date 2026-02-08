import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.database import init_db, set_db_path
from backend.routers import health, conversations, chat, documents, voice, costs, analytics

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    # Ensure data directories exist
    db_path = Path(settings.database_url)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    Path("data/uploads").mkdir(parents=True, exist_ok=True)

    # Initialize database
    set_db_path(settings.database_url)
    await init_db()
    logger.info("JijnasaAI backend started")

    yield

    logger.info("JijnasaAI backend shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="JijnasaAI API",
        description="Multi-model AI chat with RAG, voice & cost tracking",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Routers
    app.include_router(health.router)
    app.include_router(conversations.router)
    app.include_router(chat.router)
    app.include_router(documents.router)
    app.include_router(voice.router)
    app.include_router(costs.router)
    app.include_router(analytics.router)

    # CORS: read allowed origins from env, with sensible defaults
    import os

    extra_origins = os.environ.get("ALLOWED_ORIGINS", "")
    origins = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ]

    if extra_origins:
        origins.extend(
            o.strip()
            for o in extra_origins.split(",")
            if o.strip()
        )

    # Railway: auto-add the public domain so browser→backend CORS works
    railway_domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "")
    if railway_domain:
        origins.append(f"https://{railway_domain}")

    # Railway may assign a custom PORT; add localhost:<PORT> for in-container requests
    railway_port = os.environ.get("PORT", "")
    if railway_port and railway_port != "8501":
        origins.append(f"http://localhost:{railway_port}")
        origins.append(f"http://127.0.0.1:{railway_port}")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


app = create_app()


@app.get("/")
async def read_root():
    return {
        "status": "ok",
        "message": "JijnasaAI backend is running",
        "docs": "/docs",
    }
