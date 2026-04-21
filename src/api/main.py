"""FastAPI application entry point."""
from __future__ import annotations
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.utils.config import get_settings
from src.utils.logger import logger

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up heavy models on startup so first request isn't slow."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    # Pre-load singletons (triggers model downloads if needed)
    from src.api.dependencies import get_ingestion_pipeline, get_answer_generator
    get_ingestion_pipeline()
    get_answer_generator()
    logger.info("Models loaded. API ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Production-grade RAG document intelligence platform. "
        "Ingest PDFs, ask natural language questions, get cited answers."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["system"])
def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/v1/health",
    }
