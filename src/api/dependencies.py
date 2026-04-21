"""FastAPI dependency injection for shared components."""
from __future__ import annotations
from functools import lru_cache

from src.ingestion.pipeline import IngestionPipeline
from src.generation.answer_generator import AnswerGenerator
from src.retrieval.vector_store import VectorStore
from src.utils.config import get_settings


@lru_cache()
def get_ingestion_pipeline() -> IngestionPipeline:
    """Singleton ingestion pipeline (heavy models loaded once)."""
    return IngestionPipeline()


@lru_cache()
def get_answer_generator() -> AnswerGenerator:
    """Singleton answer generator (heavy models loaded once)."""
    return AnswerGenerator()


@lru_cache()
def get_vector_store() -> VectorStore:
    """Singleton vector store connection."""
    return VectorStore()


def get_settings_dep():
    return get_settings()
