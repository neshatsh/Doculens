"""Pydantic request/response schemas for the DocuLens API."""
from __future__ import annotations
from typing import List, Optional, Any
from pydantic import BaseModel, ConfigDict, Field, field_validator


# ------------------------------------------------------------------ #
# Ingest                                                               #
# ------------------------------------------------------------------ #

class IngestResponse(BaseModel):
    document_id: str
    document_name: str
    total_pages: int
    total_chunks: int
    skipped_pages: int
    success: bool
    error: Optional[str] = None
    message: str = ""

    @classmethod
    def from_result(cls, result: Any) -> "IngestResponse":
        msg = (
            f"Successfully ingested {result.total_chunks} chunks from {result.total_pages} pages."
            if result.success
            else f"Ingestion failed: {result.error}"
        )
        return cls(
            document_id=result.document_id,
            document_name=result.document_name,
            total_pages=result.total_pages,
            total_chunks=result.total_chunks,
            skipped_pages=result.skipped_pages,
            success=result.success,
            error=result.error,
            message=msg,
        )


class IngestBatchResponse(BaseModel):
    total_files: int
    successful: int
    failed: int
    results: List[IngestResponse]


# ------------------------------------------------------------------ #
# Query                                                                #
# ------------------------------------------------------------------ #

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000, description="Natural language question")
    document_ids: Optional[List[str]] = Field(
        default=None,
        description="Restrict retrieval to these document IDs. Leave empty to search all.",
    )
    retrieval_top_k: int = Field(default=20, ge=1, le=100)
    rerank_top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip()


class SourceCitation(BaseModel):
    document_name: str
    page_number: Any
    rerank_score: float
    excerpt: str


class QueryResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    query: str
    answer: str
    sources: List[SourceCitation]
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    total_chunks_retrieved: int
    total_chunks_after_rerank: int
    model_info: dict

    @classmethod
    def from_result(cls, result: Any) -> "QueryResponse":
        return cls(
            query=result.query,
            answer=result.answer,
            sources=[SourceCitation(**s) for s in result.sources],
            retrieval_time_ms=result.retrieval_time_ms,
            generation_time_ms=result.generation_time_ms,
            total_time_ms=result.total_time_ms,
            total_chunks_retrieved=result.total_chunks_retrieved,
            total_chunks_after_rerank=result.total_chunks_after_rerank,
            model_info=result.model_info,
        )


# ------------------------------------------------------------------ #
# Documents                                                            #
# ------------------------------------------------------------------ #

class DocumentInfo(BaseModel):
    document_id: str
    document_name: str
    total_chunks: int


class DocumentListResponse(BaseModel):
    total_documents: int
    documents: List[DocumentInfo]


class SummaryRequest(BaseModel):
    document_id: str
    document_name: str = ""


class SummaryResponse(BaseModel):
    document_id: str
    document_name: str
    summary: str


# ------------------------------------------------------------------ #
# Health                                                               #
# ------------------------------------------------------------------ #

class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store_chunks: int
    llm_provider: str
    embedding_model: str
