"""API route handlers."""
from __future__ import annotations
import tempfile
import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status

from src.api.schemas import (
    IngestResponse, IngestBatchResponse,
    QueryRequest, QueryResponse,
    DocumentListResponse, DocumentInfo,
    SummaryRequest, SummaryResponse,
    HealthResponse,
)
from src.api.dependencies import get_ingestion_pipeline, get_answer_generator, get_vector_store
from src.ingestion.pipeline import IngestionPipeline
from src.generation.answer_generator import AnswerGenerator
from src.retrieval.vector_store import VectorStore
from src.utils.config import get_settings
from src.utils.logger import logger

settings = get_settings()
router = APIRouter()


# ------------------------------------------------------------------ #
# Health                                                               #
# ------------------------------------------------------------------ #

@router.get("/health", response_model=HealthResponse, tags=["system"])
def health_check(store: VectorStore = Depends(get_vector_store)):
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        vector_store_chunks=store.count(),
        llm_provider=settings.llm_provider,
        embedding_model=settings.embedding_model,
    )


# ------------------------------------------------------------------ #
# Ingestion                                                            #
# ------------------------------------------------------------------ #

@router.post("/ingest", response_model=IngestResponse, tags=["ingestion"])
async def ingest_document(
    file: UploadFile = File(..., description="PDF file to ingest"),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
):
    """
    Upload and ingest a PDF document.
    Runs the full pipeline: extract -> clean -> chunk -> embed -> store.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    # Save upload to temp file (pipeline works on file paths)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        logger.info(f"Received upload: {file.filename}")
        result = pipeline.ingest_file(tmp_path)
        # Rename internally so stored metadata has the original filename
        result.document_name = file.filename
        return IngestResponse.from_result(result)
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/ingest/batch", response_model=IngestBatchResponse, tags=["ingestion"])
async def ingest_batch(
    files: List[UploadFile] = File(...),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
):
    """Upload and ingest multiple PDF documents at once."""
    results = []
    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            continue
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)
        try:
            result = pipeline.ingest_file(tmp_path)
            result.document_name = file.filename
            results.append(IngestResponse.from_result(result))
        finally:
            tmp_path.unlink(missing_ok=True)

    successful = sum(1 for r in results if r.success)
    return IngestBatchResponse(
        total_files=len(results),
        successful=successful,
        failed=len(results) - successful,
        results=results,
    )


# ------------------------------------------------------------------ #
# Query                                                                #
# ------------------------------------------------------------------ #

@router.post("/query", response_model=QueryResponse, tags=["query"])
def query_documents(
    request: QueryRequest,
    generator: AnswerGenerator = Depends(get_answer_generator),
):
    """
    Ask a natural language question over ingested documents.
    Returns an LLM-generated answer with cited sources.
    """
    try:
        result = generator.answer(
            query=request.query,
            document_filter=request.document_ids,
            retrieval_top_k=request.retrieval_top_k,
            rerank_top_k=request.rerank_top_k,
        )
        return QueryResponse.from_result(result)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}",
        )


# ------------------------------------------------------------------ #
# Documents                                                            #
# ------------------------------------------------------------------ #

@router.get("/documents", response_model=DocumentListResponse, tags=["documents"])
def list_documents(store: VectorStore = Depends(get_vector_store)):
    """List all ingested documents."""
    doc_ids = store.list_documents()
    docs = []
    for doc_id in doc_ids:
        chunks = store.get_document_chunks(doc_id)
        name = chunks[0]["metadata"].get("document_name", doc_id) if chunks else doc_id
        docs.append(DocumentInfo(
            document_id=doc_id,
            document_name=name,
            total_chunks=len(chunks),
        ))
    return DocumentListResponse(total_documents=len(docs), documents=docs)


@router.delete("/documents/{document_id}", tags=["documents"])
def delete_document(
    document_id: str,
    store: VectorStore = Depends(get_vector_store),
):
    """Remove a document and all its chunks from the vector store."""
    store.delete_document(document_id)
    return {"message": f"Document {document_id} deleted successfully."}


@router.post("/documents/summarize", response_model=SummaryResponse, tags=["documents"])
def summarize_document(
    request: SummaryRequest,
    generator: AnswerGenerator = Depends(get_answer_generator),
):
    """Generate an executive summary for a specific document."""
    summary = generator.summarize_document(request.document_id, request.document_name)
    return SummaryResponse(
        document_id=request.document_id,
        document_name=request.document_name,
        summary=summary,
    )
