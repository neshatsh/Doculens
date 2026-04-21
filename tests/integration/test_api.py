"""Integration tests for FastAPI endpoints using TestClient."""
import pytest
import io
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.dependencies import get_ingestion_pipeline, get_answer_generator, get_vector_store


# ------------------------------------------------------------------ #
# Fixtures — mock heavy dependencies so tests run without GPU/API     #
# ------------------------------------------------------------------ #

@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    result = MagicMock()
    result.document_id = "test_doc_id"
    result.document_name = "test.pdf"
    result.total_pages = 5
    result.total_chunks = 20
    result.skipped_pages = 0
    result.success = True
    result.error = None
    pipeline.ingest_file.return_value = result
    return pipeline


@pytest.fixture
def mock_generator():
    generator = MagicMock()
    result = MagicMock()
    result.query = "What are the payment terms?"
    result.answer = "Payment terms are net-30 from invoice date. [Source: contract.pdf, Page 7]"
    result.sources = [{
        "document_name": "contract.pdf",
        "page_number": 7,
        "rerank_score": 0.92,
        "excerpt": "Payment terms shall be net-30 from invoice date.",
    }]
    result.retrieval_time_ms = 45.2
    result.generation_time_ms = 820.0
    result.total_time_ms = 865.2
    result.total_chunks_retrieved = 20
    result.total_chunks_after_rerank = 5
    result.model_info = {"provider": "openai", "model": "gpt-4o-mini"}
    generator.answer.return_value = result
    return generator


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.count.return_value = 42
    store.list_documents.return_value = ["doc1", "doc2"]
    store.get_document_chunks.side_effect = lambda doc_id: [
        {"id": f"{doc_id}_chunk1", "text": "Sample text", "metadata": {"document_name": f"{doc_id}.pdf", "page_number": 1}}
    ]
    return store


@pytest.fixture
def client(mock_pipeline, mock_generator, mock_store):
    app.dependency_overrides[get_ingestion_pipeline] = lambda: mock_pipeline
    app.dependency_overrides[get_answer_generator] = lambda: mock_generator
    app.dependency_overrides[get_vector_store] = lambda: mock_store
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ------------------------------------------------------------------ #
# Tests                                                                #
# ------------------------------------------------------------------ #

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_response_shape(self, client):
        resp = client.get("/api/v1/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "vector_store_chunks" in data
        assert data["vector_store_chunks"] == 42


class TestIngestEndpoint:
    def test_ingest_valid_pdf(self, client):
        fake_pdf = io.BytesIO(b"%PDF-1.4 fake content")
        resp = client.post(
            "/api/v1/ingest",
            files={"file": ("test.pdf", fake_pdf, "application/pdf")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["total_chunks"] == 20

    def test_ingest_non_pdf_rejected(self, client):
        fake_txt = io.BytesIO(b"just text")
        resp = client.post(
            "/api/v1/ingest",
            files={"file": ("document.txt", fake_txt, "text/plain")},
        )
        assert resp.status_code == 400

    def test_ingest_response_has_document_id(self, client):
        fake_pdf = io.BytesIO(b"%PDF-1.4")
        resp = client.post(
            "/api/v1/ingest",
            files={"file": ("contract.pdf", fake_pdf, "application/pdf")},
        )
        data = resp.json()
        assert "document_id" in data
        assert len(data["document_id"]) > 0


class TestQueryEndpoint:
    def test_query_returns_200(self, client):
        resp = client.post(
            "/api/v1/query",
            json={"query": "What are the payment terms?"},
        )
        assert resp.status_code == 200

    def test_query_response_has_answer(self, client):
        resp = client.post(
            "/api/v1/query",
            json={"query": "What are the payment terms?"},
        )
        data = resp.json()
        assert "answer" in data
        assert len(data["answer"]) > 0

    def test_query_response_has_sources(self, client):
        resp = client.post(
            "/api/v1/query",
            json={"query": "What are the payment terms?"},
        )
        data = resp.json()
        assert "sources" in data
        assert isinstance(data["sources"], list)

    def test_query_response_has_timing(self, client):
        resp = client.post(
            "/api/v1/query",
            json={"query": "What are the payment terms?"},
        )
        data = resp.json()
        assert "retrieval_time_ms" in data
        assert "generation_time_ms" in data
        assert "total_time_ms" in data

    def test_short_query_rejected(self, client):
        resp = client.post("/api/v1/query", json={"query": "Hi"})
        assert resp.status_code == 422

    def test_query_with_document_filter(self, client, mock_generator):
        resp = client.post(
            "/api/v1/query",
            json={"query": "What are the terms?", "document_ids": ["doc1"]},
        )
        assert resp.status_code == 200
        mock_generator.answer.assert_called_once()
        call_kwargs = mock_generator.answer.call_args[1]
        assert call_kwargs.get("document_filter") == ["doc1"]


class TestDocumentsEndpoint:
    def test_list_documents(self, client):
        resp = client.get("/api/v1/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_documents" in data
        assert "documents" in data
        assert data["total_documents"] == 2

    def test_delete_document(self, client, mock_store):
        resp = client.delete("/api/v1/documents/doc1")
        assert resp.status_code == 200
        mock_store.delete_document.assert_called_once_with("doc1")


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_has_docs_link(self, client):
        data = client.get("/").json()
        assert "docs" in data
