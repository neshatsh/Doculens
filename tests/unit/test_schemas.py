"""Unit tests for Pydantic API schemas — validation logic."""
import pytest
from pydantic import ValidationError
from src.api.schemas import QueryRequest, IngestResponse, SourceCitation


class TestQueryRequest:
    def test_valid_query(self):
        req = QueryRequest(query="What are the termination clauses?")
        assert req.query == "What are the termination clauses?"

    def test_query_stripped(self):
        req = QueryRequest(query="  What is the payment term?  ")
        assert req.query == "What is the payment term?"

    def test_query_too_short_raises(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="Hi")

    def test_query_too_long_raises(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="x" * 1001)

    def test_default_top_k_values(self):
        req = QueryRequest(query="What are the clauses?")
        assert req.retrieval_top_k == 20
        assert req.rerank_top_k == 5

    def test_custom_top_k(self):
        req = QueryRequest(query="What are the clauses?", retrieval_top_k=10, rerank_top_k=3)
        assert req.retrieval_top_k == 10
        assert req.rerank_top_k == 3

    def test_retrieval_top_k_below_min_raises(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="Valid query here", retrieval_top_k=0)

    def test_retrieval_top_k_above_max_raises(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="Valid query here", retrieval_top_k=101)

    def test_document_ids_optional(self):
        req = QueryRequest(query="Valid query here")
        assert req.document_ids is None

    def test_document_ids_accepted(self):
        req = QueryRequest(query="Valid query here", document_ids=["doc1", "doc2"])
        assert req.document_ids == ["doc1", "doc2"]


class TestIngestResponse:
    def test_from_result_success(self):
        result = type("R", (), {
            "document_id": "abc123",
            "document_name": "test.pdf",
            "total_pages": 10,
            "total_chunks": 42,
            "skipped_pages": 1,
            "success": True,
            "error": None,
        })()
        resp = IngestResponse.from_result(result)
        assert resp.success is True
        assert resp.total_chunks == 42
        assert "42 chunks" in resp.message

    def test_from_result_failure(self):
        result = type("R", (), {
            "document_id": "abc123",
            "document_name": "bad.pdf",
            "total_pages": 0,
            "total_chunks": 0,
            "skipped_pages": 0,
            "success": False,
            "error": "Extraction failed",
        })()
        resp = IngestResponse.from_result(result)
        assert resp.success is False
        assert "Ingestion failed" in resp.message


class TestSourceCitation:
    def test_valid_citation(self):
        citation = SourceCitation(
            document_name="contract.pdf",
            page_number=5,
            rerank_score=0.92,
            excerpt="The term shall be 12 months...",
        )
        assert citation.document_name == "contract.pdf"
        assert citation.rerank_score == 0.92

    def test_unknown_page_number(self):
        # page_number can be "?" string from metadata
        citation = SourceCitation(
            document_name="contract.pdf",
            page_number="?",
            rerank_score=0.5,
            excerpt="Some excerpt",
        )
        assert citation.page_number == "?"
