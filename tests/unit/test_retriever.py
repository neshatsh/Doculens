"""Unit tests for Retriever — mocks embedder and vector store."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.retrieval.retriever import Retriever


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed_query.return_value = np.ones(384, dtype=np.float32)
    return embedder


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.query.return_value = [
        {
            "id": "chunk1",
            "text": "This is a relevant contract clause about termination.",
            "metadata": {"document_name": "contract.pdf", "page_number": 3},
            "score": 0.92,
            "distance": 0.08,
        },
        {
            "id": "chunk2",
            "text": "Payment terms shall be net-30 from invoice date.",
            "metadata": {"document_name": "contract.pdf", "page_number": 7},
            "score": 0.85,
            "distance": 0.15,
        },
    ]
    return store


@pytest.fixture
def retriever(mock_embedder, mock_vector_store):
    return Retriever(
        embedder=mock_embedder,
        vector_store=mock_vector_store,
        top_k=20,
    )


class TestRetrieve:
    def test_returns_list(self, retriever):
        results = retriever.retrieve("What are the payment terms?")
        assert isinstance(results, list)

    def test_calls_embed_query(self, retriever, mock_embedder):
        retriever.retrieve("test query")
        mock_embedder.embed_query.assert_called_once_with("test query")

    def test_calls_vector_store_query(self, retriever, mock_vector_store):
        retriever.retrieve("test query", top_k=10)
        mock_vector_store.query.assert_called_once()
        call_kwargs = mock_vector_store.query.call_args
        assert call_kwargs.kwargs.get("top_k") == 10 or call_kwargs[1].get("top_k") == 10

    def test_results_have_expected_keys(self, retriever):
        results = retriever.retrieve("termination clause")
        for r in results:
            assert "id" in r
            assert "text" in r
            assert "score" in r

    def test_document_filter_single(self, retriever, mock_vector_store):
        retriever.retrieve("query", document_filter=["doc123"])
        call_kwargs = mock_vector_store.query.call_args[1]
        where = call_kwargs.get("where")
        assert where is not None
        assert where.get("document_id") == "doc123"

    def test_document_filter_multiple(self, retriever, mock_vector_store):
        retriever.retrieve("query", document_filter=["doc1", "doc2"])
        call_kwargs = mock_vector_store.query.call_args[1]
        where = call_kwargs.get("where")
        assert "$in" in where.get("document_id", {})

    def test_no_filter_passes_none(self, retriever, mock_vector_store):
        retriever.retrieve("query")
        call_kwargs = mock_vector_store.query.call_args[1]
        assert call_kwargs.get("where") is None

    def test_retrieve_with_scores_returns_tuples(self, retriever):
        results = retriever.retrieve_with_scores("query")
        assert all(isinstance(item, tuple) and len(item) == 2 for item in results)
        assert all(isinstance(score, float) for _, score in results)

    def test_empty_store_returns_empty(self, mock_embedder):
        empty_store = MagicMock()
        empty_store.query.return_value = []
        retriever = Retriever(embedder=mock_embedder, vector_store=empty_store)
        results = retriever.retrieve("query")
        assert results == []
