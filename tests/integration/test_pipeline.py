"""Integration tests for the full ingestion pipeline with mocked heavy components."""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

from src.ingestion.pipeline import IngestionPipeline, IngestionResult


@pytest.fixture
def sample_text():
    return (
        "This Agreement is made between Acme Corp (the 'Company') and "
        "John Doe (the 'Contractor'). The term shall commence on January 1, 2024 "
        "and terminate on December 31, 2024. Payment terms are net-30 from invoice. "
        "Either party may terminate this agreement with 30 days written notice. "
        "Confidentiality obligations survive termination for a period of 2 years. "
        "This agreement is governed by the laws of the State of California."
    )


@pytest.fixture
def mock_pdf_extractor(sample_text):
    extractor = MagicMock()
    doc = MagicMock()
    doc.total_pages = 2
    doc.metadata = {"Author": "Test", "Title": "Test Contract"}
    page1 = MagicMock()
    page1.page_number = 1
    page1.text = sample_text
    page1.images = []
    page2 = MagicMock()
    page2.page_number = 2
    page2.text = "Signatures and annexures follow."
    page2.images = []
    doc.pages = [page1, page2]
    extractor.extract.return_value = doc
    extractor.extract_page_images.return_value = []
    return extractor


@pytest.fixture
def mock_vlm():
    vlm = MagicMock()
    vlm.is_available.return_value = False
    vlm.should_use_vlm_for_page.return_value = False
    return vlm


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed_chunks.side_effect = lambda chunks: (
        chunks,
        np.random.rand(len(chunks), 384).astype(np.float32),
    )
    return embedder


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.add_chunks.return_value = 5
    return store


@pytest.fixture
def pipeline(mock_pdf_extractor, mock_vlm, mock_embedder, mock_store):
    return IngestionPipeline(
        pdf_extractor=mock_pdf_extractor,
        vlm_extractor=mock_vlm,
        embedder=mock_embedder,
        vector_store=mock_store,
    )


class TestIngestionPipeline:
    def test_ingest_returns_result(self, pipeline, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        result = pipeline.ingest_file(pdf)
        assert isinstance(result, IngestionResult)

    def test_ingest_success_flag(self, pipeline, tmp_path):
        pdf = tmp_path / "contract.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        result = pipeline.ingest_file(pdf)
        assert result.success is True

    def test_ingest_produces_chunks(self, pipeline, tmp_path, mock_store):
        pdf = tmp_path / "contract.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        pipeline.ingest_file(pdf)
        mock_store.add_chunks.assert_called_once()
        args = mock_store.add_chunks.call_args[0]
        chunks = args[0]
        assert len(chunks) > 0

    def test_ingest_calls_embedder(self, pipeline, tmp_path, mock_embedder):
        pdf = tmp_path / "contract.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        pipeline.ingest_file(pdf)
        mock_embedder.embed_chunks.assert_called_once()

    def test_missing_file_returns_error_result(self, pipeline):
        result = pipeline.ingest_file("/nonexistent/path/file.pdf")
        assert result.success is False
        assert result.error is not None

    def test_document_id_is_stable(self, pipeline, tmp_path):
        pdf = tmp_path / "contract.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        id1 = pipeline._make_document_id(pdf)
        id2 = pipeline._make_document_id(pdf)
        assert id1 == id2

    def test_document_id_differs_per_file(self, pipeline, tmp_path):
        pdf1 = tmp_path / "contract1.pdf"
        pdf2 = tmp_path / "contract2.pdf"
        pdf1.write_bytes(b"fake")
        pdf2.write_bytes(b"fake")
        assert pipeline._make_document_id(pdf1) != pipeline._make_document_id(pdf2)

    def test_empty_page_skipped(self, pipeline, tmp_path, mock_pdf_extractor, mock_store):
        # Make page 1 return empty text
        mock_pdf_extractor.extract.return_value.pages[0].text = ""
        mock_pdf_extractor.extract.return_value.pages[1].text = ""
        pdf = tmp_path / "empty.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        result = pipeline.ingest_file(pdf)
        # Should either succeed with 0 chunks or fail gracefully
        assert isinstance(result, IngestionResult)
        if not result.success:
            assert result.error is not None

    def test_ingest_directory(self, pipeline, tmp_path):
        for i in range(3):
            (tmp_path / f"contract_{i}.pdf").write_bytes(b"%PDF-1.4 fake")
        results = pipeline.ingest_directory(tmp_path)
        assert len(results) == 3
        assert all(isinstance(r, IngestionResult) for r in results)
