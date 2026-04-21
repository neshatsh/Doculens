"""Unit tests for SemanticChunker."""
import pytest
from unittest.mock import MagicMock, patch
from src.ingestion.chunker import SemanticChunker, TextChunk


@pytest.fixture
def chunker():
    """Use a small fast tokenizer for tests."""
    with patch("src.ingestion.chunker.AutoTokenizer") as mock_tok_cls:
        mock_tok = MagicMock()
        # Simulate tokenizer: 1 token per word
        mock_tok.encode.side_effect = lambda text, **kw: text.split()
        mock_tok.decode.side_effect = lambda tokens, **kw: " ".join(tokens)
        mock_tok_cls.from_pretrained.return_value = mock_tok
        yield SemanticChunker(max_tokens=20, overlap_tokens=5)


class TestChunkDocument:
    def test_returns_list_of_chunks(self, chunker):
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = chunker.chunk_document(
            text=text, document_id="doc1", document_name="test.pdf"
        )
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(c, TextChunk) for c in chunks)

    def test_empty_text_returns_empty(self, chunker):
        chunks = chunker.chunk_document(
            text="", document_id="doc1", document_name="test.pdf"
        )
        assert chunks == []

    def test_whitespace_only_returns_empty(self, chunker):
        chunks = chunker.chunk_document(
            text="   \n  \t  ", document_id="doc1", document_name="test.pdf"
        )
        assert chunks == []

    def test_chunk_has_required_fields(self, chunker):
        text = "This is a test sentence. Another sentence here."
        chunks = chunker.chunk_document(
            text=text, document_id="doc42", document_name="contract.pdf"
        )
        chunk = chunks[0]
        assert chunk.document_id == "doc42"
        assert chunk.document_name == "contract.pdf"
        assert chunk.chunk_index == 0
        assert isinstance(chunk.chunk_id, str)
        assert len(chunk.chunk_id) == 32  # MD5 hex

    def test_chunk_ids_are_unique(self, chunker):
        text = ". ".join([f"Sentence number {i}" for i in range(30)])
        chunks = chunker.chunk_document(
            text=text, document_id="doc1", document_name="test.pdf"
        )
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_metadata_passed_through(self, chunker):
        text = "A sentence with metadata. Another sentence."
        meta = {"category": "finance", "year": "2024"}
        chunks = chunker.chunk_document(
            text=text, document_id="doc1", document_name="test.pdf", metadata=meta
        )
        assert all(c.metadata == meta for c in chunks)

    def test_page_number_assigned(self, chunker):
        text = "Some content on this page. More content here."
        chunks = chunker.chunk_document(
            text=text, document_id="doc1", document_name="test.pdf", page_number=7
        )
        assert all(c.page_number == 7 for c in chunks)


class TestChunkPages:
    def test_multiple_pages(self, chunker):
        pages = [
            {"text": "Page one content. More text here.", "page_number": 1},
            {"text": "Page two content. Additional text.", "page_number": 2},
        ]
        chunks = chunker.chunk_pages(
            pages=pages, document_id="doc1", document_name="test.pdf"
        )
        assert len(chunks) > 0

    def test_empty_pages_skipped(self, chunker):
        pages = [
            {"text": "", "page_number": 1},
            {"text": "Real content here.", "page_number": 2},
        ]
        chunks = chunker.chunk_pages(
            pages=pages, document_id="doc1", document_name="test.pdf"
        )
        assert len(chunks) > 0
        assert all(c.page_number == 2 for c in chunks)


class TestMakeId:
    def test_same_inputs_same_id(self):
        id1 = TextChunk.make_id("doc1", 0, "hello world")
        id2 = TextChunk.make_id("doc1", 0, "hello world")
        assert id1 == id2

    def test_different_inputs_different_id(self):
        id1 = TextChunk.make_id("doc1", 0, "hello world")
        id2 = TextChunk.make_id("doc1", 1, "hello world")
        assert id1 != id2
