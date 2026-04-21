"""Semantic text chunker with overlap for RAG ingestion."""
from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional
from transformers import AutoTokenizer
from src.utils.config import get_settings
from src.utils.logger import logger

settings = get_settings()


@dataclass
class TextChunk:
    chunk_id: str
    document_id: str
    document_name: str
    text: str
    page_number: Optional[int]
    chunk_index: int
    token_count: int
    metadata: dict = field(default_factory=dict)

    @classmethod
    def make_id(cls, document_id: str, chunk_index: int, text: str) -> str:
        content = f"{document_id}:{chunk_index}:{text[:50]}"
        return hashlib.md5(content.encode()).hexdigest()


class SemanticChunker:
    """
    Token-aware chunker that respects sentence boundaries.

    Strategy:
    1. Split text into sentences
    2. Greedily pack sentences into chunks up to max_tokens
    3. Add token overlap between adjacent chunks for context continuity
    4. Each chunk gets a stable hash ID for deduplication
    """

    def __init__(
        self,
        max_tokens: int | None = None,
        overlap_tokens: int | None = None,
        tokenizer_name: str | None = None,
    ):
        cfg = get_settings()
        self.max_tokens = max_tokens or cfg.chunk_size
        self.overlap_tokens = overlap_tokens or cfg.chunk_overlap
        tokenizer_name = tokenizer_name or cfg.embedding_model

        logger.info(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def chunk_document(
        self,
        text: str,
        document_id: str,
        document_name: str,
        page_number: Optional[int] = None,
        metadata: dict | None = None,
    ) -> List[TextChunk]:
        """Split a document's text into overlapping token-bounded chunks."""
        if not text.strip():
            return []

        sentences = self._split_into_sentences(text)
        token_batches = self._pack_sentences(sentences)
        chunks = []

        for i, batch_tokens in enumerate(token_batches):
            chunk_text = self.tokenizer.decode(batch_tokens, skip_special_tokens=True)
            chunk_id = TextChunk.make_id(document_id, i, chunk_text)
            chunks.append(TextChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                document_name=document_name,
                text=chunk_text.strip(),
                page_number=page_number,
                chunk_index=i,
                token_count=len(batch_tokens),
                metadata=metadata or {},
            ))

        logger.debug(f"Chunked '{document_name}' into {len(chunks)} chunks")
        return chunks

    def chunk_pages(
        self,
        pages: List[dict],
        document_id: str,
        document_name: str,
        metadata: dict | None = None,
    ) -> List[TextChunk]:
        """Chunk a list of page dicts (each with 'text' and 'page_number')."""
        all_chunks = []
        for page in pages:
            page_chunks = self.chunk_document(
                text=page.get("text", ""),
                document_id=document_id,
                document_name=document_name,
                page_number=page.get("page_number"),
                metadata=metadata,
            )
            all_chunks.extend(page_chunks)
        return all_chunks

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple rule-based sentence splitter (no NLTK dependency)."""
        import re
        # Split on sentence-ending punctuation followed by whitespace + capital
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\(])", text)
        return [s.strip() for s in sentences if s.strip()]

    def _pack_sentences(self, sentences: List[str]) -> List[List[int]]:
        """
        Pack sentences into token batches respecting max_tokens.
        Adds overlap_tokens from the end of the previous batch.
        """
        if not sentences:
            return []

        # Tokenize all sentences
        tokenized = [
            self.tokenizer.encode(s, add_special_tokens=False)
            for s in sentences
        ]

        batches: List[List[int]] = []
        current_batch: List[int] = []
        overlap_buffer: List[int] = []

        for tokens in tokenized:
            # If adding this sentence exceeds max, save current and start new
            if len(current_batch) + len(tokens) > self.max_tokens and current_batch:
                batches.append(current_batch)
                # Start next chunk with overlap from end of current
                overlap_buffer = current_batch[-self.overlap_tokens:]
                current_batch = overlap_buffer + tokens
            else:
                current_batch.extend(tokens)

        if current_batch:
            batches.append(current_batch)

        return batches

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))
