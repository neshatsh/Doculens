"""End-to-end ingestion pipeline orchestrator."""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from src.ingestion.pdf_extractor import PDFExtractor
from src.ingestion.vlm_extractor import VLMExtractor
from src.ingestion.text_cleaner import TextCleaner
from src.ingestion.chunker import SemanticChunker, TextChunk
from src.ingestion.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.utils.logger import logger
from src.utils.config import get_settings

settings = get_settings()


@dataclass
class IngestionResult:
    document_id: str
    document_name: str
    total_pages: int
    total_chunks: int
    skipped_pages: int
    success: bool
    error: Optional[str] = None


class IngestionPipeline:
    """
    Orchestrates the full ingestion flow:
    PDF -> extract -> VLM fallback -> clean -> chunk -> embed -> store

    Each stage is independently swappable for testing.
    """

    def __init__(
        self,
        pdf_extractor: PDFExtractor | None = None,
        vlm_extractor: VLMExtractor | None = None,
        cleaner: TextCleaner | None = None,
        chunker: SemanticChunker | None = None,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.pdf_extractor = pdf_extractor or PDFExtractor()
        self.vlm_extractor = vlm_extractor or VLMExtractor()
        self.cleaner = cleaner or TextCleaner()
        self.chunker = chunker or SemanticChunker()
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()

    def ingest_file(self, file_path: str | Path) -> IngestionResult:
        """Ingest a single PDF file end-to-end."""
        path = Path(file_path)
        document_id = self._make_document_id(path)
        document_name = path.name
        logger.info(f"Starting ingestion: {document_name} (id={document_id})")

        try:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            # Stage 1: Extract text from PDF
            doc_content = self.pdf_extractor.extract(path)

            # Stage 2: VLM fallback for low-text pages
            page_images: List[bytes] = []
            if self.vlm_extractor.is_available():
                page_images = self.pdf_extractor.extract_page_images(path)

            skipped = 0
            page_dicts = []
            for page in doc_content.pages:
                text = page.text

                # Use VLM if page has very little text (likely scanned)
                if self.vlm_extractor.is_available() and \
                   self.vlm_extractor.should_use_vlm_for_page(text) and \
                   page_images:
                    idx = page.page_number - 1
                    if idx < len(page_images):
                        logger.debug(f"Using VLM for page {page.page_number}")
                        text = self.vlm_extractor.extract_page(page_images[idx])

                # Stage 3: Clean
                cleaned = self.cleaner.clean(text)
                if not cleaned.strip():
                    skipped += 1
                    continue

                page_dicts.append({
                    "text": cleaned,
                    "page_number": page.page_number,
                })

            # Stage 4: Chunk
            chunks: List[TextChunk] = self.chunker.chunk_pages(
                pages=page_dicts,
                document_id=document_id,
                document_name=document_name,
                metadata={"source": str(path), **doc_content.metadata},
            )

            if not chunks:
                return IngestionResult(
                    document_id=document_id,
                    document_name=document_name,
                    total_pages=doc_content.total_pages,
                    total_chunks=0,
                    skipped_pages=skipped,
                    success=False,
                    error="No text chunks produced after cleaning",
                )

            # Stage 5: Embed
            _, embeddings = self.embedder.embed_chunks(chunks)

            # Stage 6: Store
            self.vector_store.add_chunks(chunks, embeddings)

            logger.info(
                f"Ingested '{document_name}': "
                f"{len(chunks)} chunks, {skipped} skipped pages"
            )
            return IngestionResult(
                document_id=document_id,
                document_name=document_name,
                total_pages=doc_content.total_pages,
                total_chunks=len(chunks),
                skipped_pages=skipped,
                success=True,
            )

        except Exception as e:
            logger.error(f"Ingestion failed for '{document_name}': {e}")
            return IngestionResult(
                document_id=document_id,
                document_name=document_name,
                total_pages=0,
                total_chunks=0,
                skipped_pages=0,
                success=False,
                error=str(e),
            )

    def ingest_directory(self, dir_path: str | Path) -> List[IngestionResult]:
        """Ingest all PDFs in a directory."""
        dir_path = Path(dir_path)
        pdf_files = list(dir_path.glob("**/*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDFs in {dir_path}")
        return [self.ingest_file(f) for f in pdf_files]

    @staticmethod
    def _make_document_id(path: Path) -> str:
        """Stable document ID from file path."""
        return hashlib.md5(str(path.resolve()).encode()).hexdigest()[:16]
