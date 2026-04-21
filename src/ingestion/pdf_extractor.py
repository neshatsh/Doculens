"""PDF text extraction using pdfplumber (primary) with PyMuPDF fallback."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import pdfplumber
import fitz  # PyMuPDF
from src.utils.logger import logger


@dataclass
class PageContent:
    page_number: int
    text: str
    tables: List[List[List[str]]] = field(default_factory=list)
    has_images: bool = False
    width: float = 0.0
    height: float = 0.0


@dataclass
class DocumentContent:
    file_path: str
    file_name: str
    total_pages: int
    pages: List[PageContent] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())


class PDFExtractor:
    """
    Two-stage PDF extractor:
    1. pdfplumber — best for text + table extraction from digital PDFs
    2. PyMuPDF fallback — handles more edge cases and malformed PDFs
    """

    def __init__(self, min_text_length: int = 20):
        self.min_text_length = min_text_length

    def extract(self, file_path: str | Path) -> DocumentContent:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        logger.info(f"Extracting: {path.name}")
        try:
            return self._extract_with_pdfplumber(path)
        except Exception as e:
            logger.warning(f"pdfplumber failed ({e}), falling back to PyMuPDF")
            return self._extract_with_pymupdf(path)

    def _extract_with_pdfplumber(self, path: Path) -> DocumentContent:
        pages: List[PageContent] = []
        metadata: dict = {}

        with pdfplumber.open(path) as pdf:
            metadata = pdf.metadata or {}
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                tables = page.extract_tables() or []
                # Convert tables to string rows and append to text
                table_text = self._tables_to_text(tables)
                if table_text:
                    text = text + "\n\n" + table_text

                pages.append(PageContent(
                    page_number=i + 1,
                    text=text.strip(),
                    tables=tables,
                    has_images=bool(page.images),
                    width=float(page.width),
                    height=float(page.height),
                ))

        return DocumentContent(
            file_path=str(path),
            file_name=path.name,
            total_pages=len(pages),
            pages=pages,
            metadata=metadata,
        )

    def _extract_with_pymupdf(self, path: Path) -> DocumentContent:
        pages: List[PageContent] = []

        doc = fitz.open(str(path))
        metadata = doc.metadata or {}

        for i, page in enumerate(doc):
            text = page.get_text("text") or ""
            images = page.get_images(full=False)
            rect = page.rect

            pages.append(PageContent(
                page_number=i + 1,
                text=text.strip(),
                has_images=bool(images),
                width=rect.width,
                height=rect.height,
            ))

        doc.close()
        return DocumentContent(
            file_path=str(path),
            file_name=path.name,
            total_pages=len(pages),
            pages=pages,
            metadata=metadata,
        )

    @staticmethod
    def _tables_to_text(tables: List[List[List[Optional[str]]]]) -> str:
        """Convert extracted tables to pipe-separated text."""
        lines = []
        for table in tables:
            for row in table:
                cleaned = [str(cell).strip() if cell is not None else "" for cell in row]
                lines.append(" | ".join(cleaned))
            lines.append("")  # blank line between tables
        return "\n".join(lines)

    def extract_page_images(self, file_path: str | Path) -> List[bytes]:
        """Extract raw image bytes from pages for VLM processing."""
        path = Path(file_path)
        doc = fitz.open(str(path))
        images = []
        for page in doc:
            # Render page as image at 150 DPI
            mat = fitz.Matrix(150 / 72, 150 / 72)
            pix = page.get_pixmap(matrix=mat)
            images.append(pix.tobytes("png"))
        doc.close()
        return images
