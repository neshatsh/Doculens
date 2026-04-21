"""VLM-based extraction for scanned pages, tables, and charts using GPT-4V."""
from __future__ import annotations
import base64
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from src.utils.config import get_settings
from src.utils.logger import logger

settings = get_settings()


class VLMExtractor:
    """
    Uses GPT-4V (or compatible VLM) to extract structured text from page images.
    Falls back gracefully when API key is not set or use_vlm=False.

    Designed for:
    - Scanned PDFs where pdfplumber returns empty text
    - Pages with complex tables that lose structure in text extraction
    - Charts and diagrams that need caption-level description
    """

    EXTRACTION_PROMPT = """You are a document analysis expert. Extract ALL text content from this document page.

Instructions:
1. Extract all visible text exactly as it appears
2. For tables: preserve structure using | separators, one row per line
3. For charts/graphs: describe key values and trends in a structured format
4. For headers/footers: include them with [HEADER] and [FOOTER] tags
5. Maintain the reading order (top-to-bottom, left-to-right)
6. Do not add interpretation — only extract what is present

Output the extracted content directly with no preamble."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=settings.openai_api_key)
        return self._client

    def is_available(self) -> bool:
        return settings.use_vlm and bool(settings.openai_api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def extract_page(self, image_bytes: bytes) -> str:
        """Extract text from a single page image using VLM."""
        if not self.is_available():
            logger.debug("VLM not available, skipping")
            return ""

        client = self._get_client()
        b64 = base64.b64encode(image_bytes).decode("utf-8")

        try:
            response = client.chat.completions.create(
                model=settings.vlm_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.EXTRACTION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=2048,
                temperature=0.0,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"VLM extraction failed: {e}")
            return ""

    def extract_pages(self, image_bytes_list: List[bytes]) -> List[str]:
        """Extract text from multiple page images."""
        results = []
        for i, img_bytes in enumerate(image_bytes_list):
            logger.info(f"VLM extracting page {i + 1}/{len(image_bytes_list)}")
            text = self.extract_page(img_bytes)
            results.append(text)
        return results

    def should_use_vlm_for_page(self, extracted_text: str) -> bool:
        """
        Heuristic: use VLM if pdfplumber returned very little text
        (likely a scanned page or image-heavy layout).
        """
        return len(extracted_text.strip()) < 100
