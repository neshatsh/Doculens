"""Text cleaning and normalisation pipeline for extracted document text."""
from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass
from typing import List


@dataclass
class CleaningConfig:
    remove_headers_footers: bool = True
    normalize_whitespace: bool = True
    fix_hyphenation: bool = True
    remove_page_numbers: bool = True
    min_line_length: int = 3
    max_repeated_chars: int = 3


class TextCleaner:
    """
    Production-grade text cleaner for legal/financial document text.
    Each step is independently testable (unit test friendly).
    """

    def __init__(self, config: CleaningConfig | None = None):
        self.config = config or CleaningConfig()

    def clean(self, text: str) -> str:
        """Run the full cleaning pipeline."""
        if not text or not text.strip():
            return ""

        text = self.normalize_unicode(text)
        text = self.fix_encoding_artifacts(text)

        if self.config.fix_hyphenation:
            text = self.fix_hyphenation(text)

        if self.config.remove_page_numbers:
            text = self.remove_page_numbers(text)

        if self.config.remove_headers_footers:
            text = self.remove_repeated_lines(text)

        text = self.remove_short_lines(text, self.config.min_line_length)
        text = self.remove_excessive_repetition(text, self.config.max_repeated_chars)

        if self.config.normalize_whitespace:
            text = self.normalize_whitespace(text)

        return text.strip()

    def clean_batch(self, texts: List[str]) -> List[str]:
        return [self.clean(t) for t in texts]

    # ------------------------------------------------------------------ #
    # Individual cleaning steps (each independently unit-testable)        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize unicode to NFC form and replace smart quotes."""
        text = unicodedata.normalize("NFC", text)
        replacements = {
            "\u2018": "'", "\u2019": "'",   # smart single quotes
            "\u201c": '"', "\u201d": '"',   # smart double quotes
            "\u2013": "-", "\u2014": "--",  # en/em dash
            "\u00a0": " ",                   # non-breaking space
            "\u2022": "-",                   # bullet
            "\u00b7": "-",                   # middle dot
        }
        for orig, replacement in replacements.items():
            text = text.replace(orig, replacement)
        return text

    @staticmethod
    def fix_encoding_artifacts(text: str) -> str:
        """Remove common OCR/encoding garbage characters."""
        text = re.sub(r"[^\x00-\x7F\u00C0-\u024F\u0370-\u03FF]", " ", text)
        text = re.sub(r"\ufffd", "", text)     # replacement character
        text = re.sub(r"\x00", "", text)       # null bytes
        return text

    @staticmethod
    def fix_hyphenation(text: str) -> str:
        """
        Re-join words broken across lines by hyphens.
        e.g. 'termin-\nation' -> 'termination'
        """
        return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    @staticmethod
    def remove_page_numbers(text: str) -> str:
        """Remove standalone page number lines like 'Page 1 of 10' or '- 3 -'."""
        patterns = [
            r"^[Pp]age\s+\d+\s*(of\s*\d+)?\s*$",
            r"^\s*[-–]\s*\d+\s*[-–]\s*$",
            r"^\s*\d+\s*$",
        ]
        lines = text.split("\n")
        cleaned = [
            line for line in lines
            if not any(re.match(p, line.strip()) for p in patterns)
        ]
        return "\n".join(cleaned)

    @staticmethod
    def remove_repeated_lines(text: str, threshold: int = 3) -> str:
        """
        Remove lines that appear more than `threshold` times
        (typical of headers/footers repeated on every page).
        """
        lines = text.split("\n")
        from collections import Counter
        counts = Counter(line.strip() for line in lines if line.strip())
        return "\n".join(
            line for line in lines
            if not line.strip() or counts[line.strip()] <= threshold
        )

    @staticmethod
    def remove_short_lines(text: str, min_length: int = 3) -> str:
        """Drop lines that are too short to be meaningful."""
        lines = text.split("\n")
        cleaned = [
            line for line in lines
            if len(line.strip()) >= min_length or line.strip() == ""
        ]
        return "\n".join(cleaned)

    @staticmethod
    def remove_excessive_repetition(text: str, max_rep: int = 3) -> str:
        """Collapse runs of the same character e.g. '.....' -> '...'"""
        return re.sub(r"(.)\1{" + str(max_rep) + r",}", r"\1" * max_rep, text)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Collapse multiple spaces/tabs; keep single blank lines."""
        text = re.sub(r"[ \t]+", " ", text)          # collapse spaces/tabs
        text = re.sub(r"\n{3,}", "\n\n", text)        # max 2 consecutive newlines
        return text
