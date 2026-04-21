"""Unit tests for TextCleaner — each cleaning step tested independently."""
import pytest
from src.ingestion.text_cleaner import TextCleaner, CleaningConfig


@pytest.fixture
def cleaner():
    return TextCleaner()


class TestNormalizeUnicode:
    def test_smart_quotes_replaced(self, cleaner):
        text = "\u2018Hello\u2019 and \u201cWorld\u201d"
        result = cleaner.normalize_unicode(text)
        assert "'" in result and '"' in result
        assert "\u2018" not in result

    def test_em_dash_replaced(self, cleaner):
        result = cleaner.normalize_unicode("term\u2014definition")
        assert "--" in result

    def test_non_breaking_space(self, cleaner):
        result = cleaner.normalize_unicode("hello\u00a0world")
        assert "\u00a0" not in result
        assert "hello world" in result


class TestFixHyphenation:
    def test_rejoins_broken_word(self, cleaner):
        result = cleaner.fix_hyphenation("termin-\nation")
        assert "termination" in result

    def test_leaves_normal_hyphens_alone(self, cleaner):
        result = cleaner.fix_hyphenation("well-known")
        assert "well-known" in result

    def test_multiple_breaks(self, cleaner):
        text = "re-\nview and sub-\nmit"
        result = cleaner.fix_hyphenation(text)
        assert "review" in result
        assert "submit" in result


class TestRemovePageNumbers:
    def test_removes_page_n_of_m(self, cleaner):
        text = "Some content\nPage 3 of 10\nMore content"
        result = cleaner.remove_page_numbers(text)
        assert "Page 3 of 10" not in result
        assert "Some content" in result

    def test_removes_standalone_number(self, cleaner):
        text = "Content\n   5   \nMore"
        result = cleaner.remove_page_numbers(text)
        lines = [l.strip() for l in result.split("\n")]
        assert "5" not in lines

    def test_removes_dash_number_dash(self, cleaner):
        text = "Content\n- 12 -\nMore"
        result = cleaner.remove_page_numbers(text)
        assert "- 12 -" not in result


class TestRemoveRepeatedLines:
    def test_removes_header_repeated_many_times(self, cleaner):
        header = "CONFIDENTIAL - ACME CORP"
        text = "\n".join([header] * 5 + ["Real content"])
        result = cleaner.remove_repeated_lines(text, threshold=3)
        assert "Real content" in result
        assert result.count(header) <= 3

    def test_keeps_non_repeated_lines(self, cleaner):
        text = "Line one\nLine two\nLine three"
        result = cleaner.remove_repeated_lines(text)
        assert "Line one" in result
        assert "Line two" in result


class TestNormalizeWhitespace:
    def test_collapses_multiple_spaces(self, cleaner):
        result = cleaner.normalize_whitespace("hello     world")
        assert "hello world" in result

    def test_collapses_triple_newlines(self, cleaner):
        result = cleaner.normalize_whitespace("a\n\n\n\nb")
        assert "\n\n\n" not in result


class TestCleanPipeline:
    def test_empty_string_returns_empty(self, cleaner):
        assert cleaner.clean("") == ""
        assert cleaner.clean("   ") == ""

    def test_full_pipeline_on_realistic_text(self, cleaner):
        raw = (
            "CONFIDENTIAL\n"
            "Page 1 of 20\n"
            "This Agree-\nment is made between Party A and Party B.\n"
            "CONFIDENTIAL\n"
            "The terms include: payment, deliv-\nery, and liability.\n"
            "Page 2 of 20\n"
            "CONFIDENTIAL\n"
            "CONFIDENTIAL\n"
        )
        result = cleaner.clean(raw)
        assert "Agreement" in result or "Agree-\nment" not in result
        assert "delivery" in result or "deliv-\nery" not in result
        assert "Page 1 of 20" not in result

    def test_batch_clean(self, cleaner):
        texts = ["Hello world", "  spaces  ", ""]
        results = cleaner.clean_batch(texts)
        assert len(results) == 3
        assert results[0] == "Hello world"
        assert results[1] == "spaces"
        assert results[2] == ""

    def test_custom_config_no_page_number_removal(self):
        config = CleaningConfig(remove_page_numbers=False)
        cleaner = TextCleaner(config)
        text = "Content\nPage 1 of 5\nMore"
        result = cleaner.clean(text)
        assert "Page 1 of 5" in result
