"""Unit tests for PromptBuilder RAG prompt construction."""
import pytest
from src.generation.prompt_builder import PromptBuilder, SYSTEM_PROMPT, NO_CONTEXT_RESPONSE


def make_chunk(text: str, doc_name: str = "contract.pdf", page: int = 1, score: float = 0.9) -> dict:
    return {
        "text": text,
        "score": score,
        "metadata": {"document_name": doc_name, "page_number": page},
    }


class TestBuildRagPrompt:
    def test_returns_system_prompt_unchanged(self):
        builder = PromptBuilder()
        system, _ = builder.build_rag_prompt("What is the term?", [make_chunk("text")])
        assert system == SYSTEM_PROMPT

    def test_no_chunks_returns_no_context_message(self):
        builder = PromptBuilder()
        _, user = builder.build_rag_prompt("What is the term?", [])
        assert "no relevant document excerpts" in user.lower()

    def test_no_chunks_includes_query_in_message(self):
        builder = PromptBuilder()
        _, user = builder.build_rag_prompt("What is the payment term?", [])
        assert "What is the payment term?" in user

    def test_user_message_contains_query(self):
        builder = PromptBuilder()
        _, user = builder.build_rag_prompt("What are the penalties?", [make_chunk("Some text")])
        assert "What are the penalties?" in user

    def test_user_message_contains_chunk_text(self):
        builder = PromptBuilder()
        _, user = builder.build_rag_prompt("query", [make_chunk("Important clause here")])
        assert "Important clause here" in user

    def test_user_message_contains_doc_name(self):
        builder = PromptBuilder()
        _, user = builder.build_rag_prompt("query", [make_chunk("text", doc_name="lease.pdf")])
        assert "lease.pdf" in user

    def test_user_message_contains_page_number(self):
        builder = PromptBuilder()
        _, user = builder.build_rag_prompt("query", [make_chunk("text", page=7)])
        assert "7" in user

    def test_multiple_chunks_all_appear(self):
        builder = PromptBuilder()
        chunks = [make_chunk("first chunk"), make_chunk("second chunk")]
        _, user = builder.build_rag_prompt("query", chunks)
        assert "first chunk" in user
        assert "second chunk" in user

    def test_rerank_score_preferred_over_score(self):
        builder = PromptBuilder()
        chunk = make_chunk("text", score=0.5)
        chunk["rerank_score"] = 0.99
        _, user = builder.build_rag_prompt("query", [chunk])
        assert "0.99" in user

    def test_missing_metadata_uses_defaults(self):
        builder = PromptBuilder()
        chunk = {"text": "bare chunk", "score": 0.5}
        _, user = builder.build_rag_prompt("query", [chunk])
        assert "Unknown document" in user
        assert "?" in user


class TestContextTruncation:
    def test_truncates_when_over_limit(self):
        builder = PromptBuilder(max_context_chars=100)
        chunks = [make_chunk("x" * 60), make_chunk("y" * 60)]
        _, user = builder.build_rag_prompt("query", chunks)
        assert "truncated" in user

    def test_no_truncation_when_under_limit(self):
        builder = PromptBuilder(max_context_chars=10_000)
        chunks = [make_chunk("short text"), make_chunk("another short")]
        _, user = builder.build_rag_prompt("query", chunks)
        assert "truncated" not in user

    def test_truncation_message_shows_remaining_count(self):
        builder = PromptBuilder(max_context_chars=100)
        chunks = [make_chunk("x" * 60), make_chunk("y" * 30), make_chunk("z" * 30)]
        _, user = builder.build_rag_prompt("query", chunks)
        assert "truncated" in user


class TestBuildSummaryPrompt:
    def test_returns_two_strings(self):
        builder = PromptBuilder()
        result = builder.build_summary_prompt("report.pdf", [make_chunk("text")])
        assert len(result) == 2

    def test_system_mentions_summary(self):
        builder = PromptBuilder()
        system, _ = builder.build_summary_prompt("report.pdf", [make_chunk("text")])
        assert "summar" in system.lower()

    def test_user_contains_document_name(self):
        builder = PromptBuilder()
        _, user = builder.build_summary_prompt("annual_report.pdf", [make_chunk("text")])
        assert "annual_report.pdf" in user

    def test_user_contains_chunk_text(self):
        builder = PromptBuilder()
        _, user = builder.build_summary_prompt("doc.pdf", [make_chunk("key financial data")])
        assert "key financial data" in user

    def test_empty_chunks_still_returns_prompts(self):
        builder = PromptBuilder()
        system, user = builder.build_summary_prompt("empty.pdf", [])
        assert isinstance(system, str)
        assert isinstance(user, str)