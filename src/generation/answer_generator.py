"""Full RAG answer generation: retrieve -> rerank -> generate."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time

from src.retrieval.retriever import Retriever
from src.retrieval.reranker import Reranker
from src.generation.prompt_builder import PromptBuilder
from src.generation.llm_client import LLMClient
from src.utils.logger import logger
from src.utils.config import get_settings

settings = get_settings()


@dataclass
class AnswerResult:
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    retrieval_time_ms: float
    generation_time_ms: float
    total_chunks_retrieved: int
    total_chunks_after_rerank: int
    model_info: dict = field(default_factory=dict)

    @property
    def total_time_ms(self) -> float:
        return self.retrieval_time_ms + self.generation_time_ms


class AnswerGenerator:
    """
    Orchestrates the full RAG pipeline:
    1. Dense retrieval (Retriever)
    2. Cross-encoder reranking (Reranker)
    3. Prompt construction (PromptBuilder)
    4. LLM generation (LLMClient)

    Returns structured AnswerResult with citations and timing metadata.
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        reranker: Reranker | None = None,
        prompt_builder: PromptBuilder | None = None,
        llm_client: LLMClient | None = None,
    ):
        self.retriever = retriever or Retriever()
        self.reranker = reranker or Reranker()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.llm_client = llm_client or LLMClient()

    def answer(
        self,
        query: str,
        document_filter: Optional[List[str]] = None,
        retrieval_top_k: int | None = None,
        rerank_top_k: int | None = None,
    ) -> AnswerResult:
        """
        Full RAG pipeline: retrieve -> rerank -> generate.

        Args:
            query: User's natural language question
            document_filter: Optionally limit retrieval to specific doc IDs
            retrieval_top_k: How many chunks to retrieve before reranking
            rerank_top_k: How many chunks to keep after reranking

        Returns:
            AnswerResult with answer, sources, and timing info
        """
        logger.info(f"Answering query: '{query[:80]}'")

        # Stage 1 + 2: Retrieve and rerank
        t0 = time.perf_counter()
        raw_chunks = self.retriever.retrieve(
            query=query,
            top_k=retrieval_top_k or settings.retrieval_top_k,
            document_filter=document_filter,
        )
        reranked_chunks = self.reranker.rerank(
            query=query,
            chunks=raw_chunks,
            top_k=rerank_top_k or settings.reranker_top_k,
        )
        retrieval_ms = (time.perf_counter() - t0) * 1000

        # Stage 3: Build prompt
        system_prompt, user_message = self.prompt_builder.build_rag_prompt(
            query=query,
            chunks=reranked_chunks,
        )

        # Stage 4: Generate
        t1 = time.perf_counter()
        answer_text = self.llm_client.complete(
            system_prompt=system_prompt,
            user_message=user_message,
        )
        generation_ms = (time.perf_counter() - t1) * 1000

        # Build source citations from top chunks
        sources = [
            {
                "document_name": c.get("metadata", {}).get("document_name", "Unknown"),
                "page_number": c.get("metadata", {}).get("page_number", "?"),
                "rerank_score": round(c.get("rerank_score", 0), 4),
                "excerpt": c.get("text", "")[:300] + "...",
            }
            for c in reranked_chunks
        ]

        logger.info(
            f"Answer generated | retrieval={retrieval_ms:.0f}ms | "
            f"generation={generation_ms:.0f}ms | sources={len(sources)}"
        )

        return AnswerResult(
            query=query,
            answer=answer_text,
            sources=sources,
            retrieval_time_ms=round(retrieval_ms, 2),
            generation_time_ms=round(generation_ms, 2),
            total_chunks_retrieved=len(raw_chunks),
            total_chunks_after_rerank=len(reranked_chunks),
            model_info=self.llm_client.provider_info,
        )

    def summarize_document(self, document_id: str, document_name: str) -> str:
        """Generate an executive summary for a full document."""
        from src.retrieval.vector_store import VectorStore
        store = VectorStore()
        chunks = store.get_document_chunks(document_id)
        if not chunks:
            return f"No content found for document: {document_name}"

        system, user = self.prompt_builder.build_summary_prompt(document_name, chunks)
        return self.llm_client.complete(system, user, max_tokens=2048)
