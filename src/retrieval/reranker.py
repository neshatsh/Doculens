"""BERT cross-encoder reranker for precision scoring of retrieved chunks."""
from __future__ import annotations
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from src.utils.config import get_settings
from src.utils.logger import logger

settings = get_settings()


class Reranker:
    """
    Cross-encoder reranker using ms-marco-MiniLM.

    Why cross-encoder vs bi-encoder:
    - Bi-encoder (retriever): fast, encodes query + doc independently
    - Cross-encoder (reranker): slow but accurate, sees query+doc jointly
    - We use bi-encoder for speed (top-20), then cross-encoder for accuracy (top-5)

    This is the standard two-stage retrieval architecture used in production RAG.
    """

    def __init__(self, model_name: str | None = None, top_k: int | None = None):
        model_name = model_name or settings.reranker_model
        self.top_k = top_k or settings.reranker_top_k
        logger.info(f"Loading reranker: {model_name}")
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank chunks by cross-encoder score.

        Args:
            query: The user's query
            chunks: Chunk dicts from the retriever (each must have 'text')
            top_k: Return this many top chunks

        Returns:
            Chunks sorted by reranker score (descending), limited to top_k.
            Each chunk gets a 'rerank_score' field added.
        """
        k = top_k or self.top_k
        if not chunks:
            return []

        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = self.model.predict(pairs, show_progress_bar=False)

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        top = reranked[:k]

        logger.info(
            f"Reranked {len(chunks)} -> {len(top)} chunks | "
            f"top score: {top[0]['rerank_score']:.3f}"
        )
        return top

    def rerank_and_threshold(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        score_threshold: float = 0.0,
        top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Rerank and filter out chunks below a minimum score threshold."""
        reranked = self.rerank(query, chunks, top_k)
        filtered = [c for c in reranked if c["rerank_score"] >= score_threshold]
        if len(filtered) < len(reranked):
            logger.debug(
                f"Threshold {score_threshold} filtered {len(reranked) - len(filtered)} chunks"
            )
        return filtered
