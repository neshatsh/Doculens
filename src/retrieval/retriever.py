"""RAG retriever: embeds query, fetches top-k chunks from vector store."""
from __future__ import annotations
from typing import List, Dict, Any, Optional
from src.utils.config import get_settings
from src.utils.logger import logger
from src.ingestion.embedder import Embedder
from src.retrieval.vector_store import VectorStore

settings = get_settings()


class Retriever:
    """
    Two-stage retrieval:
    1. Dense retrieval: embed query -> cosine search in ChromaDB (top_k=20)
    2. Hands results to Reranker for precision scoring (top_k=5)

    This class handles stage 1 only.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        top_k: int | None = None,
    ):
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()
        self.top_k = top_k or settings.retrieval_top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top_k chunks for a query.

        Args:
            query: Natural language question
            top_k: Override default top_k
            document_filter: Optionally restrict to specific document_ids

        Returns:
            List of chunk dicts sorted by similarity score (descending)
        """
        k = top_k or self.top_k
        logger.info(f"Retrieving top-{k} chunks for query: '{query[:80]}...'")

        query_embedding = self.embedder.embed_query(query)

        where_filter = None
        if document_filter:
            if len(document_filter) == 1:
                where_filter = {"document_id": document_filter[0]}
            else:
                where_filter = {"document_id": {"$in": document_filter}}

        hits = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=k,
            where=where_filter,
        )

        logger.info(f"Retrieved {len(hits)} chunks (top score: {hits[0]['score']:.3f})" if hits else "No chunks found")
        return hits

    def retrieve_with_scores(
        self, query: str, top_k: int | None = None
    ) -> List[tuple[Dict[str, Any], float]]:
        """Returns list of (chunk, score) tuples."""
        hits = self.retrieve(query, top_k)
        return [(h, h["score"]) for h in hits]
