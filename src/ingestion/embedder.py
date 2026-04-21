"""Embedding generation using sentence-transformers."""
from __future__ import annotations
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils.config import get_settings
from src.utils.logger import logger
from src.ingestion.chunker import TextChunk

settings = get_settings()


class Embedder:
    """
    Wraps sentence-transformers for batch embedding of text chunks.
    Uses mean pooling with L2 normalisation (standard for cosine similarity search).
    """

    def __init__(self, model_name: str | None = None):
        model_name = model_name or settings.embedding_model
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dim: {self.embedding_dim}")

    def embed_texts(self, texts: List[str], batch_size: int | None = None) -> np.ndarray:
        """
        Embed a list of strings.
        Returns shape (N, embedding_dim) float32 array.
        """
        batch_size = batch_size or settings.embedding_batch_size
        logger.info(f"Embedding {len(texts)} texts (batch_size={batch_size})")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,   # L2 norm -> cosine sim = dot product
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_chunks(
        self, chunks: List[TextChunk], batch_size: int | None = None
    ) -> tuple[List[TextChunk], np.ndarray]:
        """
        Embed a list of TextChunk objects.
        Returns (chunks, embeddings) where embeddings[i] corresponds to chunks[i].
        """
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts, batch_size)
        return chunks, embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns shape (embedding_dim,)."""
        return self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
