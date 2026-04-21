"""Evaluation metrics for retrieval and generation quality."""
from __future__ import annotations
from typing import List
import math


def mean_reciprocal_rank(relevant_ids: List[str], retrieved_ids: List[str]) -> float:
    """MRR — how early the first relevant doc appears in ranking."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def recall_at_k(relevant_ids: List[str], retrieved_ids: List[str], k: int) -> float:
    """Fraction of relevant docs found in top-k retrieved."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = sum(1 for r in relevant_ids if r in top_k)
    return hits / len(relevant_ids)


def precision_at_k(relevant_ids: List[str], retrieved_ids: List[str], k: int) -> float:
    """Fraction of top-k retrieved docs that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for r in top_k if r in relevant_set)
    return hits / k


def ndcg_at_k(relevant_ids: List[str], retrieved_ids: List[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain at k."""
    relevant_set = set(relevant_ids)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved_ids[:k], start=1)
        if doc_id in relevant_set
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def compute_retrieval_metrics(
    relevant_ids: List[str],
    retrieved_ids: List[str],
    k: int = 5,
) -> dict:
    return {
        "mrr": mean_reciprocal_rank(relevant_ids, retrieved_ids),
        f"recall@{k}": recall_at_k(relevant_ids, retrieved_ids, k),
        f"precision@{k}": precision_at_k(relevant_ids, retrieved_ids, k),
        f"ndcg@{k}": ndcg_at_k(relevant_ids, retrieved_ids, k),
    }
