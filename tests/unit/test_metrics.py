"""Unit tests for retrieval evaluation metrics."""
import math
import pytest
from src.utils.metrics import (
    mean_reciprocal_rank,
    recall_at_k,
    precision_at_k,
    ndcg_at_k,
    compute_retrieval_metrics,
)


class TestMeanReciprocalRank:
    def test_first_result_relevant(self):
        assert mean_reciprocal_rank(["a"], ["a", "b", "c"]) == 1.0

    def test_second_result_relevant(self):
        assert mean_reciprocal_rank(["b"], ["a", "b", "c"]) == pytest.approx(0.5)

    def test_third_result_relevant(self):
        assert mean_reciprocal_rank(["c"], ["a", "b", "c"]) == pytest.approx(1 / 3)

    def test_no_relevant_returns_zero(self):
        assert mean_reciprocal_rank(["x"], ["a", "b", "c"]) == 0.0

    def test_empty_retrieved_returns_zero(self):
        assert mean_reciprocal_rank(["a"], []) == 0.0

    def test_multiple_relevant_uses_first_hit(self):
        # "b" is at rank 2, "c" at rank 3 — MRR should be 1/2
        assert mean_reciprocal_rank(["b", "c"], ["a", "b", "c"]) == pytest.approx(0.5)


class TestRecallAtK:
    def test_all_relevant_found(self):
        assert recall_at_k(["a", "b"], ["a", "b", "c"], k=2) == 1.0

    def test_partial_recall(self):
        assert recall_at_k(["a", "b", "c"], ["a", "b", "x"], k=3) == pytest.approx(2 / 3)

    def test_none_found(self):
        assert recall_at_k(["x"], ["a", "b", "c"], k=3) == 0.0

    def test_k_larger_than_retrieved(self):
        assert recall_at_k(["a"], ["a"], k=10) == 1.0

    def test_empty_relevant_returns_zero(self):
        assert recall_at_k([], ["a", "b"], k=2) == 0.0

    def test_k_zero_cuts_list(self):
        assert recall_at_k(["a"], ["a", "b"], k=0) == 0.0


class TestPrecisionAtK:
    def test_perfect_precision(self):
        assert precision_at_k(["a", "b"], ["a", "b"], k=2) == 1.0

    def test_half_precision(self):
        assert precision_at_k(["a"], ["a", "b"], k=2) == pytest.approx(0.5)

    def test_zero_precision(self):
        assert precision_at_k(["x"], ["a", "b"], k=2) == 0.0

    def test_k_zero_returns_zero(self):
        assert precision_at_k(["a"], ["a", "b"], k=0) == 0.0

    def test_k_one_relevant_at_top(self):
        assert precision_at_k(["a"], ["a", "b", "c"], k=1) == 1.0

    def test_k_one_irrelevant_at_top(self):
        assert precision_at_k(["b"], ["a", "b", "c"], k=1) == 0.0


class TestNdcgAtK:
    def test_perfect_ranking(self):
        assert ndcg_at_k(["a", "b"], ["a", "b", "c"], k=2) == pytest.approx(1.0)

    def test_reversed_ranking_lower_than_perfect(self):
        # only one relevant doc; placing it at rank 2 vs rank 1 gives lower NDCG
        perfect = ndcg_at_k(["a"], ["a", "b", "c"], k=3)
        worse = ndcg_at_k(["a"], ["b", "a", "c"], k=3)
        assert worse < perfect

    def test_no_relevant_returns_zero(self):
        assert ndcg_at_k(["x"], ["a", "b", "c"], k=3) == 0.0

    def test_empty_relevant_returns_zero(self):
        assert ndcg_at_k([], ["a", "b"], k=2) == 0.0

    def test_single_relevant_at_rank1(self):
        expected = 1.0 / math.log2(2)  # dcg == idcg
        assert ndcg_at_k(["a"], ["a", "b"], k=2) == pytest.approx(1.0)

    def test_single_relevant_at_rank2(self):
        # dcg = 1/log2(3), idcg = 1/log2(2)
        expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
        assert ndcg_at_k(["b"], ["a", "b", "c"], k=3) == pytest.approx(expected)

    def test_k_limits_window(self):
        # relevant doc is outside k=1 window
        assert ndcg_at_k(["b"], ["a", "b"], k=1) == 0.0


class TestComputeRetrievalMetrics:
    def test_returns_all_keys(self):
        result = compute_retrieval_metrics(["a"], ["a", "b"], k=5)
        assert set(result.keys()) == {"mrr", "recall@5", "precision@5", "ndcg@5"}

    def test_custom_k_in_keys(self):
        result = compute_retrieval_metrics(["a"], ["a"], k=3)
        assert "recall@3" in result
        assert "precision@3" in result
        assert "ndcg@3" in result

    def test_perfect_retrieval_values(self):
        result = compute_retrieval_metrics(["a"], ["a"], k=1)
        assert result["mrr"] == 1.0
        assert result["recall@1"] == 1.0
        assert result["precision@1"] == 1.0
        assert result["ndcg@1"] == 1.0

    def test_no_match_all_zeros(self):
        result = compute_retrieval_metrics(["x"], ["a", "b"], k=2)
        assert result["mrr"] == 0.0
        assert result["recall@2"] == 0.0
        assert result["precision@2"] == 0.0
        assert result["ndcg@2"] == 0.0
