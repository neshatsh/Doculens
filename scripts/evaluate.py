"""
Evaluate retrieval quality on CUAD Q&A pairs and log results to MLflow.
Run: python scripts/evaluate.py
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
from datasets import load_dataset
from src.retrieval.retriever import Retriever
from src.retrieval.reranker import Reranker
from src.utils.metrics import compute_retrieval_metrics
from src.utils.config import get_settings
from src.utils.logger import logger

settings = get_settings()


def load_cuad_qa_pairs(max_samples: int = 200) -> list[dict]:
    """Load CUAD question-answer pairs for evaluation."""
    logger.info("Loading CUAD QA pairs...")
    dataset = load_dataset("theatticusproject/cuad-qa", split="train", trust_remote_code=True)
    pairs = []
    for item in dataset:
        question = item.get("question", "")
        answers = item.get("answers", {})
        answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
        context = item.get("context", "")
        title = item.get("title", "unknown")

        if not question or not answer_texts or not context:
            continue

        pairs.append({
            "question": question,
            "answer": answer_texts[0] if answer_texts else "",
            "document_name": title,
        })

        if len(pairs) >= max_samples:
            break

    logger.info(f"Loaded {len(pairs)} QA pairs")
    return pairs


def evaluate_retrieval(pairs: list[dict], top_k: int = 5) -> dict:
    """
    Evaluate retrieval quality:
    - Metric: does the retrieved text contain the gold answer?
    - Reports: hit_rate@k, MRR (approximated)
    """
    retriever = Retriever(top_k=20)
    reranker = Reranker(top_k=top_k)

    hits = 0
    mrr_total = 0.0
    total = 0

    for i, pair in enumerate(pairs):
        question = pair["question"]
        gold_answer = pair["answer"].lower().strip()

        try:
            raw_chunks = retriever.retrieve(question)
            reranked = reranker.rerank(question, raw_chunks, top_k=top_k)
        except Exception as e:
            logger.warning(f"Retrieval failed for pair {i}: {e}")
            continue

        # Check if gold answer appears in any retrieved chunk
        hit_rank = None
        for rank, chunk in enumerate(reranked, start=1):
            if gold_answer[:50] in chunk["text"].lower():
                hit_rank = rank
                break

        if hit_rank:
            hits += 1
            mrr_total += 1.0 / hit_rank

        total += 1
        if (i + 1) % 20 == 0:
            logger.info(f"  Evaluated {i+1}/{len(pairs)} pairs | hit_rate={hits/total:.3f}")

    metrics = {
        f"hit_rate_at_{top_k}": hits / total if total > 0 else 0.0,
        "mrr": mrr_total / total if total > 0 else 0.0,
        "total_evaluated": total,
        "top_k": top_k,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate DocuLens retrieval on CUAD")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    with mlflow.start_run(run_name="retrieval_eval"):
        mlflow.log_params({
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "retrieval_top_k": settings.retrieval_top_k,
            "reranker_top_k": args.top_k,
            "max_samples": args.max_samples,
        })

        pairs = load_cuad_qa_pairs(max_samples=args.max_samples)
        metrics = evaluate_retrieval(pairs, top_k=args.top_k)

        mlflow.log_metrics(metrics)
        logger.info(f"Results: {metrics}")
        print("\n=== Evaluation Results ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
