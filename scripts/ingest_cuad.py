"""
Download and ingest the CUAD dataset (Contract Understanding Atticus Dataset).
500 real legal contracts across multiple categories.
Run: python scripts/ingest_cuad.py
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/ingest_cuad.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset
from src.utils.logger import logger
from src.utils.config import get_settings

settings = get_settings()


def download_cuad_texts(output_dir: Path, max_contracts: int = 50) -> list[Path]:
    """
    Download CUAD from HuggingFace and save contract texts as text files.
    (CUAD provides raw text, not PDFs — we save as .txt and treat as plain documents.)
    """
    logger.info("Loading CUAD dataset from HuggingFace...")
    dataset = load_dataset("theatticusproject/cuad", split="train", trust_remote_code=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    for item in dataset:
        # The theatticusproject/cuad dataset has a single "text" column containing
        # contract passages; there is no separate "title" or "context" field.
        text = item.get("text") or item.get("context", "")
        if len(text) < 500:
            continue

        idx = len(saved_files)
        out_path = output_dir / f"cuad_{idx:04d}.txt"
        out_path.write_text(text, encoding="utf-8")
        saved_files.append(out_path)

        if len(saved_files) >= max_contracts:
            break

    logger.info(f"Saved {len(saved_files)} contracts to {output_dir}")
    return saved_files


def ingest_text_files(file_paths: list[Path]) -> None:
    """
    Ingest plain text contracts through a simplified pipeline
    (skips PDF extraction, goes straight to clean -> chunk -> embed -> store).
    """
    from src.ingestion.text_cleaner import TextCleaner
    from src.ingestion.chunker import SemanticChunker, TextChunk
    from src.ingestion.embedder import Embedder
    from src.retrieval.vector_store import VectorStore
    import hashlib

    cleaner = TextCleaner()
    chunker = SemanticChunker()
    embedder = Embedder()
    store = VectorStore()

    for i, path in enumerate(file_paths):
        logger.info(f"[{i+1}/{len(file_paths)}] Ingesting: {path.name}")
        text = path.read_text(encoding="utf-8", errors="replace")
        cleaned = cleaner.clean(text)
        if not cleaned.strip():
            logger.warning(f"  Skipping empty: {path.name}")
            continue

        doc_id = hashlib.md5(str(path).encode()).hexdigest()[:16]
        chunks = chunker.chunk_document(
            text=cleaned,
            document_id=doc_id,
            document_name=path.name,
            metadata={"source": "CUAD", "file": path.name},
        )
        if not chunks:
            continue

        _, embeddings = embedder.embed_chunks(chunks)
        store.add_chunks(chunks, embeddings)
        logger.info(f"  -> {len(chunks)} chunks stored")

    logger.info(f"Done. Total chunks in store: {store.count()}")


def main():
    parser = argparse.ArgumentParser(description="Download and ingest CUAD contracts")
    parser.add_argument("--max", type=int, default=50, help="Max contracts to ingest")
    parser.add_argument("--output-dir", type=str, default="data/raw/cuad")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    files = download_cuad_texts(output_dir, max_contracts=args.max)
    ingest_text_files(files)


if __name__ == "__main__":
    main()
