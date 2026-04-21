from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # App
    app_name: str = "DocuLens"
    app_version: str = "1.0.0"
    debug: bool = False

    # Paths
    data_raw_dir: Path = BASE_DIR / "data" / "raw"
    data_processed_dir: Path = BASE_DIR / "data" / "processed"
    embeddings_dir: Path = BASE_DIR / "data" / "embeddings"
    chroma_persist_dir: Path = BASE_DIR / "data" / "embeddings" / "chroma"

    # LLM
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    llm_provider: str = "openai"          # "openai" | "anthropic"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    # Embedding model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 64

    # Reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 5

    # Retrieval
    retrieval_top_k: int = 20             # fetch more, rerank down to top_k
    chunk_size: int = 512                 # tokens per chunk
    chunk_overlap: int = 64

    # VLM (optional — falls back to pdfplumber if key missing)
    use_vlm: bool = False
    vlm_model: str = "gpt-4o"

    # MLflow
    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment: str = "doculens"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
