"""Application configuration loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"


class Config:
    """Central configuration for the Trade Assistant RAG application."""

    # Flask
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-in-production")
    DEBUG = os.getenv("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")

    # Website ingestion
    TARGET_WEBSITE_BASE_URL = os.getenv(
        "TARGET_WEBSITE_BASE_URL", "https://www.investopedia.com"
    ).rstrip("/")
    MAX_PAGES = int(os.getenv("MAX_PAGES", "50"))
    MAX_DEPTH = int(os.getenv("MAX_DEPTH", "2"))
    CRAWL_DELAY_SECONDS = float(os.getenv("CRAWL_DELAY_SECONDS", "1.0"))
    USER_AGENT = os.getenv(
        "USER_AGENT",
        "TradeScope-AI-Bot/1.0 (+https://github.com/trade-assistant; educational)",
    )

    # Chunking
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

    # Embeddings
    EMBEDDING_MODEL_NAME = os.getenv(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    HF_HOME = os.getenv("HF_HOME", str(BASE_DIR / ".cache" / "huggingface"))

    # Vector store
    VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "pinecone").lower()
    FAISS_INDEX_PATH = os.getenv(
        "FAISS_INDEX_PATH", str(VECTOR_STORE_DIR / "faiss.index")
    )
    FAISS_METADATA_PATH = os.getenv(
        "FAISS_METADATA_PATH", str(VECTOR_STORE_DIR / "metadata.json")
    )
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "trade-assistant")
    PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")
    PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
    PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
    # Legacy alias
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", PINECONE_REGION)

    # Retrieval
    TOP_K = int(os.getenv("TOP_K", "5"))

    # LLM
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
    API_KEY = os.getenv("API_KEY", "")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")  # For local / custom endpoints
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

    # Protected rebuild endpoint
    REBUILD_API_KEY = os.getenv("REBUILD_API_KEY", "")

    @classmethod
    def ensure_directories(cls) -> None:
        """Create required data directories if they do not exist."""
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_llm_config(cls) -> None:
        """Raise ValueError if LLM provider requires an API key that is missing."""
        if cls.LLM_PROVIDER not in ("openai", "anthropic"):
            return
        key = (cls.API_KEY or "").strip()
        if not key or key.startswith("your-"):
            raise ValueError(
                "API_KEY is missing or still a placeholder. "
                "Set your real OpenAI/Anthropic key in .env and save the file (Ctrl+S)."
            )

    @classmethod
    def validate_vector_store_config(cls) -> None:
        """Raise ValueError if Pinecone is selected but not configured."""
        if cls.VECTOR_STORE_TYPE != "pinecone":
            return
        key = cls.PINECONE_API_KEY.strip()
        if not key or key.startswith("your-"):
            raise ValueError(
                "PINECONE_API_KEY is missing or still a placeholder. "
                "Set your real key from https://app.pinecone.io in the .env file."
            )
