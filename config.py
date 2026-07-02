# config.py

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # API Keys
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")

    # Pinecone
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "rag-pdf-index")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")

    # Embedding
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Groq
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # RAG defaults
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    similarity_threshold: float = 0.3
    max_pdf_size_mb: int = 20


def get_settings() -> Settings:
    return Settings()


def validate_settings(settings: Settings) -> list[str]:
    problems = []
    if not settings.pinecone_api_key:
        problems.append("PINECONE_API_KEY is missing.")
    if not settings.groq_api_key:
        problems.append("GROQ_API_KEY is missing.")
    if not settings.pinecone_index_name:
        problems.append("PINECONE_INDEX_NAME is missing.")
    return problems

