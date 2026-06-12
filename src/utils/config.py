from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # LLM Providers — pick ONE
    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    MOONSHOT_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    GROQ_API_KEY: str = ""

    # Provider: anthropic | openai | moonshot | openrouter | groq
    LLM_PROVIDER: str = "groq"

    # Model overrides (provider defaults used if empty)
    LLM_MODEL: str = ""

    # Embeddings
    EMBEDDING_MODEL: str = "small"
    USE_SPACY_FALLBACK: bool = False

    # Storage
    DATABASE_URL: str = "sqlite:///data/news_kg.db"
    CHROMA_PERSIST_DIR: str = "data/chroma_db"

    SEED_ONTOLOGY_PATH: str = "config/seed_ontology.yaml"
    ENABLE_TYPE_INDUCTION: bool = True
    ENABLE_RELATION_CLUSTERING: bool = True

    TEMPORAL_WINDOW_DAYS: int = 7
    MIN_CLUSTER_SIZE: int = 2
    MAX_CLUSTERS_PER_WINDOW: int = 200

    SIMILARITY_THRESHOLD: float = 0.95
    ENTITY_MERGE_THRESHOLD: float = 0.88

    NER_BATCH_SIZE: int = 32
    EMBEDDING_BATCH_SIZE: int = 64

    class Config:
        env_file = ".env"

settings = Settings()