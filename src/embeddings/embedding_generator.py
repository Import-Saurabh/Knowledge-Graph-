import numpy as np
from typing import List
from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

class EmbeddingGenerator:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or (
            "BAAI/bge-large-en-v1.5" if settings.EMBEDDING_MODEL == "large" 
            else "BAAI/bge-small-en-v1.5"
        )
        self._model = None
        self._dimension = 384 if "small" in self.model_name else 1024

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                log.info("embedding_model_loaded", model=self.model_name)
            except Exception as e:
                log.error("failed_to_load_embedding_model", error=str(e))
                raise
        return self._model

    def embed_text(self, text: str) -> List[float]:
        model = self._load_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        model = self._load_model()
        embeddings = model.encode(texts, normalize_embeddings=True, batch_size=settings.EMBEDDING_BATCH_SIZE)
        return embeddings.tolist()

    def embed_articles(self, articles) -> List[tuple]:
        texts = [f"{a.title}. {a.content[:2000]}" for a in articles]
        embeddings = self.embed_texts(texts)
        return [(a.id, emb) for a, emb in zip(articles, embeddings)]
