"""
src/embeddings/embedding_generator.py
Wraps sentence-transformers (BAAI/bge-small-en-v1.5 by default).

No bug-list items target this file. Minor additions:
  • `embed_articles` now guards against None content (mirrors bug #6 style).
  • Model load is lazy and cached; dimension is inferred from model name.
"""

from typing import List, Tuple
from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


class EmbeddingGenerator:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or (
            "BAAI/bge-large-en-v1.5"
            if settings.EMBEDDING_MODEL == "large"
            else "BAAI/bge-small-en-v1.5"
        )
        self._model = None
        # Dimension heuristic: small → 384, everything else → 1024
        self._dimension = 384 if "small" in self.model_name else 1024

    @property
    def dimension(self) -> int:
        return self._dimension

    # ------------------------------------------------------------------
    # Model loading (lazy, cached)
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                log.info("embedding_model_loaded", model=self.model_name,
                         dimension=self._dimension)
            except Exception as e:
                log.error("failed_to_load_embedding_model", error=str(e))
                raise
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> List[float]:
        """Embed a single string. Returns a normalised float list."""
        model = self._load_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Batch-embed a list of strings."""
        model = self._load_model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
        )
        return embeddings.tolist()

    def embed_articles(self, articles) -> List[Tuple[str, List[float]]]:
        """
        Return [(article_id, embedding), …] in the same order as `articles`.
        Guards against None content (mirrors bug #6 style — avoids TypeError
        on `None[:2000]`).
        """
        texts = [
            f"{a.title or ''}. {(a.content or '')[:2000]}"
            for a in articles
        ]
        embeddings = self.embed_texts(texts)
        return [(a.id, emb) for a, emb in zip(articles, embeddings)]