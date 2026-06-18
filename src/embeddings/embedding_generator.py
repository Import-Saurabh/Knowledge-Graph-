"""
src/embeddings/embedding_generator.py
Wraps sentence-transformers (BAAI/bge-small-en-v1.5 by default).

No bug-list items target this file. Minor additions:
  • `embed_articles` now guards against None content (mirrors bug #6 style).
  • Model load is lazy and cached; dimension is inferred from model name.
  • _load_model suppresses the benign BERT 'embeddings.position_ids UNEXPECTED'
    buffer warning that fires on every cold load of bge-small/large-en-v1.5.
"""

import logging
import warnings
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
        if self._model is not None:
            return self._model

        # Suppress the harmless 'embeddings.position_ids UNEXPECTED' warning.
        # Root cause: newer BERT checkpoints register position_ids as a
        # persistent *buffer* (not a parameter); the sentence-transformers
        # loader only expects parameters, so it flags every buffer as
        # UNEXPECTED.  No weights are wrong — the buffer is rebuilt correctly
        # at runtime.  We mute the two loggers that print the load-report
        # table and restore them immediately after the model is ready.
        st_logger = logging.getLogger("sentence_transformers")
        hf_logger = logging.getLogger("transformers.modeling_utils")
        old_st    = st_logger.level
        old_hf    = hf_logger.level
        st_logger.setLevel(logging.ERROR)
        hf_logger.setLevel(logging.ERROR)

        try:
            from sentence_transformers import SentenceTransformer
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*position_ids.*",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message=r".*UNEXPECTED.*",
                    category=UserWarning,
                )
                self._model = SentenceTransformer(self.model_name)
            log.info("embedding_model_loaded", model=self.model_name,
                     dimension=self._dimension)
        except Exception as e:
            log.error("failed_to_load_embedding_model", error=str(e))
            raise
        finally:
            # Always restore — even if load fails
            st_logger.setLevel(old_st)
            hf_logger.setLevel(old_hf)

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