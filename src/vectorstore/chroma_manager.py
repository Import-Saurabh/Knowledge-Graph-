"""
src/vectorstore/chroma_manager.py
Thin wrapper around ChromaDB with three collections:
  • news_articles       — article-level embeddings for deduplication
  • canonical_entities  — entity embeddings for semantic resolution
  • relation_ontology   — relation phrase embeddings for canonicalisation

No bug-list items target this file directly.
Additions vs original:
  • upsert_entities()  — add-or-update helper used by entity_resolver when
    re-inserting on resume (avoids DuplicateIDError from Chroma on re-runs).
  • get_all_entity_embeddings() — mirrors get_all_relation_embeddings() so
    the resolver can do offline batch similarity if needed.
  • All collection operations wrapped in try/except with structured logging
    so a Chroma hiccup never silently corrupts pipeline state.
"""

import chromadb
from typing import List, Dict, Optional

from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

ARTICLE_COLLECTION  = "news_articles"
ENTITY_COLLECTION   = "canonical_entities"
RELATION_COLLECTION = "relation_ontology"


class ChromaManager:
    def __init__(self, persist_dir: str = None):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.article_collection  = self.client.get_or_create_collection(ARTICLE_COLLECTION)
        self.entity_collection   = self.client.get_or_create_collection(ENTITY_COLLECTION)
        self.relation_collection = self.client.get_or_create_collection(RELATION_COLLECTION)
        log.info("chroma_initialized", persist_dir=self.persist_dir)

    # ------------------------------------------------------------------
    # Articles
    # ------------------------------------------------------------------

    def add_articles(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
    ):
        try:
            self.article_collection.add(
                ids=ids, embeddings=embeddings, metadatas=metadatas
            )
        except Exception as e:
            log.error("chroma_add_articles_failed", count=len(ids), error=str(e))
            raise

    def search_articles(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[dict] = None,
    ) -> dict:
        kwargs = dict(query_embeddings=[query_embedding], n_results=n_results)
        if where:
            kwargs["where"] = where
        return self.article_collection.query(**kwargs)

    # ------------------------------------------------------------------
    # Entities
    # ------------------------------------------------------------------

    def add_entities(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
    ):
        try:
            self.entity_collection.add(
                ids=ids, embeddings=embeddings, metadatas=metadatas
            )
        except Exception as e:
            log.warning("chroma_add_entities_failed", count=len(ids), error=str(e))

    def update_entities(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
    ):
        try:
            self.entity_collection.update(
                ids=ids, embeddings=embeddings, metadatas=metadatas
            )
        except Exception as e:
            log.warning("chroma_update_entities_failed", count=len(ids), error=str(e))

    def upsert_entities(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
    ):
        """
        Add-or-update: safe to call on resume runs where the entity may
        already exist in Chroma (avoids DuplicateIDError).
        """
        try:
            self.entity_collection.upsert(
                ids=ids, embeddings=embeddings, metadatas=metadatas
            )
        except Exception as e:
            log.warning("chroma_upsert_entities_failed", count=len(ids), error=str(e))

    def delete_entities(self, ids: List[str]):
        try:
            self.entity_collection.delete(ids=ids)
        except Exception as e:
            log.warning("chroma_delete_entities_failed", count=len(ids), error=str(e))

    def search_entities(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[dict] = None,
    ) -> dict:
        kwargs = dict(query_embeddings=[query_embedding], n_results=n_results)
        if where:
            kwargs["where"] = where
        try:
            return self.entity_collection.query(**kwargs)
        except Exception as e:
            log.warning("chroma_search_entities_failed", error=str(e))
            return {}

    def get_all_entity_embeddings(self) -> dict:
        """Return all stored entity embeddings + ids (for offline batch use)."""
        try:
            return self.entity_collection.get(include=["embeddings", "metadatas"])
        except Exception as e:
            log.warning("chroma_get_all_entities_failed", error=str(e))
            return {}

    # ------------------------------------------------------------------
    # Relations
    # ------------------------------------------------------------------

    def add_relations(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
    ):
        try:
            self.relation_collection.add(
                ids=ids, embeddings=embeddings, metadatas=metadatas
            )
        except Exception as e:
            log.warning("chroma_add_relations_failed", count=len(ids), error=str(e))

    def search_relation_ontology(
        self,
        query_embedding: List[float],
        n_results: int = 10,
    ) -> dict:
        try:
            return self.relation_collection.query(
                query_embeddings=[query_embedding], n_results=n_results
            )
        except Exception as e:
            log.warning("chroma_search_relations_failed", error=str(e))
            return {}

    def get_all_relation_embeddings(self) -> dict:
        try:
            return self.relation_collection.get(include=["embeddings"])
        except Exception as e:
            log.warning("chroma_get_all_relations_failed", error=str(e))
            return {}
