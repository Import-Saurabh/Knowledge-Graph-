import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Optional
from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

ARTICLE_COLLECTION = "news_articles"
ENTITY_COLLECTION = "canonical_entities"
RELATION_COLLECTION = "relation_ontology"

class ChromaManager:
    def __init__(self, persist_dir: str = None):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.article_collection = self.client.get_or_create_collection(ARTICLE_COLLECTION)
        self.entity_collection = self.client.get_or_create_collection(ENTITY_COLLECTION)
        self.relation_collection = self.client.get_or_create_collection(RELATION_COLLECTION)
        log.info("chroma_initialized", persist_dir=self.persist_dir)

    def add_articles(self, ids: List[str], embeddings: List[List[float]], metadatas: List[dict]):
        self.article_collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def search_articles(self, query_embedding: List[float], n_results: int = 10, where: dict = None):
        return self.article_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )

    def add_entities(self, ids: List[str], embeddings: List[List[float]], metadatas: List[dict]):
        self.entity_collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def update_entities(self, ids: List[str], embeddings: List[List[float]], metadatas: List[dict]):
        self.entity_collection.update(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def search_entities(self, query_embedding: List[float], n_results: int = 10, where: dict = None):
        return self.entity_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )

    def add_relations(self, ids: List[str], embeddings: List[List[float]], metadatas: List[dict]):
        self.relation_collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def search_relation_ontology(self, query_embedding: List[float], n_results: int = 10):
        return self.relation_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

    def get_all_relation_embeddings(self):
        data = self.relation_collection.get(include=["embeddings"])
        return data
