import numpy as np
from typing import List, Tuple
from src.utils.logger import get_logger
from src.utils.config import settings

log = get_logger(__name__)

class DuplicateDetector:
    def __init__(self, chroma_manager, embedding_generator):
        self.chroma = chroma_manager
        self.embedder = embedding_generator

    def find_duplicates(self, articles) -> List[Tuple[str, str, float]]:
        """Returns list of (article_id_1, article_id_2, similarity) for duplicates."""
        duplicates = []

        for article in articles:
            try:
                text = f"{article.title}. {article.content[:2000]}"
                embedding = self.embedder.embed_text(text)
                results = self.chroma.search_articles(embedding, n_results=10)

                if not results or not results.get("ids"):
                    continue

                ids = results["ids"][0]
                distances = results.get("distances", [[]])[0]

                for i, other_id in enumerate(ids):
                    if other_id == article.id:
                        continue
                    dist = distances[i] if i < len(distances) else 1.0
                    similarity = 1.0 - min(dist, 1.0)
                    if similarity >= settings.SIMILARITY_THRESHOLD:
                        duplicates.append((article.id, other_id, similarity))
                        log.info("duplicate_found", id1=article.id, id2=other_id, similarity=similarity)
            except Exception as e:
                log.warning("duplicate_check_failed", article_id=article.id, error=str(e))

        return duplicates

    def mark_duplicates(self, duplicate_pairs: List[Tuple[str, str, float]]):
        from src.utils.db import get_session, ArticleDB
        session = get_session()

        for id1, id2, sim in duplicate_pairs:
            group_id = f"dup_{min(id1, id2)}"
            session.query(ArticleDB).filter(ArticleDB.id.in_([id1, id2])).update({
                ArticleDB.duplicate_group_id: group_id,
                ArticleDB.status: "deduplicated"
            }, synchronize_session=False)

        session.commit()
        session.close()
