"""
src/relations/relation_ontology.py
Normalises free-text relation phrases to canonical forms via embedding
similarity + HDBSCAN clustering.

Hardening vs original:
  • _create_relation_entry already captured `canonical` before session.close()
    in the original — preserved as-is.
  • Added explicit `is not None` guard on embedding (mirrors bug-#4 fix style).
  • cluster_relations: guard against < 5 embeddings returns early gracefully.
"""

import uuid
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional

from src.utils.db import get_session, RelationOntologyDB
from src.utils.logger import get_logger
from src.utils.config import settings

log = get_logger(__name__)

RELATION_SIMILARITY_THRESHOLD = 0.85


class RelationOntologyManager:
    def __init__(self, chroma_manager, embedding_generator):
        self.chroma = chroma_manager
        self.embedder = embedding_generator

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, relation_text: str) -> str:
        import re
        text = relation_text.lower().strip()
        text = re.sub(r"\b(the|a|an|to|of|for|with|by|on|in|at)\b", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize_relation(self, relation_text: str) -> str:
        """
        Map a free-text relation phrase to its canonical form.
        Tries ChromaDB cosine similarity first; falls back to creating a new
        entry.
        """
        cleaned = self._preprocess(relation_text)

        embedding: list | None = None
        try:
            embedding = self.embedder.embed_text(cleaned)
        except Exception as e:
            log.warning("relation_embedding_failed", error=str(e))

        if embedding is not None and self.chroma:
            try:
                matches = self.chroma.search_relation_ontology(embedding, n_results=5)
                if matches and matches.get("ids") and matches["ids"][0]:
                    ids = matches["ids"][0]
                    distances = matches.get("distances", [[]])[0]
                    if ids and distances:
                        similarity = 1.0 - min(distances[0], 1.0)
                        if similarity > RELATION_SIMILARITY_THRESHOLD:
                            self._increment_usage(ids[0])
                            session = get_session()
                            rel = session.query(RelationOntologyDB).filter_by(
                                relation_id=ids[0]
                            ).first()
                            # Capture value before close to avoid DetachedInstanceError
                            canonical = rel.relation_canonical if rel else cleaned
                            session.close()
                            return canonical
            except Exception as e:
                log.warning("relation_search_failed", error=str(e))

        return self._create_relation_entry(cleaned, embedding, relation_text)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_relation_entry(
        self, cleaned: str, embedding: list | None, original_text: str
    ) -> str:
        session = get_session()
        existing = session.query(RelationOntologyDB).filter_by(
            relation_text=original_text
        ).first()
        if existing:
            # Read BEFORE closing (original bug fix already present, kept here)
            canonical = existing.relation_canonical or cleaned
            session.close()
            return canonical

        relation_id = str(uuid.uuid4())
        emb_bytes = None
        if embedding is not None:
            emb_bytes = np.array(embedding, dtype=np.float32).tobytes()

        canonical = cleaned.upper().replace(" ", "_")
        db_rel = RelationOntologyDB(
            relation_id=relation_id,
            relation_text=original_text,
            relation_canonical=canonical,
            relation_embedding=emb_bytes,
            usage_count=1,
            first_seen=datetime.utcnow(),
            last_seen=datetime.utcnow(),
        )
        session.add(db_rel)
        session.commit()
        # Capture BEFORE session.close() — avoids DetachedInstanceError
        canonical = db_rel.relation_canonical
        session.close()

        if embedding is not None and self.chroma:
            try:
                self.chroma.add_relations(
                    ids=[relation_id],
                    embeddings=[embedding],
                    metadatas=[{
                        "relation_text": original_text,
                        "relation_canonical": canonical,
                        "usage_count": 1,
                    }],
                )
            except Exception as e:
                log.warning("relation_chroma_add_failed", error=str(e))

        log.info("new_relation_created", text=original_text, canonical=canonical)
        return canonical

    def _increment_usage(self, relation_id: str):
        session = get_session()
        rel = session.query(RelationOntologyDB).filter_by(
            relation_id=relation_id
        ).first()
        if rel:
            rel.usage_count += 1
            rel.last_seen = datetime.utcnow()
            session.commit()
        session.close()

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def cluster_relations(self) -> dict:
        """
        HDBSCAN clustering over all stored relation embeddings.
        Returns {cluster_id: [relation_ids]}.
        """
        try:
            import hdbscan

            data = self.chroma.get_all_relation_embeddings()
            if (
                not data
                or not data.get("embeddings")
                or len(data["embeddings"]) < 5
            ):
                log.info("cluster_relations_skipped", reason="fewer than 5 embeddings")
                return {}

            embeddings = np.array(data["embeddings"])
            ids = data["ids"]

            clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric="euclidean")
            labels = clusterer.fit_predict(embeddings)

            clusters: Dict[int, List[str]] = {}
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                clusters.setdefault(label, []).append(ids[i])

            session = get_session()
            for cluster_id, relation_ids in clusters.items():
                rels = (
                    session.query(RelationOntologyDB)
                    .filter(RelationOntologyDB.relation_id.in_(relation_ids))
                    .order_by(RelationOntologyDB.usage_count.desc())
                    .all()
                )
                if rels:
                    # Most-used relation becomes the cluster canonical
                    canonical = rels[0].relation_canonical
                    for rel in rels:
                        rel.relation_canonical = canonical
                        rel.cluster_id = int(cluster_id)

            session.commit()
            session.close()
            return clusters

        except Exception as e:
            log.warning("relation_clustering_failed", error=str(e))
            return {}

    # ------------------------------------------------------------------
    # Taxonomy report
    # ------------------------------------------------------------------

    def get_relation_taxonomy(self) -> List[dict]:
        session = get_session()
        relations = session.query(RelationOntologyDB).all()

        clusters: Dict[int, list] = {}
        for rel in relations:
            cid = rel.cluster_id if rel.cluster_id is not None else -1
            clusters.setdefault(cid, []).append({
                "relation_text": rel.relation_text,
                "relation_canonical": rel.relation_canonical,
                "usage_count": rel.usage_count,
            })

        session.close()
        return [
            {
                "cluster_id": cid,
                "relations": items,
                "canonical": items[0]["relation_canonical"] if items else "",
            }
            for cid, items in clusters.items()
        ]