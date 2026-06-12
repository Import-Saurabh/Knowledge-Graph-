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

    def _preprocess(self, relation_text: str) -> str:
        import re
        text = relation_text.lower().strip()
        text = re.sub(r"\b(the|a|an|to|of|for|with|by|on|in|at)\b", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def normalize_relation(self, relation_text: str) -> str:
        cleaned = self._preprocess(relation_text)

        # Search existing relations
        embedding = None
        try:
            embedding = self.embedder.embed_text(cleaned)
        except Exception as e:
            log.warning("relation_embedding_failed", error=str(e))

        if embedding and self.chroma:
            try:
                matches = self.chroma.search_relation_ontology(embedding, n_results=5)
                if matches and matches.get("ids") and matches["ids"][0]:
                    ids = matches["ids"][0]
                    distances = matches.get("distances", [[]])[0]
                    if ids and distances:
                        top_dist = distances[0]
                        similarity = 1.0 - min(top_dist, 1.0)
                        if similarity > RELATION_SIMILARITY_THRESHOLD:
                            self._increment_usage(ids[0])
                            # Get canonical form
                            session = get_session()
                            rel = session.query(RelationOntologyDB).filter_by(relation_id=ids[0]).first()
                            canonical = rel.relation_canonical if rel else cleaned
                            session.close()
                            return canonical
            except Exception as e:
                log.warning("relation_search_failed", error=str(e))

        # Create new relation entry
        return self._create_relation_entry(cleaned, embedding, relation_text)

    def _create_relation_entry(self, cleaned: str, embedding, original_text: str) -> str:
        session = get_session()
        existing = session.query(RelationOntologyDB).filter_by(relation_text=original_text).first()
        if existing:
            session.close()
            return existing.relation_canonical or cleaned

        relation_id = str(uuid.uuid4())
        emb_bytes = None
        if embedding:
            emb_bytes = np.array(embedding, dtype=np.float32).tobytes()

        db_rel = RelationOntologyDB(
            relation_id=relation_id,
            relation_text=original_text,
            relation_canonical=cleaned.upper().replace(" ", "_"),
            relation_embedding=emb_bytes,
            usage_count=1,
            first_seen=datetime.utcnow(),
            last_seen=datetime.utcnow()
        )
        session.add(db_rel)
        session.commit()
        session.close()

        if embedding and self.chroma:
            try:
                self.chroma.add_relations(
                    ids=[relation_id],
                    embeddings=[embedding],
                    metadatas=[{
                        "relation_text": original_text,
                        "relation_canonical": cleaned.upper().replace(" ", "_"),
                        "usage_count": 1
                    }]
                )
            except Exception as e:
                log.warning("relation_chroma_add_failed", error=str(e))

        log.info("new_relation_created", text=original_text, canonical=db_rel.relation_canonical)
        return db_rel.relation_canonical

    def _increment_usage(self, relation_id: str):
        session = get_session()
        rel = session.query(RelationOntologyDB).filter_by(relation_id=relation_id).first()
        if rel:
            rel.usage_count += 1
            rel.last_seen = datetime.utcnow()
            session.commit()
        session.close()

    def cluster_relations(self) -> dict:
        try:
            import hdbscan
            data = self.chroma.get_all_relation_embeddings()
            if not data or not data.get("embeddings") or len(data["embeddings"]) < 5:
                return {}

            embeddings = np.array(data["embeddings"])
            ids = data["ids"]

            clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric="euclidean")
            labels = clusterer.fit_predict(embeddings)

            clusters = {}
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(ids[i])

            # Update canonical forms based on clusters
            session = get_session()
            for cluster_id, relation_ids in clusters.items():
                # Use most frequent as canonical
                rels = session.query(RelationOntologyDB).filter(
                    RelationOntologyDB.relation_id.in_(relation_ids)
                ).order_by(RelationOntologyDB.usage_count.desc()).all()

                if rels:
                    canonical = rels[0].relation_canonical
                    for rel in rels[1:]:
                        rel.relation_canonical = canonical
                        rel.cluster_id = int(cluster_id)
                    rels[0].cluster_id = int(cluster_id)

            session.commit()
            session.close()

            return clusters
        except Exception as e:
            log.warning("relation_clustering_failed", error=str(e))
            return {}

    def get_relation_taxonomy(self) -> List[dict]:
        session = get_session()
        relations = session.query(RelationOntologyDB).all()

        clusters = {}
        for rel in relations:
            cid = rel.cluster_id if rel.cluster_id is not None else -1
            if cid not in clusters:
                clusters[cid] = []
            clusters[cid].append({
                "relation_text": rel.relation_text,
                "relation_canonical": rel.relation_canonical,
                "usage_count": rel.usage_count
            })

        result = []
        for cid, items in clusters.items():
            result.append({
                "cluster_id": cid,
                "relations": items,
                "canonical": items[0]["relation_canonical"] if items else ""
            })

        session.close()
        return result
