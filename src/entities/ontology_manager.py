import os
import uuid
import json
import numpy as np
from datetime import datetime
from typing import List, Optional, Dict
from src.utils.db import get_session, EntityOntologyDB, CanonicalEntityDB
from src.utils.logger import get_logger
from src.utils.config import settings
import yaml

log = get_logger(__name__)

TYPE_SIMILARITY_THRESHOLD = 0.82

class OntologyManager:
    def __init__(self, chroma_manager=None, embedding_generator=None):
        self.chroma = chroma_manager
        self.embedder = embedding_generator
        self._seed_loaded = False

    def _ensure_seed_loaded(self):
        if self._seed_loaded:
            return
        self._seed_loaded = True
        seed_path = settings.SEED_ONTOLOGY_PATH
        if not seed_path or not os.path.exists(seed_path):
            return
        try:
            with open(seed_path, "r") as f:
                seed = yaml.safe_load(f)
            if seed and "entity_types" in seed:
                for et in seed["entity_types"]:
                    self._create_type_if_missing(et["name"], et.get("parent"))
        except Exception as e:
            log.warning("seed_ontology_load_failed", error=str(e))

    def _create_type_if_missing(self, type_name: str, parent: str = None):
        session = get_session()
        existing = session.query(EntityOntologyDB).filter_by(type_name=type_name).first()
        if not existing:
            type_id = str(uuid.uuid4())
            parent_id = None
            if parent:
                p = session.query(EntityOntologyDB).filter_by(type_name=parent).first()
                if p:
                    parent_id = p.type_id

            embedding = None
            if self.embedder:
                try:
                    emb = self.embedder.embed_text(type_name)
                    embedding = np.array(emb, dtype=np.float32).tobytes()
                except:
                    pass

            db_type = EntityOntologyDB(
                type_id=type_id,
                type_name=type_name,
                type_embedding=embedding,
                parent_type_id=parent_id,
                is_auto_discovered=False,
                is_confirmed=True
            )
            session.add(db_type)
            session.commit()
            log.info("seed_type_created", type_name=type_name)
        session.close()

    def get_active_labels(self, limit: int = 50) -> List[str]:
        self._ensure_seed_loaded()
        session = get_session()
        types = session.query(EntityOntologyDB).order_by(
            EntityOntologyDB.mention_count.desc()
        ).limit(limit).all()
        labels = [t.type_name for t in types]
        session.close()

        # Add generic fallbacks if ontology is empty
        if not labels:
            labels = ["Person", "Organization", "Location", "Event"]
        return labels

    def get_type_by_name(self, type_name: str) -> Optional[dict]:
        session = get_session()
        t = session.query(EntityOntologyDB).filter_by(type_name=type_name).first()
        result = None
        if t:
            result = {
                "type_id": t.type_id,
                "type_name": t.type_name,
                "mention_count": t.mention_count
            }
        session.close()
        return result

    def induce_type(self, entity_name: str, context: str, suggested_type: str = None) -> str:
        self._ensure_seed_loaded()
        session = get_session()

        if suggested_type:
            type_embedding = None
            if self.embedder:
                try:
                    type_embedding = self.embedder.embed_text(suggested_type)
                except:
                    pass
            existing = self._find_similar_type(type_embedding, session) if type_embedding else None
            if existing:
                session.close()
                return existing["type_name"]
            else:
                type_id = str(uuid.uuid4())
                emb_bytes = None
                if type_embedding:
                    emb_bytes = np.array(type_embedding, dtype=np.float32).tobytes()
                db_type = EntityOntologyDB(
                    type_id=type_id,
                    type_name=suggested_type,
                    type_embedding=emb_bytes,
                    mention_count=1,
                    first_seen=datetime.utcnow(),
                    last_seen=datetime.utcnow()
                )
                session.add(db_type)
                session.commit()
                session.close()
                log.info("new_type_induced", type_name=suggested_type, entity=entity_name)
                return suggested_type

        # Fallback: use context to find nearest type
        if self.embedder:
            try:
                context_embedding = self.embedder.embed_text(context)
                nearest = self._find_similar_type(context_embedding, session)
                if nearest and nearest.get("score", 0) > TYPE_SIMILARITY_THRESHOLD:
                    session.close()
                    return nearest["type_name"]
            except Exception as e:
                log.warning("type_induction_context_failed", error=str(e))

        session.close()
        return "Unknown"

    def _find_similar_type(self, embedding: List[float], session) -> Optional[dict]:
        if not embedding:
            return None
        types = session.query(EntityOntologyDB).all()
        if not types:
            return None

        best = None
        best_score = 0
        query_vec = np.array(embedding)

        for t in types:
            if t.type_embedding is None:
                continue
            type_vec = np.frombuffer(t.type_embedding, dtype=np.float32)
            if len(type_vec) != len(query_vec):
                continue
            score = float(np.dot(query_vec, type_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(type_vec)))
            if score > best_score:
                best_score = score
                best = {"type_id": t.type_id, "type_name": t.type_name, "score": score}

        return best

    def merge_types(self, type_id_1: str, type_id_2: str) -> str:
        session = get_session()
        t1 = session.query(EntityOntologyDB).filter_by(type_id=type_id_1).first()
        t2 = session.query(EntityOntologyDB).filter_by(type_id=type_id_2).first()
        if not t1 or not t2:
            session.close()
            return ""

        # Merge t2 into t1
        session.query(CanonicalEntityDB).filter_by(type_id=type_id_2).update({
            CanonicalEntityDB.type_id: type_id_1
        }, synchronize_session=False)

        t1.mention_count += t2.mention_count
        t1.last_seen = datetime.utcnow()
        session.delete(t2)
        session.commit()
        session.close()
        log.info("types_merged", kept=t1.type_name, removed=t2.type_name)
        return t1.type_name

    def get_ontology_report(self) -> dict:
        session = get_session()
        types = session.query(EntityOntologyDB).all()
        type_count = len(types)
        top_types = sorted(
            [{"name": t.type_name, "count": t.mention_count, "confirmed": t.is_confirmed} for t in types],
            key=lambda x: x["count"],
            reverse=True
        )[:20]
        unconfirmed = [t.type_name for t in types if not t.is_confirmed]
        recently_discovered = [t.type_name for t in types if t.is_auto_discovered][:10]
        session.close()

        return {
            "type_count": type_count,
            "top_types": top_types,
            "unconfirmed_types": unconfirmed,
            "recently_discovered": recently_discovered
        }
