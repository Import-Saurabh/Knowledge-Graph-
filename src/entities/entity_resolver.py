import json
import re
import uuid
import numpy as np
from datetime import datetime
from typing import List, Optional
from rapidfuzz import fuzz
from src.models.entity import EntityMention, CanonicalEntity
from src.utils.db import get_session, CanonicalEntityDB, EntityAliasDB, MentionDB
from src.utils.logger import get_logger
from src.utils.config import settings

log = get_logger(__name__)

SIMILARITY_THRESHOLD = 0.88
FUZZY_THRESHOLD = 85

class EntityResolver:
    def __init__(self, chroma_manager, embedding_generator):
        self.chroma = chroma_manager
        self.embedder = embedding_generator
        self._cache = {}

    def _normalize(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"^(President|Dr\.|Mr\.|Mrs\.|Ms\.|CEO|Chairman|Secretary|Minister|Prime Minister|)", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"[.,;:!?]$", "", text)
        return text.lower()

    def _fuzzy_search_aliases(self, normalized: str, top_k: int = 5) -> List[dict]:
        session = get_session()
        aliases = session.query(EntityAliasDB).all()
        candidates = []
        for alias in aliases:
            score = fuzz.token_sort_ratio(normalized, alias.alias_text.lower())
            if score > FUZZY_THRESHOLD:
                candidates.append({
                    "alias": alias,
                    "score": score / 100.0,
                    "canonical_id": alias.canonical_id
                })
        candidates.sort(key=lambda x: x["score"], reverse=True)
        session.close()
        return candidates[:top_k]

    def _get_entity_by_id(self, canonical_id: str) -> Optional[CanonicalEntity]:
        session = get_session()
        db = session.query(CanonicalEntityDB).filter_by(canonical_id=canonical_id).first()
        if not db:
            session.close()
            return None
        entity = self._db_to_model(db)
        session.close()
        return entity

    def _db_to_model(self, db) -> CanonicalEntity:
        aliases = []
        try:
            aliases = json.loads(db.aliases) if db.aliases else []
        except:
            pass
        embedding = []
        if db.embedding_vector:
            try:
                embedding = np.frombuffer(db.embedding_vector, dtype=np.float32).tolist()
            except:
                pass
        return CanonicalEntity(
            canonical_id=db.canonical_id,
            canonical_name=db.canonical_name,
            entity_type=db.type_id or "Unknown",
            aliases=aliases,
            mention_count=db.mention_count,
            first_seen=db.first_seen,
            last_seen=db.last_seen,
            embedding_vector=embedding
        )

    def _fuse_scores(self, fuzzy_candidates: List[dict], semantic_matches: dict) -> Optional[dict]:
        # semantic_matches is chroma query result
        if not semantic_matches or not semantic_matches.get("ids"):
            return fuzzy_candidates[0] if fuzzy_candidates else None

        ids = semantic_matches["ids"][0]
        distances = semantic_matches.get("distances", [[]])[0]
        metadatas = semantic_matches.get("metadatas", [[]])[0]

        best = None
        best_score = 0

        # Check fuzzy candidates
        for cand in fuzzy_candidates:
            entity = self._get_entity_by_id(cand["canonical_id"])
            if not entity:
                continue
            score = cand["score"]
            # Find semantic score for same entity
            sem_score = 0
            for i, sid in enumerate(ids):
                if sid == cand["canonical_id"]:
                    # Chroma returns distances (lower is closer), convert to similarity
                    dist = distances[i] if i < len(distances) else 1.0
                    sem_score = 1.0 - min(dist, 1.0)
                    break
            fused = 0.4 * score + 0.6 * sem_score
            if fused > best_score and fused > SIMILARITY_THRESHOLD:
                best_score = fused
                best = {"entity": entity, "score": fused}

        # If no fuzzy match, check pure semantic
        if best is None and ids:
            for i, sid in enumerate(ids[:3]):
                dist = distances[i] if i < len(distances) else 1.0
                sim = 1.0 - min(dist, 1.0)
                if sim > best_score and sim > SIMILARITY_THRESHOLD:
                    entity = self._get_entity_by_id(sid)
                    if entity:
                        best_score = sim
                        best = {"entity": entity, "score": sim}

        return best

    def resolve(self, mention: EntityMention) -> CanonicalEntity:
        normalized = self._normalize(mention.text)

        # Pass 2: Fuzzy search
        fuzzy_candidates = self._fuzzy_search_aliases(normalized, top_k=5)

        # Pass 3: Semantic search
        mention_embedding = None
        try:
            mention_embedding = self.embedder.embed_text(normalized)
        except Exception as e:
            log.warning("embedding_failed", text=normalized, error=str(e))

        semantic_matches = None
        if mention_embedding and self.chroma:
            try:
                semantic_matches = self.chroma.search_entities(
                    mention_embedding,
                    n_results=5,
                    where={"entity_type": mention.entity_type} if mention.entity_type != "Unknown" else None
                )
            except Exception as e:
                log.warning("semantic_search_failed", error=str(e))

        # Pass 4: Fuse scores
        best_match = self._fuse_scores(fuzzy_candidates, semantic_matches)

        if best_match and best_match["score"] > SIMILARITY_THRESHOLD:
            return self._update_entity(best_match["entity"], mention, normalized)
        else:
            return self._create_entity(mention, normalized, mention_embedding)

    def resolve_batch(self, mentions: List[EntityMention]) -> List[CanonicalEntity]:
        return [self.resolve(m) for m in mentions]

    def _update_entity(self, entity: CanonicalEntity, mention: EntityMention, normalized: str) -> CanonicalEntity:
        session = get_session()
        db = session.query(CanonicalEntityDB).filter_by(canonical_id=entity.canonical_id).first()
        if not db:
            session.close()
            return entity

        db.mention_count += 1
        db.last_seen = datetime.utcnow()

        # Update aliases
        aliases = []
        try:
            aliases = json.loads(db.aliases) if db.aliases else []
        except:
            pass
        if mention.text not in aliases and mention.text != db.canonical_name:
            aliases.append(mention.text)
            db.aliases = json.dumps(aliases)

            # Add to alias table
            alias_db = EntityAliasDB(
                alias_id=str(uuid.uuid4()),
                alias_text=mention.text,
                canonical_id=entity.canonical_id,
                match_score=1.0,
                source_article_id=mention.article_id
            )
            session.add(alias_db)

        # Update embedding (running average)
        if mention.entity_type != "Unknown" and db.type_id != mention.entity_type:
            # Update type if more specific
            from src.utils.db import EntityOntologyDB
            type_db = session.query(EntityOntologyDB).filter_by(type_name=mention.entity_type).first()
            if type_db:
                db.type_id = type_db.type_id

        if mention.entity_type != "Unknown":
            type_db = session.query(EntityOntologyDB).filter_by(type_name=mention.entity_type).first()
            if type_db:
                type_db.mention_count += 1
                type_db.last_seen = datetime.utcnow()

        session.commit()
        session.close()

        # Update chroma
        if self.chroma and entity.embedding_vector:
            try:
                self.chroma.update_entities(
                    ids=[entity.canonical_id],
                    embeddings=[entity.embedding_vector],
                    metadatas=[{
                        "canonical_name": entity.canonical_name,
                        "entity_type": entity.entity_type,
                        "mention_count": entity.mention_count + 1
                    }]
                )
            except Exception as e:
                log.warning("chroma_update_failed", error=str(e))

        entity.mention_count += 1
        entity.last_seen = datetime.utcnow()
        if mention.text not in entity.aliases and mention.text != entity.canonical_name:
            entity.aliases.append(mention.text)

        return entity

    def _create_entity(self, mention: EntityMention, normalized: str, mention_embedding: Optional[List[float]]) -> CanonicalEntity:
        session = get_session()

        canonical_id = str(uuid.uuid4())
        canonical_name = mention.text.strip()

        # Determine type
        type_id = None
        type_name = mention.entity_type
        if mention.entity_type != "Unknown":
            from src.utils.db import EntityOntologyDB
            type_db = session.query(EntityOntologyDB).filter_by(type_name=mention.entity_type).first()
            if type_db:
                type_id = type_db.type_id
                type_db.mention_count += 1
                type_db.last_seen = datetime.utcnow()

        embedding_bytes = None
        if mention_embedding:
            embedding_bytes = np.array(mention_embedding, dtype=np.float32).tobytes()

        db_entity = CanonicalEntityDB(
            canonical_id=canonical_id,
            canonical_name=canonical_name,
            type_id=type_id,
            aliases=json.dumps([mention.text]),
            embedding_vector=embedding_bytes,
            mention_count=1,
            first_seen=datetime.utcnow(),
            last_seen=datetime.utcnow()
        )
        session.add(db_entity)

        alias_db = EntityAliasDB(
            alias_id=str(uuid.uuid4()),
            alias_text=mention.text,
            canonical_id=canonical_id,
            match_score=1.0,
            source_article_id=mention.article_id
        )
        session.add(alias_db)

        mention_db = MentionDB(
            mention_id=str(uuid.uuid4()),
            text=mention.text,
            entity_type=mention.entity_type,
            confidence=mention.confidence,
            article_id=mention.article_id,
            canonical_id=canonical_id,
            span_start=mention.span_start,
            span_end=mention.span_end
        )
        session.add(mention_db)

        session.commit()
        session.close()

        # Add to ChromaDB
        if self.chroma and mention_embedding:
            try:
                self.chroma.add_entities(
                    ids=[canonical_id],
                    embeddings=[mention_embedding],
                    metadatas=[{
                        "canonical_name": canonical_name,
                        "entity_type": type_name,
                        "mention_count": 1
                    }]
                )
            except Exception as e:
                log.warning("chroma_add_failed", error=str(e))

        entity = CanonicalEntity(
            canonical_id=canonical_id,
            canonical_name=canonical_name,
            entity_type=type_name,
            aliases=[mention.text],
            mention_count=1,
            embedding_vector=mention_embedding or []
        )

        log.info("entity_created", canonical_name=canonical_name, type=type_name)
        return entity
