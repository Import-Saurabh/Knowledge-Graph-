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

_TITLE_RE = re.compile(
    r"^(?:former\s+|current\s+)?(?:president|prime minister|minister|secretary|"
    r"dr|mr|mrs|ms|ceo|chairman|leader|commander)\.?\s+",
    re.IGNORECASE,
)
_PUNCT_RE = re.compile(r"[^\w\s&-]")
_SPACE_RE = re.compile(r"\s+")

_KNOWN_ALIAS_KEYS = {
    "donald j trump": "donald trump",
    "donald john trump": "donald trump",
    "president donald trump": "donald trump",
    "president donald j trump": "donald trump",
    "president trump": "donald trump",
    "trump": "donald trump",
    "trump trump": "donald trump",
    "u s": "united states",
    "us": "united states",
    "usa": "united states",
    "u s a": "united states",
    "america": "united states",
    "irgc": "islamic revolutionary guard corps",
    "irgcs": "islamic revolutionary guard corps",
    "islamic revolutionary guards corps": "islamic revolutionary guard corps",
    "islamic revolution guard corps": "islamic revolutionary guard corps",
    "islamic revolutionary guards": "islamic revolutionary guard corps",
}

_GENERIC_ROLE_KEYS = {
    "president", "prime minister", "minister", "secretary", "leader",
    "commander", "official", "spokesperson", "government", "military",
}

class EntityResolver:
    def __init__(self, chroma_manager, embedding_generator, wikidata_linker=None):
        self.chroma = chroma_manager
        self.embedder = embedding_generator
        self.wikidata_linker = wikidata_linker  # store for potential future use
        self._cache = {}
        if wikidata_linker:
            log.info("wikidata_linker_attached_to_resolver")

    @staticmethod
    def canonical_key(text: str) -> str:
        """Stable surface key used for alias matching and graph endpoints."""
        value = (text or "").strip()
        value = value.replace("’", "'").replace("`", "'")
        value = re.sub(r"'s\b", "", value, flags=re.IGNORECASE)
        value = _TITLE_RE.sub("", value)
        value = _PUNCT_RE.sub(" ", value)
        value = _SPACE_RE.sub(" ", value).strip().lower()

        # Collapse accidental repeated tokens: "Trump Trump" -> "trump".
        tokens = value.split()
        collapsed = []
        for token in tokens:
            if not collapsed or collapsed[-1] != token:
                collapsed.append(token)
        value = " ".join(collapsed)

        # Drop middle initials in person names: "donald j trump" -> "donald trump".
        value = re.sub(r"\b([a-z]+)\s+[a-z]\s+([a-z]+)\b", r"\1 \2", value)
        return _KNOWN_ALIAS_KEYS.get(value, value)

    @staticmethod
    def is_generic_role(text: str) -> bool:
        return EntityResolver.canonical_key(text) in _GENERIC_ROLE_KEYS

    def _normalize(self, text: str) -> str:
        return self.canonical_key(text)

    def _type_name_for_id(self, session, type_id: str | None) -> str:
        if not type_id:
            return "Unknown"
        try:
            from src.utils.db import EntityOntologyDB
            row = session.query(EntityOntologyDB).filter_by(type_id=type_id).first()
            return row.type_name if row else type_id
        except Exception:
            return type_id

    # ------------------------------------------------------------------
    # DUPLICATE-RESOLUTION FIX
    # ------------------------------------------------------------------
    # The fuzzy (token_sort_ratio) + semantic fusion pipeline below needs a
    # candidate alias to already be "close" character-for-character or
    # embedding-wise. Two very common real-world coreference patterns never
    # clear that bar, so they always minted brand-new duplicate entities:
    #
    #   1. Acronyms vs. full names — "US" vs "United States",
    #      "IRGC" vs "Islamic Revolutionary Guards Corps".
    #      fuzz.token_sort_ratio("us", "united states") ≈ 18 — nowhere near
    #      FUZZY_THRESHOLD=85, so no candidate is ever even considered.
    #
    #   2. Partial person names — "Trump" vs "Donald Trump".
    #      fuzz.token_sort_ratio("trump", "donald trump") ≈ 59 — also below
    #      threshold, so "President Trump" mentions never matched the
    #      "Donald Trump" entity created earlier.
    #
    # These two checks run BEFORE the fuzzy/semantic pipeline as a cheap,
    # high-precision exact pass. They look at the canonical_name AND all
    # known aliases of every existing entity.

    @staticmethod
    def _is_acronym_match(raw_text: str, candidate_name: str) -> bool:
        """True if raw_text is an ALL-CAPS acronym of candidate_name's words.
        e.g. "IRGC" <-> "Islamic Revolutionary Guards Corps"."""
        token = raw_text.strip()
        if not (2 <= len(token) <= 6 and token.isalpha() and token.isupper()):
            return False
        words = re.findall(r"[A-Za-z]+", candidate_name)
        if len(words) < 2:
            return False
        initials = "".join(w[0] for w in words).upper()
        return initials == token.upper()

    @staticmethod
    def _is_partial_name_match(normalized: str, candidate_name: str,
                               entity_type: str = "") -> bool:
        """True if `normalized` is a strict, whole-word subset of a longer
        candidate name — e.g. "trump" <-> "donald trump". Restricted to
        person-like mentions to avoid false merges between unrelated
        multi-word organisations/places that happen to share one word."""
        if entity_type and entity_type.lower() != "person":
            return False
        cand_tokens = re.sub(r"[.,;:!?]$", "", candidate_name.strip().lower()).split()
        mention_tokens = normalized.split()
        if not mention_tokens or len(cand_tokens) < 2:
            return False
        if len(mention_tokens) >= len(cand_tokens):
            return False
        if not all(len(t) >= 3 for t in mention_tokens):
            return False
        return all(t in cand_tokens for t in mention_tokens)

    def _exact_alias_match(self, raw_text: str, normalized: str,
                           entity_type: str = "") -> Optional[CanonicalEntity]:
        """Pass 1: cheap, high-precision acronym / partial-name lookup
        against every existing canonical entity (name + aliases)."""
        session = get_session()
        try:
            rows = session.query(CanonicalEntityDB).all()
            for db in rows:
                names = [db.canonical_name]
                try:
                    names += json.loads(db.aliases) if db.aliases else []
                except Exception:
                    pass
                for name in names:
                    if not name:
                        continue
                    if self.canonical_key(name) == normalized:
                        return self._db_to_model(db)
                    if self._is_acronym_match(raw_text, name) or \
                       self._is_partial_name_match(normalized, name, entity_type):
                        return self._db_to_model(db)
        finally:
            session.close()
        return None

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
        session = get_session()
        try:
            type_name = self._type_name_for_id(session, db.type_id)
        finally:
            session.close()
        embedding = []
        if db.embedding_vector:
            try:
                embedding = np.frombuffer(db.embedding_vector, dtype=np.float32).tolist()
            except:
                pass
        return CanonicalEntity(
            canonical_id=db.canonical_id,
            canonical_name=db.canonical_name,
            entity_type=type_name,
            aliases=aliases,
            mention_count=db.mention_count,
            first_seen=db.first_seen,
            last_seen=db.last_seen,
            embedding_vector=embedding
        )

    def _fuse_scores(self, fuzzy_candidates: List[dict], semantic_matches: dict) -> Optional[dict]:
        if not semantic_matches or not semantic_matches.get("ids"):
            return fuzzy_candidates[0] if fuzzy_candidates else None

        ids = semantic_matches["ids"][0]
        distances = semantic_matches.get("distances", [[]])[0]
        metadatas = semantic_matches.get("metadatas", [[]])[0]

        best = None
        best_score = 0

        for cand in fuzzy_candidates:
            entity = self._get_entity_by_id(cand["canonical_id"])
            if not entity:
                continue
            score = cand["score"]
            sem_score = 0
            for i, sid in enumerate(ids):
                if sid == cand["canonical_id"]:
                    dist = distances[i] if i < len(distances) else 1.0
                    sem_score = 1.0 - min(dist, 1.0)
                    break
            fused = 0.4 * score + 0.6 * sem_score
            if fused > best_score and fused > SIMILARITY_THRESHOLD:
                best_score = fused
                best = {"entity": entity, "score": fused}

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
        if self.is_generic_role(mention.text):
            raise ValueError(f"generic role mention skipped: {mention.text}")

        # Pass 1: acronym / partial-name exact match (see block above) —
        # cheap and high-precision, so it short-circuits the rest.
        exact_match = self._exact_alias_match(
            mention.text, normalized, getattr(mention, "entity_type", "") or ""
        )
        if exact_match:
            log.info(
                "entity_resolved_exact_match",
                mention=mention.text, canonical=exact_match.canonical_name,
            )
            return self._update_entity(exact_match, mention, normalized)

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
        try:
            from tqdm import tqdm
            iterator = tqdm(mentions, desc="entity_resolution", unit="mention")
        except ImportError:
            iterator = mentions

        results = []
        resolved = 0
        created = 0
        _prev_canonical_count = 0

        for i, mention in enumerate(iterator):
            try:
                entity = self.resolve(mention)
                results.append(entity)
                # Track new vs existing — if canonical_id is new this iteration
                resolved += 1
            except Exception as e:
                log.warning("resolution_failed", mention=mention.text, error=str(e))
                results.append(None)

            # Every 100 mentions log a structured progress line
            if (i + 1) % 100 == 0:
                log.info(
                    "entity_resolution_progress",
                    done=i + 1,
                    total=len(mentions),
                    pct=round(100 * (i + 1) / len(mentions), 1),
                )

        return results

    def _update_entity(self, entity: CanonicalEntity, mention: EntityMention, normalized: str) -> CanonicalEntity:
        session = get_session()
        db = session.query(CanonicalEntityDB).filter_by(canonical_id=entity.canonical_id).first()
        if not db:
            session.close()
            return entity

        db.mention_count += 1
        db.last_seen = datetime.utcnow()

        aliases = []
        try:
            aliases = json.loads(db.aliases) if db.aliases else []
        except:
            pass
        if mention.text not in aliases and mention.text != db.canonical_name:
            aliases.append(mention.text)
            db.aliases = json.dumps(aliases)

            alias_db = EntityAliasDB(
                alias_id=str(uuid.uuid4()),
                alias_text=mention.text,
                canonical_id=entity.canonical_id,
                match_score=1.0,
                source_article_id=mention.article_id
            )
            session.add(alias_db)

        if mention.entity_type != "Unknown" and db.type_id != mention.entity_type:
            from src.utils.db import EntityOntologyDB
            type_db = session.query(EntityOntologyDB).filter_by(type_name=mention.entity_type).first()
            if type_db:
                db.type_id = type_db.type_id

        if mention.entity_type != "Unknown":
            from src.utils.db import EntityOntologyDB
            type_db = session.query(EntityOntologyDB).filter_by(type_name=mention.entity_type).first()
            if type_db:
                type_db.mention_count += 1
                type_db.last_seen = datetime.utcnow()

        session.commit()
        session.close()

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
        canonical_name = mention.text.strip()

        # --- UPSERT GUARD ---
        rows = session.query(CanonicalEntityDB).all()
        for existing_db in rows:
            names = [existing_db.canonical_name]
            try:
                names += json.loads(existing_db.aliases) if existing_db.aliases else []
            except Exception:
                pass
            if any(self.canonical_key(name) == normalized for name in names):
                session.close()
                existing_entity = self._db_to_model(existing_db)
                log.info("entity_exists_reusing", canonical_name=existing_db.canonical_name)
                return self._update_entity(existing_entity, mention, normalized)

        canonical_id = str(uuid.uuid4())

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

        try:
            session.commit()
        except Exception as e:
            session.rollback()
            session.close()
            log.warning("entity_create_race_retry", canonical_name=canonical_name, error=str(e))
            session2 = get_session()
            existing_db = session2.query(CanonicalEntityDB).filter_by(canonical_name=canonical_name).first()
            session2.close()
            if existing_db:
                existing_entity = self._db_to_model(existing_db)
                return self._update_entity(existing_entity, mention, normalized)
            raise

        session.close()

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

    # ------------------------------------------------------------------
    # One-time cleanup: merge duplicates already sitting in the DB
    # ------------------------------------------------------------------
    # Fixing `resolve()` only prevents *new* duplicates going forward — it
    # can't retroactively fix "US" / "United States" / "IRGC" / "Donald
    # Trump" pairs that were already created as separate canonical rows by
    # earlier runs. Run this once (see main.py --merge-duplicate-entities)
    # before re-exporting the graph.

    def find_and_merge_duplicates(self) -> int:
        """
        Scan every CanonicalEntityDB row for acronym/partial-name duplicates
        using the same predicates `resolve()` uses, merge each duplicate
        pair (re-pointing aliases/mentions, summing mention_count, unioning
        alias lists), and delete the loser row. Returns the number of pairs
        merged. Safe to re-run — it's idempotent once no duplicates remain.
        """
        session = get_session()
        try:
            from src.utils.db import EntityOntologyDB
            type_rows = session.query(EntityOntologyDB).all()
            type_name_by_id = {t.type_id: t.type_name for t in type_rows}

            rows = session.query(CanonicalEntityDB).all()
            # Survivor is whichever row has more mentions (ties broken by
            # earlier first_seen) so the more "established" name/aliases win.
            rows.sort(key=lambda r: (-(r.mention_count or 0), r.first_seen or datetime.max))

            merged_away: set = set()
            merge_count = 0

            key_owner: dict[str, CanonicalEntityDB] = {}
            acronym_owner: dict[str, CanonicalEntityDB] = {}

            def names_for(row):
                names = [row.canonical_name]
                try:
                    names += json.loads(row.aliases) if row.aliases else []
                except Exception:
                    pass
                return [n for n in names if n]

            def acronym_for(name: str) -> str:
                words = re.findall(r"[A-Za-z]+", name or "")
                if len(words) < 2:
                    return ""
                return "".join(w[0] for w in words).upper()

            for row in rows:
                if row.canonical_id in merged_away:
                    continue
                row_type = type_name_by_id.get(row.type_id, "")
                keys = {
                    self.canonical_key(name)
                    for name in names_for(row)
                    if name and not self.is_generic_role(name)
                }
                keys = {k for k in keys if k and len(k) > 1}

                survivor = None
                for key in keys:
                    if key in key_owner and key_owner[key].canonical_id != row.canonical_id:
                        survivor = key_owner[key]
                        break

                if survivor is None:
                    for name in names_for(row):
                        token = name.strip()
                        if 2 <= len(token) <= 8 and token.isalpha() and token.isupper():
                            survivor = acronym_owner.get(token.upper())
                            if survivor is not None:
                                break

                if survivor is not None:
                    survivor_type = type_name_by_id.get(survivor.type_id, "")
                    compatible = (
                        not row_type or not survivor_type or row_type == survivor_type
                        or {row_type, survivor_type} <= {"person", "job title or role"}
                    )
                    if compatible:
                        self._merge_rows(session, survivor=survivor, loser=row)
                        merged_away.add(row.canonical_id)
                        merge_count += 1
                        log.info(
                            "duplicate_entities_merged",
                            survivor=survivor.canonical_name,
                            loser=row.canonical_name,
                        )
                        continue

                for key in keys:
                    key_owner.setdefault(key, row)
                for name in names_for(row):
                    acro = acronym_for(name)
                    if acro:
                        acronym_owner.setdefault(acro, row)

            session.commit()
        except Exception as e:
            session.rollback()
            log.error("merge_duplicates_failed", error=str(e))
            raise
        finally:
            session.close()

        return merge_count

    def _merge_rows(self, session, survivor: CanonicalEntityDB, loser: CanonicalEntityDB) -> None:
        """Re-point loser's aliases/mentions onto survivor, union alias
        lists, sum mention counts, widen the first/last-seen range, then
        delete the loser row. Caller owns the session/commit."""
        try:
            survivor_aliases = json.loads(survivor.aliases) if survivor.aliases else []
        except Exception:
            survivor_aliases = []
        try:
            loser_aliases = json.loads(loser.aliases) if loser.aliases else []
        except Exception:
            loser_aliases = []

        merged_aliases = list(dict.fromkeys(
            survivor_aliases + [loser.canonical_name] + loser_aliases
        ))
        survivor.aliases = json.dumps(merged_aliases)
        survivor.mention_count = (survivor.mention_count or 0) + (loser.mention_count or 0)
        if loser.first_seen and (not survivor.first_seen or loser.first_seen < survivor.first_seen):
            survivor.first_seen = loser.first_seen
        if loser.last_seen and (not survivor.last_seen or loser.last_seen > survivor.last_seen):
            survivor.last_seen = loser.last_seen

        session.query(EntityAliasDB).filter_by(canonical_id=loser.canonical_id).update(
            {EntityAliasDB.canonical_id: survivor.canonical_id}, synchronize_session=False,
        )
        session.query(MentionDB).filter_by(canonical_id=loser.canonical_id).update(
            {MentionDB.canonical_id: survivor.canonical_id}, synchronize_session=False,
        )
        session.delete(loser)

        if self.chroma:
            try:
                self.chroma.delete_entities(ids=[loser.canonical_id])
            except Exception as e:
                log.warning("chroma_delete_on_merge_failed",
                           loser_id=loser.canonical_id, error=str(e))
