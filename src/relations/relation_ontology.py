"""
src/relations/relation_ontology.py

Normalises GLiREL relation phrases to canonical labels.

No Wikidata. No LLM. Pure local semantic normalization.

Three-stage pipeline
--------------------
Stage 1 — Direct canonical check
    If the raw GLiREL phrase is already a known UPPER_SNAKE canonical, accept it.

Stage 2 — Vocab centroid cosine similarity (lazy, auto-initialised)
    Each canonical label is embedded as the centroid of
    [canonical_name] + [alias phrases]. Input phrase is compared to all
    centroids; if cos ≥ VOCAB_SIM_THRESHOLD it maps to that canonical.

Stage 3 — ChromaDB emergent-cluster nearest-neighbour
    The running ChromaDB collection contains every relation seen so far.
    If a phrase matches an existing stored relation at cos ≥ EMERGENT_SIM_THRESHOLD
    it joins that cluster and reuses its canonical.

Stage 4 — New emergent canonical
    No match → derive a new canonical from the raw phrase:
    e.g. "imposed travel ban on" → "IMPOSED_TRAVEL_BAN_ON".
    Stored in SQLite + ChromaDB and becomes a first-class citizen.

Public helpers
--------------
get_relation_taxonomy()   — live taxonomy sorted by usage from SQLite
cluster_relations()       — {canonical: [raw_phrases]} dict
get_canonical_labels()    — seed labels + any emergent labels in DB
"""

import re
import threading
import uuid
import numpy as np
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from src.utils.db import get_session, RelationOntologyDB
from src.utils.logger import get_logger
from src.utils.config import settings

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
VOCAB_SIM_THRESHOLD    = 0.72   # centroid cosine → map to seed canonical
EMERGENT_SIM_THRESHOLD = 0.82   # ChromaDB cosine → join existing emergent cluster

# ---------------------------------------------------------------------------
# Seed vocabulary
# ---------------------------------------------------------------------------
# Each canonical maps to representative alias phrases used to compute the
# centroid embedding used for Stage 2.  Not used for substring matching.
CONTROLLED_VOCABULARY: Dict[str, List[str]] = {
    "MILITARY_ATTACK": [
        "launched airstrike against", "bombed", "shelled", "struck",
        "fired missiles at", "attacked", "conducted military operation against",
        "deployed forces against",
    ],
    "MILITARY_OCCUPATION": [
        "occupied territory", "seized territory", "captured city",
        "took control of region", "invaded", "moved troops into",
    ],
    "MILITARY_WITHDRAWAL": [
        "withdrew troops from", "retreated from", "pulled back from",
        "evacuated forces from", "ended military presence in",
    ],
    "MILITARY_SUPPORT": [
        "provided military aid to", "supplied weapons to", "armed forces of",
        "sent troops to support", "provided defence assistance to",
    ],
    "CEASEFIRE": [
        "agreed ceasefire with", "declared ceasefire", "signed ceasefire",
        "halted fighting with", "paused hostilities with",
    ],
    "DIPLOMATIC_MEETING": [
        "met with", "held talks with", "convened summit with",
        "held bilateral meeting with", "visited for diplomatic talks",
    ],
    "DIPLOMATIC_AGREEMENT": [
        "signed agreement with", "signed deal with", "reached accord with",
        "concluded treaty with", "finalised pact with",
    ],
    "DIPLOMATIC_RECOGNITION": [
        "recognised government of", "established diplomatic relations with",
        "normalised relations with",
    ],
    "DIPLOMATIC_EXPULSION": [
        "expelled ambassador of", "broke diplomatic relations with",
        "recalled ambassador from", "downgraded relations with",
    ],
    "DIPLOMATIC_STATEMENT": [
        "condemned actions of", "criticised government of", "denounced",
        "issued statement against", "called on government of",
    ],
    "PEACE_NEGOTIATION": [
        "negotiated peace with", "mediated conflict between",
        "brokered peace deal between", "facilitated peace negotiations",
        "proposed peace plan for",
    ],
    "SANCTIONS_IMPOSED": [
        "imposed sanctions on", "sanctioned government of",
        "placed embargo on", "froze assets of", "banned trade with",
    ],
    "SANCTIONS_LIFTED": [
        "lifted sanctions on", "removed sanctions from",
        "eased restrictions on", "ended embargo against",
    ],
    "TRADE_AGREEMENT": [
        "signed trade deal with", "established trade agreement with",
        "concluded free trade agreement with", "opened trade corridor with",
    ],
    "ECONOMIC_AID": [
        "provided economic aid to", "granted loan to",
        "donated funds to", "pledged financial assistance to",
    ],
    "INVESTMENT": [
        "invested in", "acquired stake in", "financed project in",
        "funded development in",
    ],
    "LEADER_OF": [
        "leads", "heads organisation", "commands military of",
        "governs", "is president of", "is prime minister of",
    ],
    "APPOINTED": [
        "appointed as", "elected as", "nominated for",
        "sworn in as", "confirmed as",
    ],
    "RESIGNED": [
        "resigned from", "stepped down from", "ousted from office",
        "removed from position", "forced to resign",
    ],
    "ALLY_OF": [
        "allied with", "partnered with", "is coalition partner of",
        "backs government of", "supports position of",
    ],
    "OPPOSES": [
        "opposes policy of", "protests against", "challenges authority of",
        "is rival of", "stands against",
    ],
    "ACCUSED_OF": [
        "accused of wrongdoing", "charged with crimes", "indicted for",
        "alleged to have committed", "suspected of",
    ],
    "CONVICTED_OF": [
        "convicted of crime", "found guilty of", "sentenced for",
    ],
    "ARRESTED": [
        "arrested by authorities", "detained by", "taken into custody by",
    ],
    "MEMBER_OF": [
        "is member of organisation", "belongs to group",
        "joined alliance", "is part of coalition",
    ],
    "FOUNDED": [
        "founded organisation", "established group", "created institution",
        "set up agency",
    ],
    "HEADQUARTERED_IN": [
        "based in city", "headquartered in country", "operates from",
    ],
    "HUMANITARIAN_AID": [
        "provided humanitarian aid to", "delivered aid to",
        "sent relief supplies to", "funded humanitarian operation in",
    ],
    "REFUGEE_MOVEMENT": [
        "refugees fled to", "population displaced to",
        "civilians evacuated to", "people sought refuge in",
    ],
    "NUCLEAR_ACTIVITY": [
        "conducted nuclear test", "enriched uranium to",
        "developed nuclear weapons", "restarted nuclear programme",
        "violated nuclear agreement",
    ],
    "RELATED_TO": [
        "is related to", "is associated with", "is linked to",
        "has connection with",
    ],
}

# Fast set for Stage 1 direct-canonical check
SEED_SET: set = set(CONTROLLED_VOCABULARY.keys())

DIRECT_RELATION_ALIASES: Dict[str, str] = {
    "attacked": "MILITARY_ATTACK",
    "attack": "MILITARY_ATTACK",
    "targeted": "MILITARY_ATTACK",
    "struck": "MILITARY_ATTACK",
    "bombed": "MILITARY_ATTACK",
    "invaded": "MILITARY_OCCUPATION",
    "occupied": "MILITARY_OCCUPATION",
    "controlled by": "MILITARY_OCCUPATION",
    "allied with": "ALLY_OF",
    "ally of": "ALLY_OF",
    "supported by": "MILITARY_SUPPORT",
    "supports": "MILITARY_SUPPORT",
    "supplied weapons to": "MILITARY_SUPPORT",
    "funded by": "ECONOMIC_AID",
    "sanctioned by": "SANCTIONS_IMPOSED",
    "sanctioned": "SANCTIONS_IMPOSED",
    "accused": "ACCUSED_OF",
    "accused of": "ACCUSED_OF",
    "negotiated with": "PEACE_NEGOTIATION",
    "signed agreement with": "DIPLOMATIC_AGREEMENT",
    "leader of": "LEADER_OF",
    "head of state of": "LEADER_OF",
    "member of": "MEMBER_OF",
    "part of": "MEMBER_OF",
    "headquartered in": "HEADQUARTERED_IN",
    "based in": "HEADQUARTERED_IN",
    "located in": "HEADQUARTERED_IN",
    "capital of": "CAPITAL_OF",
    "occurred in": "OCCURRED_IN",
    "citizen of": "CITIZEN_OF",
    "participant in": "PARTICIPANT_IN",
    "killed": "MILITARY_ATTACK",
    "at war with": "OPPOSES",
    "threatened": "OPPOSES",
    "responded to": "DIPLOMATIC_STATEMENT",
    "spokesperson of": "MEMBER_OF",
    "founded by": "FOUNDED",
}


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class RelationOntologyManager:
    """
    Normalises GLiREL relation phrases to canonical labels and tracks a live
    taxonomy in SQLite + ChromaDB.

    Parameters
    ----------
    chroma_manager      : ChromaManager instance (for Stage 3 cluster lookup)
    embedding_generator : EmbeddingGenerator instance (BGE embeddings)
    """

    def __init__(
        self,
        chroma_manager,
        embedding_generator,
    ):
        self.chroma   = chroma_manager
        self.embedder = embedding_generator

        # canonical → normalised centroid vector (lazy-populated on first use)
        self._vocab_embeddings: Dict[str, np.ndarray] = {}
        self._vocab_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Vocab embedding helpers
    # ------------------------------------------------------------------

    def precompute_vocab_embeddings(self) -> None:
        """
        Embed each canonical as the centroid of its label + alias phrases.
        Call this explicitly at pipeline startup to avoid a latency spike on
        the first normalize_relation() call during a run.
        """
        with self._vocab_lock:
            for canonical, aliases in CONTROLLED_VOCABULARY.items():
                if canonical in self._vocab_embeddings:
                    continue
                texts = [canonical.lower().replace("_", " ")] + aliases
                vecs  = []
                for t in texts:
                    try:
                        v = np.array(self.embedder.embed_text(t), dtype=np.float32)
                        if np.linalg.norm(v) > 0:
                            vecs.append(v)
                    except Exception as e:
                        log.warning("alias_embed_failed", text=t, error=str(e))
                if vecs:
                    centroid = np.mean(vecs, axis=0)
                    self._vocab_embeddings[canonical] = (
                        centroid / (np.linalg.norm(centroid) + 1e-9)
                    )
        log.info("vocab_embeddings_ready", count=len(self._vocab_embeddings))

    def _ensure_vocab_embeddings(self) -> None:
        if not self._vocab_embeddings:
            self.precompute_vocab_embeddings()

    @staticmethod
    def is_seed_canonical(canonical: str) -> bool:
        return canonical in SEED_SET

    # ------------------------------------------------------------------
    # Public: normalisation
    # ------------------------------------------------------------------

    def normalize_relation(self, relation_text: str) -> str:
        """
        Normalise a raw GLiREL relation phrase to a canonical label.

        Stage 1 — direct canonical check (O(1) set lookup)
        Stage 2 — vocab centroid cosine similarity
        Stage 3 — ChromaDB emergent nearest-neighbour
        Stage 4 — new emergent canonical derived from the phrase
        """
        if not relation_text or not relation_text.strip():
            return "RELATED_TO"

        # ── Stage 1: direct canonical ──────────────────────────────────────
        upper = relation_text.strip().upper()
        if upper in SEED_SET:
            log.debug("relation_normalized_direct", raw=relation_text, canonical=upper)
            return upper

        cleaned = self._preprocess(relation_text)
        alias_hit = DIRECT_RELATION_ALIASES.get(cleaned)
        if alias_hit:
            return self._persist_relation(
                cleaned, None, relation_text, canonical=alias_hit
            )

        # ── Embed ──────────────────────────────────────────────────────────
        embedding_l: Optional[list]        = None
        embedding_np: Optional[np.ndarray] = None
        try:
            embedding_l  = self.embedder.embed_text(cleaned)
            embedding_np = np.array(embedding_l, dtype=np.float32)
        except Exception as e:
            log.warning("relation_embedding_failed", text=cleaned, error=str(e))

        # ── Stage 2: vocab centroid cosine ────────────────────────────────
        if embedding_np is not None:
            self._ensure_vocab_embeddings()
            vocab_canonical, vocab_sim = self._vocab_cosine_match(embedding_np)
            if vocab_canonical and vocab_sim >= VOCAB_SIM_THRESHOLD:
                log.debug(
                    "relation_normalized_vocab",
                    raw=relation_text, canonical=vocab_canonical,
                    sim=round(vocab_sim, 3),
                )
                return self._persist_relation(
                    cleaned, embedding_l, relation_text, canonical=vocab_canonical
                )

        # ── Stage 3: ChromaDB emergent nearest-neighbour ──────────────────
        if embedding_l is not None and self.chroma:
            emergent_canonical, emergent_sim = self._chroma_cluster_match(embedding_l)
            if emergent_canonical and emergent_sim >= EMERGENT_SIM_THRESHOLD:
                log.debug(
                    "relation_normalized_emergent",
                    raw=relation_text, canonical=emergent_canonical,
                    sim=round(emergent_sim, 3),
                )
                return self._persist_relation(
                    cleaned, embedding_l, relation_text, canonical=emergent_canonical
                )

        # ── Stage 4: new emergent canonical ──────────────────────────────
        emergent = self._make_emergent_canonical(cleaned)
        log.info("new_emergent_relation", raw=relation_text, canonical=emergent)
        return self._persist_relation(
            cleaned, embedding_l, relation_text, canonical=emergent
        )

    # ------------------------------------------------------------------
    # Public: taxonomy helpers
    # ------------------------------------------------------------------

    def get_relation_taxonomy(self) -> List[dict]:
        """
        Return a live taxonomy built from what actually appeared in the data.
        Each entry: {canonical, is_in_vocab, usage_count, phrase_count,
                     phrases (top 10), first_seen, last_seen}
        Sorted by usage_count descending.
        """
        try:
            session = get_session()
            db_rows = session.query(RelationOntologyDB).all()
            session.close()
        except Exception as e:
            log.warning("get_relation_taxonomy_db_error", error=str(e))
            return []

        groups: Dict[str, dict] = {}
        for row in db_rows:
            c = row.relation_canonical or "RELATED_TO"
            if c not in groups:
                groups[c] = {
                    "canonical":    c,
                    "is_in_vocab":  c in SEED_SET,
                    "usage_count":  0,
                    "phrase_count": 0,
                    "phrases":      [],
                    "first_seen":   None,
                    "last_seen":    None,
                }
            g = groups[c]
            g["usage_count"]  += row.usage_count or 1
            g["phrase_count"] += 1
            if row.relation_text:
                g["phrases"].append(row.relation_text)
            if row.first_seen:
                ts = row.first_seen.isoformat()
                if g["first_seen"] is None or ts < g["first_seen"]:
                    g["first_seen"] = ts
            if row.last_seen:
                ts = row.last_seen.isoformat()
                if g["last_seen"] is None or ts > g["last_seen"]:
                    g["last_seen"] = ts

        for g in groups.values():
            g["phrases"] = g["phrases"][:10]

        return sorted(groups.values(), key=lambda x: x["usage_count"], reverse=True)

    def cluster_relations(self) -> Dict[str, List[str]]:
        """Return {canonical: [raw_phrase, ...]} from SQLite."""
        try:
            session = get_session()
            rows    = session.query(RelationOntologyDB).all()
            session.close()
        except Exception as e:
            log.warning("cluster_relations_db_error", error=str(e))
            return {}

        clusters: Dict[str, List[str]] = defaultdict(list)
        for row in rows:
            c = row.relation_canonical or "RELATED_TO"
            if row.relation_text:
                clusters[c].append(row.relation_text)
        return dict(clusters)

    def get_canonical_labels(self) -> List[str]:
        """Seed vocab labels + any emergent labels found in DB."""
        seed = set(SEED_SET)
        try:
            session  = get_session()
            emergent = {
                row.relation_canonical
                for row in session.query(RelationOntologyDB).all()
                if row.relation_canonical and row.relation_canonical not in seed
            }
            session.close()
        except Exception:
            emergent = set()
        return sorted(seed | emergent)

    def is_in_vocabulary(self, canonical: str) -> bool:
        return canonical in SEED_SET

    # ------------------------------------------------------------------
    # Private: similarity helpers
    # ------------------------------------------------------------------

    def _preprocess(self, text: str) -> str:
        t = text.lower().strip()
        return re.sub(r"\s+", " ", t)

    def _vocab_cosine_match(
        self, embedding_np: np.ndarray
    ) -> Tuple[Optional[str], float]:
        if not self._vocab_embeddings:
            return None, 0.0
        query = embedding_np / (np.linalg.norm(embedding_np) + 1e-9)
        best_canonical, best_sim = None, -1.0
        for canonical, vec in self._vocab_embeddings.items():
            sim = float(np.dot(query, vec))
            if sim > best_sim:
                best_sim, best_canonical = sim, canonical
        return best_canonical, best_sim

    def _chroma_cluster_match(
        self, embedding_l: list
    ) -> Tuple[Optional[str], float]:
        try:
            matches = self.chroma.search_relation_ontology(embedding_l, n_results=3)
            if not (matches and matches.get("ids") and matches["ids"][0]):
                return None, 0.0
            ids       = matches["ids"][0]
            distances = matches.get("distances", [[]])[0]
            if not ids or not distances:
                return None, 0.0
            # ChromaDB returns L2 distance; convert to cosine-similarity proxy
            similarity = 1.0 - min(float(distances[0]), 1.0)
            if similarity < EMERGENT_SIM_THRESHOLD:
                return None, similarity
            session   = get_session()
            rel       = session.query(RelationOntologyDB).filter_by(
                relation_id=ids[0]
            ).first()
            canonical = rel.relation_canonical if rel else None
            session.close()
            if canonical:
                self._increment_usage(ids[0])
            return canonical, similarity
        except Exception as e:
            log.warning("chroma_cluster_match_failed", error=str(e))
            return None, 0.0

    @staticmethod
    def _make_emergent_canonical(cleaned: str) -> str:
        """
        Convert a preprocessed phrase to UPPER_SNAKE canonical.
        e.g. "imposed travel ban on" → "IMPOSED_TRAVEL_BAN_ON"
        """
        stopwords = {
            "the", "a", "an", "to", "of", "for", "with", "by", "on",
            "in", "at", "and", "or", "is", "was", "were", "been",
        }
        tokens = [
            t for t in re.sub(r"[^a-z0-9 ]", "", cleaned.lower()).split()
            if t not in stopwords
        ]
        if not tokens:
            return "RELATED_TO"
        label = "_".join(tokens[:6]).upper()
        if label in SEED_SET:
            label = label + "_EMERGENT"
        return label

    # ------------------------------------------------------------------
    # Private: persistence
    # ------------------------------------------------------------------

    def _persist_relation(
        self,
        cleaned: str,
        embedding_l: Optional[list],
        original_text: str,
        canonical: str,
    ) -> str:
        """Upsert relation in SQLite and ChromaDB; return canonical."""
        session = get_session()
        try:
            existing = session.query(RelationOntologyDB).filter_by(
                relation_text=original_text
            ).first()
            if existing:
                existing.usage_count = (existing.usage_count or 0) + 1
                existing.last_seen   = datetime.utcnow()
                if canonical and existing.relation_canonical != canonical:
                    existing.relation_canonical = canonical
                session.commit()
                return canonical or existing.relation_canonical or "RELATED_TO"
        except Exception as e:
            log.warning("persist_relation_lookup_failed", error=str(e))
        finally:
            session.close()

        relation_id = str(uuid.uuid4())
        emb_bytes   = (
            np.array(embedding_l, dtype=np.float32).tobytes()
            if embedding_l is not None else None
        )
        session = get_session()
        try:
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
        except Exception as e:
            log.warning("persist_relation_insert_failed", error=str(e))
            session.rollback()
            return canonical
        finally:
            session.close()

        if embedding_l is not None and self.chroma:
            try:
                self.chroma.add_relations(
                    ids=[relation_id],
                    embeddings=[embedding_l],
                    metadatas=[{
                        "relation_text":      original_text,
                        "relation_canonical": canonical,
                        "usage_count":        1,
                    }],
                )
            except Exception as e:
                log.warning("relation_chroma_add_failed", error=str(e))

        return canonical

    def _increment_usage(self, relation_id: str) -> None:
        session = get_session()
        try:
            rel = session.query(RelationOntologyDB).filter_by(
                relation_id=relation_id
            ).first()
            if rel:
                rel.usage_count = (rel.usage_count or 0) + 1
                rel.last_seen   = datetime.utcnow()
                session.commit()
        except Exception as e:
            log.warning("increment_usage_failed", error=str(e))
        finally:
            session.close()
