"""
src/relations/relation_ontology.py

Normalises free-text relation phrases to canonical forms.

Changes vs previous version:
  - CONTROLLED_VOCABULARY dict defines the master taxonomy (~30 canonical
    relation types grouped by domain).  This is the source of truth.
  - normalize_relation() now runs a three-stage lookup:
      1. Exact / near-exact string match against vocab alias phrases   (fast)
      2. Cosine similarity against pre-computed vocab embeddings        (accurate)
      3. Cosine similarity against ChromaDB (previously-seen relations) (fallback)
    If nothing passes RELATION_SIMILARITY_THRESHOLD a new entry is minted,
    but it is first mapped to the nearest vocab canonical rather than being
    stored as a raw ALL_CAPS slug.
  - precompute_vocab_embeddings() eagerly warms the vocab embedding cache so
    subsequent calls are O(1) cosine comparisons.
  - get_relation_taxonomy() groups output by controlled-vocab cluster first,
    then by HDBSCAN cluster_id for uncategorised entries.
  - All existing DB / ChromaDB / HDBSCAN logic is preserved unchanged.
"""

import re
import uuid
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from src.utils.db import get_session, RelationOntologyDB
from src.utils.logger import get_logger
from src.utils.config import settings

log = get_logger(__name__)

RELATION_SIMILARITY_THRESHOLD = 0.85   # cosine similarity to map to existing canonical
VOCAB_SIMILARITY_THRESHOLD    = 0.72   # lower bar: map to nearest vocab canonical

# ---------------------------------------------------------------------------
# Controlled vocabulary
# ---------------------------------------------------------------------------
# Keys   → canonical relation labels used in the graph and Neo4j export.
# Values → representative alias phrases used for embedding-based matching.
#
# The LLM is prompted to pick the best key directly (see relation_extractor.py).
# This dict is the server-side guard that re-validates or re-maps whatever
# the LLM returned.
# ---------------------------------------------------------------------------

CONTROLLED_VOCABULARY: Dict[str, List[str]] = {
    # ── Military / Conflict ─────────────────────────────────────────────
    "MILITARY_ATTACK": [
        "launched airstrike against", "bombed", "shelled", "struck",
        "fired missiles at", "attacked", "conducted military operation against",
        "deployed forces against",
    ],
    "MILITARY_OCCUPATION": [
        "occupied", "seized", "captured", "took control of",
        "invaded", "moved troops into",
    ],
    "MILITARY_WITHDRAWAL": [
        "withdrew from", "retreated from", "pulled back from",
        "evacuated forces from", "ended military presence in",
    ],
    "MILITARY_SUPPORT": [
        "provided military aid to", "supplied weapons to", "armed",
        "sent troops to support", "provided defence assistance to",
    ],
    "CEASEFIRE": [
        "agreed ceasefire with", "declared ceasefire", "signed ceasefire",
        "halted fighting with", "paused hostilities",
    ],

    # ── Diplomatic ──────────────────────────────────────────────────────
    "DIPLOMATIC_MEETING": [
        "met with", "held talks with", "negotiated with",
        "convened summit with", "held bilateral meeting with",
    ],
    "DIPLOMATIC_AGREEMENT": [
        "signed agreement with", "signed deal with", "reached accord with",
        "concluded treaty with", "finalised pact with",
    ],
    "DIPLOMATIC_RECOGNITION": [
        "recognised", "established diplomatic relations with",
        "normalised relations with",
    ],
    "DIPLOMATIC_EXPULSION": [
        "expelled ambassador of", "broke diplomatic relations with",
        "recalled ambassador from", "downgraded relations with",
    ],
    "DIPLOMATIC_STATEMENT": [
        "condemned", "criticised", "denounced", "praised",
        "issued statement against", "called on",
    ],
    "PEACE_NEGOTIATION": [
        "negotiated peace with", "mediated between", "brokered peace deal",
        "facilitated negotiations between", "proposed peace plan",
    ],

    # ── Sanctions / Economic ────────────────────────────────────────────
    "SANCTIONS_IMPOSED": [
        "imposed sanctions on", "sanctioned", "placed embargo on",
        "froze assets of", "banned trade with",
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
        "invested in", "acquired stake in",
        "financed", "funded project in",
    ],

    # ── Leadership / Political ──────────────────────────────────────────
    "LEADER_OF": [
        "leads", "heads", "commands", "governs",
        "is president of", "is prime minister of", "is chairman of",
    ],
    "APPOINTED": [
        "appointed", "elected", "nominated", "named as",
        "sworn in as", "confirmed as",
    ],
    "RESIGNED": [
        "resigned from", "stepped down from", "ousted from",
        "removed from office", "forced to resign from",
    ],
    "ALLY_OF": [
        "allied with", "partnered with", "supports",
        "backs", "is coalition partner of",
    ],
    "OPPOSES": [
        "opposes", "protests against", "challenges",
        "is rival of", "stands against",
    ],

    # ── Legal / Criminal ────────────────────────────────────────────────
    "ACCUSED_OF": [
        "accused of", "charged with", "indicted for",
        "alleged to have", "suspected of",
    ],
    "CONVICTED_OF": [
        "convicted of", "found guilty of", "sentenced for",
    ],
    "ARRESTED": [
        "arrested", "detained", "captured", "taken into custody",
    ],

    # ── Organisational ──────────────────────────────────────────────────
    "MEMBER_OF": [
        "is member of", "belongs to", "joined", "is part of",
    ],
    "FOUNDED": [
        "founded", "established", "created", "set up",
    ],
    "HEADQUARTERED_IN": [
        "based in", "headquartered in", "operates from",
    ],

    # ── Humanitarian / Population ───────────────────────────────────────
    "HUMANITARIAN_AID": [
        "provided humanitarian aid to", "delivered aid to",
        "sent relief to", "funded humanitarian operation in",
    ],
    "REFUGEE_MOVEMENT": [
        "fled to", "displaced to", "evacuated to",
        "sought refuge in", "migrated to",
    ],

    # ── Nuclear / WMD ───────────────────────────────────────────────────
    "NUCLEAR_ACTIVITY": [
        "conducted nuclear test", "enriched uranium",
        "developed nuclear weapons", "restarted nuclear programme",
        "violated nuclear agreement",
    ],

    # ── Catch-all ───────────────────────────────────────────────────────
    "RELATED_TO": [
        "is related to", "is associated with", "is linked to",
    ],
}

# Flat alias → canonical lookup (built once at import time)
_ALIAS_TO_CANONICAL: Dict[str, str] = {}
for _canonical, _aliases in CONTROLLED_VOCABULARY.items():
    _ALIAS_TO_CANONICAL[_canonical.lower()] = _canonical   # canonical is its own alias
    for _alias in _aliases:
        _ALIAS_TO_CANONICAL[_alias.lower()] = _canonical


class RelationOntologyManager:
    def __init__(self, chroma_manager, embedding_generator):
        self.chroma    = chroma_manager
        self.embedder  = embedding_generator

        # Cache: canonical_label → embedding vector (numpy float32)
        self._vocab_embeddings: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, relation_text: str) -> str:
        text = relation_text.lower().strip()
        text = re.sub(r"\b(the|a|an|to|of|for|with|by|on|in|at)\b", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ------------------------------------------------------------------
    # Vocabulary warm-up
    # ------------------------------------------------------------------

    def precompute_vocab_embeddings(self) -> None:
        """
        Pre-embed every canonical label and a representative alias so that
        subsequent normalize_relation() calls do fast in-memory cosine
        comparisons instead of hitting the embedder repeatedly.

        Call once during pipeline initialisation (or lazily on first use).
        """
        for canonical, aliases in CONTROLLED_VOCABULARY.items():
            if canonical in self._vocab_embeddings:
                continue
            # Embed the canonical label itself (most compact representation)
            text_to_embed = canonical.lower().replace("_", " ")
            try:
                vec = np.array(
                    self.embedder.embed_text(text_to_embed), dtype=np.float32
                )
                self._vocab_embeddings[canonical] = vec / (np.linalg.norm(vec) + 1e-9)
            except Exception as e:
                log.warning("vocab_embedding_failed", canonical=canonical, error=str(e))

        log.info("vocab_embeddings_ready", count=len(self._vocab_embeddings))

    # ------------------------------------------------------------------
    # Stage 1 — exact / fuzzy string match against alias table
    # ------------------------------------------------------------------

    def _alias_match(self, cleaned: str) -> Optional[str]:
        """Return canonical if cleaned text matches a known alias exactly."""
        # Direct lookup
        if cleaned in _ALIAS_TO_CANONICAL:
            return _ALIAS_TO_CANONICAL[cleaned]

        # Substring containment (for partial matches like "imposed sanctions")
        for alias, canonical in _ALIAS_TO_CANONICAL.items():
            if alias in cleaned or cleaned in alias:
                return canonical

        return None

    # ------------------------------------------------------------------
    # Stage 2 — cosine similarity against vocab embeddings
    # ------------------------------------------------------------------

    def _vocab_embedding_match(
        self, embedding: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """
        Return (best_canonical, similarity) against the pre-computed vocab
        embeddings.  Returns (None, 0.0) if cache is empty.
        """
        if not self._vocab_embeddings:
            return None, 0.0

        query = embedding / (np.linalg.norm(embedding) + 1e-9)
        best_canonical = None
        best_sim       = -1.0

        for canonical, vec in self._vocab_embeddings.items():
            sim = float(np.dot(query, vec))
            if sim > best_sim:
                best_sim       = sim
                best_canonical = canonical

        return best_canonical, best_sim

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize_relation(self, relation_text: str) -> str:
        """
        Three-stage normalization pipeline:

        Stage 1 — Alias string match (no embedding call needed)
            If the cleaned text matches a known alias exactly or by
            containment → return that canonical immediately.

        Stage 2 — Vocab embedding similarity
            Embed the cleaned text; compare cosine similarity to pre-computed
            vocab embeddings.  If best_sim >= VOCAB_SIMILARITY_THRESHOLD →
            return that canonical.

        Stage 3 — ChromaDB lookup (previously seen relations)
            Search the stored relation ontology for near-duplicates.
            If similarity >= RELATION_SIMILARITY_THRESHOLD → reuse that
            canonical.

        Fallback — mint a new entry
            Map to nearest vocab canonical if any similarity > 0; otherwise
            store as a new unseen relation (tagged RELATED_TO as safety net).
        """
        cleaned = self._preprocess(relation_text)

        # ── Stage 1: alias table ─────────────────────────────────────────
        alias_hit = self._alias_match(cleaned)
        if alias_hit:
            log.debug("relation_normalized_alias", raw=relation_text, canonical=alias_hit)
            return alias_hit

        # ── Embed once; reuse across stages 2 & 3 ───────────────────────
        embedding: Optional[list] = None
        embedding_np: Optional[np.ndarray] = None
        try:
            embedding = self.embedder.embed_text(cleaned)
            embedding_np = np.array(embedding, dtype=np.float32)
        except Exception as e:
            log.warning("relation_embedding_failed", error=str(e))

        # ── Stage 2: vocab embedding match ───────────────────────────────
        if embedding_np is not None and self._vocab_embeddings:
            vocab_canonical, vocab_sim = self._vocab_embedding_match(embedding_np)
            if vocab_canonical and vocab_sim >= VOCAB_SIMILARITY_THRESHOLD:
                log.debug(
                    "relation_normalized_vocab_embedding",
                    raw=relation_text,
                    canonical=vocab_canonical,
                    similarity=round(vocab_sim, 3),
                )
                # Record in DB / ChromaDB for faster future lookups
                return self._create_relation_entry(
                    cleaned, embedding, relation_text,
                    forced_canonical=vocab_canonical,
                )

        # ── Stage 3: ChromaDB (previously seen relations) ─────────────────
        if embedding is not None and self.chroma:
            try:
                matches = self.chroma.search_relation_ontology(embedding, n_results=5)
                if matches and matches.get("ids") and matches["ids"][0]:
                    ids       = matches["ids"][0]
                    distances = matches.get("distances", [[]])[0]
                    if ids and distances:
                        similarity = 1.0 - min(distances[0], 1.0)
                        if similarity > RELATION_SIMILARITY_THRESHOLD:
                            self._increment_usage(ids[0])
                            session = get_session()
                            rel = session.query(RelationOntologyDB).filter_by(
                                relation_id=ids[0]
                            ).first()
                            canonical = rel.relation_canonical if rel else cleaned
                            session.close()
                            return canonical
            except Exception as e:
                log.warning("relation_search_failed", error=str(e))

        # ── Fallback: map to nearest vocab canonical or mint new entry ────
        forced = None
        if embedding_np is not None and self._vocab_embeddings:
            best_canonical, best_sim = self._vocab_embedding_match(embedding_np)
            if best_canonical and best_sim > 0.40:   # soft floor — avoids random assignment
                forced = best_canonical
                log.debug(
                    "relation_fallback_to_vocab",
                    raw=relation_text,
                    canonical=forced,
                    similarity=round(best_sim, 3),
                )

        if forced is None:
            forced = "RELATED_TO"   # ultimate safety net

        return self._create_relation_entry(
            cleaned, embedding, relation_text, forced_canonical=forced
        )

    # ------------------------------------------------------------------
    # Vocabulary introspection helpers
    # ------------------------------------------------------------------

    def get_canonical_labels(self) -> List[str]:
        """Return the sorted list of canonical relation labels."""
        return sorted(CONTROLLED_VOCABULARY.keys())

    def is_in_vocabulary(self, canonical: str) -> bool:
        return canonical in CONTROLLED_VOCABULARY

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_relation_entry(
        self,
        cleaned: str,
        embedding: Optional[list],
        original_text: str,
        forced_canonical: Optional[str] = None,
    ) -> str:
        session  = get_session()
        existing = session.query(RelationOntologyDB).filter_by(
            relation_text=original_text
        ).first()
        if existing:
            canonical = existing.relation_canonical or cleaned
            session.close()
            return canonical

        relation_id = str(uuid.uuid4())
        emb_bytes   = None
        if embedding is not None:
            emb_bytes = np.array(embedding, dtype=np.float32).tobytes()

        # Use forced_canonical if provided; otherwise derive from cleaned text
        if forced_canonical:
            canonical = forced_canonical
        else:
            canonical = cleaned.upper().replace(" ", "_")

        db_rel = RelationOntologyDB(
            relation_id        = relation_id,
            relation_text      = original_text,
            relation_canonical = canonical,
            relation_embedding = emb_bytes,
            usage_count        = 1,
            first_seen         = datetime.utcnow(),
            last_seen          = datetime.utcnow(),
        )
        session.add(db_rel)
        session.commit()
        # Capture BEFORE session.close() — avoids DetachedInstanceError
        canonical = db_rel.relation_canonical
        session.close()

        if embedding is not None and self.chroma:
            try:
                self.chroma.add_relations(
                    ids        = [relation_id],
                    embeddings = [embedding],
                    metadatas  = [{
                        "relation_text":      original_text,
                        "relation_canonical": canonical,
                        "usage_count":        1,
                    }],
                )
            except Exception as e:
                log.warning("relation_chroma_add_failed", error=str(e))

        log.info("new_relation_created",
                 text=original_text, canonical=canonical)
        return canonical

    def _increment_usage(self, relation_id: str) -> None:
        session = get_session()
        rel = session.query(RelationOntologyDB).filter_by(
            relation_id=relation_id
        ).first()
        if rel:
            rel.usage_count += 1
            rel.last_seen    = datetime.utcnow()
            session.commit()
        session.close()

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def cluster_relations(self) -> dict:
        """
        HDBSCAN clustering over all stored relation embeddings.
        Cluster canonical is set to the nearest vocab label for the
        cluster centroid rather than the most-used raw text.
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
                log.info("cluster_relations_skipped",
                         reason="fewer than 5 embeddings")
                return {}

            embeddings = np.array(data["embeddings"])
            ids        = data["ids"]

            clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric="euclidean")
            labels    = clusterer.fit_predict(embeddings)

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
                if not rels:
                    continue

                # Prefer a vocab-canonical for the cluster representative
                centroid = np.mean(
                    [
                        np.frombuffer(r.relation_embedding, dtype=np.float32)
                        for r in rels
                        if r.relation_embedding
                    ],
                    axis=0,
                ) if any(r.relation_embedding for r in rels) else None

                if centroid is not None and self._vocab_embeddings:
                    vocab_canonical, _ = self._vocab_embedding_match(centroid)
                    canonical = vocab_canonical or rels[0].relation_canonical
                else:
                    canonical = rels[0].relation_canonical

                for rel in rels:
                    rel.relation_canonical = canonical
                    rel.cluster_id         = int(cluster_id)

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
        """
        Returns the relation taxonomy in two sections:
          1. Controlled-vocabulary relations (vocab_hit=True)
          2. Uncategorised / HDBSCAN clusters (vocab_hit=False)
        """
        session   = get_session()
        relations = session.query(RelationOntologyDB).all()

        # Group by canonical label
        by_canonical: Dict[str, list] = {}
        for rel in relations:
            canon = rel.relation_canonical or "RELATED_TO"
            by_canonical.setdefault(canon, []).append({
                "relation_text":      rel.relation_text,
                "relation_canonical": canon,
                "usage_count":        rel.usage_count,
                "cluster_id":         rel.cluster_id,
            })

        session.close()

        # Partition into vocab vs unseen
        vocab_set = set(CONTROLLED_VOCABULARY.keys())
        taxonomy  = []

        # Vocab entries first (in defined order)
        for canonical in CONTROLLED_VOCABULARY:
            items = by_canonical.get(canonical, [])
            taxonomy.append({
                "canonical":  canonical,
                "vocab_hit":  True,
                "domain":     _vocab_domain(canonical),
                "usage_count": sum(i["usage_count"] for i in items),
                "relations":   sorted(items,
                                      key=lambda x: x["usage_count"],
                                      reverse=True),
            })

        # Unseen / drift entries (not in vocab)
        for canonical, items in by_canonical.items():
            if canonical not in vocab_set:
                taxonomy.append({
                    "canonical":   canonical,
                    "vocab_hit":   False,
                    "domain":      "uncategorised",
                    "usage_count": sum(i["usage_count"] for i in items),
                    "relations":   sorted(items,
                                         key=lambda x: x["usage_count"],
                                         reverse=True),
                })

        return taxonomy


# ---------------------------------------------------------------------------
# Helper — domain tag from canonical name
# ---------------------------------------------------------------------------

_DOMAIN_MAP = {
    "MILITARY_ATTACK":       "military",
    "MILITARY_OCCUPATION":   "military",
    "MILITARY_WITHDRAWAL":   "military",
    "MILITARY_SUPPORT":      "military",
    "CEASEFIRE":             "military",
    "DIPLOMATIC_MEETING":    "diplomatic",
    "DIPLOMATIC_AGREEMENT":  "diplomatic",
    "DIPLOMATIC_RECOGNITION":"diplomatic",
    "DIPLOMATIC_EXPULSION":  "diplomatic",
    "DIPLOMATIC_STATEMENT":  "diplomatic",
    "PEACE_NEGOTIATION":     "diplomatic",
    "SANCTIONS_IMPOSED":     "economic",
    "SANCTIONS_LIFTED":      "economic",
    "TRADE_AGREEMENT":       "economic",
    "ECONOMIC_AID":          "economic",
    "INVESTMENT":            "economic",
    "LEADER_OF":             "political",
    "APPOINTED":             "political",
    "RESIGNED":              "political",
    "ALLY_OF":               "political",
    "OPPOSES":               "political",
    "ACCUSED_OF":            "legal",
    "CONVICTED_OF":          "legal",
    "ARRESTED":              "legal",
    "MEMBER_OF":             "organisational",
    "FOUNDED":               "organisational",
    "HEADQUARTERED_IN":      "organisational",
    "HUMANITARIAN_AID":      "humanitarian",
    "REFUGEE_MOVEMENT":      "humanitarian",
    "NUCLEAR_ACTIVITY":      "security",
    "RELATED_TO":            "generic",
}


def _vocab_domain(canonical: str) -> str:
    return _DOMAIN_MAP.get(canonical, "uncategorised")