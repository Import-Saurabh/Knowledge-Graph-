"""
src/graph/graph_builder.py

Builds a NetworkX DiGraph from:
  • LLM relation triples  (event-scoped, high-precision)
  • GLiREL raw triples    (article-scoped, zero-shot, high-recall)

Changes vs previous version:
  - Confidence thresholds: LLM edges require confidence > LLM_CONF_THRESHOLD (0.7),
    GLiREL edges require score > GLIREL_SCORE_THRESHOLD (0.5).
  - Low-degree pruning: after graph build, nodes with degree < MIN_NODE_DEGREE (2)
    are removed (Event nodes are exempt).
  - Community detection: Louvain communities computed on the undirected projection;
    each node receives a `community_id` attribute used by the exporter for colouring.
  - All thresholds are class-level constants so callers can override them.
"""

import networkx as nx
from typing import List, Dict, Optional

from src.models.event import EventModel
from src.models.relation import RelationTriple, LLMRelationResponse
from src.models.entity import CanonicalEntity
from src.utils.logger import get_logger

log = get_logger(__name__)


class GraphBuilder:
    # ------------------------------------------------------------------
    # Tuneable thresholds  (override on the instance if needed)
    # ------------------------------------------------------------------
    LLM_CONF_THRESHOLD:   float = 0.70   # drop LLM triples below this
    GLIREL_SCORE_THRESHOLD: float = 0.50  # drop GLiREL triples below this
    MIN_NODE_DEGREE:      int   = 2       # prune nodes with degree < this
    #   Event nodes are never pruned regardless of degree.

    def __init__(self):
        self.graph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Node helpers
    # ------------------------------------------------------------------

    def add_event_node(self, event: EventModel, event_label: str) -> None:
        self.graph.add_node(
            event.event_id,
            name=event_label,
            type="Event",
            cluster_id=event.cluster_id,
            temporal_window=event.temporal_window,
            article_count=len(event.article_ids),
        )

    def add_entity_node(self, canonical_entity: CanonicalEntity) -> None:
        self.graph.add_node(
            canonical_entity.canonical_id,
            name=canonical_entity.canonical_name,
            type=canonical_entity.entity_type,
            mention_count=canonical_entity.mention_count,
            aliases=canonical_entity.aliases,
        )

    def _ensure_entity_stub(self, node_id: str, name: str) -> None:
        """Add a minimal node if it doesn't exist yet (for GLiREL-only entities)."""
        if node_id not in self.graph:
            self.graph.add_node(
                node_id,
                name=name,
                type="Unknown",
                mention_count=0,
                aliases=[],
            )

    # ------------------------------------------------------------------
    # Edge helpers
    # ------------------------------------------------------------------

    def add_relation(self, triple: RelationTriple, event_id: str) -> None:
        """Add LLM triple edges.  Silently drops if confidence is below threshold."""
        if triple.confidence < self.LLM_CONF_THRESHOLD:
            return

        # Entity → PARTICIPATES_IN → Event
        if triple.source and event_id:
            self.graph.add_edge(
                triple.source, event_id,
                relation="PARTICIPATES_IN",
                confidence=triple.confidence,
                event_id=event_id,
                source="llm",
            )
        if triple.target and event_id:
            self.graph.add_edge(
                triple.target, event_id,
                relation="PARTICIPATES_IN",
                confidence=triple.confidence,
                event_id=event_id,
                source="llm",
            )

        # Entity → [relation] → Entity
        if triple.source and triple.target and triple.source != triple.target:
            rel_label = (
                triple.relation_canonical
                if triple.relation_canonical
                else triple.relation
            )
            self.graph.add_edge(
                triple.source, triple.target,
                relation=rel_label,
                confidence=triple.confidence,
                event_id=event_id,
                original_relation=triple.relation,
                source="llm",
            )

    def _add_glirel_edge(
        self,
        head_id: str,
        tail_id: str,
        relation: str,
        score: float,
        article_id: str,
    ) -> None:
        """
        Add a GLiREL-sourced entity→entity edge.
        Scores below GLIREL_SCORE_THRESHOLD are silently dropped.
        If the same edge already exists from LLM extraction, keep the LLM
        version (higher precision) but record that GLiREL also confirmed it.
        """
        if score < self.GLIREL_SCORE_THRESHOLD:
            return

        if self.graph.has_edge(head_id, tail_id):
            existing = self.graph[head_id][tail_id]
            existing.setdefault("glirel_confirmed", True)
            existing.setdefault("glirel_score", score)
            return

        self.graph.add_edge(
            head_id, tail_id,
            relation=relation,
            confidence=score,
            event_id="",
            article_id=article_id,
            original_relation=relation,
            source="glirel",
            glirel_confirmed=True,
            glirel_score=score,
        )

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _prune_low_degree_nodes(self) -> int:
        """
        Remove nodes whose (undirected) degree is below MIN_NODE_DEGREE.
        Event nodes are always kept.
        Returns the number of nodes removed.
        """
        to_remove = [
            n for n, data in self.graph.nodes(data=True)
            if data.get("type") != "Event"
            and self.graph.degree(n) < self.MIN_NODE_DEGREE
        ]
        self.graph.remove_nodes_from(to_remove)
        return len(to_remove)

    def _assign_communities(self) -> int:
        """
        Run Louvain community detection on the undirected projection of the
        graph and write `community_id` (int) onto every node.
        Returns the number of communities found.

        Falls back gracefully if python-louvain / networkx-community is not
        installed — all nodes get community_id = 0.
        """
        undirected = self.graph.to_undirected()
        try:
            # networkx >= 3.x ships greedy_modularity_communities
            from networkx.algorithms.community import louvain_communities
            communities = louvain_communities(undirected, seed=42)
        except (ImportError, AttributeError):
            try:
                from networkx.algorithms.community import (
                    greedy_modularity_communities,
                )
                communities = list(greedy_modularity_communities(undirected))
            except Exception:
                log.warning(
                    "community_detection_unavailable",
                    hint="Install networkx >= 3.0 for Louvain support. "
                         "All nodes assigned community_id=0.",
                )
                for n in self.graph.nodes():
                    self.graph.nodes[n]["community_id"] = 0
                return 1

        for community_id, members in enumerate(communities):
            for node in members:
                if node in self.graph:
                    self.graph.nodes[node]["community_id"] = community_id

        return len(communities)

    # ------------------------------------------------------------------
    # Main builder
    # ------------------------------------------------------------------

    def build_from_relations(
        self,
        events: List[EventModel],
        llm_responses: List[LLMRelationResponse],
        entity_map: Dict[str, CanonicalEntity] = None,
        glirel_triples: Optional[List[dict]] = None,
    ) -> nx.DiGraph:
        """
        Build the graph from three sources:
          1. entity_map     — canonical entity nodes (from GLiNER + resolution)
          2. llm_responses  — LLM relation triples (event-scoped, high precision)
          3. glirel_triples — GLiREL raw triples (article-scoped, high recall)

        Post-processing applied automatically:
          • Confidence/score filtering (edges below threshold are never added).
          • Low-degree node pruning (nodes with degree < MIN_NODE_DEGREE removed).
          • Community detection (Louvain); community_id written to every node.
        """

        # ── Step 0: name → canonical_id lookup ──────────────────────────
        name_to_id: Dict[str, str] = {}
        if entity_map:
            for entity in entity_map.values():
                name_to_id[entity.canonical_name.lower()] = entity.canonical_id
                for alias in (entity.aliases or []):
                    name_to_id[alias.lower()] = entity.canonical_id

        # ── Step 1: LLM triples (event-scoped) ──────────────────────────
        llm_kept = 0
        llm_dropped = 0
        for event, response in zip(events, llm_responses):
            self.add_event_node(event, response.event_label)

            for triple in response.triples:
                if triple.confidence < self.LLM_CONF_THRESHOLD:
                    llm_dropped += 1
                    continue

                resolved = triple.model_copy(deep=True)
                if resolved.source:
                    resolved.source = name_to_id.get(
                        resolved.source.lower(), resolved.source
                    )
                if resolved.target:
                    resolved.target = name_to_id.get(
                        resolved.target.lower(), resolved.target
                    )
                self.add_relation(resolved, event.event_id)
                llm_kept += 1

        log.info(
            "llm_triples_filtered",
            kept=llm_kept,
            dropped=llm_dropped,
            threshold=self.LLM_CONF_THRESHOLD,
        )

        # ── Step 2: canonical entity nodes ──────────────────────────────
        if entity_map:
            for entity in entity_map.values():
                if entity.canonical_id in self.graph:
                    self.graph.nodes[entity.canonical_id].update({
                        "name":          entity.canonical_name,
                        "type":          entity.entity_type,
                        "mention_count": entity.mention_count,
                        "aliases":       entity.aliases,
                    })
                else:
                    self.add_entity_node(entity)

        # ── Step 3: GLiREL triples (article-scoped) ──────────────────────
        glirel_added   = 0
        glirel_merged  = 0
        glirel_dropped = 0

        if glirel_triples:
            for t in glirel_triples:
                head_raw = (t.get("head") or "").strip()
                tail_raw = (t.get("tail") or "").strip()
                relation = (t.get("relation") or "").strip()
                score    = float(t.get("score", 0.0))
                art_id   = t.get("article_id", "")

                if not head_raw or not tail_raw or not relation:
                    continue
                if head_raw.lower() == tail_raw.lower():
                    continue

                if score < self.GLIREL_SCORE_THRESHOLD:
                    glirel_dropped += 1
                    continue

                head_id = name_to_id.get(head_raw.lower(), head_raw)
                tail_id = name_to_id.get(tail_raw.lower(), tail_raw)

                self._ensure_entity_stub(head_id, head_raw)
                self._ensure_entity_stub(tail_id, tail_raw)

                existed = self.graph.has_edge(head_id, tail_id)
                self._add_glirel_edge(head_id, tail_id, relation, score, art_id)

                if existed:
                    glirel_merged += 1
                else:
                    glirel_added += 1

        log.info(
            "glirel_triples_filtered",
            added=glirel_added,
            merged=glirel_merged,
            dropped=glirel_dropped,
            threshold=self.GLIREL_SCORE_THRESHOLD,
        )

        # ── Step 4: Prune low-degree nodes ───────────────────────────────
        pruned = self._prune_low_degree_nodes()
        log.info(
            "low_degree_pruning",
            nodes_removed=pruned,
            min_degree=self.MIN_NODE_DEGREE,
        )

        # ── Step 5: Community detection ───────────────────────────────────
        n_communities = self._assign_communities()
        log.info(
            "community_detection_complete",
            communities=n_communities,
        )

        log.info(
            "graph_built",
            nodes=self.graph.number_of_nodes(),
            edges=self.graph.number_of_edges(),
            llm_events=len(events),
            glirel_new_edges=glirel_added,
            glirel_merged_edges=glirel_merged,
            communities=n_communities,
        )
        return self.graph

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        types     = set()
        relations = set()
        sources   = {"llm": 0, "glirel": 0, "unknown": 0}
        communities: set = set()

        for _, data in self.graph.nodes(data=True):
            if "type" in data:
                types.add(data["type"])
            if "community_id" in data:
                communities.add(data["community_id"])

        for _, _, data in self.graph.edges(data=True):
            if "relation" in data:
                relations.add(data["relation"])
            src = data.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1

        return {
            "node_count":                self.graph.number_of_nodes(),
            "edge_count":                self.graph.number_of_edges(),
            "entity_types_discovered":   len(types) - (1 if "Event" in types else 0),
            "relation_types_discovered": len(relations),
            "edges_by_source":           sources,
            "communities_detected":      len(communities),
            "llm_conf_threshold":        self.LLM_CONF_THRESHOLD,
            "glirel_score_threshold":    self.GLIREL_SCORE_THRESHOLD,
            "min_node_degree":           self.MIN_NODE_DEGREE,
        }