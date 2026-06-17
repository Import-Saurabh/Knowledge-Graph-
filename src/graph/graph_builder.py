"""
src/graph/graph_builder.py
Builds a NetworkX DiGraph from:
  • LLM relation triples  (event-scoped, high-precision)
  • GLiREL raw triples    (article-scoped, zero-shot, high-recall)

GLiREL integration:
  build_from_relations() now accepts glirel_triples: List[dict].
  Each dict has keys: head, relation, tail, score, article_id
  These become direct entity→entity edges tagged source="glirel".
  They go through the same name→canonical_id resolver so they land
  on the correct node keys — not orphaned string nodes.
"""

import networkx as nx
from typing import List, Dict, Optional

from src.models.event import EventModel
from src.models.relation import RelationTriple, LLMRelationResponse
from src.models.entity import CanonicalEntity
from src.utils.logger import get_logger

log = get_logger(__name__)


class GraphBuilder:
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
        If the same edge already exists from LLM extraction, keep the LLM
        version (higher precision) but record that GLiREL also saw it.
        """
        if self.graph.has_edge(head_id, tail_id):
            existing = self.graph[head_id][tail_id]
            # Merge: note glirel confirmation on the existing edge
            existing.setdefault("glirel_confirmed", True)
            existing.setdefault("glirel_score", score)
            return

        self.graph.add_edge(
            head_id, tail_id,
            relation=relation,
            confidence=score,
            event_id="",          # GLiREL triples are article-scoped, not event-scoped
            article_id=article_id,
            original_relation=relation,
            source="glirel",
            glirel_confirmed=True,
            glirel_score=score,
        )

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
          1. entity_map   — canonical entity nodes (from GLiNER + resolution)
          2. llm_responses — LLM relation triples (event-scoped, high precision)
          3. glirel_triples — GLiREL raw triples (article-scoped, high recall)

        All entity name strings in LLM and GLiREL triples are resolved to
        canonical UUIDs via `name_to_canonical_id` so every edge lands on
        the correct node rather than creating orphaned string nodes.
        """

        # Step 0: build name → canonical_id lookup (used by both LLM + GLiREL)
        name_to_id: Dict[str, str] = {}
        if entity_map:
            for entity in entity_map.values():
                name_to_id[entity.canonical_name.lower()] = entity.canonical_id
                for alias in (entity.aliases or []):
                    name_to_id[alias.lower()] = entity.canonical_id

        # ------------------------------------------------------------------
        # Step 1: LLM triples (event-scoped)
        # ------------------------------------------------------------------
        for event, response in zip(events, llm_responses):
            self.add_event_node(event, response.event_label)

            for triple in response.triples:
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

        # ------------------------------------------------------------------
        # Step 2: Enrich / add canonical entity nodes
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Step 3: GLiREL triples (article-scoped, direct entity→entity edges)
        # ------------------------------------------------------------------
        glirel_added   = 0
        glirel_merged  = 0

        if glirel_triples:
            for t in glirel_triples:
                head_raw = (t.get("head") or "").strip()
                tail_raw = (t.get("tail") or "").strip()
                relation = (t.get("relation") or "").strip()
                score    = float(t.get("score", 0.5))
                art_id   = t.get("article_id", "")

                if not head_raw or not tail_raw or not relation:
                    continue
                if head_raw.lower() == tail_raw.lower():
                    continue  # skip self-loops

                # Resolve to canonical IDs; fall back to the raw name string
                # so the entity still appears in the graph even if it was
                # never resolved (e.g., only appears in one article).
                head_id = name_to_id.get(head_raw.lower(), head_raw)
                tail_id = name_to_id.get(tail_raw.lower(), tail_raw)

                # Ensure both endpoints exist as nodes
                self._ensure_entity_stub(head_id, head_raw)
                self._ensure_entity_stub(tail_id, tail_raw)

                existed = self.graph.has_edge(head_id, tail_id)
                self._add_glirel_edge(head_id, tail_id, relation, score, art_id)

                if existed:
                    glirel_merged += 1
                else:
                    glirel_added += 1

        log.info(
            "graph_built",
            nodes=self.graph.number_of_nodes(),
            edges=self.graph.number_of_edges(),
            llm_events=len(events),
            glirel_new_edges=glirel_added,
            glirel_merged_edges=glirel_merged,
        )
        return self.graph

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        types     = set()
        relations = set()
        sources   = {"llm": 0, "glirel": 0, "unknown": 0}

        for _, data in self.graph.nodes(data=True):
            if "type" in data:
                types.add(data["type"])

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
        }