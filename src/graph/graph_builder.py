"""
src/graph/graph_builder.py

Builds a NetworkX DiGraph from LLM, GLiREL, and infobox triples.
Now also accepts entity_enrichment from Wikidata and infobox_triples.
"""

import networkx as nx
from typing import List, Dict, Optional, Tuple

from src.models.event import EventModel
from src.models.relation import RelationTriple, LLMRelationResponse
from src.models.entity import CanonicalEntity
from src.utils.logger import get_logger

log = get_logger(__name__)

class GraphBuilder:
    LLM_CONF_THRESHOLD:   float = 0.70
    GLIREL_SCORE_THRESHOLD: float = 0.50
    MIN_NODE_DEGREE:      int   = 2

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_event_node(self, event: EventModel, event_label: str) -> None:
        self.graph.add_node(
            event.event_id,
            name=event_label,
            type="Event",
            cluster_id=event.cluster_id,
            temporal_window=event.temporal_window,
            article_count=len(event.article_ids),
        )

    def add_entity_node(self, canonical_entity: CanonicalEntity,
                        extra_attrs: Optional[Dict] = None) -> None:
        attrs = {
            "name": canonical_entity.canonical_name,
            "type": canonical_entity.entity_type,
            "mention_count": canonical_entity.mention_count,
            "aliases": canonical_entity.aliases,
        }
        if extra_attrs:
            attrs.update(extra_attrs)
        self.graph.add_node(canonical_entity.canonical_id, **attrs)

    def _ensure_entity_stub(self, node_id: str, name: str) -> None:
        if node_id not in self.graph:
            self.graph.add_node(
                node_id,
                name=name,
                type="Unknown",
                mention_count=0,
                aliases=[],
            )

    def add_relation(self, triple: RelationTriple, event_id: str) -> None:
        if triple.confidence < self.LLM_CONF_THRESHOLD:
            return

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

        if triple.source and triple.target and triple.source != triple.target:
            rel_label = (
                triple.relation_canonical
                if triple.relation_canonical
                else triple.relation
            )
            edge_attrs = {
                "relation": rel_label,
                "confidence": triple.confidence,
                "event_id": event_id,
                "original_relation": triple.relation,
                "source": "llm",
                "wikidata_property": triple.wikidata_property,
                "wikidata_date": triple.wikidata_date,
                "wikidata_description": triple.wikidata_description,
                "needs_review": triple.needs_review,
            }
            self.graph.add_edge(triple.source, triple.target, **edge_attrs)

    def _add_glirel_edge(self, head_id: str, tail_id: str, relation: str,
                         score: float, article_id: str) -> None:
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

    def _prune_low_degree_nodes(self) -> int:
        to_remove = [
            n for n, data in self.graph.nodes(data=True)
            if data.get("type") != "Event"
            and self.graph.degree(n) < self.MIN_NODE_DEGREE
        ]
        self.graph.remove_nodes_from(to_remove)
        return len(to_remove)

    def _assign_communities(self) -> int:
        undirected = self.graph.to_undirected()
        try:
            from networkx.algorithms.community import louvain_communities
            communities = louvain_communities(undirected, seed=42)
        except (ImportError, AttributeError):
            try:
                from networkx.algorithms.community import greedy_modularity_communities
                communities = list(greedy_modularity_communities(undirected))
            except Exception:
                log.warning("community_detection_unavailable")
                for n in self.graph.nodes():
                    self.graph.nodes[n]["community_id"] = 0
                return 1

        for community_id, members in enumerate(communities):
            for node in members:
                if node in self.graph:
                    self.graph.nodes[node]["community_id"] = community_id
        return len(communities)

    def _enrich_nodes_with_wikidata(self, entity_enrichment: Dict[str, Dict]) -> None:
        if not entity_enrichment:
            return
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "Event":
                continue
            name = data.get("name")
            if name and name in entity_enrichment:
                self.graph.nodes[node].update(entity_enrichment[name])

    def build_from_relations(
        self,
        events: List[EventModel],
        llm_responses: List[LLMRelationResponse],
        entity_map: Dict[str, CanonicalEntity] = None,
        glirel_triples: Optional[List[dict]] = None,
        entity_enrichment: Optional[Dict[str, Dict]] = None,
        infobox_triples: Optional[List[Tuple[str, str, str]]] = None,
    ) -> nx.DiGraph:
        name_to_id: Dict[str, str] = {}
        if entity_map:
            for entity in entity_map.values():
                name_to_id[entity.canonical_name.lower()] = entity.canonical_id
                for alias in (entity.aliases or []):
                    name_to_id[alias.lower()] = entity.canonical_id

        # LLM triples
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
                    resolved.source = name_to_id.get(resolved.source.lower(), resolved.source)
                if resolved.target:
                    resolved.target = name_to_id.get(resolved.target.lower(), resolved.target)
                self.add_relation(resolved, event.event_id)
                llm_kept += 1
        log.info("llm_triples_filtered", kept=llm_kept, dropped=llm_dropped)

        # Canonical entity nodes
        if entity_map:
            for entity in entity_map.values():
                if entity.canonical_id in self.graph:
                    self.graph.nodes[entity.canonical_id].update({
                        "name": entity.canonical_name,
                        "type": entity.entity_type,
                        "mention_count": entity.mention_count,
                        "aliases": entity.aliases,
                    })
                else:
                    self.add_entity_node(entity)

        # GLiREL triples
        glirel_added = 0
        glirel_merged = 0
        glirel_dropped = 0
        if glirel_triples:
            for t in glirel_triples:
                head_raw = (t.get("head") or "").strip()
                tail_raw = (t.get("tail") or "").strip()
                relation = (t.get("relation") or "").strip()
                score = float(t.get("score", 0.0))
                art_id = t.get("article_id", "")
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
        log.info("glirel_triples_filtered", added=glirel_added, merged=glirel_merged, dropped=glirel_dropped)

        # Infobox triples (high confidence)
        if infobox_triples:
            added = 0
            for subject, relation, obj in infobox_triples:
                subj_id = name_to_id.get(subject.lower(), subject)
                obj_id = name_to_id.get(obj.lower(), obj)
                self._ensure_entity_stub(subj_id, subject)
                self._ensure_entity_stub(obj_id, obj)
                if not self.graph.has_edge(subj_id, obj_id):
                    self.graph.add_edge(
                        subj_id, obj_id,
                        relation=relation,
                        confidence=0.95,
                        source="infobox",
                        event_id="",
                        original_relation=relation,
                    )
                    added += 1
            log.info("infobox_triples_added", count=added)

        # Enrich nodes with Wikidata
        self._enrich_nodes_with_wikidata(entity_enrichment or {})

        # Prune and assign communities
        pruned = self._prune_low_degree_nodes()
        n_communities = self._assign_communities()

        log.info("graph_built",
                 nodes=self.graph.number_of_nodes(),
                 edges=self.graph.number_of_edges(),
                 communities=n_communities,
                 pruned_nodes=pruned)
        return self.graph

    def get_stats(self) -> dict:
        # (unchanged)
        pass