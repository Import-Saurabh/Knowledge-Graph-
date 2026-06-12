import networkx as nx
from typing import List, Dict
from src.models.event import EventModel
from src.models.relation import RelationTriple, LLMRelationResponse
from src.models.entity import CanonicalEntity
from src.utils.logger import get_logger
import uuid

log = get_logger(__name__)

class GraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_event_node(self, event: EventModel, event_label: str) -> None:
        self.graph.add_node(
            event.event_id,
            name=event_label,
            type="Event",
            cluster_id=event.cluster_id,
            temporal_window=event.temporal_window,
            article_count=len(event.article_ids)
        )

    def add_entity_node(self, canonical_entity: CanonicalEntity) -> None:
        self.graph.add_node(
            canonical_entity.canonical_id,
            name=canonical_entity.canonical_name,
            type=canonical_entity.entity_type,
            mention_count=canonical_entity.mention_count,
            aliases=canonical_entity.aliases
        )

    def add_relation(self, triple: RelationTriple, event_id: str) -> None:
        # Entity -> PARTICIPATES_IN -> Event
        if triple.source and event_id:
            self.graph.add_edge(
                triple.source,
                event_id,
                relation="PARTICIPATES_IN",
                confidence=triple.confidence,
                event_id=event_id
            )

        if triple.target and event_id:
            self.graph.add_edge(
                triple.target,
                event_id,
                relation="PARTICIPATES_IN",
                confidence=triple.confidence,
                event_id=event_id
            )

        # Entity -> [relation] -> Entity
        if triple.source and triple.target and triple.source != triple.target:
            rel_label = triple.relation_canonical if triple.relation_canonical else triple.relation
            self.graph.add_edge(
                triple.source,
                triple.target,
                relation=rel_label,
                confidence=triple.confidence,
                event_id=event_id,
                original_relation=triple.relation
            )

    def build_from_relations(self, events: List[EventModel], llm_responses: List[LLMRelationResponse], 
                            entity_map: Dict[str, CanonicalEntity] = None) -> nx.DiGraph:
        for event, response in zip(events, llm_responses):
            self.add_event_node(event, response.event_label)

            for triple in response.triples:
                self.add_relation(triple, event.event_id)

        if entity_map:
            for entity in entity_map.values():
                if entity.canonical_id in self.graph:
                    # Update node attributes if entity already added via edge
                    self.graph.nodes[entity.canonical_id].update({
                        "name": entity.canonical_name,
                        "type": entity.entity_type,
                        "mention_count": entity.mention_count,
                        "aliases": entity.aliases
                    })
                else:
                    self.add_entity_node(entity)

        log.info("graph_built", nodes=self.graph.number_of_nodes(), edges=self.graph.number_of_edges())
        return self.graph

    def get_stats(self) -> dict:
        types = set()
        relations = set()
        for node, data in self.graph.nodes(data=True):
            if "type" in data:
                types.add(data["type"])
        for u, v, data in self.graph.edges(data=True):
            if "relation" in data:
                relations.add(data["relation"])

        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "entity_types_discovered": len(types) - 1 if "Event" in types else len(types),
            "relation_types_discovered": len(relations)
        }
