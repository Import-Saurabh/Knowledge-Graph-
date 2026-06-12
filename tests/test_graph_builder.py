import networkx as nx
from src.graph.graph_builder import GraphBuilder
from src.models.entity import CanonicalEntity

def test_dynamic_node_types_in_graph():
    graph_builder = GraphBuilder()
    graph_builder.add_entity_node(CanonicalEntity(
        canonical_name="SpaceX",
        entity_type="Aerospace Manufacturer"
    ))
    graph_builder.add_entity_node(CanonicalEntity(
        canonical_name="Elon Musk",
        entity_type="Technology CEO"
    ))

    types = set(nx.get_node_attributes(graph_builder.graph, 'type').values())
    assert "Aerospace Manufacturer" in types
    assert "Technology CEO" in types
