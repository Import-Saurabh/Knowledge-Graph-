import networkx as nx
from typing import Dict, List
from src.utils.logger import get_logger

log = get_logger(__name__)

class GraphMetrics:
    def compute_all(self, graph: nx.DiGraph) -> dict:
        metrics = {}

        try:
            metrics["degree_centrality"] = nx.degree_centrality(graph)
        except Exception as e:
            log.warning("degree_centrality_failed", error=str(e))
            metrics["degree_centrality"] = {}

        try:
            metrics["betweenness_centrality"] = nx.betweenness_centrality(graph)
        except Exception as e:
            log.warning("betweenness_centrality_failed", error=str(e))
            metrics["betweenness_centrality"] = {}

        # Top entities by mentions
        top_by_mentions = []
        for node, data in graph.nodes(data=True):
            if data.get("type") != "Event":
                top_by_mentions.append((node, data.get("name", node), data.get("mention_count", 0)))
        top_by_mentions.sort(key=lambda x: x[2], reverse=True)
        metrics["top_entities_by_mentions"] = top_by_mentions[:20]

        # Top entities by connections
        degrees = dict(graph.degree())
        top_by_connections = []
        for node, data in graph.nodes(data=True):
            if data.get("type") != "Event":
                top_by_connections.append((node, data.get("name", node), degrees.get(node, 0)))
        top_by_connections.sort(key=lambda x: x[2], reverse=True)
        metrics["top_entities_by_connections"] = top_by_connections[:20]

        # Event frequency by window
        event_windows = {}
        for node, data in graph.nodes(data=True):
            if data.get("type") == "Event":
                window = data.get("temporal_window", "unknown")
                event_windows[window] = event_windows.get(window, 0) + 1
        metrics["event_frequency_by_window"] = event_windows

        # Graph stats
        from src.graph.graph_builder import GraphBuilder
        gb = GraphBuilder()
        gb.graph = graph
        metrics["graph_stats"] = gb.get_stats()

        # Ontology stats
        types = set()
        relations = set()
        for node, data in graph.nodes(data=True):
            if "type" in data and data["type"] != "Event":
                types.add(data["type"])
        for u, v, data in graph.edges(data=True):
            if "relation" in data and data["relation"] != "PARTICIPATES_IN":
                relations.add(data["relation"])

        metrics["ontology_stats"] = {
            "entity_types_discovered": len(types),
            "relation_types_discovered": len(relations)
        }

        return metrics

    def save_report(self, metrics: dict, output_path: str) -> None:
        import json
        # Convert to serializable
        serializable = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                serializable[k] = {str(kk): vv for kk, vv in v.items()}
            elif isinstance(v, list):
                serializable[k] = [list(item) if isinstance(item, tuple) else item for item in v]
            else:
                serializable[k] = v

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        log.info("analytics_report_saved", path=output_path)
