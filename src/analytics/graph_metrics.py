"""
src/analytics/graph_metrics.py
Computes degree / betweenness centrality, top entities, event frequency,
and ontology stats from the final NetworkX DiGraph.

No bug-list items target this file.
Minor change: serialisation loop handles numpy scalars that json.dumps
would otherwise choke on (common when NetworkX returns numpy float64).
"""

import json
import networkx as nx
from typing import Dict, List

from src.utils.logger import get_logger

log = get_logger(__name__)


class GraphMetrics:
    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute_all(self, graph: nx.DiGraph) -> dict:
        metrics: dict = {}

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

        # Top entities by mention count
        top_by_mentions = [
            (node, data.get("name", node), data.get("mention_count", 0))
            for node, data in graph.nodes(data=True)
            if data.get("type") != "Event"
        ]
        top_by_mentions.sort(key=lambda x: x[2], reverse=True)
        metrics["top_entities_by_mentions"] = top_by_mentions[:20]

        # Top entities by graph degree
        degrees = dict(graph.degree())
        top_by_connections = [
            (node, data.get("name", node), degrees.get(node, 0))
            for node, data in graph.nodes(data=True)
            if data.get("type") != "Event"
        ]
        top_by_connections.sort(key=lambda x: x[2], reverse=True)
        metrics["top_entities_by_connections"] = top_by_connections[:20]

        # Event frequency per temporal window
        event_windows: Dict[str, int] = {}
        for node, data in graph.nodes(data=True):
            if data.get("type") == "Event":
                window = data.get("temporal_window", "unknown")
                event_windows[window] = event_windows.get(window, 0) + 1
        metrics["event_frequency_by_window"] = event_windows

        # Graph-level stats (delegate to GraphBuilder helper)
        try:
            from src.graph.graph_builder import GraphBuilder
            gb = GraphBuilder()
            gb.graph = graph
            metrics["graph_stats"] = gb.get_stats()
        except Exception as e:
            log.warning("graph_stats_failed", error=str(e))
            metrics["graph_stats"] = {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
            }

        # Ontology coverage
        types = {
            data["type"]
            for _, data in graph.nodes(data=True)
            if "type" in data and data["type"] != "Event"
        }
        relations = {
            data["relation"]
            for _, _, data in graph.edges(data=True)
            if "relation" in data and data["relation"] != "PARTICIPATES_IN"
        }
        metrics["ontology_stats"] = {
            "entity_types_discovered": len(types),
            "relation_types_discovered": len(relations),
            "entity_type_list": sorted(types),
            "relation_type_list": sorted(relations),
        }

        return metrics

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save_report(self, metrics: dict, output_path: str) -> None:
        """
        Write metrics to JSON.
        Converts numpy scalars / tuples to plain Python types so json.dumps
        never raises TypeError on float64 / int64 from NetworkX centrality dicts.
        """
        import numpy as np

        def _coerce(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, tuple):
                return list(obj)
            return obj

        serializable: dict = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                serializable[k] = {str(kk): _coerce(vv) for kk, vv in v.items()}
            elif isinstance(v, list):
                serializable[k] = [
                    [_coerce(x) for x in item] if isinstance(item, (tuple, list))
                    else _coerce(item)
                    for item in v
                ]
            else:
                serializable[k] = _coerce(v)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        log.info("analytics_report_saved", path=output_path)