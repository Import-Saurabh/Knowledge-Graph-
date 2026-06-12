import csv
import networkx as nx
from typing import Dict
from src.utils.logger import get_logger

log = get_logger(__name__)

class Neo4jExporter:
    def export_nodes_csv(self, graph: nx.DiGraph, output_path: str) -> None:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "type", "mention_count", "cluster_id", "temporal_window"])
            for node, data in graph.nodes(data=True):
                writer.writerow([
                    node,
                    data.get("name", ""),
                    data.get("type", ""),
                    data.get("mention_count", 0),
                    data.get("cluster_id", ""),
                    data.get("temporal_window", "")
                ])
        log.info("nodes_exported", path=output_path, count=graph.number_of_nodes())

    def export_relationships_csv(self, graph: nx.DiGraph, output_path: str) -> None:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["source", "target", "relation", "confidence", "event_id"])
            for u, v, data in graph.edges(data=True):
                writer.writerow([
                    u,
                    v,
                    data.get("relation", ""),
                    data.get("confidence", 0.0),
                    data.get("event_id", "")
                ])
        log.info("relationships_exported", path=output_path, count=graph.number_of_edges())

    def export_pyvis_html(self, graph: nx.DiGraph, output_path: str) -> None:
        try:
            from pyvis.network import Network
        except ImportError:
            log.error("pyvis_not_installed")
            return

        net = Network(height="900px", width="100%", directed=True, bgcolor="#ffffff", font_color="black")

        # Generate color palette dynamically
        type_colors = {}
        all_types = set()
        for node, data in graph.nodes(data=True):
            if "type" in data:
                all_types.add(data["type"])

        import hashlib
        for t in all_types:
            h = hashlib.md5(t.encode()).hexdigest()
            r = int(h[0:2], 16)
            g = int(h[2:4], 16)
            b = int(h[4:6], 16)
            type_colors[t] = f"#{r:02x}{g:02x}{b:02x}"

        type_colors["Event"] = "#FF6B35"  # Orange for events

        for node, data in graph.nodes(data=True):
            node_type = data.get("type", "Unknown")
            color = type_colors.get(node_type, "#999999")
            size = 10 + data.get("mention_count", 0) * 2
            if node_type == "Event":
                size = 20

            net.add_node(
                node,
                label=data.get("name", node)[:50],
                title=f"Type: {node_type}\nMentions: {data.get('mention_count', 0)}",
                color=color,
                size=size
            )

        for u, v, data in graph.edges(data=True):
            rel = data.get("relation", "")
            color = "#666666"
            if rel == "PARTICIPATES_IN":
                color = "#AAAAAA"
                dashes = True
            else:
                dashes = False

            net.add_edge(
                u, v,
                title=rel,
                label=rel[:20],
                color=color,
                arrows="to",
                dashes=dashes
            )

        net.set_options("""
        var options = {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {"iterations": 150}
          }
        }
        """)

        net.write_html(output_path)
        log.info("graph_html_exported", path=output_path)
