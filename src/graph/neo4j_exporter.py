"""
src/graph/neo4j_exporter.py
Exports the NetworkX DiGraph to:
  • nodes.csv          — neo4j-admin import ready
  • relationships.csv  — now includes `source` (llm/glirel) and `article_id`
  • graph.html         — PyVis interactive visualisation
    – GLiREL edges rendered as dashed blue lines (distinct from LLM edges)
    – Event nodes stay orange
    – All other node types get stable hash-based colours
"""

import csv
import hashlib
import networkx as nx

from src.utils.logger import get_logger

log = get_logger(__name__)


def _type_color(type_name: str) -> str:
    """Deterministic hex color from a type string (MD5 hash)."""
    h = hashlib.md5(type_name.encode()).hexdigest()
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    # Clamp to mid-range so colours aren't too dark / too light
    r = max(60, min(r, 210))
    g = max(60, min(g, 210))
    b = max(60, min(b, 210))
    return f"#{r:02x}{g:02x}{b:02x}"


class Neo4jExporter:
    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def export_nodes_csv(self, graph: nx.DiGraph, output_path: str) -> None:
        """
        Columns: id, name, type, mention_count, cluster_id, temporal_window
        Compatible with neo4j-admin database import full --nodes=...
        """
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id:ID", "name", "type:LABEL",
                "mention_count:INT", "cluster_id", "temporal_window",
            ])
            for node, data in graph.nodes(data=True):
                writer.writerow([
                    node,
                    data.get("name", ""),
                    data.get("type", "Unknown"),
                    data.get("mention_count", 0),
                    data.get("cluster_id", ""),
                    data.get("temporal_window", ""),
                ])
        log.info("nodes_exported", path=output_path, count=graph.number_of_nodes())

    def export_relationships_csv(self, graph: nx.DiGraph, output_path: str) -> None:
        """
        Columns: source, target, relation, confidence, event_id, article_id, edge_source
        `edge_source` is "llm" or "glirel" — useful for downstream filtering.
        Compatible with neo4j-admin database import full --relationships=...
        """
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                ":START_ID", ":END_ID", ":TYPE",
                "confidence:FLOAT", "event_id", "article_id", "edge_source",
            ])
            for u, v, data in graph.edges(data=True):
                writer.writerow([
                    u,
                    v,
                    data.get("relation", "RELATED_TO"),
                    round(float(data.get("confidence", 0.0)), 4),
                    data.get("event_id", ""),
                    data.get("article_id", ""),
                    data.get("source", "unknown"),
                ])
        log.info("relationships_exported", path=output_path,
                 count=graph.number_of_edges())

    # ------------------------------------------------------------------
    # PyVis HTML
    # ------------------------------------------------------------------

    def export_pyvis_html(self, graph: nx.DiGraph, output_path: str) -> None:
        try:
            from pyvis.network import Network
        except ImportError:
            log.error("pyvis_not_installed",
                      hint="pip install pyvis")
            return

        net = Network(
            height="900px", width="100%",
            directed=True,
            bgcolor="#ffffff",      # clean white background
            font_color="#1f2937",   # dark text for contrast
        )

        # Build type → colour map
        all_types = {
            data.get("type", "Unknown")
            for _, data in graph.nodes(data=True)
        }
        type_colors = {t: _type_color(t) for t in all_types}
        type_colors["Event"] = "#f97316"    # vivid orange
        type_colors["Unknown"] = "#9ca3af"  # neutral grey for unresolved GLiREL stubs

        # Nodes
        for node, data in graph.nodes(data=True):
            node_type = data.get("type", "Unknown")
            color = type_colors.get(node_type, "#9ca3af")
            mentions = data.get("mention_count", 0)
            size = 20 if node_type == "Event" else max(12, 10 + mentions * 1.6)

            aliases_str = ", ".join(data.get("aliases") or [])
            tooltip = (
                f"Type: {node_type}\n"
                f"Mentions: {mentions}\n"
                f"Aliases: {aliases_str or '—'}"
            )
            net.add_node(
                node,
                label=str(data.get("name", node))[:50],
                title=tooltip,
                color={
                    "background": color,
                    "border": "#111827",
                    "highlight": {"background": color, "border": "#000000"},
                    "hover": {"background": color, "border": "#000000"},
                },
                borderWidth=2,
                borderWidthSelected=4,
                size=size,
                font={"color": "#111827", "size": 16, "face": "arial"},
                shadow=True,
            )

        # Edges — thicker, darker, and more readable on white
        for u, v, data in graph.edges(data=True):
            rel = data.get("relation", "")
            src = data.get("source", "unknown")
            score = data.get("confidence", data.get("glirel_score", 0.0))

            if rel == "PARTICIPATES_IN":
                color = "#6b7280"
                dashes = True
                width = 1.8
            elif src == "glirel":
                color = "#2563eb"   # strong blue for GLiREL
                dashes = True
                width = 2.8
            else:
                color = "#374151"   # dark gray for LLM
                dashes = False
                width = 2.4

            tooltip = f"{rel}\nsource: {src}\nconf: {score:.2f}"
            if data.get("glirel_confirmed"):
                tooltip += "\n✓ GLiREL confirmed"

            net.add_edge(
                u, v,
                title=tooltip,
                label=rel[:25],
                color=color,
                arrows="to",
                dashes=dashes,
                width=width,
                shadow=True,
                selectionWidth=4,
                hoverWidth=3,
                smooth={"type": "dynamic", "roundness": 0.25},
            )

        net.set_options("""
        var options = {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -55,
              "centralGravity": 0.01,
              "springLength": 135,
              "springConstant": 0.07,
              "damping": 0.4,
              "avoidOverlap": 1
            },
            "maxVelocity": 45,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": { "iterations": 250 }
          },
          "nodes": {
            "shape": "dot",
            "shadow": true,
            "font": {
              "color": "#111827",
              "size": 16,
              "face": "arial"
            }
          },
          "edges": {
            "smooth": { "type": "dynamic", "roundness": 0.25 },
            "font": {
              "color": "#374151",
              "size": 12,
              "face": "arial",
              "strokeWidth": 3,
              "strokeColor": "#ffffff"
            }
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": true
          }
        }
        """)

        net.write_html(output_path)
        log.info("graph_html_exported", path=output_path,
                 nodes=graph.number_of_nodes(),
                 edges=graph.number_of_edges())
