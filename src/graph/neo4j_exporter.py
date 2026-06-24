"""
src/graph/neo4j_exporter.py

Exports the NetworkX DiGraph to nodes.csv, relationships.csv, and graph.html.
Includes Wikidata fields and infobox source.
"""

import csv
import hashlib
import re
import networkx as nx

from src.utils.logger import get_logger

log = get_logger(__name__)

_COMMUNITY_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
    "#1f77b4", "#aec7e8", "#ffbb78", "#2ca02c", "#98df8a",
    "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b",
]

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _is_junk_node(node_id, data: dict) -> bool:
    """
    DROP-JUNK-NODES FIX (defensive copy of graph_builder._is_junk_node):
    skip nodes that never resolved to a real canonical entity — no UUID id,
    no recognised type, zero recorded mentions (e.g. "Yak-130", "53",
    "23 provinces"). GraphBuilder already prunes these before export, but
    the exporter checks again so it stays correct even if it's ever called
    on a graph that wasn't built/pruned through GraphBuilder.
    """
    if data.get("type") == "Event":
        return False
    has_uuid_id = bool(_UUID_RE.match(str(node_id)))
    is_unknown_type = data.get("type", "Unknown") == "Unknown"
    has_no_mentions = int(data.get("mention_count", 0) or 0) <= 0
    return is_unknown_type and not has_uuid_id and has_no_mentions


def _community_color(community_id: int | None) -> str:
    if community_id is None:
        return "#9ca3af"
    return _COMMUNITY_PALETTE[int(community_id) % len(_COMMUNITY_PALETTE)]

def _edge_width(confidence: float) -> float:
    return 1.5 + float(confidence) * 4.0

def _edge_opacity(confidence: float, threshold: float = 0.60) -> str:
    return "1.0" if float(confidence) >= threshold else "0.35"

class Neo4jExporter:
    EDGE_OPACITY_THRESHOLD: float = 0.60

    def export_nodes_csv(self, graph: nx.DiGraph, output_path: str) -> None:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id:ID", "name", "type:LABEL",
                "mention_count:INT", "cluster_id",
                "temporal_window", "community_id:INT",
                "wikidata_id", "description", "types", "url",
            ])
            exported = 0
            for node, data in graph.nodes(data=True):
                if _is_junk_node(node, data):
                    continue
                types_list = data.get("types", [])
                types_str = ";".join(types_list) if types_list else ""
                writer.writerow([
                    node,
                    data.get("name", ""),
                    data.get("type", "Unknown"),
                    data.get("mention_count", 0),
                    data.get("cluster_id", ""),
                    data.get("temporal_window", ""),
                    data.get("community_id", -1),
                    data.get("wikidata_id", ""),
                    data.get("description", ""),
                    types_str,
                    data.get("url", ""),
                ])
                exported += 1
        log.info("nodes_exported", path=output_path, count=exported,
                 skipped_junk=graph.number_of_nodes() - exported)

    def export_relationships_csv(self, graph: nx.DiGraph, output_path: str) -> None:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                ":START_ID", ":END_ID", ":TYPE",
                "confidence:FLOAT", "event_id", "article_id", "edge_source",
                "wikidata_property", "wikidata_date", "needs_review:BOOLEAN",
            ])
            exported = 0
            for u, v, data in graph.edges(data=True):
                if _is_junk_node(u, graph.nodes[u]) or _is_junk_node(v, graph.nodes[v]):
                    continue
                writer.writerow([
                    u,
                    v,
                    data.get("relation", "RELATED_TO"),
                    round(float(data.get("confidence", 0.0)), 4),
                    data.get("event_id", ""),
                    data.get("article_id", ""),
                    data.get("source", "unknown"),
                    data.get("wikidata_property", ""),
                    data.get("wikidata_date", ""),
                    data.get("needs_review", False),
                ])
                exported += 1
        log.info("relationships_exported", path=output_path, count=exported,
                 skipped_junk=graph.number_of_edges() - exported)

    def export_pyvis_html(self, graph: nx.DiGraph, output_path: str,
                          top_n_nodes: int | None = None) -> None:
        try:
            from pyvis.network import Network
        except ImportError:
            log.error("pyvis_not_installed", hint="pip install pyvis")
            return

        clean_nodes = [
            n for n, data in graph.nodes(data=True) if not _is_junk_node(n, data)
        ]
        render_graph = graph.subgraph(clean_nodes).copy() \
            if len(clean_nodes) != graph.number_of_nodes() else graph

        if top_n_nodes is not None and render_graph.number_of_nodes() > top_n_nodes:
            top_nodes = sorted(
                render_graph.nodes(), key=lambda n: render_graph.degree(n), reverse=True
            )[:top_n_nodes]
            render_graph = render_graph.subgraph(top_nodes).copy()
            log.info("html_export_subgraph",
                     total_nodes=graph.number_of_nodes(),
                     rendered_nodes=render_graph.number_of_nodes(),
                     rendered_edges=render_graph.number_of_edges())

        net = Network(height="950px", width="100%",
                      directed=True, bgcolor="#f8fafc", font_color="#1f2937")

        for node, data in render_graph.nodes(data=True):
            node_type = data.get("type", "Unknown")
            community = data.get("community_id", None)
            mentions = data.get("mention_count", 0)
            if node_type == "Event":
                color = "#f97316"
                size = 22
            else:
                color = _community_color(community)
                size = max(12, 10 + mentions * 1.6)

            aliases_str = ", ".join(data.get("aliases") or [])
            tooltip_lines = [
                f"Type: {node_type}",
                f"Community: {community if community is not None else '—'}",
                f"Mentions: {mentions}",
                f"Aliases: {aliases_str or '—'}",
            ]
            if data.get("wikidata_id"):
                tooltip_lines.append(f"Wikidata: {data['wikidata_id']}")
            if data.get("description"):
                tooltip_lines.append(f"Description: {data['description']}")
            if data.get("types"):
                tooltip_lines.append(f"Types: {', '.join(data['types'])}")
            if data.get("url"):
                tooltip_lines.append(f"URL: {data['url']}")
            tooltip = "\n".join(tooltip_lines)

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
                group=community if community is not None else -1,
            )

        for u, v, data in render_graph.edges(data=True):
            rel = data.get("relation", "")
            src = data.get("source", "unknown")
            confidence = float(data.get("confidence", data.get("glirel_score", 0.0)))
            width = _edge_width(confidence)
            opacity = _edge_opacity(confidence, self.EDGE_OPACITY_THRESHOLD)

            if rel == "PARTICIPATES_IN":
                hex_color = "#6b7280"
                dashes = True
                width = 1.5
            elif src == "glirel":
                hex_color = "#2563eb"
                dashes = True
            else:
                hex_color = "#374151"
                dashes = False

            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            edge_color = f"rgba({r},{g},{b},{opacity})"

            tooltip_lines = [
                f"{rel}",
                f"source: {src}",
                f"confidence: {confidence:.2f}",
            ]
            if data.get("glirel_confirmed"):
                tooltip_lines.append("✓ GLiREL confirmed")
            if data.get("wikidata_property"):
                tooltip_lines.append(f"Wikidata property: {data['wikidata_property']}")
            if data.get("wikidata_date"):
                tooltip_lines.append(f"Date: {data['wikidata_date']}")
            if data.get("needs_review"):
                tooltip_lines.append("⚠️ Needs review")
            tooltip = "\n".join(tooltip_lines)

            net.add_edge(
                u, v,
                title=tooltip,
                label=rel[:25] if confidence >= self.EDGE_OPACITY_THRESHOLD else "",
                color=edge_color,
                arrows="to",
                dashes=dashes,
                width=width,
                shadow=False,
                selectionWidth=4,
                hoverWidth=3,
                smooth={"type": "dynamic", "roundness": 0.25},
            )

        net.set_options("""
        var options = {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -80,
              "centralGravity": 0.015,
              "springLength": 160,
              "springConstant": 0.06,
              "damping": 0.45,
              "avoidOverlap": 1
            },
            "maxVelocity": 40,
            "solver": "forceAtlas2Based",
            "timestep": 0.30,
            "stabilization": {
              "enabled": true,
              "iterations": 300,
              "updateInterval": 25,
              "fit": true
            }
          },
          "nodes": {
            "shape": "dot",
            "shadow": true,
            "font": { "color": "#111827", "size": 16, "face": "arial" }
          },
          "edges": {
            "smooth": { "type": "dynamic", "roundness": 0.25 },
            "font": {
              "color": "#374151",
              "size": 11,
              "face": "arial",
              "strokeWidth": 3,
              "strokeColor": "#f8fafc"
            }
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 80,
            "navigationButtons": true,
            "keyboard": true,
            "hideEdgesOnDrag": true,
            "hideEdgesOnZoom": false
          },
          "layout": { "improvedLayout": true }
        }
        """)
        net.write_html(output_path)
        log.info("graph_html_exported", path=output_path,
                 nodes=render_graph.number_of_nodes(),
                 edges=render_graph.number_of_edges(),
                 top_n_cap=top_n_nodes)