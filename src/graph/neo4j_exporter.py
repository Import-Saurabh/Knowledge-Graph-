"""
src/graph/neo4j_exporter.py

Exports the NetworkX DiGraph to:
  • nodes.csv          — neo4j-admin import ready (+ community_id column)
  • relationships.csv  — includes `source` (llm/glirel) and `article_id`
  • graph.html         — PyVis interactive visualisation

Changes vs previous version:
  - Node colour now driven by `community_id` (assigned by GraphBuilder's
    Louvain pass) so clusters stand out visually.  Event nodes stay orange.
  - Edge width and opacity are confidence-scaled:
      width  = 1.5 + confidence * 4.0   (range ≈ 2 – 5.5 px)
  - Low-confidence edges (< EDGE_OPACITY_THRESHOLD) rendered semi-transparent
    to reduce clutter without removing them from the export.
  - GLiREL edges still dashed-blue; LLM edges solid dark-gray.
  - PARTICIPATES_IN edges remain thin dashed-gray (structural, not semantic).
  - Physics preset tightened to reduce spaghetti on large graphs.
  - export_pyvis_html() accepts an optional `top_n_nodes` param:
    when set, only the top-N nodes by degree are rendered in the HTML
    (CSV exports always contain the full graph).
"""

import csv
import hashlib
import networkx as nx

from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

# Palette for up to 20 communities.  Additional communities wrap around.
_COMMUNITY_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
    "#1f77b4", "#aec7e8", "#ffbb78", "#2ca02c", "#98df8a",
    "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b",
]


def _community_color(community_id: int | None) -> str:
    if community_id is None:
        return "#9ca3af"
    return _COMMUNITY_PALETTE[int(community_id) % len(_COMMUNITY_PALETTE)]


def _type_color(type_name: str) -> str:
    """Deterministic hex color from a type string (MD5 hash) — used as fallback."""
    h = hashlib.md5(type_name.encode()).hexdigest()
    r = max(60, min(int(h[0:2], 16), 210))
    g = max(60, min(int(h[2:4], 16), 210))
    b = max(60, min(int(h[4:6], 16), 210))
    return f"#{r:02x}{g:02x}{b:02x}"


def _edge_width(confidence: float) -> float:
    """Map [0, 1] confidence → [1.5, 5.5] edge width."""
    return 1.5 + float(confidence) * 4.0


def _edge_opacity(confidence: float, threshold: float = 0.60) -> str:
    """
    Return rgba opacity string.  Edges below threshold are semi-transparent
    (0.35) so high-confidence edges dominate visually.
    """
    return "1.0" if float(confidence) >= threshold else "0.35"


class Neo4jExporter:
    EDGE_OPACITY_THRESHOLD: float = 0.60  # edges below this are faded

    # ------------------------------------------------------------------
    # CSV export  (always full graph — no pruning here)
    # ------------------------------------------------------------------

    def export_nodes_csv(self, graph: nx.DiGraph, output_path: str) -> None:
        """
        Columns: id, name, type, mention_count, cluster_id,
                 temporal_window, community_id
        Compatible with neo4j-admin database import full --nodes=...
        """
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id:ID", "name", "type:LABEL",
                "mention_count:INT", "cluster_id",
                "temporal_window", "community_id:INT",
            ])
            for node, data in graph.nodes(data=True):
                writer.writerow([
                    node,
                    data.get("name", ""),
                    data.get("type", "Unknown"),
                    data.get("mention_count", 0),
                    data.get("cluster_id", ""),
                    data.get("temporal_window", ""),
                    data.get("community_id", -1),
                ])
        log.info("nodes_exported", path=output_path, count=graph.number_of_nodes())

    def export_relationships_csv(self, graph: nx.DiGraph, output_path: str) -> None:
        """
        Columns: source, target, relation, confidence, event_id,
                 article_id, edge_source
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

    def export_pyvis_html(
        self,
        graph: nx.DiGraph,
        output_path: str,
        top_n_nodes: int | None = None,
    ) -> None:
        """
        Render an interactive PyVis graph.

        Parameters
        ----------
        graph        : the full NetworkX DiGraph from GraphBuilder
        output_path  : where to write graph.html
        top_n_nodes  : if set, only the top-N nodes by degree are included
                       in the HTML visualisation (CSV exports are unaffected).
                       Useful for keeping large graphs readable; recommended
                       value: 150–300.
        """
        try:
            from pyvis.network import Network
        except ImportError:
            log.error("pyvis_not_installed", hint="pip install pyvis")
            return

        # ── Optionally restrict to top-N nodes by degree ────────────────
        render_graph = graph
        if top_n_nodes is not None and graph.number_of_nodes() > top_n_nodes:
            top_nodes = sorted(
                graph.nodes(), key=lambda n: graph.degree(n), reverse=True
            )[:top_n_nodes]
            render_graph = graph.subgraph(top_nodes).copy()
            log.info(
                "html_export_subgraph",
                total_nodes=graph.number_of_nodes(),
                rendered_nodes=render_graph.number_of_nodes(),
                rendered_edges=render_graph.number_of_edges(),
            )

        net = Network(
            height="950px", width="100%",
            directed=True,
            bgcolor="#f8fafc",      # very light gray — easier on the eyes than white
            font_color="#1f2937",
        )

        # ── Nodes ────────────────────────────────────────────────────────
        for node, data in render_graph.nodes(data=True):
            node_type   = data.get("type", "Unknown")
            community   = data.get("community_id", None)
            mentions    = data.get("mention_count", 0)

            # Event nodes stay vivid orange; others coloured by community
            if node_type == "Event":
                color = "#f97316"
                size  = 22
            else:
                color = _community_color(community)
                size  = max(12, 10 + mentions * 1.6)

            aliases_str = ", ".join(data.get("aliases") or [])
            tooltip = (
                f"Type: {node_type}\n"
                f"Community: {community if community is not None else '—'}\n"
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
                    "hover":     {"background": color, "border": "#000000"},
                },
                borderWidth=2,
                borderWidthSelected=4,
                size=size,
                font={"color": "#111827", "size": 16, "face": "arial"},
                shadow=True,
                group=community if community is not None else -1,
            )

        # ── Edges (confidence-scaled width + opacity) ────────────────────
        for u, v, data in render_graph.edges(data=True):
            rel        = data.get("relation", "")
            src        = data.get("source", "unknown")
            confidence = float(data.get("confidence", data.get("glirel_score", 0.0)))
            width      = _edge_width(confidence)
            opacity    = _edge_opacity(confidence, self.EDGE_OPACITY_THRESHOLD)

            if rel == "PARTICIPATES_IN":
                hex_color = "#6b7280"
                dashes    = True
                width     = 1.5
            elif src == "glirel":
                hex_color = "#2563eb"   # strong blue for GLiREL
                dashes    = True
            else:
                hex_color = "#374151"   # dark gray for LLM
                dashes    = False

            # Apply opacity by converting to rgba
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            edge_color = f"rgba({r},{g},{b},{opacity})"

            tooltip = (
                f"{rel}\n"
                f"source: {src}\n"
                f"confidence: {confidence:.2f}"
            )
            if data.get("glirel_confirmed"):
                tooltip += "\n✓ GLiREL confirmed"

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
          "layout": {
            "improvedLayout": true
          }
        }
        """)

        net.write_html(output_path)
        log.info(
            "graph_html_exported",
            path=output_path,
            nodes=render_graph.number_of_nodes(),
            edges=render_graph.number_of_edges(),
            top_n_cap=top_n_nodes,
        )