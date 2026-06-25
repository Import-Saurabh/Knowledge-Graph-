"""
src/graph/graph_builder.py

Builds a NetworkX DiGraph from GLiREL triples and infobox triples.

LLM relation extraction has been removed. The only edge sources are:
  • GLiREL        — zero-shot relation triples (source="glirel")
  • Infobox       — Wikipedia infobox facts   (source="infobox")

Event nodes are created from EventModel cluster metadata (temporal_window +
cluster_id) — no LLM-generated event labels.

Edge confidence thresholds
--------------------------
  GLIREL_SCORE_THRESHOLD = 0.50   (edges below this are dropped before
                                    add_glirel_edge is even called)

Node pruning
------------
Nodes with degree < MIN_NODE_DEGREE and no UUID canonical id are removed
after graph construction. Event nodes are always kept.
"""

import re
import networkx as nx
from typing import List, Dict, Optional, Tuple

from src.models.event import EventModel
from src.models.entity import CanonicalEntity
from src.utils.logger import get_logger

log = get_logger(__name__)

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _is_junk_node(node_id: str, data: Dict) -> bool:
    """
    A node is junk if it never went through entity resolution:
    no UUID canonical id, no recognised type, zero recorded mentions.
    These are stub nodes created purely as edge endpoints for a GLiREL triple
    whose head/tail never matched anything in name_to_id.
    Event nodes are always kept.
    """
    if data.get("type") == "Event":
        return False
    has_uuid_id      = bool(_UUID_RE.match(str(node_id)))
    is_unknown_type  = data.get("type", "Unknown") == "Unknown"
    has_no_mentions  = int(data.get("mention_count", 0) or 0) <= 0
    return is_unknown_type and not has_uuid_id and has_no_mentions


class GraphBuilder:
    GLIREL_SCORE_THRESHOLD: float = 0.50
    MIN_NODE_DEGREE:        int   = 2

    def __init__(self):
        self.graph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Node builders
    # ------------------------------------------------------------------

    def add_event_node(self, event: EventModel) -> None:
        """
        Create an Event node labeled from cluster metadata.
        Label format: "Event <window> C<cluster_id>"
        e.g. "Event 2024-W03 C2"
        """
        label = f"Event {event.temporal_window} C{event.cluster_id}"
        self.graph.add_node(
            event.event_id,
            name=label,
            type="Event",
            cluster_id=event.cluster_id,
            temporal_window=event.temporal_window,
            article_count=len(event.article_ids),
        )

    def add_entity_node(
        self,
        canonical_entity: CanonicalEntity,
        extra_attrs: Optional[Dict] = None,
    ) -> None:
        attrs = {
            "name":          canonical_entity.canonical_name,
            "type":          canonical_entity.entity_type,
            "mention_count": canonical_entity.mention_count,
            "aliases":       canonical_entity.aliases,
        }
        if extra_attrs:
            attrs.update(extra_attrs)
        self.graph.add_node(canonical_entity.canonical_id, **attrs)

    def _ensure_entity_stub(self, node_id: str, name: str) -> None:
        """Add a placeholder node if node_id is not yet in the graph."""
        if node_id not in self.graph:
            self.graph.add_node(
                node_id,
                name=name,
                type="Unknown",
                mention_count=0,
                aliases=[],
            )

    # ------------------------------------------------------------------
    # Edge builders
    # ------------------------------------------------------------------

    def _add_glirel_edge(
        self,
        head_id:  str,
        tail_id:  str,
        relation: str,
        score:    float,
        article_id: str,
        original_relation: Optional[str] = None,
    ) -> None:
        """
        Add a GLiREL edge. If the edge already exists (e.g. from a prior
        article), annotate it as multi-confirmed rather than duplicating.
        """
        if score < self.GLIREL_SCORE_THRESHOLD:
            return
        if self.graph.has_edge(head_id, tail_id):
            existing = self.graph[head_id][tail_id]
            # Upgrade confidence to the higher of the two scores
            if score > existing.get("confidence", 0.0):
                existing["confidence"] = score
            existing["glirel_confirmed"] = True
            existing.setdefault("glirel_score", score)
            return
        self.graph.add_edge(
            head_id, tail_id,
            relation=relation,
            confidence=score,
            article_id=article_id,
            original_relation=original_relation or relation,
            source="glirel",
            glirel_confirmed=True,
            glirel_score=score,
        )

    # ------------------------------------------------------------------
    # Main builder
    # ------------------------------------------------------------------

    def build_from_relations(
        self,
        events:            List[EventModel],
        entity_map:        Optional[Dict[str, CanonicalEntity]] = None,
        glirel_triples:    Optional[List[dict]]                 = None,
        entity_enrichment: Optional[Dict[str, Dict]]            = None,
        infobox_triples:   Optional[List[Tuple[str, str, str]]] = None,
        relation_ontology=None,
    ) -> nx.DiGraph:
        """
        Build the knowledge graph from GLiREL triples + infobox facts.

        Parameters
        ----------
        events            : EventModel list from EventClusterer
        entity_map        : {canonical_id: CanonicalEntity} from EntityResolver
        glirel_triples    : flat list of {head, relation, tail, score, article_id}
                            dicts from EntityExtractor.extract_with_glirel()
        entity_enrichment : {entity_name: {wikidata_id, description, ...}}
                            from WikidataValidator (optional)
        infobox_triples   : [(subject, relation, object), ...] from
                            InfoboxExtractor (optional)
        relation_ontology : RelationOntologyManager instance for label
                            normalisation (optional — raw labels used if None)
        """
        # ── Build name → canonical_id lookup ────────────────────────────────
        name_to_id: Dict[str, str] = {}
        if entity_map:
            for entity in entity_map.values():
                name_to_id[entity.canonical_name.lower()] = entity.canonical_id
                for alias in (entity.aliases or []):
                    name_to_id[alias.lower()] = entity.canonical_id

        # ── Event nodes (from cluster metadata) ─────────────────────────────
        for event in events:
            self.add_event_node(event)
        log.info("event_nodes_added", count=len(events))

        # ── Canonical entity nodes ───────────────────────────────────────────
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

        # ── GLiREL triples ──────────────────────────────────────────────────
        glirel_added   = 0
        glirel_merged  = 0
        glirel_dropped = 0
        if glirel_triples:
            for t in glirel_triples:
                head_raw     = (t.get("head")     or "").strip()
                tail_raw     = (t.get("tail")     or "").strip()
                relation_raw = (t.get("relation") or "").strip()
                score        = float(t.get("score", 0.0))
                art_id       = t.get("article_id", "")

                if not head_raw or not tail_raw or not relation_raw:
                    continue
                if head_raw.lower() == tail_raw.lower():
                    continue
                if score < self.GLIREL_SCORE_THRESHOLD:
                    glirel_dropped += 1
                    continue

                # Normalise relation label through the ontology (if available)
                if relation_ontology:
                    relation_canonical = relation_ontology.normalize_relation(relation_raw)
                    # Fallback: if normalization returns empty, use raw phrase
                    if not relation_canonical:
                        relation_canonical = relation_raw
                else:
                    relation_canonical = relation_raw

                head_id = name_to_id.get(head_raw.lower(), head_raw)
                tail_id = name_to_id.get(tail_raw.lower(), tail_raw)
                self._ensure_entity_stub(head_id, head_raw)
                self._ensure_entity_stub(tail_id, tail_raw)

                existed = self.graph.has_edge(head_id, tail_id)
                self._add_glirel_edge(
                    head_id, tail_id,
                    relation_canonical, score, art_id,
                    original_relation=relation_raw,
                )
                if existed:
                    glirel_merged += 1
                else:
                    glirel_added += 1

            log.info(
                "glirel_triples_processed",
                added=glirel_added,
                merged=glirel_merged,
                dropped=glirel_dropped,
            )

        # ── Infobox triples (static Wikipedia facts) ────────────────────────
        if infobox_triples:
            infobox_added = 0
            for subject, relation, obj in infobox_triples:
                subj_id = name_to_id.get(subject.lower(), subject)
                obj_id  = name_to_id.get(obj.lower(), obj)
                self._ensure_entity_stub(subj_id, subject)
                self._ensure_entity_stub(obj_id, obj)
                if not self.graph.has_edge(subj_id, obj_id):
                    self.graph.add_edge(
                        subj_id, obj_id,
                        relation=relation,
                        confidence=0.95,
                        source="infobox",
                        article_id="",
                        original_relation=relation,
                    )
                    infobox_added += 1
            log.info("infobox_triples_added", count=infobox_added)

        # ── Wikidata enrichment on entity nodes ─────────────────────────────
        self._enrich_nodes_with_wikidata(entity_enrichment or {})

        # ── Prune low-degree junk nodes ──────────────────────────────────────
        pruned = self._prune_low_degree_nodes()

        # ── Community detection ──────────────────────────────────────────────
        n_communities = self._assign_communities()

        log.info(
            "graph_built",
            nodes=self.graph.number_of_nodes(),
            edges=self.graph.number_of_edges(),
            communities=n_communities,
            pruned_nodes=pruned,
        )
        return self.graph

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _prune_low_degree_nodes(self) -> int:
        to_remove = [
            n for n, data in self.graph.nodes(data=True)
            if data.get("type") != "Event"
            and (
                self.graph.degree(n) < self.MIN_NODE_DEGREE
                or _is_junk_node(n, data)
            )
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

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        sources: Dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            src = data.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        return {
            "nodes":      self.graph.number_of_nodes(),
            "edges":      self.graph.number_of_edges(),
            "by_source":  sources,
        }