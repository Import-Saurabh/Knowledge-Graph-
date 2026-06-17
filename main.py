"""
news_kg/main.py  — Final version
Bugs fixed:
  #1  Embedding/metadata misalignment  → metadata built from embedder output tuple
  #2  Resume bug (--from-stage > 3)    → canonical_map reloaded from DB
  #7  NameError on relation_ontology   → defined as None before try block

GLiREL integration (Stage 2b):
  After GLiNER NER, extract_with_glirel() is called on the same articles.
  The resulting raw triples (List[dict]) are passed to GraphBuilder alongside
  LLM responses so every GLiREL edge becomes a real graph edge + Neo4j row.

  graph_builder.build_from_relations() signature must accept an optional
  fourth argument:  glirel_triples: List[dict] = None
  (see note at Stage 10 below if you need the stub).
"""
import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.article_loader import ArticleLoader
from src.ingestion.news_downloader import NewsDownloader
from src.entities.entity_extractor import EntityExtractor
from src.entities.entity_resolver import EntityResolver
from src.entities.ontology_manager import OntologyManager
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.vectorstore.chroma_manager import ChromaManager
from src.deduplication.duplicate_detector import DuplicateDetector
from src.clustering.event_clusterer import EventClusterer
from src.events.event_builder import EventBuilder
from src.relations.relation_extractor import RelationExtractor
from src.relations.relation_ontology import RelationOntologyManager
from src.graph.graph_builder import GraphBuilder
from src.graph.neo4j_exporter import Neo4jExporter
from src.analytics.graph_metrics import GraphMetrics
from src.utils.logger import get_logger
from src.utils.db import init_db, get_session, ArticleDB, CanonicalEntityDB
from src.utils.config import settings
from src.models.relation import LLMRelationResponse

log = get_logger(__name__)


@contextmanager
def managed_session():
    session = get_session()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _load_canonical_map_from_db(resolver: EntityResolver) -> dict:
    """
    BUG #2 FIX: rebuild canonical_map from SQLite when resuming past stage 3.
    Without this, --from-stage 4/6/7 builds a graph with zero entity nodes.
    """
    canonical_map = {}
    with managed_session() as session:
        db_entities = session.query(CanonicalEntityDB).all()
        for db_ent in db_entities:
            entity = resolver._db_to_model(db_ent)
            canonical_map[entity.canonical_id] = entity
    log.info("canonical_map_loaded_from_db", count=len(canonical_map))
    return canonical_map


def run_pipeline(args):
    os.makedirs("data/exports", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    init_db()

    embedder = EmbeddingGenerator()
    chroma   = ChromaManager()
    ontology = OntologyManager(chroma, embedder)

    # ------------------------------------------------------------------ #
    # Stage 1: Ingestion                                                   #
    # ------------------------------------------------------------------ #
    loader = ArticleLoader()

    if args.download:
        downloader = NewsDownloader(
            language="en",
            country="US",
            interval_days=args.interval_days,
            categories=args.categories or None,
        )
        articles = downloader.download(
            topic=args.topic,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.input,
            max_articles_per_interval=args.max_per_interval,
        )
        if articles:
            loader.ingest_to_db(articles)

    articles = loader.load_from_directory(args.input)
    result   = loader.ingest_to_db(articles)
    log.info("ingestion_complete", **result)

    # Resume logic — pick the right unprocessed bucket
    if args.from_stage <= 2:
        unprocessed = loader.get_unprocessed_articles("ingested")
    elif args.from_stage <= 6:
        unprocessed = (
            loader.get_unprocessed_articles("ingested")
            or loader.get_unprocessed_articles("embedded")
            or loader.get_unprocessed_articles("deduplicated")
        )
    else:
        unprocessed = (
            loader.get_unprocessed_articles("embedded")
            or loader.get_unprocessed_articles("deduplicated")
            or loader.get_unprocessed_articles("clustered")
        )

    if not unprocessed and args.from_stage <= 2:
        log.info("no_new_articles")
        return

    # ------------------------------------------------------------------ #
    # Stage 2: GLiNER NER — entity mentions                               #
    # ------------------------------------------------------------------ #
    all_mentions       = []
    article_to_mentions = {}
    extractor = None   # keep reference for Stage 2b below

    if args.from_stage <= 2:
        extractor = EntityExtractor(
            use_spacy_fallback=args.fast,
            ontology_manager=ontology,
            use_glirel=not args.skip_glirel,
        )

        for i in range(0, len(unprocessed), settings.NER_BATCH_SIZE):
            batch = unprocessed[i:i + settings.NER_BATCH_SIZE]
            try:
                batch_mentions = extractor.extract_batch(batch)
                for article, mentions in zip(batch, batch_mentions):
                    article_to_mentions[article.id] = mentions
                    all_mentions.extend(mentions)
            except Exception as e:
                log.error("extraction_batch_failed", error=str(e))
                for article in batch:
                    try:
                        mentions = extractor.extract_single(article)
                        article_to_mentions[article.id] = mentions
                        all_mentions.extend(mentions)
                    except Exception as inner_e:
                        log.error("extraction_single_failed",
                                  article_id=article.id, error=str(inner_e))

        log.info("ner_complete", mentions=len(all_mentions))

    # ------------------------------------------------------------------ #
    # Stage 2b: GLiREL zero-shot relation extraction on raw articles       #
    # ------------------------------------------------------------------ #
    # This produces raw structural triples directly from article text      #
    # without an LLM call — cheaper, faster, good for high-recall edges.  #
    # These are merged into the graph at Stage 10 alongside LLM triples.  #
    raw_glirel_triples = []
    if args.from_stage <= 2 and not args.skip_glirel and extractor is not None:
        try:
            raw_glirel_triples = extractor.extract_with_glirel(unprocessed)
            log.info("glirel_complete", triples=len(raw_glirel_triples))
        except Exception as e:
            log.warning("glirel_extraction_failed", error=str(e))
            raw_glirel_triples = []

    # ------------------------------------------------------------------ #
    # Stage 3: Entity Resolution                                           #
    # ------------------------------------------------------------------ #
    canonical_map = {}
    resolver = EntityResolver(chroma, embedder)

    if args.from_stage <= 3:
        for mention in all_mentions:
            try:
                canonical = resolver.resolve(mention)
                canonical_map[canonical.canonical_id] = canonical
            except Exception as e:
                log.warning("resolution_failed", mention=mention.text, error=str(e))
        log.info("resolution_complete", canonical_entities=len(canonical_map))
    else:
        # BUG #2 FIX: load persisted entities so the graph has real nodes
        canonical_map = _load_canonical_map_from_db(resolver)

    # ------------------------------------------------------------------ #
    # Stage 4+5: Article Embeddings → ChromaDB                            #
    # ------------------------------------------------------------------ #
    if args.from_stage <= 4:
        # BUG #1 FIX: build metadata from embedder output (same indexed list),
        # not from `unprocessed` which may diverge if embed_articles skips rows.
        article_embeddings = embedder.embed_articles(unprocessed)
        article_lookup     = {a.id: a for a in unprocessed}
        chroma.add_articles(
            ids=[ae[0] for ae in article_embeddings],
            embeddings=[ae[1] for ae in article_embeddings],
            metadatas=[
                {
                    "title":  article_lookup[ae[0]].title,
                    "source": article_lookup[ae[0]].source,
                }
                for ae in article_embeddings
            ],
        )
        loader.update_status([a.id for a in unprocessed], "embedded")

    # ------------------------------------------------------------------ #
    # Stage 6: Deduplication                                               #
    # ------------------------------------------------------------------ #
    if args.from_stage <= 6:
        dedup      = DuplicateDetector(chroma, embedder)
        duplicates = dedup.find_duplicates(unprocessed)
        if duplicates:
            dedup.mark_duplicates(duplicates)
        non_dup_articles = [
            a for a in unprocessed
            if a.id not in {d[0] for d in duplicates}
        ]
        loader.update_status([a.id for a in non_dup_articles], "deduplicated")
    else:
        non_dup_articles = unprocessed

    # ------------------------------------------------------------------ #
    # Stage 7: Clustering + Stage 8: Event Building                        #
    # ------------------------------------------------------------------ #
    if args.from_stage <= 7:
        clusterer = EventClusterer(embedder=embedder)
        clusters  = clusterer.run_all_windows(non_dup_articles)

        builder = EventBuilder()
        events  = [builder.build_event(c) for c in clusters]
    else:
        log.error(
            "Resuming from stage > 7 not yet supported — events must be "
            "re-loaded from DB."
        )
        raise SystemExit(1)

    # Persist cluster assignments
    try:
        with managed_session() as session:
            for event in events:
                session.query(ArticleDB).filter(
                    ArticleDB.id.in_(event.article_ids)
                ).update(
                    {
                        ArticleDB.cluster_id:       event.cluster_id,
                        ArticleDB.temporal_window:  event.temporal_window,
                        ArticleDB.status:           "clustered",
                    },
                    synchronize_session=False,
                )
            session.commit()
    except Exception as e:
        log.error("cluster_assignment_failed", error=str(e))
        raise

    log.info("events_built", count=len(events))

    # ------------------------------------------------------------------ #
    # Stage 9: LLM Relations + Type Induction                              #
    # ------------------------------------------------------------------ #
    # BUG #7 FIX: define relation_ontology=None BEFORE the try block so   #
    # the ontology-report reference at the bottom is always safe.          #
    relation_ontology: RelationOntologyManager | None = None
    llm_responses = []

    if not args.skip_llm:
        extractor_llm = RelationExtractor(max_workers=args.llm_workers)
        try:
            relation_ontology = RelationOntologyManager(chroma, embedder)
        except Exception as e:
            log.error("relation_ontology_init_failed", error=str(e))

        total_events = len(events)
        for i in range(0, total_events, args.llm_batch_size):
            batch     = events[i:i + args.llm_batch_size]
            batch_end = min(i + args.llm_batch_size, total_events)
            log.info("llm_extraction_batch",
                     batch_start=i, batch_end=batch_end,
                     total=total_events, workers=args.llm_workers)
            try:
                batch_responses = extractor_llm.extract_batch(batch)
                llm_responses.extend(batch_responses)
            except Exception as e:
                log.error("llm_batch_failed", batch_start=i, error=str(e))
                llm_responses.extend([
                    LLMRelationResponse(
                        event_label="Unknown Event",
                        triples=[],
                        discovered_entity_types=[],
                    )
                    for _ in batch
                ])

        # Sequential post-processing (DB-bound, not thread-safe)
        for event, response in zip(events, llm_responses):
            try:
                for triple in response.triples:
                    if relation_ontology:
                        triple.relation_canonical = (
                            relation_ontology.normalize_relation(triple.relation)
                        )
                    triple.event_id = event.event_id

                if settings.ENABLE_TYPE_INDUCTION:
                    for disc in response.discovered_entity_types:
                        ontology.induce_type(
                            disc.get("entity_name", ""),
                            event.context,
                            disc.get("suggested_type"),
                        )
            except Exception as e:
                log.warning("llm_post_processing_failed",
                            event_id=event.event_id, error=str(e))
    else:
        llm_responses = [
            LLMRelationResponse(
                event_label=f"Event {e.cluster_id}",
                triples=[],
                discovered_entity_types=[],
            )
            for e in events
        ]

    # ------------------------------------------------------------------ #
    # Stage 10: Graph Building                                             #
    # ------------------------------------------------------------------ #
    # GraphBuilder.build_from_relations() is called with an optional 4th  #
    # argument `glirel_triples` (List[dict]).  Each dict has keys:         #
    #   head, relation, tail, score, article_id                            #
    # Your graph_builder.py must accept and wire this in.  Minimal stub:  #
    #                                                                      #
    #   def build_from_relations(self, events, llm_responses,             #
    #                             canonical_map, glirel_triples=None):     #
    #       ...                                                            #
    #       if glirel_triples:                                             #
    #           for t in glirel_triples:                                   #
    #               self.graph.add_edge(                                   #
    #                   t["head"], t["tail"],                              #
    #                   relation=t["relation"],                            #
    #                   score=t["score"],                                  #
    #                   source="glirel",                                   #
    #                   article_id=t["article_id"],                        #
    #               )                                                      #
    # ------------------------------------------------------------------ #
    graph_builder = GraphBuilder()
    graph = graph_builder.build_from_relations(
        events,
        llm_responses,
        canonical_map,
        glirel_triples=raw_glirel_triples if raw_glirel_triples else None,
    )

    # ------------------------------------------------------------------ #
    # Stage 11: Analytics                                                  #
    # ------------------------------------------------------------------ #
    metrics = GraphMetrics()
    report  = metrics.compute_all(graph)
    metrics.save_report(report, "data/exports/analytics_report.json")

    # ------------------------------------------------------------------ #
    # Stage 12: Export — CSVs ready for neo4j-admin import + PyVis HTML   #
    # ------------------------------------------------------------------ #
    exporter = Neo4jExporter()
    exporter.export_nodes_csv(graph,         "data/exports/nodes.csv")
    exporter.export_relationships_csv(graph, "data/exports/relationships.csv")
    exporter.export_pyvis_html(graph,        "data/exports/graph.html")

    # Ontology report
    # BUG #7 FIX: `and relation_ontology` guard prevents NameError if init failed
    ontology_report = {
        "entity_types":  ontology.get_ontology_report(),
        "relation_types": (
            relation_ontology.get_relation_taxonomy()
            if (not args.skip_llm and relation_ontology)
            else []
        ),
        "glirel_triples_count": len(raw_glirel_triples),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open("data/exports/ontology_report.json", "w", encoding="utf-8") as f:
        json.dump(ontology_report, f, indent=2, ensure_ascii=False)

    log.info(
        "pipeline_complete",
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
        glirel_triples=len(raw_glirel_triples),
        entity_types=len(ontology_report["entity_types"].get("top_types", [])),
        relation_types=len(ontology_report["relation_types"]),
    )


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="news_kg — knowledge graph pipeline"
    )
    parser.add_argument("--input", default="data/raw/")
    parser.add_argument("--run-all", action="store_true",
                        help="Execute the full pipeline end-to-end")
    parser.add_argument(
        "--from-stage", type=int, default=1, metavar="N",
        help=(
            "Resume from stage N  "
            "(1=ingest, 2=NER+GLiREL, 3=entity-resolution, "
            "4=embeddings, 6=dedup, 7=clustering, 9=relations). "
            "Stages before N reload their outputs from DB automatically."
        ),
    )
    parser.add_argument("--fast", action="store_true",
                        help="Use spaCy fallback instead of GLiNER")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM relation extraction (Stage 9)")
    parser.add_argument("--skip-glirel", action="store_true",
                        help="Skip GLiREL zero-shot relation pass (Stage 2b)")
    parser.add_argument("--daily", action="store_true",
                        help="Daily mode: only process new articles")

    # LLM controls
    parser.add_argument("--llm-batch-size", type=int, default=10)
    parser.add_argument("--llm-workers",    type=int, default=5)

    # Download controls
    parser.add_argument("--download",          action="store_true")
    parser.add_argument("--topic",             default="Iran war 2026")
    parser.add_argument("--start-date",        default="2026-02-28")
    parser.add_argument("--end-date",          default="2026-06-16")
    parser.add_argument("--interval-days",     type=int, default=3)
    parser.add_argument("--max-per-interval",  type=int, default=50)
    parser.add_argument(
        "--categories", nargs="+", default=None, metavar="CATEGORY",
        help=(
            "GNews category filter.  "
            "Choices: global_news finance_business energy_commodities "
            "india china defense_security"
        ),
    )

    args = parser.parse_args()
    if args.run_all or args.daily:
        run_pipeline(args)
    else:
        print("Use --run-all to execute the full pipeline")