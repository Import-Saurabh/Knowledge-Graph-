import argparse
import json
import os
import sys
import time
import urllib.error
from contextlib import contextmanager
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.article_loader import ArticleLoader
from src.ingestion.news_downloader import NewsDownloader
from src.ingestion.infobox_extractor import InfoboxExtractor
from src.entities.entity_extractor import EntityExtractor
from src.entities.entity_resolver import EntityResolver
from src.entities.ontology_manager import OntologyManager
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.vectorstore.chroma_manager import ChromaManager
from src.deduplication.duplicate_detector import DuplicateDetector
from src.clustering.event_clusterer import EventClusterer
from src.events.event_builder import EventBuilder
from src.relations.relation_ontology import RelationOntologyManager
from src.graph.graph_builder import GraphBuilder
from src.graph.neo4j_exporter import Neo4jExporter
from src.analytics.graph_metrics import GraphMetrics
from src.utils.logger import get_logger
from src.utils.db import init_db, get_session, ArticleDB, CanonicalEntityDB
from src.utils.config import settings

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
    """Rebuild canonical_map from SQLite when resuming past stage 3."""
    canonical_map = {}
    with managed_session() as session:
        db_entities = session.query(CanonicalEntityDB).all()
        for db_ent in db_entities:
            entity = resolver._db_to_model(db_ent)
            canonical_map[entity.canonical_id] = entity
    log.info("canonical_map_loaded_from_db", count=len(canonical_map))
    return canonical_map


def _run_wiki_download(args) -> (str, list):
    """Fetch Wikipedia articles via wiki_loader."""
    from src.ingestion.wiki_loader import fetch_all, ingest_to_db

    start = datetime.fromisoformat(args.start_date).replace(tzinfo=timezone.utc)
    end   = datetime.fromisoformat(args.end_date).replace(tzinfo=timezone.utc)
    out_dir = args.wiki_dir

    log.info("wiki_fetch_start",
             start=args.start_date, end=args.end_date, output=out_dir)

    articles = fetch_all(start_date=start, end_date=end, output_dir=out_dir)
    titles = []
    if articles:
        titles = [a["title"] if isinstance(a, dict) else a.title for a in articles]
        ingest_to_db(articles)
        log.info("wiki_fetch_complete", articles=len(articles), output=out_dir)
    else:
        log.warning("wiki_fetch_empty",
                    hint="Wikipedia may not have pages for this date range yet.")

    return out_dir, titles


def _get_all_articles() -> list:
    """Fetch all articles from the database (regardless of status)."""
    from src.models.article import ArticleModel
    with managed_session() as session:
        db_articles = session.query(ArticleDB).all()
        return [
            ArticleModel(
                id=a.id,
                title=a.title,
                content=a.content,
                source=getattr(a, "source", "unknown"),
                published_at=getattr(a, "published_at", None),
                url=getattr(a, "url", f"https://example.com/article/{a.id}"),
            )
            for a in db_articles
        ]


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

    # --wiki: Wikipedia fetch
    if args.wiki:
        wiki_dir, _ = _run_wiki_download(args)
        if not args.download:
            args.input = wiki_dir

    # --download: GNews fetch
    if args.download:
        downloader = NewsDownloader(
            language="en",
            country="US",
            interval_days=args.interval_days,
            categories=args.categories or None,
        )
        gnews_articles = downloader.download(
            topic=args.topic,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.input,
            max_articles_per_interval=args.max_per_interval,
        )
        if gnews_articles:
            loader.ingest_to_db(gnews_articles)

    articles = loader.load_from_directory(args.input)
    result   = loader.ingest_to_db(articles)
    log.info("ingestion_complete", **result)

    # ------------------------------------------------------------------ #
    # Stage 1b: Infobox extraction (Wikipedia articles only)              #
    # ------------------------------------------------------------------ #

    infobox_triples = []
    if args.extract_infoboxes:
        wiki_articles = [a for a in articles if getattr(a, "source", "") == "wikipedia"]
        if not wiki_articles:
            with managed_session() as session:
                try:
                    db_articles = session.query(ArticleDB).filter(
                        ArticleDB.source == "wikipedia"
                    ).all()
                    from src.models.article import ArticleModel
                    wiki_articles = [
                        ArticleModel(
                            id=a.id, title=a.title, content=a.content,
                            source=a.source, published_at=a.published_at,
                        )
                        for a in db_articles
                    ]
                except AttributeError:
                    db_articles   = session.query(ArticleDB).all()
                    wiki_articles = [
                        ArticleModel(id=a.id, title=a.title, content=a.content)
                        for a in db_articles
                    ]
                log.info("infobox_titles_from_db", count=len(wiki_articles))
        if wiki_articles:
            extractor_ib = InfoboxExtractor()
            for article in wiki_articles:
                triples = extractor_ib.extract_from_wikitext(
                    article.title, article.content or ""
                )
                infobox_triples.extend(triples)
            log.info("infobox_extraction_complete", triples=len(infobox_triples))
        else:
            log.warning("infobox_skipped_no_titles",
                        hint="No Wikipedia articles found.")

    # ------------------------------------------------------------------ #
    # Resume logic — determine which articles to process                  #
    # ------------------------------------------------------------------ #

    if args.force:
        unprocessed = _get_all_articles()
        log.info("force_mode_active", article_count=len(unprocessed))
    else:
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

    if not unprocessed and args.from_stage <= 2 and not args.force:
        log.info("no_new_articles")
        return

    log.info("articles_to_process", count=len(unprocessed))

    # ------------------------------------------------------------------ #
    # Stage 2: GLiNER NER                                                  #
    # ------------------------------------------------------------------ #

    all_mentions       = []
    article_to_mentions = {}
    extractor: EntityExtractor | None = None

    if args.from_stage <= 2:
        extractor = EntityExtractor(
            ontology_manager=ontology,
            use_glirel=not args.skip_glirel,
        )
        for i in range(0, len(unprocessed), settings.NER_BATCH_SIZE):
            batch = unprocessed[i : i + settings.NER_BATCH_SIZE]
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
    # Stage 2b: GLiREL zero-shot relation extraction                       #
    # ------------------------------------------------------------------ #

    raw_glirel_triples = []

    if args.from_stage <= 2 and not args.skip_glirel and extractor is not None:
        try:
            raw_glirel_triples = extractor.extract_with_glirel(unprocessed)
            log.info("glirel_complete", triples=len(raw_glirel_triples))
            import pickle
            with open("data/processed/glirel_triples.pkl", "wb") as _f:
                pickle.dump(raw_glirel_triples, _f)
            log.info("glirel_triples_saved",
                     path="data/processed/glirel_triples.pkl")
        except Exception as e:
            log.warning("glirel_extraction_failed", error=str(e))
            raw_glirel_triples = []

    # ------------------------------------------------------------------ #
    # Stage 3: Entity Resolution (with optional Wikidata linkage)         #
    # ------------------------------------------------------------------ #

    canonical_map = {}
    wikidata_linker = None
    if args.use_wikidata_resolution and not args.skip_wikidata:
        from src.enrichment.wikidata_validator import WikidataValidator
        wikidata_linker = WikidataValidator()

    resolver = EntityResolver(chroma, embedder, wikidata_linker=wikidata_linker)

    if args.from_stage <= 3:
        for mention in all_mentions:
            try:
                canonical = resolver.resolve(mention)
                canonical_map[canonical.canonical_id] = canonical
            except Exception as e:
                log.warning("resolution_failed",
                            mention=mention.text, error=str(e))
        log.info("resolution_complete", canonical_entities=len(canonical_map))
    else:
        canonical_map = _load_canonical_map_from_db(resolver)

    # ------------------------------------------------------------------ #
    # Stage 4+5: Article Embeddings → ChromaDB                            #
    # ------------------------------------------------------------------ #

    if args.from_stage <= 4:
        log.info("stage4_embedding_start", articles=len(unprocessed))
        article_embeddings = embedder.embed_articles(unprocessed)
        log.info("stage4_embedding_done", embedded=len(article_embeddings))
        article_lookup = {a.id: a for a in unprocessed}
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
        log.info("stage4_complete", articles_embedded=len(article_embeddings))

    # ------------------------------------------------------------------ #
    # Stage 6: Deduplication                                               #
    # ------------------------------------------------------------------ #

    if args.from_stage <= 6:
        log.info("stage6_dedup_start", articles=len(unprocessed))
        dedup      = DuplicateDetector(chroma, embedder)
        duplicates = dedup.find_duplicates(unprocessed)
        if duplicates:
            dedup.mark_duplicates(duplicates)
            log.info("stage6_duplicates_found", count=len(duplicates))
        non_dup_articles = [
            a for a in unprocessed
            if a.id not in {d[0] for d in duplicates}
        ]
        loader.update_status([a.id for a in non_dup_articles], "deduplicated")
        log.info("stage6_complete", unique_articles=len(non_dup_articles))
    else:
        non_dup_articles = unprocessed

    # ------------------------------------------------------------------ #
    # Stage 7: Clustering  +  Stage 8: Event Building                     #
    # ------------------------------------------------------------------ #

    if args.from_stage <= 7:
        log.info("stage7_clustering_start", articles=len(non_dup_articles))
        clusterer = EventClusterer(embedder=embedder)
        clusters  = clusterer.run_all_windows(non_dup_articles)
        log.info("stage7_clustering_done", clusters=len(clusters))
        builder = EventBuilder()
        events  = [builder.build_event(c) for c in clusters]
        log.info("stage8_events_built", events=len(events))
    else:
        log.info("resuming_events_from_db", from_stage=args.from_stage)
        events = EventBuilder.load_from_db()
        if not events:
            log.error(
                "no_clustered_events_in_db",
                hint="Re-run from --from-stage 7 to rebuild clusters first.",
            )
            raise SystemExit(1)

        # Reload GLiREL triples from disk when resuming past stage 2
        _glirel_path = "data/processed/glirel_triples.pkl"
        if os.path.exists(_glirel_path):
            import pickle
            with open(_glirel_path, "rb") as _f:
                raw_glirel_triples = pickle.load(_f)
            log.info("glirel_triples_loaded",
                     path=_glirel_path, count=len(raw_glirel_triples))
        else:
            raw_glirel_triples = []
            log.warning(
                "glirel_triples_not_found",
                hint="Run --from-stage 2 once to generate and cache GLiREL triples.",
            )

    # Stamp articles with their cluster assignment
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
    # Stage 9: Relation Ontology init + optional Wikidata entity enrich    #
    # ------------------------------------------------------------------ #
    # No LLM. No triple validation. RelationOntologyManager is only used   #
    # to normalise GLiREL relation labels inside graph_builder.            #
    # Wikidata is kept solely for entity-node metadata enrichment.         #
    # ------------------------------------------------------------------ #

    relation_ontology: RelationOntologyManager | None = None
    entity_enrichment_cache: dict = {}

    try:
        relation_ontology = RelationOntologyManager(chroma, embedder)
        # Pre-warm vocab embeddings so the first normalize_relation() call
        # during graph building doesn't block on cold embedding computation.
        relation_ontology.precompute_vocab_embeddings()
        log.info("relation_ontology_ready")
    except Exception as e:
        log.error("relation_ontology_init_failed", error=str(e))

    # Optional Wikidata entity enrichment (node metadata only — no SPARQL
    # for relations; Wikidata property mapping was removed with the LLM path)
    if not args.skip_wikidata and canonical_map:
        from src.enrichment.wikidata_validator import WikidataValidator
        validator = WikidataValidator()

        _WIKIDATA_MAX_RETRIES  = 3
        _WIKIDATA_RETRY_DELAY  = 65  # seconds (WDQS: 1 req / 60 s)
        _wikidata_rate_limited = False

        def _wikidata_call(fn, *fn_args):
            nonlocal _wikidata_rate_limited
            if _wikidata_rate_limited:
                return None
            for attempt in range(_WIKIDATA_MAX_RETRIES):
                try:
                    return fn(*fn_args)
                except urllib.error.HTTPError as exc:
                    if exc.code == 429:
                        if attempt < _WIKIDATA_MAX_RETRIES - 1:
                            log.warning(
                                "wikidata_rate_limited_retrying",
                                attempt=attempt + 1,
                                sleep_seconds=_WIKIDATA_RETRY_DELAY,
                            )
                            time.sleep(_WIKIDATA_RETRY_DELAY)
                        else:
                            log.warning(
                                "wikidata_rate_limit_exceeded",
                                hint=(
                                    "Skipping all remaining Wikidata enrichment. "
                                    "Use --skip-wikidata to suppress entirely."
                                ),
                            )
                            _wikidata_rate_limited = True
                            return None
                    else:
                        log.warning("wikidata_http_error",
                                    code=exc.code, error=str(exc))
                        return None
                except Exception as exc:
                    log.warning("wikidata_call_failed", error=str(exc))
                    return None
            return None

        # Enrich canonical entity nodes with Wikidata metadata
        log.info("wikidata_entity_enrichment_start",
                 entities=len(canonical_map))
        enriched = 0
        for entity in canonical_map.values():
            if _wikidata_rate_limited:
                break
            if entity.canonical_name not in entity_enrichment_cache:
                info = _wikidata_call(validator.enrich_entity,
                                      entity.canonical_name)
                if info:
                    entity_enrichment_cache[entity.canonical_name] = info
                    enriched += 1
        log.info("wikidata_entity_enrichment_complete",
                 enriched=enriched,
                 skipped=len(canonical_map) - enriched)

    # ------------------------------------------------------------------ #
    # Stage 10: Graph Building                                             #
    # ------------------------------------------------------------------ #

    graph_builder = GraphBuilder()
    graph = graph_builder.build_from_relations(
        events=events,
        entity_map=canonical_map,
        glirel_triples=raw_glirel_triples or None,
        entity_enrichment=entity_enrichment_cache or None,
        infobox_triples=infobox_triples or None,
        relation_ontology=relation_ontology,
    )

    # ------------------------------------------------------------------ #
    # Stage 11: Analytics                                                  #
    # ------------------------------------------------------------------ #

    metrics = GraphMetrics()
    report  = metrics.compute_all(graph)
    metrics.save_report(report, "data/exports/analytics_report.json")

    # ------------------------------------------------------------------ #
    # Stage 12: Export                                                     #
    # ------------------------------------------------------------------ #

    exporter = Neo4jExporter()
    exporter.export_nodes_csv(graph, "data/exports/nodes.csv")
    exporter.export_relationships_csv(graph, "data/exports/relationships.csv")
    exporter.export_pyvis_html(graph, "data/exports/graph.html",
                               top_n_nodes=200)

    ontology_report = {
        "entity_types": ontology.get_ontology_report(),
        "relation_types": (
            relation_ontology.get_relation_taxonomy()
            if relation_ontology else []
        ),
        "glirel_triples_count":  len(raw_glirel_triples),
        "infobox_triples_count": len(infobox_triples),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open("data/exports/ontology_report.json", "w", encoding="utf-8") as f:
        json.dump(ontology_report, f, indent=2, ensure_ascii=False)

    log.info(
        "pipeline_complete",
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
        glirel_triples=len(raw_glirel_triples),
        infobox_triples=len(infobox_triples),
        entity_types=len(
            ontology_report["entity_types"].get("top_types", [])
        ),
        relation_types=len(ontology_report["relation_types"]),
    )


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="news_kg — knowledge graph pipeline (GLiNER + GLiREL)"
    )

    # ── Input ──────────────────────────────────────────────────────────
    parser.add_argument("--input", default="data/raw/",
                        help="Directory of JSON/JSONL articles to load")
    parser.add_argument("--run-all", action="store_true",
                        help="Execute the full pipeline end-to-end")
    parser.add_argument(
        "--from-stage", type=int, default=1, metavar="N",
        help=(
            "Resume from stage N  "
            "(1=ingest, 2=NER+GLiREL, 3=entity-resolution, "
            "4=embeddings, 6=dedup, 7=clustering, 9=graph-build). "
            "Earlier stages reload their outputs from DB automatically."
        ),
    )
    parser.add_argument("--daily", action="store_true",
                        help="Daily mode: only process new articles")

    # ── Reprocessing ───────────────────────────────────────────────────
    parser.add_argument("--force", action="store_true",
                        help="Ignore article status and reprocess ALL articles "
                             "from the given --from-stage.")

    # ── One-time duplicate merge ────────────────────────────────────────
    parser.add_argument("--merge-duplicate-entities", action="store_true",
                        help="One-time cleanup: merge duplicate canonical "
                             "entities in the DB and exit.")

    # ── NER / GLiREL flags ─────────────────────────────────────────────
    parser.add_argument("--skip-glirel", action="store_true",
                        help="Skip GLiREL zero-shot relation pass (Stage 2b)")

    # ── Wikidata enrichment ────────────────────────────────────────────
    # Wikidata is used ONLY for:
    #   (a) entity node metadata enrichment  (Stage 9, always on unless skipped)
    #   (b) entity resolution linking        (Stage 3, opt-in via flag below)
    # It is NOT used for relation validation or property mapping.
    parser.add_argument("--skip-wikidata", action="store_true",
                        help="Skip Wikidata entity metadata enrichment (Stage 9)")
    parser.add_argument("--use-wikidata-resolution", action="store_true",
                        help="Use Wikidata to link entity mentions in Stage 3")

    # ── Infobox extraction ─────────────────────────────────────────────
    parser.add_argument("--extract-infoboxes", action="store_true",
                        help="Extract infobox triples from Wikipedia articles")

    # ── Wikipedia download ─────────────────────────────────────────────
    parser.add_argument(
        "--wiki", action="store_true",
        help="Fetch articles from Wikipedia via wiki_loader "
             "(uses --start-date / --end-date / --wiki-dir)",
    )
    parser.add_argument(
        "--wiki-dir", default="data/raw/wiki",
        help="Output directory for Wikipedia JSONL files (default: data/raw/wiki)",
    )

    # ── GNews download ─────────────────────────────────────────────────
    parser.add_argument("--download", action="store_true",
                        help="Download articles from Google News via gnews")
    parser.add_argument("--topic", default="Iran war 2026")
    parser.add_argument("--interval-days", type=int, default=3)
    parser.add_argument("--max-per-interval", type=int, default=50)
    parser.add_argument(
        "--categories", nargs="+", default=None, metavar="CATEGORY",
        help=(
            "GNews category filter. "
            "Choices: global_news finance_business energy_commodities "
            "india china defense_security"
        ),
    )

    # ── Shared date range ──────────────────────────────────────────────
    parser.add_argument("--start-date", default="2026-02-28",
                        help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", default="2026-04-01",
                        help="End date YYYY-MM-DD")

    args = parser.parse_args()

    if args.merge_duplicate_entities:
        init_db()
        embedder = EmbeddingGenerator()
        chroma   = ChromaManager()
        resolver = EntityResolver(chroma, embedder)
        merged   = resolver.find_and_merge_duplicates()
        log.info("duplicate_merge_complete", pairs_merged=merged)
        sys.exit(0)

    if args.wiki and not args.download and args.input == "data/raw/":
        args.input = args.wiki_dir

    if args.run_all or args.daily:
        run_pipeline(args)
    else:
        print("Use --run-all to execute the full pipeline")