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
    end = datetime.fromisoformat(args.end_date).replace(tzinfo=timezone.utc)
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
                source=getattr(a, 'source', 'unknown'),
                published_at=getattr(a, 'published_at', None),
                url=getattr(a, 'url', f"https://example.com/article/{a.id}"),  # default if missing
            )
            for a in db_articles
        ]


def run_pipeline(args):
    os.makedirs("data/exports", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    init_db()

    embedder = EmbeddingGenerator()
    chroma = ChromaManager()
    ontology = OntologyManager(chroma, embedder)

    # ------------------------------------------------------------------ #
    # Stage 1: Ingestion                                                   #
    # ------------------------------------------------------------------ #

    loader = ArticleLoader()
    wiki_titles = []

    # --wiki: Wikipedia fetch
    if args.wiki:
        wiki_dir, wiki_titles = _run_wiki_download(args)
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

    # Load from directory
    articles = loader.load_from_directory(args.input)
    result = loader.ingest_to_db(articles)
    log.info("ingestion_complete", **result)

    # ------------------------------------------------------------------ #
    # Stage 1b: Infobox extraction (for Wikipedia articles)               #
    # ------------------------------------------------------------------ #
    infobox_triples = []
    if args.extract_infoboxes:
        # Use the articles loaded from directory (already have content)
        # Filter to only Wikipedia articles
        wiki_articles = [a for a in articles if getattr(a, 'source', '') == 'wikipedia']
        if not wiki_articles:
            # Fallback: fetch from DB
            with managed_session() as session:
                try:
                    db_articles = session.query(ArticleDB).filter(ArticleDB.source == 'wikipedia').all()
                    from src.models.article import ArticleModel
                    wiki_articles = [
                        ArticleModel(
                            id=a.id,
                            title=a.title,
                            content=a.content,
                            source=a.source,
                            published_at=a.published_at
                        )
                        for a in db_articles
                    ]
                except AttributeError:
                    # No source column, get all
                    db_articles = session.query(ArticleDB).all()
                    wiki_articles = [
                        ArticleModel(id=a.id, title=a.title, content=a.content)
                        for a in db_articles
                    ]
                log.info("infobox_titles_from_db", count=len(wiki_articles))
        if wiki_articles:
            extractor = InfoboxExtractor()
            for idx, article in enumerate(wiki_articles):
                # Extract from local content – no API calls
                triples = extractor.extract_from_wikitext(article.title, article.content or "")
                infobox_triples.extend(triples)
            log.info("infobox_extraction_complete", triples=len(infobox_triples))
        else:
            log.warning("infobox_skipped_no_titles", hint="No Wikipedia articles found.")

    # ------------------------------------------------------------------ #
    # Resume logic – determine which articles to process                #
    # ------------------------------------------------------------------ #

    # If --force is used, fetch ALL articles regardless of status.
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

    all_mentions = []
    article_to_mentions = {}
    extractor = None

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
    # Stage 2b: GLiREL zero-shot relation extraction                       #
    # ------------------------------------------------------------------ #

    raw_glirel_triples = []

    if args.from_stage <= 2 and not args.skip_glirel and extractor is not None:
        try:
            raw_glirel_triples = extractor.extract_with_glirel(unprocessed)
            log.info("glirel_complete", triples=len(raw_glirel_triples))
            import pickle
            os.makedirs("data/processed", exist_ok=True)
            with open("data/processed/glirel_triples.pkl", "wb") as _f:
                pickle.dump(raw_glirel_triples, _f)
            log.info("glirel_triples_saved", path="data/processed/glirel_triples.pkl")
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
                log.warning("resolution_failed", mention=mention.text, error=str(e))
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
        log.info("stage4_chroma_insert_start", count=len(article_embeddings))
        chroma.add_articles(
            ids=[ae[0] for ae in article_embeddings],
            embeddings=[ae[1] for ae in article_embeddings],
            metadatas=[
                {
                    "title": article_lookup[ae[0]].title,
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
        dedup = DuplicateDetector(chroma, embedder)
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
        clusters = clusterer.run_all_windows(non_dup_articles)
        log.info("stage7_clustering_done", clusters=len(clusters))
        builder = EventBuilder()
        events = [builder.build_event(c) for c in clusters]
        log.info("stage8_events_built", events=len(events))
    else:
        log.info("resuming_events_from_db", from_stage=args.from_stage)
        events = EventBuilder.load_from_db()
        if not events:
            log.error("no_clustered_events_in_db",
                      hint="Re-run from --from-stage 7 to rebuild clusters first.")
            raise SystemExit(1)
        import pickle
        _glirel_path = "data/processed/glirel_triples.pkl"
        if os.path.exists(_glirel_path):
            with open(_glirel_path, "rb") as _f:
                raw_glirel_triples = pickle.load(_f)
            log.info("glirel_triples_loaded",
                     path=_glirel_path, count=len(raw_glirel_triples))
        else:
            raw_glirel_triples = []
            log.warning("glirel_triples_not_found",
                        hint="Run --from-stage 2 once to generate and save GLiREL triples.")

    try:
        with managed_session() as session:
            for event in events:
                session.query(ArticleDB).filter(
                    ArticleDB.id.in_(event.article_ids)
                ).update(
                    {
                        ArticleDB.cluster_id: event.cluster_id,
                        ArticleDB.temporal_window: event.temporal_window,
                        ArticleDB.status: "clustered",
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

    relation_ontology: RelationOntologyManager | None = None
    llm_responses = []

    if not args.skip_llm:
        extractor_llm = RelationExtractor(max_workers=args.llm_workers)
        try:
            relation_ontology = RelationOntologyManager(chroma, embedder)
            if args.use_wikidata_relation_normalization and not args.skip_wikidata:
                relation_ontology.precompute_vocab_embeddings()
        except Exception as e:
            log.error("relation_ontology_init_failed", error=str(e))

        total_events = len(events)
        for i in range(0, total_events, args.llm_batch_size):
            batch = events[i:i + args.llm_batch_size]
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

        for event, response in zip(events, llm_responses):
            try:
                for triple in response.triples:
                    # --- FIX: only normalize if LLM gave a non-seed label ---
                    if relation_ontology:
                        # If the LLM's canonical is already a seed label, keep it.
                        # Otherwise, try to map from the raw phrase.
                        if not relation_ontology.is_seed_canonical(triple.relation_canonical):
                            triple.relation_canonical = relation_ontology.normalize_relation(triple.relation)

                        # Optionally fetch Wikidata property without altering canonical
                        if args.use_wikidata_relation_normalization and not args.skip_wikidata:
                            triple.wikidata_property = relation_ontology.get_wikidata_property(triple.relation)

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
    # Stage 9b: Wikidata Validation & Enrichment                           #
    # ------------------------------------------------------------------ #

    entity_enrichment_cache = {}
    if not args.skip_wikidata and any(
        triple.source and triple.target
        for response in llm_responses
        for triple in response.triples
    ):
        from src.enrichment.wikidata_validator import WikidataValidator
        validator = WikidataValidator()

        _WIKIDATA_MAX_RETRIES = 3
        _WIKIDATA_RETRY_DELAY = 65  # seconds — just over the 1 req/min limit
        _wikidata_rate_limited = False  # once True, skip remaining SPARQL calls

        def _wikidata_call(fn, *fn_args):
            """Call a WikidataValidator method with retry-on-429 and global skip."""
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
                                    "Wikidata SPARQL is aggressively rate-limiting. "
                                    "Skipping all remaining Wikidata enrichment. "
                                    "Re-run with --skip-wikidata to suppress this entirely."
                                ),
                            )
                            _wikidata_rate_limited = True
                            return None
                    else:
                        log.warning("wikidata_http_error", code=exc.code, error=str(exc))
                        return None
                except Exception as exc:
                    log.warning("wikidata_call_failed", error=str(exc))
                    return None
            return None

        for event, response in zip(events, llm_responses):
            if _wikidata_rate_limited:
                break
            for triple in response.triples:
                if triple.source and triple.target:
                    validation = _wikidata_call(
                        validator.validate_relation,
                        triple.source, triple.relation, triple.target,
                    )
                    if validation is not None:
                        if validation["exists"]:
                            triple.wikidata_property = validation["wikidata_property"]
                            triple.wikidata_date = validation.get("date")
                            triple.wikidata_description = validation.get("description")
                            triple.confidence = min(1.0, triple.confidence + validation["confidence_boost"])
                            triple.needs_review = False
                        else:
                            triple.needs_review = True
                            triple.wikidata_property = None
                            triple.wikidata_date = None
                            triple.wikidata_description = None
                for entity_name in [triple.source, triple.target]:
                    if entity_name and entity_name not in entity_enrichment_cache:
                        info = _wikidata_call(validator.enrich_entity, entity_name)
                        if info:
                            entity_enrichment_cache[entity_name] = info

    # ------------------------------------------------------------------ #
    # Stage 10: Graph Building                                             #
    # ------------------------------------------------------------------ #

    graph_builder = GraphBuilder()
    graph = graph_builder.build_from_relations(
        events,
        llm_responses,
        canonical_map,
        glirel_triples=raw_glirel_triples or None,
        entity_enrichment=entity_enrichment_cache,
        infobox_triples=infobox_triples,
    )

    # ------------------------------------------------------------------ #
    # Stage 11: Analytics                                                  #
    # ------------------------------------------------------------------ #

    metrics = GraphMetrics()
    report = metrics.compute_all(graph)
    metrics.save_report(report, "data/exports/analytics_report.json")

    # ------------------------------------------------------------------ #
    # Stage 12: Export                                                     #
    # ------------------------------------------------------------------ #

    exporter = Neo4jExporter()
    exporter.export_nodes_csv(graph, "data/exports/nodes.csv")
    exporter.export_relationships_csv(graph, "data/exports/relationships.csv")
    exporter.export_pyvis_html(graph, "data/exports/graph.html", top_n_nodes=200)

    ontology_report = {
        "entity_types": ontology.get_ontology_report(),
        "relation_types": (
            relation_ontology.get_relation_taxonomy() or []
            if (not args.skip_llm and relation_ontology)
            else []
        ),
        "glirel_triples_count": len(raw_glirel_triples),
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
            "4=embeddings, 6=dedup, 7=clustering, 9=relations). "
            "Earlier stages reload their outputs from DB automatically."
        ),
    )
    parser.add_argument("--daily", action="store_true",
                        help="Daily mode: only process new articles")

    # ── Force reprocess all articles ──────────────────────────────────
    parser.add_argument("--force", action="store_true",
                        help="Ignore article status and reprocess ALL articles from the given --from-stage.")

    # ── One‑time duplicate merge ─────────────────────────────────────
    parser.add_argument("--merge-duplicate-entities", action="store_true",
                        help="One-time cleanup: merge duplicate canonical entities in the DB and exit.")

    # ── NER / relation flags ───────────────────────────────────────────
    parser.add_argument("--fast", action="store_true",
                        help="Use spaCy fallback instead of GLiNER")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM relation extraction (Stage 9)")
    parser.add_argument("--skip-glirel", action="store_true",
                        help="Skip GLiREL zero-shot relation pass (Stage 2b)")

    # ── LLM controls ──────────────────────────────────────────────────
    parser.add_argument("--llm-batch-size", type=int, default=10)
    parser.add_argument("--llm-workers", type=int, default=5)

    # ── Wikidata enrichment ───────────────────────────────────────────
    parser.add_argument("--skip-wikidata", action="store_true",
                        help="Skip Wikidata validation/enrichment (Stage 9b)")
    parser.add_argument("--use-wikidata-resolution", action="store_true",
                        help="Use Wikidata to resolve entity mentions in Stage 3")
    parser.add_argument("--use-wikidata-relation-normalization", action="store_true",
                        help="Use Wikidata to normalize relation types")

    # ── Infobox extraction ────────────────────────────────────────────
    parser.add_argument("--extract-infoboxes", action="store_true",
                        help="Extract infobox triples from Wikipedia articles (titles fetched from DB)")

    # ── Wikipedia download ────────────────────────────────────────────
    parser.add_argument(
        "--wiki", action="store_true",
        help=(
            "Fetch articles from Wikipedia via wiki_loader "
            "(uses --start-date / --end-date / --wiki-dir)"
        ),
    )
    parser.add_argument(
        "--wiki-dir", default="data/raw/wiki",
        help="Output directory for Wikipedia JSONL files (default: data/raw/wiki)",
    )

    # ── GNews download ────────────────────────────────────────────────
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

    # ── Shared date range ─────────────────────────────────────────────
    parser.add_argument("--start-date", default="2026-02-28",
                        help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", default="2026-04-01",
                        help="End date YYYY-MM-DD")

    args = parser.parse_args()

    # If --merge-duplicate-entities is given, run the merge and exit.
    if args.merge_duplicate_entities:
        init_db()
        embedder = EmbeddingGenerator()
        chroma = ChromaManager()
        resolver = EntityResolver(chroma, embedder)
        merged = resolver.find_and_merge_duplicates()
        log.info("duplicate_merge_complete", pairs_merged=merged)
        sys.exit(0)

    if args.wiki and not args.download and args.input == "data/raw/":
        args.input = args.wiki_dir

    if args.run_all or args.daily:
        run_pipeline(args)
    else:
        print("Use --run-all to execute the full pipeline")