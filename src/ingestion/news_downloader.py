import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlparse
from src.models.article import ArticleModel
from src.utils.logger import get_logger

log = get_logger(__name__)

# Approved media houses — organized by category.
# GNews site: queries are built from these domains directly.
# newspaper4k only extracts text from URLs that GNews returns.
MEDIA_HOUSES: Dict[str, List[str]] = {
    "global_news": [
        "bbc.co.uk", "bbc.com", "cnn.com", "nbcnews.com", "abcnews.go.com",
        "cbsnews.com", "nytimes.com", "washingtonpost.com", "usatoday.com",
        "latimes.com", "aljazeera.com", "france24.com", "dw.com",
        "euronews.com", "trtworld.com", "apnews.com", "reuters.com",
    ],
    "finance_business": [
        "bloomberg.com", "economictimes.indiatimes.com", "business-standard.com",
        "thehindubusinessline.com", "investopedia.com", "spglobal.com",
        "financialtimes.com", "wsj.com", "marketwatch.com", "cnbc.com",
        "fortune.com", "forbes.com", "moneycontrol.com", "businessinsider.com",
    ],
    "energy_commodities": [
        "oilprice.com", "world-nuclear-news.org", "energyintel.com",
        "commodities-insights.spglobal.com", "tradewindsnews.com",
        "shippingwatch.com", "gcaptain.com", "metalbulletin.com",
        "mining.com", "fastmarkets.com",
    ],
    "india": [
        "thehindu.com", "hindustantimes.com", "indiatoday.in", "ndtv.com",
        "timesofindia.indiatimes.com", "indianexpress.com",
    ],
    "china": [
        "chinadaily.com.cn", "scmp.com", "globaltimes.cn",
    ],
    "defense_security": [
        "defensenews.com", "janes.com", "aviationweek.com", "flightglobal.com",
        "foreignpolicy.com", "politico.com", "axios.com",
    ],
}

# Flat set of all approved domains for fast membership checks.
ALL_APPROVED_DOMAINS: List[str] = [
    domain for domains in MEDIA_HOUSES.values() for domain in domains
]


class NewsDownloader:
    """
    Fetches articles through GNews (Google News RSS) and extracts full text
    via newspaper4k.  Both GNews queries and article extractions are now run
    in parallel, reducing a typical 3-month fetch from hours down to
    5-15 minutes depending on network conditions.

    Key architecture:
      _fetch_interval()
        Phase 1 — parallel GNews queries  (_GNEWS_WORKERS concurrent workers)
        Phase 2 — parallel text extraction (_EXTRACT_WORKERS concurrent workers)

    Each GNews call creates its own client instance (thread-safe).
    newspaper4k calls go to different domains — high concurrency is safe.
    """

    # Maximum domains per GNews OR-query; Google News degrades above ~5.
    _DOMAINS_PER_QUERY: int = 5

    # Parallelism knobs — tune if you hit rate limits.
    _GNEWS_WORKERS: int = 4      # concurrent GNews RSS queries (conservative)
    _EXTRACT_WORKERS: int = 20   # concurrent newspaper4k fetches (different domains)

    # Timeouts
    _GNEWS_TIMEOUT: int = 30
    _NEWSPAPER_TIMEOUT: int = 15

    def __init__(
        self,
        language: str = "en",
        country: str = "US",
        interval_days: int = 3,
        categories: List[str] = None,
    ):
        self.language = language
        self.country = country
        self.interval_days = interval_days
        self.active_categories: Dict[str, List[str]] = (
            {k: MEDIA_HOUSES[k] for k in categories if k in MEDIA_HOUSES}
            if categories
            else MEDIA_HOUSES
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(
        self,
        topic: str,
        start_date: str,
        end_date: str = None,
        output_dir: str = "data/raw/",
        max_articles_per_interval: int = 50,
    ) -> List[ArticleModel]:
        """
        Download articles for *topic* between *start_date* and *end_date*.
        All GNews queries within an interval run in parallel; all newspaper4k
        extractions within an interval also run in parallel.
        """
        os.makedirs(output_dir, exist_ok=True)
        end_date = end_date or datetime.utcnow().strftime("%Y-%m-%d")
        intervals = self._generate_date_intervals(start_date, end_date)

        log.info(
            "download_plan",
            topic=topic,
            intervals=len(intervals),
            start=start_date,
            end=end_date,
            interval_days=self.interval_days,
            categories=list(self.active_categories.keys()),
            gnews_workers=self._GNEWS_WORKERS,
            extract_workers=self._EXTRACT_WORKERS,
        )

        all_articles: List[ArticleModel] = []
        seen_urls: set = set()

        for interval_start, interval_end in intervals:
            interval_articles = self._fetch_interval(
                topic, interval_start, interval_end,
                max_articles_per_interval, seen_urls,
            )
            all_articles.extend(interval_articles)
            for a in interval_articles:
                seen_urls.add(a.url)

            log.info(
                "interval_complete",
                start=interval_start,
                end=interval_end,
                fetched=len(interval_articles),
                total_so_far=len(all_articles),
            )
            # Reduced from 2s → 0.5s; GNews RSS is not session-stateful.
            time.sleep(0.5)

        if all_articles:
            self._save_to_jsonl(all_articles, output_dir, topic)

        log.info(
            "download_complete",
            topic=topic,
            total=len(all_articles),
            intervals=len(intervals),
            unique_sources=len({a.source for a in all_articles}),
        )
        return all_articles

    # ------------------------------------------------------------------
    # Interval-level orchestration (two-phase parallel)
    # ------------------------------------------------------------------

    def _fetch_interval(
        self,
        topic: str,
        start: str,
        end: str,
        max_per_interval: int,
        seen_urls: set,
    ) -> List[ArticleModel]:
        """
        Phase 1 — parallel GNews queries
            Build every category×batch query for this window, then fire them
            all concurrently (_GNEWS_WORKERS).  Also run the broad sweep in
            the same pool.

        Phase 2 — parallel newspaper4k extraction
            Deduplicate and filter candidates, then extract full text for all
            of them concurrently (_EXTRACT_WORKERS).
        """
        # Build all queries for this interval.
        queries: List[Tuple[str, List[str], str]] = []  # (category, batch, query)
        for category, domains in self.active_categories.items():
            for batch in self._chunk(domains, self._DOMAINS_PER_QUERY):
                site_clause = " OR ".join(f"site:{d}" for d in batch)
                queries.append((category, batch, f"{topic} {site_clause}"))
        # Add broad sweep as a pseudo-query.
        queries.append(("_broad", [], topic))

        log.info(
            "interval_start",
            start=start, end=end,
            total_queries=len(queries),
            categories=list(self.active_categories.keys()),
        )

        # ---- Phase 1: parallel GNews queries ----
        raw_items: List[dict] = []

        def _run_query(cat: str, batch: List[str], q: str) -> List[dict]:
            results = self._gnews_search(q, start, end, max_per_interval)
            log.info(
                "query_done",
                category=cat,
                domains=batch,
                found=len(results),
            )
            return results

        with ThreadPoolExecutor(max_workers=self._GNEWS_WORKERS) as pool:
            futures = {
                pool.submit(_run_query, cat, batch, q): q
                for cat, batch, q in queries
            }
            for future in as_completed(futures):
                try:
                    raw_items.extend(future.result())
                except Exception as e:
                    log.warning("gnews_parallel_failed", query=futures[future], error=str(e))

        # ---- Deduplicate and domain-filter candidates ----
        seen_candidate_urls: set = set()
        candidates: List[dict] = []
        for item in raw_items:
            url = str(item.get("url", "")).strip()
            if not url or url in seen_urls or url in seen_candidate_urls:
                continue
            if not self._is_approved_result(item, url):
                continue
            seen_candidate_urls.add(url)
            candidates.append(item)

        log.info(
            "candidates_collected",
            count=len(candidates),
            start=start,
            end=end,
        )

        # ---- Phase 2: parallel newspaper4k extraction ----
        articles = self._extract_articles_parallel(
            candidates[:max_per_interval],
            seen_urls,
        )
        return articles[:max_per_interval]

    # ------------------------------------------------------------------
    # GNews client — thread-safe (fresh instance per call)
    # ------------------------------------------------------------------

    def _gnews_search(
        self, query: str, start: str, end: str, max_results: int = 100
    ) -> List[dict]:
        """
        Thread-safe GNews search.  Creates a fresh GNews instance per call so
        multiple threads can set different date windows without racing on shared
        state.  Wrapped in a hard timeout to prevent silent hangs.
        """
        log.info("gnews_querying", query=query, start=start, end=end)

        def _do_search() -> List[dict]:
            from gnews import GNews
            client = GNews(
                language=self.language,
                country=self.country,
                max_results=max_results,
            )
            client.start_date = datetime.strptime(start, "%Y-%m-%d")
            client.end_date   = datetime.strptime(end,   "%Y-%m-%d")
            return client.get_news(query) or []

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_do_search)
            try:
                results = future.result(timeout=self._GNEWS_TIMEOUT)
                log.info("gnews_query_done", query=query, hits=len(results))
                return results
            except FuturesTimeoutError:
                log.warning("gnews_query_timeout", query=query, timeout=self._GNEWS_TIMEOUT)
                future.cancel()
                return []
            except Exception as e:
                log.warning("gnews_query_failed", query=query, error=str(e))
                return []

    # ------------------------------------------------------------------
    # Parallel extraction
    # ------------------------------------------------------------------

    def _extract_articles_parallel(
        self,
        candidates: List[dict],
        seen_urls: set,
    ) -> List[ArticleModel]:
        """
        Extract full text for all *candidates* concurrently using
        _EXTRACT_WORKERS threads.  Each URL goes to a different domain, so
        high concurrency here is safe and fast.
        """
        articles: List[ArticleModel] = []

        def _process_item(item: dict) -> Optional[ArticleModel]:
            url         = str(item.get("url", "")).strip()
            title       = str(item.get("title", "")).strip()
            description = str(item.get("description", "")).strip()
            published   = item.get("published date") or item.get("published_date") or ""
            source_name = self._get_source_name(item)

            if not url or not title:
                return None

            pub_dt    = self._parse_published_date(published)
            extracted = self._extract_full_text(url, fallback="")
            content   = self._build_content(title=title, description=description, extracted=extracted)

            if len(content.strip()) < 40:
                log.info("article_too_short", source=source_name, length=len(content))
                return None

            return ArticleModel(
                title=title,
                content=content,
                source=source_name,
                published_at=pub_dt,
                url=url,
            )

        with ThreadPoolExecutor(max_workers=self._EXTRACT_WORKERS) as pool:
            futures = {pool.submit(_process_item, item): item for item in candidates}
            for future in as_completed(futures):
                try:
                    article = future.result()
                    if article:
                        articles.append(article)
                        log.info(
                            "article_extracted",
                            source=article.source,
                            title=article.title[:60],
                        )
                except Exception as e:
                    log.warning("parallel_extraction_failed", error=str(e))

        return articles

    # ------------------------------------------------------------------
    # newspaper4k text extraction
    # ------------------------------------------------------------------

    def _extract_full_text(self, url: str, fallback: str) -> str:
        """
        Use newspaper4k to download and parse the article at *url*.
        Hard timeout prevents any single slow article from stalling the pool.
        """
        def _download_and_parse() -> str:
            from newspaper import Article
            art = Article(url, language=self.language)
            art.download()
            art.parse()
            return art.text or ""

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_download_and_parse)
            try:
                text = future.result(timeout=self._NEWSPAPER_TIMEOUT)
                if text and len(text) > 100:
                    log.info("article_extracted_full", chars=len(text), url=url)
                return text or fallback
            except FuturesTimeoutError:
                log.warning("newspaper4k_timeout", url=url, timeout=self._NEWSPAPER_TIMEOUT)
                future.cancel()
                return fallback
            except Exception as e:
                log.warning("newspaper4k_extraction_failed", url=url, error=str(e))
                return fallback

    def _build_content(self, title: str, description: str, extracted: str) -> str:
        """Compose the best available article text."""
        parts = []
        for part in (title, description, extracted):
            part = (part or "").strip()
            if part and part not in parts:
                parts.append(part)
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Domain / source helpers
    # ------------------------------------------------------------------

    def _get_source_name(self, item: dict) -> str:
        publisher = item.get("publisher", "Unknown")
        if isinstance(publisher, dict):
            return str(publisher.get("title") or publisher.get("name") or "Unknown").strip() or "Unknown"
        return str(publisher).strip() or "Unknown"

    def _is_approved_result(self, item: dict, url: str) -> bool:
        if self._is_approved_domain(url):
            return True
        publisher = item.get("publisher", {})
        candidates = []
        if isinstance(publisher, dict):
            candidates.extend([publisher.get("href", ""), publisher.get("title", "")])
        elif isinstance(publisher, str):
            candidates.append(publisher)
        for candidate in candidates:
            if not candidate:
                continue
            candidate = str(candidate).lower()
            for approved in ALL_APPROVED_DOMAINS:
                if approved in candidate or candidate.endswith(approved):
                    return True
        return False

    def _is_approved_domain(self, url: str) -> bool:
        domain = urlparse(url).netloc.lower().removeprefix("www.")
        return any(
            approved in domain or domain.endswith(approved)
            for approved in ALL_APPROVED_DOMAINS
        )

    # ------------------------------------------------------------------
    # Date helpers
    # ------------------------------------------------------------------

    def _parse_published_date(self, published: str) -> datetime:
        if published:
            try:
                return datetime.strptime(published, "%a, %d %b %Y %H:%M:%S %Z")
            except ValueError:
                pass
        return datetime.utcnow()

    def _generate_date_intervals(self, start_date: str, end_date: str) -> List[Tuple[str, str]]:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt   = datetime.strptime(end_date,   "%Y-%m-%d")
        intervals, current = [], start_dt
        while current <= end_dt:
            window_end = min(current + timedelta(days=self.interval_days - 1), end_dt)
            intervals.append((current.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d")))
            current = window_end + timedelta(days=1)
        return intervals

    @staticmethod
    def _chunk(lst: List, size: int) -> List[List]:
        return [lst[i : i + size] for i in range(0, len(lst), size)]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_to_jsonl(self, articles: List[ArticleModel], output_dir: str, topic: str) -> None:
        safe_topic = topic.replace(" ", "_").replace("/", "_")[:50]
        filename   = f"{safe_topic}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
        filepath   = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            for article in articles:
                f.write(json.dumps(article.model_dump(), default=str, ensure_ascii=False) + "\n")
        log.info("articles_saved", filepath=filepath, count=len(articles))