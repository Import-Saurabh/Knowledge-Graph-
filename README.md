import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
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
    Fetches articles exclusively through GNews (Google News RSS) and extracts
    full text via newspaper4k.

    Discovery strategy (GNews only, no direct site crawling):
      1. Per-category site-grouped queries  → e.g. "ukraine war site:reuters.com OR site:bbc.com"
         run for every MEDIA_HOUSES category.
      2. Broad topic search as a final sweep to catch any approved sources
         not covered in step 1.

    Text extraction (newspaper4k only):
      newspaper4k downloads and parses only the article URLs that GNews
      already returned — it is never used to crawl media house homepages or
      section pages independently.

    Searches run in configurable-day intervals to maximise Google News
    coverage and stay within rate limits.
    """


    def __init__(
        self,
        language: str = "en",
        country: str = "US",
        interval_days: int = 3,
        categories: List[str] = None,
    ):
        """
        Args:
            language: GNews language code.
            country: GNews country code.
            interval_days: Width of each date window sent to GNews.
            categories: Subset of MEDIA_HOUSES keys to query. None = all.
        """
        self.language = language
        self.country = country
        self.interval_days = interval_days
        self.active_categories: Dict[str, List[str]] = (
            {k: MEDIA_HOUSES[k] for k in categories if k in MEDIA_HOUSES}
            if categories
            else MEDIA_HOUSES
        )
        self._gnews = None

    # ------------------------------------------------------------------
    # GNews client (lazy singleton)
    # ------------------------------------------------------------------

    def _get_gnews(self):
        if self._gnews is None:
            try:
                from gnews import GNews
                self._gnews = GNews(
                    language=self.language,
                    country=self.country,
                    max_results=100,
                )
                log.info("gnews_client_initialized")
            except ImportError:
                log.error("gnews_not_installed", hint="pip install gnews")
                raise
        return self._gnews

    def _set_gnews_window(self, start: str, end: str, max_results: int) -> None:
        """Apply date window and result cap to the shared GNews client."""
        gnews = self._get_gnews()
        gnews.start_date = datetime.strptime(start, "%Y-%m-%d")
        gnews.end_date   = datetime.strptime(end,   "%Y-%m-%d")
        gnews.max_results = max_results

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

        Args:
            topic: Search keyword / phrase (e.g. "Ukraine Russia war").
            start_date: Inclusive start date, YYYY-MM-DD.
            end_date: Inclusive end date, YYYY-MM-DD. Defaults to today (UTC).
            output_dir: Directory where the .jsonl file is written.
            max_articles_per_interval: Hard cap per date-window per GNews query.

        Returns:
            List of ArticleModel instances collected across all intervals.
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
            time.sleep(2)   # polite pause between intervals

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
    # Interval-level orchestration
    # ------------------------------------------------------------------

    # High-value domains always queried individually via get_news_by_site
    # (in addition to the broad topic sweep) to maximise recall.
    _PRIORITY_DOMAINS: List[str] = [
        "reuters.com", "bloomberg.com", "apnews.com", "bbc.com",
        "aljazeera.com", "ft.com", "wsj.com", "cnbc.com",
        "economictimes.indiatimes.com", "thehindu.com",
    ]

    def _fetch_interval(
        self,
        topic: str,
        start: str,
        end: str,
        max_per_interval: int,
        seen_urls: set,
    ) -> List[ArticleModel]:
        """
        Two-pass GNews strategy for one date window.

        GNews wraps Google News RSS — the site: operator is NOT supported in
        RSS query strings, so OR-grouped site: queries always return 0 results.
        The two passes below use only what GNews actually supports:

        Pass 1 — Broad topic search (get_news)
            Single query for the topic, returns up to max_results items.
            Results are filtered post-hoc to approved domains.
            This is the primary, highest-yield pass.

        Pass 2 — Per-domain site search (get_news_by_site)
            For each priority domain, fetch that domain's latest news and
            keep articles whose title/description contains a topic keyword.
            Fills gaps for high-value outlets not caught in pass 1.
        """
        articles: List[ArticleModel] = []
        topic_keywords = [w.lower() for w in topic.split() if len(w) > 3]
        log.info("interval_start", start=start, end=end, max=max_per_interval)

        # ── Pass 1: broad topic search → approved-domain filter ──────────────
        self._set_gnews_window(start, end, max_per_interval)
        raw = self._gnews_search(topic, start, end)
        new = self._process_raw_results(raw, seen_urls, max_per_interval, require_approved=True)
        articles.extend(new)
        for a in new:
            seen_urls.add(a.url)
        log.info("pass1_broad_done", raw_hits=len(raw), accepted=len(new), total=len(articles))

        # ── Pass 2: per-domain get_news_by_site → topic keyword filter ───────
        remaining = max_per_interval - len(articles)
        if remaining > 0:
            for domain in self._PRIORITY_DOMAINS:
                if len(articles) >= max_per_interval:
                    break
                self._set_gnews_window(start, end, 20)
                domain_raw = self._gnews_search_by_site(domain, start, end)

                # Keep only items whose title contains at least one topic keyword
                relevant = [
                    item for item in domain_raw
                    if any(kw in item.get("title", "").lower() for kw in topic_keywords)
                    and item.get("url", "") not in seen_urls
                ]
                log.info(
                    "pass2_site_done",
                    domain=domain,
                    raw=len(domain_raw),
                    relevant=len(relevant),
                )

                new = self._process_raw_results(
                    relevant, seen_urls,
                    max_per_interval - len(articles),
                    require_approved=False,   # domain already trusted
                )
                articles.extend(new)
                for a in new:
                    seen_urls.add(a.url)
                time.sleep(1)

        return articles

    # ------------------------------------------------------------------
    # GNews query helper
    # ------------------------------------------------------------------

    # Seconds to wait for a single GNews HTTP call before giving up.
    _GNEWS_TIMEOUT: int = 30

    def _gnews_search(self, query: str, start: str, end: str) -> List[dict]:
        """
        Execute a single GNews query with a hard timeout.
        Logs before and after so silent hangs are immediately visible.
        """
        log.info("gnews_querying", query=query, start=start, end=end)
        gnews = self._get_gnews()

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(gnews.get_news, query)
            try:
                results = future.result(timeout=self._GNEWS_TIMEOUT)
                hits = len(results) if results else 0
                log.info("gnews_query_done", query=query, hits=hits)
                return results or []
            except FuturesTimeoutError:
                log.warning("gnews_query_timeout", query=query, timeout=self._GNEWS_TIMEOUT)
                future.cancel()
                return []
            except Exception as e:
                log.warning("gnews_query_failed", query=query, error=str(e))
                return []

    def _gnews_search_by_site(self, domain: str, start: str, end: str) -> List[dict]:
        """
        Fetch recent articles from a single domain using get_news_by_site.
        Falls back to an empty list on timeout or error.
        """
        log.info("gnews_querying_site", domain=domain, start=start, end=end)
        gnews = self._get_gnews()

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(gnews.get_news_by_site, domain)
            try:
                results = future.result(timeout=self._GNEWS_TIMEOUT)
                hits = len(results) if results else 0
                log.info("gnews_site_done", domain=domain, hits=hits)
                return results or []
            except FuturesTimeoutError:
                log.warning("gnews_site_timeout", domain=domain, timeout=self._GNEWS_TIMEOUT)
                future.cancel()
                return []
            except Exception as e:
                log.warning("gnews_site_failed", domain=domain, error=str(e))
                return []

    # ------------------------------------------------------------------
    # Raw-result → ArticleModel conversion
    # ------------------------------------------------------------------

    def _process_raw_results(
        self,
        raw_results: List[dict],
        seen_urls: set,
        quota: int,
        require_approved: bool = True,
    ) -> List[ArticleModel]:
        """
        Convert GNews raw dicts to ArticleModel instances.

        Steps:
          1. Skip missing/duplicate URLs.
          2. Optionally enforce approved-domain allowlist.
          3. Extract full text via newspaper4k from the GNews-provided URL.
          4. Skip articles whose extracted text is too short.
        """
        articles: List[ArticleModel] = []

        for item in raw_results:
            if len(articles) >= quota:
                break

            url         = item.get("url", "").strip()
            title       = item.get("title", "").strip()
            published   = item.get("published date", "")
            source_name = item.get("publisher", {}).get("title", "Unknown")

            if not url or not title:
                continue
            if url in seen_urls:
                continue
            if require_approved and not self._is_approved_domain(url):
                continue

            pub_dt = self._parse_published_date(published)

            # newspaper4k: extract full text ONLY from the GNews-returned URL.
            log.info("extracting_article", n=len(articles)+1, quota=quota, source=source_name, title=title[:60])
            content = self._extract_full_text(url, fallback=title)
            if len(content) < 200:
                log.info("article_too_short", source=source_name, length=len(content))
                continue

            articles.append(
                ArticleModel(
                    title=title,
                    content=content,
                    source=source_name,
                    published_at=pub_dt,
                    url=url,
                )
            )
            time.sleep(0.3)     # be polite to article servers

        return articles

    # ------------------------------------------------------------------
    # newspaper4k text extraction
    # ------------------------------------------------------------------

    # Seconds to wait for newspaper4k download+parse before giving up.
    _NEWSPAPER_TIMEOUT: int = 15

    def _extract_full_text(self, url: str, fallback: str) -> str:
        """
        Use newspaper4k to download and parse the article at *url*.

        This is the ONLY place where an external HTTP request to a media
        house URL occurs — and only because GNews already surfaced that URL.
        A hard timeout prevents any single slow article from stalling the pipeline.
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
                    log.info("article_extracted", chars=len(text), url=url)
                    return text
                return text or fallback
            except FuturesTimeoutError:
                log.warning("newspaper4k_timeout", url=url, timeout=self._NEWSPAPER_TIMEOUT)
                future.cancel()
                return fallback
            except Exception as e:
                log.warning("newspaper4k_extraction_failed", url=url, error=str(e))
                return fallback

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _is_approved_domain(self, url: str) -> bool:
        """Return True if the URL's domain is in the approved allowlist."""
        domain = urlparse(url).netloc.lower().removeprefix("www.")
        return any(
            approved in domain or domain.endswith(approved)
            for approved in ALL_APPROVED_DOMAINS
        )

    def _parse_published_date(self, published: str) -> datetime:
        """Parse GNews 'published date' string; fall back to UTC now."""
        if published:
            try:
                return datetime.strptime(published, "%a, %d %b %Y %H:%M:%S %Z")
            except ValueError:
                pass
        return datetime.utcnow()

    def _generate_date_intervals(self, start_date: str, end_date: str) -> List[Tuple[str, str]]:
        """Split [start_date, end_date] into windows of self.interval_days width."""
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt   = datetime.strptime(end_date,   "%Y-%m-%d")
        intervals, current = [], start_dt
        while current <= end_dt:
            window_end = min(current + timedelta(days=self.interval_days - 1), end_dt)
            intervals.append((current.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d")))
            current = window_end + timedelta(days=1)
        return intervals

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_to_jsonl(self, articles: List[ArticleModel], output_dir: str, topic: str) -> None:
        """Append-write articles to a timestamped .jsonl file."""
        safe_topic = topic.replace(" ", "_").replace("/", "_")[:50]
        filename   = f"{safe_topic}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
        filepath   = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            for article in articles:
                f.write(json.dumps(article.model_dump(), default=str, ensure_ascii=False) + "\n")

        log.info("articles_saved", filepath=filepath, count=len(articles))