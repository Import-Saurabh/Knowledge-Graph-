"""
wiki_loader.py
==============
Fetch Iran-war-2026 content from Wikipedia, split into dated chunks,
and save as JSONL files that ArticleLoader / the main pipeline can ingest.

Usage (standalone):
    python wiki_loader.py                        # fetch + save to data/raw/wiki/
    python wiki_loader.py --ingest               # fetch + save + ingest to DB
    python wiki_loader.py --dry-run              # print article count only

Then run the pipeline normally:
    python main.py --run-all --from-stage 2

Design:
    Wikipedia is an encyclopedia, not a newsfeed, so articles don't carry
    real publication dates. We recover temporal signal two ways:
      1. Section-level date parsing — headings like "March 2026" or body text
         matching a known date pattern are used to stamp that section's chunk.
      2. Linear interpolation — remaining chunks are spread evenly across
         the requested [start_date, end_date] window so the clustering stage
         sees a plausible temporal spread instead of everything landing on
         the same timestamp.

    Each Wikipedia *section* → one ArticleModel.  This keeps chunks
    roughly the same size as GNews articles and avoids overloading
    the embedder with 10 000-word walls of text.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Wikipedia topics to fetch
# ---------------------------------------------------------------------------
# Each entry is a search query.  The first Wikipedia hit is used.
# Add / remove as the war develops.
SEARCH_QUERIES = [
    "Iran United States war 2026",
    "2026 attack on Iran",
    "Operation Midnight Hammer",
    "Iran nuclear program 2026",
    "Iran Israel conflict 2026",
    "Iran military 2026",
    "United States Iran military strike",
    "Iranian Revolutionary Guard 2026",
    "Israel Iran war 2026",
    "Iran nuclear deal collapse 2026",
    "Gulf of Oman conflict 2026",
    "Strait of Hormuz closure 2026",
    "Iran Houthi 2026",
    "Middle East war 2026",
    "Iran sanctions 2026",
]

# Date window we want to cover
DEFAULT_START = datetime(2026, 2, 28, tzinfo=timezone.utc)
DEFAULT_END   = datetime(2026, 4,  1, tzinfo=timezone.utc)

OUTPUT_DIR    = "data/raw/wiki"
MAX_WORDS_PER_CHUNK = 300      # words per ArticleModel — ~2 paragraphs
MIN_WORDS_PER_CHUNK = 30       # discard tiny section stubs
MAX_ARTICLES_PER_PAGE = 40     # cap so one huge page doesn't dominate

# Regex patterns used to extract inline dates from section text
_MONTH_NAMES = (
    "january|february|march|april|may|june|"
    "july|august|september|october|november|december"
)
_DATE_PATTERNS = [
    # "March 15, 2026" / "15 March 2026"
    re.compile(
        rf"(?:({_MONTH_NAMES})\s+(\d{{1,2}}),?\s+(202[5-9]))|"
        rf"(?:(\d{{1,2}})\s+({_MONTH_NAMES})\s+(202[5-9]))",
        re.IGNORECASE,
    ),
    # "2026-03-15"
    re.compile(r"(202[5-9])-(\d{2})-(\d{2})"),
]
_SECTION_DATE_RE = re.compile(
    rf"^(?:({_MONTH_NAMES})(?:\s+\d{{4}})?|"
    rf"(\d{{1,2}})\s+({_MONTH_NAMES})\s+(202[5-9]))",
    re.IGNORECASE,
)
_HEADER_RE = re.compile(r"^=+\s*(.*?)\s*=+$")


# ---------------------------------------------------------------------------
# Wikipedia fetch helpers  (mirrors extract_kg.py)
# ---------------------------------------------------------------------------

def _setup_wikipedia():
    import wikipedia
    wikipedia.set_user_agent(
        "NewsKG/1.0 (https://example.com; bot@example.com)"
    )
    wikipedia.API_URL = "https://en.wikipedia.org/w/api.php"
    return wikipedia


def _rest_fetch(title: str) -> Tuple[str, str]:
    """Direct MediaWiki API fetch as fallback."""
    import requests
    headers = {"User-Agent": "NewsKG/1.0 (bot@example.com)"}
    api = "https://en.wikipedia.org/w/api.php"
    p = requests.get(api, headers=headers, params={
        "action": "query", "prop": "extracts", "explaintext": 1,
        "titles": title, "format": "json", "redirects": 1,
    }, timeout=15).json()
    page = next(iter(p["query"]["pages"].values()))
    return page.get("title", title), page.get("extract", "")


def fetch_page(query: str, wikipedia) -> Optional[Tuple[str, str]]:
    """Return (title, full_text) or None."""
    try:
        results = wikipedia.search(query, results=3)
        if not results:
            return None
        for candidate in results:
            try:
                page = wikipedia.page(candidate, auto_suggest=False)
                return page.title, page.content
            except wikipedia.DisambiguationError as e:
                try:
                    page = wikipedia.page(e.options[0], auto_suggest=False)
                    return page.title, page.content
                except Exception:
                    continue
            except wikipedia.PageError:
                continue
    except Exception as err:
        print(f"  wikipedia lib failed for {query!r}: {err} — trying REST")
        try:
            return _rest_fetch(query)
        except Exception as e2:
            print(f"  REST also failed: {e2}")
    return None


# ---------------------------------------------------------------------------
# Chunking & date extraction
# ---------------------------------------------------------------------------

def _parse_inline_date(text: str, fallback: datetime) -> datetime:
    """Try to find a date in the first 400 chars of text."""
    snippet = text[:400]
    for pat in _DATE_PATTERNS:
        m = pat.search(snippet)
        if not m:
            continue
        groups = [g for g in m.groups() if g]
        raw = " ".join(groups)
        for fmt in ("%B %d %Y", "%d %B %Y", "%Y %m %d", "%B %Y"):
            try:
                return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return fallback


def _section_date_hint(heading: str) -> Optional[datetime]:
    """Return a date if the section heading encodes one, e.g. '== March 2026 =='."""
    m = _SECTION_DATE_RE.match(heading.strip())
    if not m:
        return None
    raw = heading.strip()
    for fmt in ("%B %Y", "%B", "%d %B %Y"):
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.year < 2020:
                dt = dt.replace(year=2026)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def split_into_sections(text: str) -> List[Tuple[str, str]]:
    """
    Split Wikipedia full-text into (heading, body) pairs.
    Top-level sections only; sub-sections are merged into their parent.
    """
    sections: List[Tuple[str, str]] = []
    current_heading = "Introduction"
    current_lines: List[str] = []

    for line in text.split("\n"):
        m = _HEADER_RE.match(line.strip())
        if m:
            body = " ".join(current_lines).strip()
            if body:
                sections.append((current_heading, body))
            current_heading = m.group(1)
            current_lines = []
        else:
            if line.strip():
                current_lines.append(line.strip())

    body = " ".join(current_lines).strip()
    if body:
        sections.append((current_heading, body))

    return sections


def chunk_section(
    heading: str,
    body: str,
    max_words: int = MAX_WORDS_PER_CHUNK,
) -> List[str]:
    """Split a section body into word-capped chunks."""
    words = body.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        if len(chunk.split()) >= MIN_WORDS_PER_CHUNK:
            chunks.append(f"{heading}: {chunk}")
    return chunks


def page_to_articles(
    title: str,
    content: str,
    start_date: datetime,
    end_date: datetime,
    source_url: str = "",
) -> List[dict]:
    """
    Convert a Wikipedia page into a list of article dicts compatible
    with ArticleLoader._parse_article().
    """
    sections  = split_into_sections(content)
    raw_chunks: List[Tuple[str, str, Optional[datetime]]] = []

    for heading, body in sections:
        heading_date = _section_date_hint(heading)
        for chunk in chunk_section(heading, body):
            raw_chunks.append((heading, chunk, heading_date))

    # Cap per page
    raw_chunks = raw_chunks[:MAX_ARTICLES_PER_PAGE]
    total = len(raw_chunks)
    if total == 0:
        return []

    # Linear interpolation fallback dates
    window_secs = (end_date - start_date).total_seconds()
    step        = window_secs / max(total - 1, 1)

    articles = []
    for i, (heading, chunk, hint_date) in enumerate(raw_chunks):
        interp_date = start_date + timedelta(seconds=i * step)
        if hint_date:
            # Clamp hinted date to our window
            pub_date = max(start_date, min(hint_date, end_date))
        else:
            pub_date = _parse_inline_date(chunk, interp_date)
            pub_date = max(start_date, min(pub_date, end_date))

        # Stable deterministic ID — same content always gets same ID
        uid = hashlib.sha256(
            f"{title}::{heading}::{chunk[:80]}".encode()
        ).hexdigest()[:16]

        articles.append({
            "id":           uid,
            "title":        f"{title} — {heading}",
            "content":      chunk,
            "source":       "wikipedia",
            "published_at": pub_date.isoformat(),
            "url":          (
                source_url or
                f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                f"#section_{i}"
            ),
        })

    return articles


# ---------------------------------------------------------------------------
# Main fetch logic
# ---------------------------------------------------------------------------

def fetch_all(
    start_date: datetime,
    end_date:   datetime,
    output_dir: str = OUTPUT_DIR,
    dry_run:    bool = False,
) -> List[dict]:
    import wikipedia as wp_lib
    _setup_wikipedia()

    os.makedirs(output_dir, exist_ok=True)

    all_articles: List[dict]  = []
    seen_titles:  set         = set()

    for query in SEARCH_QUERIES:
        print(f"\n→ Searching: {query!r}")
        result = fetch_page(query, wp_lib)
        if not result:
            print("  ✗ no page found")
            continue

        title, content = result
        if title in seen_titles:
            print(f"  (skip duplicate) {title!r}")
            continue
        seen_titles.add(title)

        word_count = len(content.split())
        print(f"  ✓ {title!r} — {word_count:,} words")

        articles = page_to_articles(title, content, start_date, end_date)
        print(f"    → {len(articles)} chunks")
        all_articles.extend(articles)

        time.sleep(0.5)   # be polite to Wikimedia

    # Deduplicate by id
    seen_ids: set = set()
    deduped: List[dict] = []
    for a in all_articles:
        if a["id"] not in seen_ids:
            seen_ids.add(a["id"])
            deduped.append(a)

    print(f"\n{'='*55}")
    print(f"  Total unique chunks : {len(deduped)}")
    print(f"  Wikipedia pages used: {len(seen_titles)}")
    print(f"  Date window         : {start_date.date()} → {end_date.date()}")
    print(f"{'='*55}")

    if dry_run:
        print("  [dry-run] nothing written")
        return deduped

    # Write one JSONL file per Wikipedia page title
    # Group by source title (first part of article title before " — ")
    from collections import defaultdict
    page_groups: dict = defaultdict(list)
    for a in deduped:
        page_title = a["title"].split(" — ")[0]
        page_groups[page_title].append(a)

    written = 0
    for page_title, articles in page_groups.items():
        safe_name = re.sub(r"[^\w\-]", "_", page_title)[:60]
        out_path  = os.path.join(output_dir, f"{safe_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for a in articles:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
        written += len(articles)
        print(f"  wrote {len(articles):3d} chunks → {out_path}")

    print(f"\n✅  {written} articles saved to {output_dir}/")
    return deduped


# ---------------------------------------------------------------------------
# Optional direct DB ingest
# ---------------------------------------------------------------------------

def ingest_to_db(articles: List[dict]) -> None:
    from src.ingestion.article_loader import ArticleLoader
    loader  = ArticleLoader()
    models  = [loader._parse_article(a) for a in articles]
    # Attach IDs so the pipeline can track status
    for model, raw in zip(models, articles):
        model.id = raw["id"]
    result = loader.ingest_to_db(models)
    print(
        f"\n✅  DB ingest: inserted={result['inserted']}  "
        f"skipped={result['skipped']}  failed={result['failed']}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch Iran-war-2026 Wikipedia pages and prepare for pipeline."
    )
    parser.add_argument(
        "--start-date", default="2026-02-28",
        help="Start of date window (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date", default="2026-04-01",
        help="End of date window (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR,
        help=f"Directory to write JSONL files (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--ingest", action="store_true",
        help="Also ingest articles directly into the SQLite DB",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch and count only — do not write files or touch DB",
    )
    args = parser.parse_args()

    start = datetime.fromisoformat(args.start_date).replace(tzinfo=timezone.utc)
    end   = datetime.fromisoformat(args.end_date).replace(tzinfo=timezone.utc)

    articles = fetch_all(
        start_date=start,
        end_date=end,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )

    if args.ingest and not args.dry_run:
        ingest_to_db(articles)

    if not args.dry_run:
        print("\nNext step — run the pipeline:")
        print(
            "  python main.py --run-all "
            f"--input {args.output_dir} "
            "--llm-batch-size 10 --llm-workers 4"
        )