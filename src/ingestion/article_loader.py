import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List, Dict
from typing import Optional

from src.models.article import ArticleModel
from src.utils.db import get_session, ArticleDB
from src.utils.logger import get_logger

log = get_logger(__name__)


class ArticleLoader:
    """
    Load articles from JSON/JSONL files and ingest them into the database.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_from_directory(self, directory: str) -> List[ArticleModel]:
        """
        Recursively load *.json and *.jsonl files from *directory*.
        Malformed lines/records are logged and skipped rather than crashing
        the entire batch.
        """
        articles: List[ArticleModel] = []
        if not os.path.isdir(directory):
            log.warning("directory_not_found", path=directory)
            return articles

        for root, _, files in os.walk(directory):
            for filename in files:
                if not (filename.endswith(".jsonl") or filename.endswith(".json")):
                    continue

                filepath = os.path.join(root, filename)
                try:
                    file_articles = self._load_file(filepath)
                    articles.extend(file_articles)
                except Exception as exc:
                    log.error("file_load_failed", path=filepath, error=str(exc))

        log.info("articles_loaded", count=len(articles), directory=directory)
        return articles

    def ingest_to_db(self, articles: List[ArticleModel]) -> Dict[str, int]:
        """
        Upsert articles into the DB.  URLs already present are skipped.
        Uses a single bulk-insert to avoid N+1 round-trips.

        Returns:
            {"inserted": int, "skipped": int, "total": int, "failed": int}
        """
        if not articles:
            return {"inserted": 0, "skipped": 0, "total": 0, "failed": 0}

        with _managed_session() as session:
            # 1. Fetch existing URLs in one query
            urls = [a.url for a in articles if a.url]
            existing_urls = {
                row[0] for row in
                session.query(ArticleDB.url).filter(ArticleDB.url.in_(urls)).all()
            }

            # 2. Build DB objects for new articles only
            new_db_articles: List[ArticleDB] = []
            skipped = 0
            for article in articles:
                if not article.url or article.url in existing_urls:
                    skipped += 1
                    continue

                # Defensive: ensure we don't pass a null ID if the DB expects
                # to auto-generate it.
                db_article = ArticleDB(
                    id=getattr(article, "id", None),
                    title=article.title,
                    content=article.content,
                    source=article.source,
                    published_at=article.published_at,
                    url=article.url,
                    status=getattr(article, "status", "ingested"),
                )
                new_db_articles.append(db_article)

            # 3. Bulk insert
            inserted = 0
            failed = 0
            if new_db_articles:
                try:
                    session.bulk_save_objects(new_db_articles)
                    session.commit()
                    inserted = len(new_db_articles)
                    log.info(
                        "ingestion_complete",
                        inserted=inserted,
                        skipped=skipped,
                        total=len(articles),
                    )
                except Exception as exc:
                    session.rollback()
                    failed = len(new_db_articles)
                    log.error("bulk_insert_failed", error=str(exc), count=failed)

        return {
            "inserted": inserted,
            "skipped": skipped,
            "total": len(articles),
            "failed": failed,
        }

    def get_unprocessed_articles(self, status: str = "ingested") -> List[ArticleModel]:
        """Return all articles with the given status."""
        with _managed_session() as session:
            try:
                db_articles = session.query(ArticleDB).filter_by(status=status).all()
                return [_to_model(db) for db in db_articles]
            except Exception as exc:
                log.error("fetch_unprocessed_failed", error=str(exc))
                return []

    def update_status(self, article_ids: List[str], status: str) -> int:
        """
        Bulk-update status for the given IDs.

        Returns:
            Number of rows updated (0 if none matched).
        """
        if not article_ids:
            return 0

        with _managed_session() as session:
            try:
                result = (
                    session.query(ArticleDB)
                    .filter(ArticleDB.id.in_(article_ids))
                    .update(
                        {
                            ArticleDB.status: status,
                            ArticleDB.updated_at: datetime.now(timezone.utc),
                        },
                        synchronize_session=False,
                    )
                )
                session.commit()
                log.info("status_updated", count=result, status=status)
                return result
            except Exception as exc:
                session.rollback()
                log.error("status_update_failed", error=str(exc))
                return 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_file(self, filepath: str) -> List[ArticleModel]:
        """Load a single .json or .jsonl file."""
        articles: List[ArticleModel] = []
        with open(filepath, "r", encoding="utf-8") as fh:
            if filepath.endswith(".jsonl"):
                for line_no, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        articles.append(self._parse_article(data))
                    except json.JSONDecodeError as exc:
                        log.warning(
                            "jsonl_decode_error",
                            path=filepath,
                            line=line_no,
                            error=str(exc),
                        )
                    except Exception as exc:
                        log.warning(
                            "jsonl_parse_error",
                            path=filepath,
                            line=line_no,
                            error=str(exc),
                        )
            else:
                try:
                    data = json.load(fh)
                except json.JSONDecodeError as exc:
                    log.error("json_decode_error", path=filepath, error=str(exc))
                    return articles

                if isinstance(data, list):
                    for item in data:
                        try:
                            articles.append(self._parse_article(item))
                        except Exception as exc:
                            log.warning("json_item_parse_error", path=filepath, error=str(exc))
                else:
                    articles.append(self._parse_article(data))

        return articles

    def _parse_article(self, data: dict) -> ArticleModel:
        """Parse a raw dict into an ArticleModel with safe date handling."""
        published_raw = data.get("published_at") or data.get("published")
        published = _safe_parse_date(published_raw)

        return ArticleModel(
            title=data.get("title", ""),
            content=data.get("content", ""),
            source=data.get("source", ""),
            published_at=published,
            url=data.get("url", ""),
        )


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _safe_parse_date(value) -> datetime:
    """
    Parse a datetime from string, epoch, or datetime object.
    Falls back to UTC now on failure.
    """
    if isinstance(value, datetime):
        return value

    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)

    if isinstance(value, str):
        value = value.strip()
        # Handle ISO formats with trailing 'Z'
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass

        # Common RSS / news formats
        formats = (
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S %Z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%d %b %Y",
        )
        for fmt in formats:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue

    log.warning("date_parse_failed", raw=value)
    return datetime.now(timezone.utc)


def _to_model(db: ArticleDB) -> ArticleModel:
    """Convert a DB row back to an ArticleModel."""
    return ArticleModel(
        id=db.id,
        title=db.title,
        content=db.content,
        source=db.source,
        published_at=db.published_at,
        url=db.url,
        status=db.status,
    )


@contextmanager
def _managed_session():
    """
    Context manager that yields a DB session and guarantees
    rollback on exception *and* proper close().
    """
    session = get_session()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()