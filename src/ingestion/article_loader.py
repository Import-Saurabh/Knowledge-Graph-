import json
import os
from datetime import datetime
from typing import List
from src.models.article import ArticleModel
from src.utils.db import get_session, ArticleDB
from src.utils.logger import get_logger

log = get_logger(__name__)

class ArticleLoader:
    def load_from_directory(self, directory: str) -> List[ArticleModel]:
        articles = []
        if not os.path.exists(directory):
            log.warning("directory_not_found", path=directory)
            return articles

        for filename in os.listdir(directory):
            if filename.endswith(".jsonl") or filename.endswith(".json"):
                filepath = os.path.join(directory, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    if filename.endswith(".jsonl"):
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            data = json.loads(line)
                            articles.append(self._parse_article(data))
                    else:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                articles.append(self._parse_article(item))
                        else:
                            articles.append(self._parse_article(data))

        log.info("articles_loaded", count=len(articles), directory=directory)
        return articles

    def _parse_article(self, data: dict) -> ArticleModel:
        published = data.get("published_at", datetime.utcnow().isoformat())
        if isinstance(published, str):
            published = datetime.fromisoformat(published.replace("Z", "+00:00"))
        return ArticleModel(
            title=data.get("title", ""),
            content=data.get("content", ""),
            source=data.get("source", ""),
            published_at=published,
            url=data.get("url", "")
        )

    def ingest_to_db(self, articles: List[ArticleModel]) -> dict:
        session = get_session()
        inserted = 0
        skipped = 0

        for article in articles:
            existing = session.query(ArticleDB).filter_by(url=article.url).first()
            if existing:
                skipped += 1
                continue

            db_article = ArticleDB(
                id=article.id,
                title=article.title,
                content=article.content,
                source=article.source,
                published_at=article.published_at,
                url=article.url,
                status=article.status
            )
            session.add(db_article)
            inserted += 1

        session.commit()
        session.close()

        log.info("ingestion_complete", inserted=inserted, skipped=skipped)
        return {"inserted": inserted, "skipped": skipped, "total": len(articles)}

    def get_unprocessed_articles(self, status: str = "ingested") -> List[ArticleModel]:
        session = get_session()
        db_articles = session.query(ArticleDB).filter_by(status=status).all()
        articles = []
        for db in db_articles:
            articles.append(ArticleModel(
                id=db.id,
                title=db.title,
                content=db.content,
                source=db.source,
                published_at=db.published_at,
                url=db.url,
                status=db.status
            ))
        session.close()
        return articles

    def update_status(self, article_ids: List[str], status: str):
        session = get_session()
        session.query(ArticleDB).filter(ArticleDB.id.in_(article_ids)).update({
            ArticleDB.status: status,
            ArticleDB.updated_at: datetime.utcnow()
        }, synchronize_session=False)
        session.commit()
        session.close()
