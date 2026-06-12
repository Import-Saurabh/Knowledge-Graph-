from typing import List
from src.models.event import EventModel
from src.utils.logger import get_logger
import uuid

log = get_logger(__name__)

class EventBuilder:
    def build_event(self, cluster_data: dict) -> EventModel:
        articles = cluster_data["articles"]
        article_ids = cluster_data["article_ids"]

        # Build context from titles and first sentences
        titles = [a.title for a in articles]
        contents = [a.content[:500] for a in articles]

        context = "\n\n".join([
            f"Title: {t}\nExcerpt: {c[:300]}..." 
            for t, c in zip(titles, contents)
        ])

        # Pick representative articles (first 3)
        reps = article_ids[:3]

        event = EventModel(
            event_id=str(uuid.uuid4()),
            cluster_id=cluster_data["cluster_id"],
            temporal_window=cluster_data["temporal_window"],
            article_ids=article_ids,
            representative_article_ids=reps,
            context=context
        )

        log.info("event_built", event_id=event.event_id, articles=len(article_ids))
        return event
