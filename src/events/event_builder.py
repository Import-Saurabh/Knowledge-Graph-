"""
src/events/event_builder.py

Additions vs original:
  • EventBuilder.load_from_db() — static method that reconstructs
    EventModel objects from ArticleDB rows whose cluster_id is set.
    Used by main.py when resuming from --from-stage 9 so that clustering
    does not need to be re-run.
"""

import uuid
from collections import defaultdict
from typing import List

from src.models.event import EventModel
from src.utils.logger import get_logger

log = get_logger(__name__)


class EventBuilder:
    def build_event(self, cluster_data: dict) -> EventModel:
        articles    = cluster_data["articles"]
        article_ids = cluster_data["article_ids"]

        titles   = [a.title for a in articles]
        contents = [a.content[:500] for a in articles]

        context = "\n\n".join([
            f"Title: {t}\nExcerpt: {c[:300]}..."
            for t, c in zip(titles, contents)
        ])

        reps = article_ids[:3]

        event = EventModel(
            event_id=str(uuid.uuid4()),
            cluster_id=cluster_data["cluster_id"],
            temporal_window=cluster_data["temporal_window"],
            article_ids=article_ids,
            representative_article_ids=reps,
            context=context,
        )

        log.info("event_built", event_id=event.event_id, articles=len(article_ids))
        return event

    # ------------------------------------------------------------------
    # DB resume path  (called by main.py when --from-stage >= 9)
    # ------------------------------------------------------------------

    @staticmethod
    def load_from_db() -> List[EventModel]:
        """
        Reconstruct EventModel objects from already-clustered ArticleDB rows.

        Articles are grouped by cluster_id.  Each group becomes one EventModel
        with the same context format that build_event() would have produced.

        Returns an empty list (with a warning) if no clustered articles are
        found, so the caller can decide whether to abort or continue.
        """
        from src.utils.db import get_session, ArticleDB  # local import avoids circular deps

        session = get_session()
        try:
            rows = (
                session.query(ArticleDB)
                .filter(
                    ArticleDB.cluster_id.isnot(None),
                    ArticleDB.status.in_(["clustered", "embedded", "deduplicated"]),
                )
                .all()
            )
        finally:
            session.close()

        if not rows:
            log.warning("load_from_db_empty",
                        hint="No clustered articles found. Run from --from-stage 7 instead.")
            return []

        # Group by cluster_id preserving insertion order
        buckets: dict[str, list] = defaultdict(list)
        for row in rows:
            buckets[row.cluster_id].append(row)

        events: List[EventModel] = []
        for cluster_id, cluster_rows in buckets.items():
            article_ids      = [r.id for r in cluster_rows]
            temporal_window  = cluster_rows[0].temporal_window or "unknown"

            context = "\n\n".join([
                f"Title: {r.title}\nExcerpt: {(r.content or '')[:300]}..."
                for r in cluster_rows[:10]   # cap context length
            ])

            event = EventModel(
                event_id=str(uuid.uuid4()),
                cluster_id=cluster_id,
                temporal_window=temporal_window,
                article_ids=article_ids,
                representative_article_ids=article_ids[:3],
                context=context,
            )
            events.append(event)

        log.info("events_loaded_from_db",
                 clusters=len(events),
                 total_articles=len(rows))
        return events