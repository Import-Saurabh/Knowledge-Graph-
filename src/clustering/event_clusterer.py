import numpy as np
import hdbscan
from typing import List, Dict
from datetime import datetime, timedelta
from src.utils.logger import get_logger
from src.utils.config import settings
from src.utils.db import get_session, ArticleDB

log = get_logger(__name__)

class EventClusterer:
    def __init__(self):
        self.window_days = settings.TEMPORAL_WINDOW_DAYS
        self.min_cluster_size = settings.MIN_CLUSTER_SIZE

    def run_all_windows(self, articles) -> List[dict]:
        # Group articles by temporal window
        windows = self._group_by_window(articles)
        all_clusters = []

        for window, window_articles in windows.items():
            if len(window_articles) < self.min_cluster_size:
                continue

            clusters = self._cluster_window(window_articles, window)
            all_clusters.extend(clusters)

        log.info("clustering_complete", clusters=len(all_clusters), windows=len(windows))
        return all_clusters

    def _group_by_window(self, articles) -> Dict[str, List]:
        windows = {}
        for article in articles:
            dt = article.published_at
            if isinstance(dt, str):
                dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
            year, week, _ = dt.isocalendar()
            window = f"{year}-W{week:02d}"
            if window not in windows:
                windows[window] = []
            windows[window].append(article)
        return windows

    def _cluster_window(self, articles, window: str) -> List[dict]:
        from src.embeddings.embedding_generator import EmbeddingGenerator
        embedder = EmbeddingGenerator()

        texts = [f"{a.title}. {a.content[:2000]}" for a in articles]
        embeddings = embedder.embed_texts(texts)
        embeddings = np.array(embeddings)

        if len(embeddings) < self.min_cluster_size:
            return []

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom"
        )
        labels = clusterer.fit_predict(embeddings)

        clusters = {}
        for i, label in enumerate(labels):
            if label == -1:
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(articles[i])

        result = []
        for cluster_id, cluster_articles in clusters.items():
            result.append({
                "cluster_id": cluster_id,
                "temporal_window": window,
                "articles": cluster_articles,
                "article_ids": [a.id for a in cluster_articles]
            })

        return result
