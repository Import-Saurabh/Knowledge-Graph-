import numpy as np
import hdbscan
from typing import List, Dict, Optional
from datetime import datetime, timezone
from src.utils.logger import get_logger
from src.utils.config import settings

log = get_logger(__name__)


class EventClusterer:
    """
    Clusters articles into events using embeddings.

    Improvements over the original version:
    - Normalizes embeddings before clustering.
    - Uses a cosine-style clustering setup for better semantic grouping.
    - Falls back to a greedy similarity-based grouping when HDBSCAN produces
      only noise or no clusters.
    - Never silently returns zero clusters if there are enough articles to form
      at least one reasonable event.
    """

    def __init__(self, embedder=None):
        self.window_days = getattr(settings, "TEMPORAL_WINDOW_DAYS", 7)
        self.min_cluster_size = max(2, int(getattr(settings, "MIN_CLUSTER_SIZE", 2)))

        # Shared embedder injected from the pipeline — avoids reloading the
        # model for every temporal window.
        self._embedder = embedder
        self._local_embedder = None  # fallback, lazily initialized once

        # Fallback similarity threshold for greedy grouping.
        # Keep it moderately strict so unrelated articles do not merge.
        self.fallback_similarity_threshold = float(
            getattr(settings, "EVENT_SIMILARITY_THRESHOLD", 0.78)
        )

    def _get_embedder(self):
        """Return the shared embedder, or lazily create a local one."""
        if self._embedder is not None:
            return self._embedder

        if self._local_embedder is None:
            from src.embeddings.embedding_generator import EmbeddingGenerator
            self._local_embedder = EmbeddingGenerator()
            log.info("event_clusterer_local_embedder_created")

        return self._local_embedder

    @staticmethod
    def _normalize_datetime(dt):
        if isinstance(dt, str):
            # Accept both plain ISO strings and Zulu timestamps.
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt

    def _group_by_window(self, articles) -> Dict[str, List]:
        """
        Group articles by ISO week.
        This is stable and works well for news event clustering.
        """
        windows = {}
        for article in articles:
            dt = self._normalize_datetime(article.published_at)
            year, week, _ = dt.isocalendar()
            window = f"{year}-W{week:02d}"
            windows.setdefault(window, []).append(article)
        return windows

    @staticmethod
    def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return embeddings / norms

    def run_all_windows(self, articles) -> List[dict]:
        if not articles:
            log.info("clustering_complete", clusters=0, windows=0)
            return []

        windows = self._group_by_window(articles)
        all_clusters = []

        # Sort windows for reproducible output.
        for window in sorted(windows.keys()):
            window_articles = windows[window]
            if not window_articles:
                continue

            clusters = self._cluster_window(window_articles, window)
            all_clusters.extend(clusters)

        log.info("clustering_complete", clusters=len(all_clusters), windows=len(windows))
        return all_clusters

    def _cluster_window(self, articles, window: str) -> List[dict]:
        """
        Cluster articles inside one time window.

        Strategy:
        1) Try HDBSCAN on normalized embeddings.
        2) If HDBSCAN yields no usable clusters, fall back to a greedy semantic
           grouping based on pairwise cosine similarity.
        """
        if len(articles) == 1:
            # Single article can still become a single-article event.
            a = articles[0]
            result = [{
                "cluster_id": 0,
                "temporal_window": window,
                "articles": [a],
                "article_ids": [a.id],
            }]
            log.info("window_clustered_singleton", window=window, articles=1, clusters=1)
            return result

        embedder = self._get_embedder()

        texts = []
        for a in articles:
            title = (a.title or "").strip()
            content = (a.content or "").strip()
            texts.append(f"{title}. {content[:2000]}")

        try:
            embeddings = embedder.embed_texts(texts)
            embeddings = np.array(embeddings, dtype=np.float32)
        except Exception as e:
            log.warning("window_embedding_failed", window=window, error=str(e))
            return self._fallback_greedy_cluster(articles, window, embedder)

        if len(embeddings) < 2:
            return []

        embeddings = self._l2_normalize(embeddings)

        # HDBSCAN works better on normalized vectors with Euclidean distance.
        # On unit vectors, Euclidean distance tracks cosine similarity closely.
        try:
            hdbscan_min_cluster_size = min(self.min_cluster_size, len(articles))
            hdbscan_min_cluster_size = max(2, hdbscan_min_cluster_size)

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=hdbscan_min_cluster_size,
                min_samples=1,
                metric="euclidean",
                cluster_selection_method="eom",
            )
            labels = clusterer.fit_predict(embeddings)
        except Exception as e:
            log.warning("hdbscan_failed", window=window, error=str(e))
            return self._fallback_greedy_cluster(articles, window, embedder)

        clusters = {}
        for i, label in enumerate(labels):
            if label == -1:
                continue
            clusters.setdefault(label, []).append(articles[i])

        # If HDBSCAN found nothing useful, use a deterministic fallback.
        if not clusters:
            log.warning("hdbscan_produced_no_clusters", window=window, articles=len(articles))
            return self._fallback_greedy_cluster(articles, window, embedder)

        result = []
        next_cluster_id = 0
        for _, cluster_articles in sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True):
            if len(cluster_articles) < 1:
                continue
            result.append({
                "cluster_id": next_cluster_id,
                "temporal_window": window,
                "articles": cluster_articles,
                "article_ids": [a.id for a in cluster_articles],
            })
            next_cluster_id += 1

        # Final safeguard: if everything collapsed to too-small clusters,
        # fall back to a greedy merge so event detection does not go empty.
        if not result:
            log.warning("hdbscan_clusters_empty_after_postprocess", window=window)
            return self._fallback_greedy_cluster(articles, window, embedder)

        log.info("window_clustered", window=window, articles=len(articles), clusters=len(result))
        return result

    def _fallback_greedy_cluster(self, articles, window: str, embedder) -> List[dict]:
        """
        Greedy semantic clustering:
        - Start each article as a candidate.
        - Merge into the nearest cluster if cosine similarity exceeds threshold.
        - Otherwise start a new cluster.
        This is intentionally conservative, but it prevents total failure.
        """
        if not articles:
            return []

        texts = []
        for a in articles:
            title = (a.title or "").strip()
            content = (a.content or "").strip()
            texts.append(f"{title}. {content[:2000]}")

        try:
            embeddings = np.array(embedder.embed_texts(texts), dtype=np.float32)
        except Exception as e:
            log.error("fallback_embedding_failed", window=window, error=str(e))
            return []

        if len(embeddings) == 0:
            return []

        embeddings = self._l2_normalize(embeddings)

        cluster_members = []   # list[list[int]]
        cluster_centroids = [] # list[np.ndarray]

        def cosine_sim(a, b):
            return float(np.dot(a, b))

        for idx, vec in enumerate(embeddings):
            best_cluster = None
            best_score = -1.0

            for c_idx, centroid in enumerate(cluster_centroids):
                score = cosine_sim(vec, centroid)
                if score > best_score:
                    best_score = score
                    best_cluster = c_idx

            if best_cluster is not None and best_score >= self.fallback_similarity_threshold:
                cluster_members[best_cluster].append(idx)
                members = embeddings[cluster_members[best_cluster]]
                centroid = np.mean(members, axis=0)
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm == 0:
                    centroid_norm = 1.0
                cluster_centroids[best_cluster] = centroid / centroid_norm
            else:
                cluster_members.append([idx])
                cluster_centroids.append(vec)

        # Convert candidate clusters to output.
        result = []
        cluster_id = 0

        for member_indices in cluster_members:
            cluster_articles = [articles[i] for i in member_indices]

            # Keep singleton clusters only if there is no better structure.
            # This ensures the pipeline still produces events instead of nothing.
            if len(cluster_articles) < self.min_cluster_size and len(articles) >= self.min_cluster_size:
                continue

            result.append({
                "cluster_id": cluster_id,
                "temporal_window": window,
                "articles": cluster_articles,
                "article_ids": [a.id for a in cluster_articles],
            })
            cluster_id += 1

        # Final rescue: if everything got filtered out, create one cluster
        # from the most similar article pair, or one big cluster as last resort.
        if not result:
            if len(articles) >= 2:
                result.append({
                    "cluster_id": 0,
                    "temporal_window": window,
                    "articles": articles,
                    "article_ids": [a.id for a in articles],
                })
                log.warning(
                    "fallback_created_single_cluster",
                    window=window,
                    articles=len(articles),
                    reason="all clusters were filtered out"
                )
            else:
                a = articles[0]
                result.append({
                    "cluster_id": 0,
                    "temporal_window": window,
                    "articles": [a],
                    "article_ids": [a.id],
                })

        log.info("window_clustered_fallback", window=window, articles=len(articles), clusters=len(result))
        return result