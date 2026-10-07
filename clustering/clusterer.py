"""
clustering/clusterer.py
Clusters research papers by semantic similarity using scikit-learn.

Supports K-Means clustering with automatic optimal-k selection
via Silhouette Score. Also provides Agglomerative Clustering as an
alternative. Returns cluster assignments and quality metrics.
"""

import logging
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from config import MIN_CLUSTERS, MAX_CLUSTERS, DEFAULT_CLUSTERS, RANDOM_STATE

logger = logging.getLogger(__name__)


def find_optimal_clusters(
    embeddings: np.ndarray,
    min_k: int = MIN_CLUSTERS,
    max_k: int = MAX_CLUSTERS,
) -> tuple[int, float]:
    """
    Evaluate K-Means for k in [min_k, max_k] and return the k with the
    highest Silhouette Score along with that score.

    Requires at least min_k + 1 samples. Falls back to DEFAULT_CLUSTERS
    if the corpus is too small.
    """
    n_samples = len(embeddings)
    max_k = min(max_k, n_samples - 1)

    if max_k < min_k:
        logger.warning("Too few papers for clustering; using %d cluster(s).", 1)
        return 1, 0.0

    best_k = min(DEFAULT_CLUSTERS, max_k)
    best_score = -1.0
    norms = normalize(embeddings)

    for k in range(min_k, max_k + 1):
        try:
            labels = KMeans(
                n_clusters=k, random_state=RANDOM_STATE, n_init=10
            ).fit_predict(norms)
            score = silhouette_score(norms, labels)
            logger.debug("k=%d  silhouette=%.4f", k, score)
            if score > best_score:
                best_score = score
                best_k = k
        except Exception as exc:  # noqa: BLE001
            logger.warning("Clustering failed for k=%d: %s", k, exc)

    return best_k, best_score


def cluster_papers(
    embeddings: np.ndarray,
    n_clusters: Optional[int] = None,
    method: str = "kmeans",
) -> tuple[np.ndarray, float]:
    """
    Cluster papers and return (labels_array, silhouette_score).

    Parameters
    ----------
    embeddings : np.ndarray of shape (N, D)
    n_clusters : int or None. If None, the optimal k is determined automatically.
    method     : 'kmeans' or 'agglomerative'

    Returns
    -------
    labels : np.ndarray of int, length N
    score  : float  (Silhouette Score; 0.0 if only 1 cluster)
    """
    if len(embeddings) < 2:
        logger.warning("Fewer than 2 papers — skipping clustering.")
        return np.zeros(len(embeddings), dtype=int), 0.0

    norms = normalize(embeddings)

    if n_clusters is None:
        n_clusters, _ = find_optimal_clusters(norms)

    n_clusters = max(1, min(n_clusters, len(embeddings) - 1))

    if n_clusters == 1:
        return np.zeros(len(embeddings), dtype=int), 0.0

    try:
        if method == "agglomerative":
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
            labels = model.fit_predict(norms)
        else:
            model = KMeans(
                n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10
            )
            labels = model.fit_predict(norms)

        score = silhouette_score(norms, labels)
        logger.info(
            "Clustering complete: method=%s, k=%d, silhouette=%.4f",
            method, n_clusters, score,
        )
        return labels, float(score)

    except Exception as exc:  # noqa: BLE001
        logger.error("Clustering failed: %s", exc)
        return np.zeros(len(embeddings), dtype=int), 0.0


def get_cluster_summary(
    papers: list[dict],
    labels: np.ndarray,
) -> dict[int, list[dict]]:
    """
    Group papers by their cluster label.

    Returns a dict mapping cluster_id -> list of paper dicts.
    """
    clusters: dict[int, list[dict]] = {}
    for paper, label in zip(papers, labels):
        cid = int(label)
        clusters.setdefault(cid, []).append(paper)
    return clusters
