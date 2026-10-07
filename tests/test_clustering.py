"""
tests/test_clustering.py
Unit tests for the clustering module.
"""

import unittest
import numpy as np
from clustering.clusterer import cluster_papers, find_optimal_clusters, get_cluster_summary


class TestFindOptimalClusters(unittest.TestCase):
    def _make_embeddings(self, n: int, dim: int = 16) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.standard_normal((n, dim))

    def test_returns_valid_k(self):
        emb = self._make_embeddings(20)
        k, score = find_optimal_clusters(emb, min_k=2, max_k=5)
        self.assertGreaterEqual(k, 2)
        self.assertLessEqual(k, 5)

    def test_returns_1_for_too_few_samples(self):
        emb = self._make_embeddings(2)
        k, score = find_optimal_clusters(emb, min_k=3, max_k=5)
        self.assertEqual(k, 1)


class TestClusterPapers(unittest.TestCase):
    def _make_embeddings(self, n: int, dim: int = 16) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.standard_normal((n, dim))

    def test_output_shape(self):
        emb = self._make_embeddings(15)
        labels, score = cluster_papers(emb, n_clusters=3)
        self.assertEqual(len(labels), 15)

    def test_labels_in_range(self):
        emb = self._make_embeddings(15)
        labels, _ = cluster_papers(emb, n_clusters=3)
        self.assertTrue(all(0 <= l < 3 for l in labels))

    def test_single_paper(self):
        emb = self._make_embeddings(1)
        labels, score = cluster_papers(emb)
        self.assertEqual(labels[0], 0)
        self.assertEqual(score, 0.0)

    def test_agglomerative_method(self):
        emb = self._make_embeddings(12)
        labels, score = cluster_papers(emb, n_clusters=3, method="agglomerative")
        self.assertEqual(len(labels), 12)

    def test_silhouette_in_valid_range(self):
        emb = self._make_embeddings(20)
        _, score = cluster_papers(emb, n_clusters=4)
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)


class TestGetClusterSummary(unittest.TestCase):
    def test_correct_grouping(self):
        papers = [{"title": "A"}, {"title": "B"}, {"title": "C"}]
        labels = np.array([0, 1, 0])
        summary = get_cluster_summary(papers, labels)
        self.assertIn(0, summary)
        self.assertIn(1, summary)
        self.assertEqual(len(summary[0]), 2)
        self.assertEqual(len(summary[1]), 1)

    def test_single_cluster(self):
        papers = [{"title": "X"}, {"title": "Y"}]
        labels = np.array([0, 0])
        summary = get_cluster_summary(papers, labels)
        self.assertEqual(len(summary), 1)
        self.assertEqual(len(summary[0]), 2)


if __name__ == "__main__":
    unittest.main()
