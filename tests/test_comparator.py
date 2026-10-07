"""
tests/test_comparator.py
Unit tests for the cross-paper comparison table and statistics.
"""

import unittest
import pandas as pd
from analysis.comparator import build_comparison_table, compute_statistics


SAMPLE_PAPERS = [
    {
        "title": "Deep Learning for NLP",
        "authors": ["Alice Smith", "Bob Jones"],
        "year": 2022,
        "source": "arxiv",
        "categories": ["cs.CL", "cs.LG"],
        "theme_name": "Neural NLP Methods",
        "analysis": {
            "research_problem": "How to improve NLP with deep learning",
            "methodology": "LSTM-based sequence model",
            "dataset_experimental_setup": "Penn Treebank",
            "key_results": "87.3% accuracy",
            "limitations": "High computational cost",
        },
    },
    {
        "title": "Transformer Models for Text Classification",
        "authors": ["Carol Brown"],
        "year": 2023,
        "source": "semantic_scholar",
        "categories": ["cs.CL"],
        "theme_name": "Transformer Architectures",
        "analysis": {
            "research_problem": "Text classification at scale",
            "methodology": "Fine-tuned BERT",
            "dataset_experimental_setup": "SST-2, IMDB",
            "key_results": "95.1% on SST-2",
            "limitations": "Not reported in the paper.",
        },
    },
]


class TestBuildComparisonTable(unittest.TestCase):
    def setUp(self):
        self.df = build_comparison_table(SAMPLE_PAPERS)

    def test_returns_dataframe(self):
        self.assertIsInstance(self.df, pd.DataFrame)

    def test_correct_column_count(self):
        expected_cols = {"Paper", "Year", "Authors", "Problem", "Method",
                         "Dataset", "Results", "Limitations", "Theme"}
        self.assertEqual(set(self.df.columns), expected_cols)

    def test_row_count_matches_papers(self):
        self.assertEqual(len(self.df), len(SAMPLE_PAPERS))

    def test_title_truncated(self):
        for title in self.df["Paper"]:
            self.assertLessEqual(len(title), 85)

    def test_missing_analysis_shows_dash(self):
        papers = [{"title": "Empty", "authors": [], "year": None,
                   "source": "arxiv", "categories": [], "theme_name": "",
                   "analysis": {}}]
        df = build_comparison_table(papers)
        self.assertEqual(df.loc[0, "Problem"], "-")


class TestComputeStatistics(unittest.TestCase):
    def setUp(self):
        self.stats = compute_statistics(SAMPLE_PAPERS)

    def test_total_count(self):
        self.assertEqual(self.stats["total"], 2)

    def test_by_year(self):
        self.assertIn("2022", self.stats["by_year"])
        self.assertIn("2023", self.stats["by_year"])

    def test_by_source(self):
        self.assertIn("arxiv", self.stats["by_source"])
        self.assertIn("semantic_scholar", self.stats["by_source"])

    def test_year_range(self):
        low, high = self.stats["year_range"]
        self.assertEqual(low, 2022)
        self.assertEqual(high, 2023)

    def test_by_theme(self):
        self.assertIn("Neural NLP Methods", self.stats["by_theme"])
        self.assertIn("Transformer Architectures", self.stats["by_theme"])

    def test_empty_papers(self):
        stats = compute_statistics([])
        self.assertEqual(stats["total"], 0)
        self.assertEqual(stats["year_range"], (None, None))


if __name__ == "__main__":
    unittest.main()
