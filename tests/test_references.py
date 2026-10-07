"""
tests/test_references.py
Unit tests for IEEE reference formatting.
"""

import unittest
from analysis.references import format_ieee_reference, build_reference_list, build_reference_map
from utils.helpers import authors_to_ieee


class TestAuthorsToIEEE(unittest.TestCase):
    def test_single_author(self):
        result = authors_to_ieee(["Alice Smith"])
        self.assertIn("Smith", result)
        self.assertIn("A.", result)

    def test_two_authors(self):
        result = authors_to_ieee(["Alice Smith", "Bob Jones"])
        self.assertIn("and", result)

    def test_three_authors(self):
        result = authors_to_ieee(["Alice Smith", "Bob Jones", "Carol Brown"])
        self.assertIn(", and", result)

    def test_empty_list(self):
        result = authors_to_ieee([])
        self.assertEqual(result, "")

    def test_single_name(self):
        result = authors_to_ieee(["Smith"])
        self.assertEqual(result, "Smith")


class TestFormatIEEEReference(unittest.TestCase):
    PAPER = {
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        "year": 2017,
        "source": "arxiv",
        "arxiv_id": "1706.03762",
        "doi": "",
        "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
    }

    def test_contains_title(self):
        ref = format_ieee_reference(self.PAPER, 1)
        self.assertIn("Attention Is All You Need", ref)

    def test_contains_year(self):
        ref = format_ieee_reference(self.PAPER, 1)
        self.assertIn("2017", ref)

    def test_contains_arxiv_id(self):
        ref = format_ieee_reference(self.PAPER, 1)
        self.assertIn("1706.03762", ref)

    def test_starts_with_number(self):
        ref = format_ieee_reference(self.PAPER, 5)
        self.assertTrue(ref.startswith("[5]"))

    def test_missing_authors(self):
        paper = dict(self.PAPER)
        paper["authors"] = []
        ref = format_ieee_reference(paper, 1)
        self.assertIn("Unknown Author", ref)


class TestBuildReferenceList(unittest.TestCase):
    def test_numbering(self):
        papers = [
            {"title": "Paper A", "authors": ["A. Author"], "year": 2020,
             "source": "arxiv", "arxiv_id": "2001.00001", "doi": "", "pdf_url": ""},
            {"title": "Paper B", "authors": ["B. Author"], "year": 2021,
             "source": "arxiv", "arxiv_id": "2101.00001", "doi": "", "pdf_url": ""},
        ]
        refs = build_reference_list(papers)
        self.assertEqual(len(refs), 2)
        self.assertTrue(refs[0].startswith("[1]"))
        self.assertTrue(refs[1].startswith("[2]"))

    def test_empty(self):
        self.assertEqual(build_reference_list([]), [])


class TestBuildReferenceMap(unittest.TestCase):
    def test_mapping(self):
        papers = [
            {"arxiv_id": "2001.00001", "title": "Paper A", "uid": "arxiv:2001.00001"},
            {"arxiv_id": "2101.00001", "title": "Paper B", "uid": "arxiv:2101.00001"},
        ]
        mapping = build_reference_map(papers)
        self.assertEqual(mapping["arxiv:2001.00001"], 1)
        self.assertEqual(mapping["arxiv:2101.00001"], 2)


if __name__ == "__main__":
    unittest.main()
