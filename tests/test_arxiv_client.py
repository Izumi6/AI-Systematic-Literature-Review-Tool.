"""
tests/test_arxiv_client.py
Unit tests for the arXiv API client and deduplication logic.
"""

import unittest
from unittest.mock import patch, MagicMock

from api.arxiv_client import ArxivClient, Paper, deduplicate_papers
from utils.helpers import normalize_title, parse_year


class TestNormalizeTitle(unittest.TestCase):
    def test_removes_punctuation(self):
        result = normalize_title("BERT: Pre-training of Deep Bidirectional Transformers!")
        self.assertNotIn(":", result)
        self.assertNotIn("!", result)
        self.assertNotIn("-", result)

    def test_lowercases(self):
        result = normalize_title("BERT Model")
        self.assertEqual(result, normalize_title("bert model"))

    def test_empty_string(self):
        self.assertEqual(normalize_title(""), "")


class TestParseYear(unittest.TestCase):
    def test_iso_date(self):
        self.assertEqual(parse_year("2023-05-15T12:00:00Z"), 2023)

    def test_year_only(self):
        self.assertEqual(parse_year("2021"), 2021)

    def test_none_on_invalid(self):
        self.assertIsNone(parse_year("not a date"))

    def test_none_on_empty(self):
        self.assertIsNone(parse_year(""))


class TestDeduplicatePapers(unittest.TestCase):
    def _make_paper(self, arxiv_id="", title="Paper", doi=""):
        return Paper(arxiv_id=arxiv_id, title=title, doi=doi)

    def test_removes_same_arxiv_id(self):
        papers = [
            self._make_paper(arxiv_id="2301.00001", title="Paper A"),
            self._make_paper(arxiv_id="2301.00001", title="Paper A duplicate"),
        ]
        unique = deduplicate_papers(papers)
        self.assertEqual(len(unique), 1)
        self.assertEqual(unique[0].title, "Paper A")

    def test_removes_same_normalized_title(self):
        papers = [
            self._make_paper(title="Attention Is All You Need"),
            self._make_paper(title="Attention is all you need!"),
        ]
        unique = deduplicate_papers(papers)
        self.assertEqual(len(unique), 1)

    def test_keeps_different_papers(self):
        papers = [
            self._make_paper(arxiv_id="2301.00001", title="Paper A"),
            self._make_paper(arxiv_id="2301.00002", title="Paper B"),
        ]
        unique = deduplicate_papers(papers)
        self.assertEqual(len(unique), 2)

    def test_empty_list(self):
        self.assertEqual(deduplicate_papers([]), [])


SAMPLE_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <entry>
    <id>http://arxiv.org/abs/2301.00001v1</id>
    <title>Test Paper on Machine Learning</title>
    <summary>This is a test abstract about machine learning methods.</summary>
    <published>2023-01-15T00:00:00Z</published>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <category term="cs.LG"/>
    <link title="pdf" href="https://arxiv.org/pdf/2301.00001"/>
    <arxiv:doi>10.1000/test.doi</arxiv:doi>
  </entry>
</feed>"""


class TestArxivClientParsing(unittest.TestCase):
    def test_parse_feed(self):
        client = ArxivClient()
        papers = client._parse_feed(SAMPLE_ARXIV_XML)
        self.assertEqual(len(papers), 1)
        paper = papers[0]
        self.assertEqual(paper.arxiv_id, "2301.00001v1")
        self.assertEqual(paper.title, "Test Paper on Machine Learning")
        self.assertIn("Alice Smith", paper.authors)
        self.assertEqual(paper.year, 2023)
        self.assertIn("cs.LG", paper.categories)
        self.assertEqual(paper.doi, "10.1000/test.doi")

    def test_parse_empty_feed(self):
        client = ArxivClient()
        papers = client._parse_feed(
            '<?xml version="1.0"?>'
            '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        )
        self.assertEqual(papers, [])

    def test_filter_by_year(self):
        papers = [
            Paper(title="Old", year=2015),
            Paper(title="Recent", year=2022),
            Paper(title="Future", year=2025),
        ]
        filtered = ArxivClient._filter_by_year(papers, 2020, 2023)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].title, "Recent")


class TestPaperUID(unittest.TestCase):
    def test_uid_prefers_arxiv_id(self):
        p = Paper(arxiv_id="2301.00001", doi="10.xxx", title="Some title")
        self.assertTrue(p.uid.startswith("arxiv:"))

    def test_uid_falls_back_to_doi(self):
        p = Paper(arxiv_id="", doi="10.1000/xyz", title="Some title")
        self.assertTrue(p.uid.startswith("doi:"))

    def test_uid_falls_back_to_title(self):
        p = Paper(arxiv_id="", doi="", title="Some Unique Title")
        self.assertTrue(p.uid.startswith("title:"))

    def test_to_dict_and_from_dict(self):
        p = Paper(arxiv_id="2301.00001", title="Round-trip test", year=2023)
        d = p.to_dict()
        p2 = Paper.from_dict(d)
        self.assertEqual(p.arxiv_id, p2.arxiv_id)
        self.assertEqual(p.title, p2.title)


if __name__ == "__main__":
    unittest.main()
