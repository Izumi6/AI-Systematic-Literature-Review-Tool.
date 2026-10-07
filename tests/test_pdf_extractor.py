"""
tests/test_pdf_extractor.py
Unit tests for PDF text extraction, cleaning, and section detection.
"""

import unittest
from pdf_processing.extractor import extract_sections, build_paper_context
from utils.helpers import clean_text


class TestCleanText(unittest.TestCase):
    def test_collapses_multiple_spaces(self):
        result = clean_text("hello    world")
        self.assertNotIn("    ", result)

    def test_removes_page_numbers(self):
        text = "Some content\n  42\nMore content"
        result = clean_text(text)
        # standalone "42" should be removed
        lines = result.splitlines()
        self.assertNotIn("42", lines)

    def test_handles_empty_string(self):
        self.assertEqual(clean_text(""), "")

    def test_preserves_meaningful_content(self):
        text = "The results show significant improvement."
        self.assertIn("The results show", clean_text(text))


class TestExtractSections(unittest.TestCase):
    SAMPLE_TEXT = """
Abstract
This paper presents a novel approach to NLP.

1. Introduction
Modern NLP systems have advanced significantly.

2. Methodology
We use a transformer-based architecture.

3. Results
Our model achieves 94% accuracy on the benchmark.

4. Conclusion
The proposed method outperforms all baselines.
"""

    def test_detects_known_sections(self):
        sections = extract_sections(self.SAMPLE_TEXT)
        self.assertIn("abstract", sections)
        self.assertIn("introduction", sections)
        self.assertIn("methodology", sections)
        self.assertIn("results", sections)
        self.assertIn("conclusion", sections)

    def test_section_content_not_empty(self):
        sections = extract_sections(self.SAMPLE_TEXT)
        for key, val in sections.items():
            self.assertTrue(len(val) > 0, f"Section '{key}' is empty")

    def test_no_headers_returns_body(self):
        plain = "Some random text without any headers."
        sections = extract_sections(plain)
        self.assertIn("body", sections)

    def test_empty_text(self):
        sections = extract_sections("")
        self.assertEqual(sections, {})


class TestBuildPaperContext(unittest.TestCase):
    SECTIONS = {
        "abstract": "This paper proposes a new method for text classification.",
        "introduction": "Text classification is a fundamental NLP task.",
        "methodology": "We fine-tune BERT on the task.",
        "results": "We achieve 97% on SST-2.",
        "conclusion": "Our method is effective and efficient.",
    }

    def test_respects_max_chars(self):
        context = build_paper_context(self.SECTIONS, max_chars=200)
        self.assertLessEqual(len(context), 250)  # small buffer for formatting

    def test_includes_abstract(self):
        context = build_paper_context(self.SECTIONS, max_chars=5000)
        self.assertIn("proposes a new method", context)

    def test_fallback_to_abstract_when_no_sections(self):
        context = build_paper_context({}, abstract="Fallback abstract text.", max_chars=5000)
        self.assertIn("Fallback abstract text.", context)

    def test_empty_sections_and_no_abstract(self):
        context = build_paper_context({}, abstract="", max_chars=5000)
        self.assertEqual(context, "")


if __name__ == "__main__":
    unittest.main()
