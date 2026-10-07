"""
tests/test_helpers.py
Unit tests for shared utility functions.
"""

import unittest
from utils.helpers import (
    normalize_title,
    clean_text,
    truncate_text,
    safe_json_loads,
    parse_year,
    format_authors,
    current_timestamp,
)
from utils.cache import DiskCache
import tempfile
from pathlib import Path


class TestTruncateText(unittest.TestCase):
    def test_short_text_unchanged(self):
        text = "Hello world"
        self.assertEqual(truncate_text(text, 100), text)

    def test_long_text_truncated(self):
        text = "word " * 100
        result = truncate_text(text, 50)
        self.assertLessEqual(len(result), 60)
        self.assertTrue(result.endswith("..."))

    def test_exact_length(self):
        text = "a" * 100
        result = truncate_text(text, 100)
        self.assertEqual(result, text)


class TestSafeJsonLoads(unittest.TestCase):
    def test_plain_json(self):
        result = safe_json_loads('{"key": "value"}')
        self.assertEqual(result["key"], "value")

    def test_markdown_fenced_json(self):
        result = safe_json_loads('```json\n{"key": "fenced"}\n```')
        self.assertIsNotNone(result)
        self.assertEqual(result["key"], "fenced")

    def test_returns_none_on_invalid(self):
        result = safe_json_loads("this is not json")
        self.assertIsNone(result)

    def test_nested_json(self):
        raw = '{"a": 1, "b": {"c": 2}}'
        result = safe_json_loads(raw)
        self.assertEqual(result["b"]["c"], 2)


class TestFormatAuthors(unittest.TestCase):
    def test_single_author(self):
        self.assertEqual(format_authors(["Alice Smith"]), "Alice Smith")

    def test_multiple_authors_truncated(self):
        authors = ["A", "B", "C", "D", "E"]
        result = format_authors(authors, max_authors=3)
        self.assertIn("et al.", result)

    def test_exactly_max_authors(self):
        authors = ["A", "B", "C"]
        result = format_authors(authors, max_authors=3)
        self.assertNotIn("et al.", result)

    def test_empty_list(self):
        self.assertEqual(format_authors([]), "Unknown authors")


class TestCurrentTimestamp(unittest.TestCase):
    def test_format(self):
        ts = current_timestamp()
        self.assertRegex(ts, r"\d{8}_\d{6}")


class TestDiskCache(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache = DiskCache(Path(self.tmpdir))

    def test_set_and_get(self):
        self.cache.set("mykey", {"data": 42})
        result = self.cache.get("mykey")
        self.assertIsNotNone(result)
        self.assertEqual(result["data"], 42)

    def test_get_missing_returns_none(self):
        self.assertIsNone(self.cache.get("nonexistent"))

    def test_exists(self):
        self.assertFalse(self.cache.exists("k"))
        self.cache.set("k", "v")
        self.assertTrue(self.cache.exists("k"))

    def test_delete(self):
        self.cache.set("del_key", "value")
        self.cache.delete("del_key")
        self.assertFalse(self.cache.exists("del_key"))

    def test_clear(self):
        self.cache.set("k1", 1)
        self.cache.set("k2", 2)
        self.cache.clear()
        self.assertEqual(self.cache.keys(), [])

    def test_handles_special_chars_in_key(self):
        self.cache.set("key/with:special.chars", {"ok": True})
        result = self.cache.get("key/with:special.chars")
        self.assertEqual(result["ok"], True)


if __name__ == "__main__":
    unittest.main()
