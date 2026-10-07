"""
analysis/comparator.py
Builds the cross-paper comparison table and computes summary statistics.
"""

import logging
from typing import Any

import pandas as pd

from utils.helpers import format_authors

logger = logging.getLogger(__name__)


def build_comparison_table(papers: list[dict]) -> pd.DataFrame:
    """
    Build a DataFrame with one row per paper for the cross-paper comparison.

    Columns:
        Paper, Year, Authors, Problem, Method, Dataset, Results, Limitations, Theme
    """
    rows = []
    for p in papers:
        analysis = p.get("analysis", {})
        authors = p.get("authors", [])

        def _get(field: str) -> str:
            val = analysis.get(field, "Not reported in the paper.")
            if not val or val == "Not reported in the paper.":
                return "-"
            return val[:250]

        rows.append({
            "Paper": p.get("title", "Unknown")[:80],
            "Year": p.get("year", "-"),
            "Authors": format_authors(authors, max_authors=2),
            "Problem": _get("research_problem"),
            "Method": _get("methodology"),
            "Dataset": _get("dataset_experimental_setup"),
            "Results": _get("key_results"),
            "Limitations": _get("limitations"),
            "Theme": p.get("theme_name", "-"),
        })

    return pd.DataFrame(rows)


def compute_statistics(papers: list[dict]) -> dict[str, Any]:
    """
    Compute summary statistics about the paper collection.

    Returns:
        total          - total number of papers
        by_year        - {year: count}
        by_source      - {source: count}
        by_theme       - {theme_name: count}
        year_range     - (min_year, max_year) tuple
        top_categories - top 5 arXiv categories
    """
    stats: dict[str, Any] = {
        "total": len(papers),
        "by_year": {},
        "by_source": {},
        "by_theme": {},
        "year_range": (None, None),
        "top_categories": [],
    }

    years = []
    categories: dict[str, int] = {}

    for p in papers:
        year = p.get("year")
        if year:
            years.append(int(year))
            stats["by_year"][str(year)] = stats["by_year"].get(str(year), 0) + 1

        source = p.get("source", "unknown")
        stats["by_source"][source] = stats["by_source"].get(source, 0) + 1

        theme = p.get("theme_name", "Unassigned")
        stats["by_theme"][theme] = stats["by_theme"].get(theme, 0) + 1

        for cat in p.get("categories", []):
            if cat:
                categories[cat] = categories.get(cat, 0) + 1

    if years:
        stats["year_range"] = (min(years), max(years))

    stats["top_categories"] = sorted(
        categories.items(), key=lambda x: x[1], reverse=True
    )[:5]

    return stats
