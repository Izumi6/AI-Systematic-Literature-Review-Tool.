"""
api/arxiv_client.py
arXiv API wrapper for searching and retrieving paper metadata.

Uses the arXiv Atom feed API (no authentication required).
Reference: https://arxiv.org/help/api/user-manual
"""

import time
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from typing import Optional
from urllib.parse import urlencode

import requests

from config import (
    ARXIV_BASE_URL,
    ARXIV_MAX_RESULTS_PER_REQUEST,
    ARXIV_REQUEST_DELAY_SECONDS,
)
from utils.helpers import parse_year, normalize_title

logger = logging.getLogger(__name__)

# XML namespace used in the arXiv Atom feed
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}


@dataclass
class Paper:
    """Canonical representation of a research paper across sources."""

    arxiv_id: str = ""
    doi: str = ""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    published: str = ""
    year: Optional[int] = None
    categories: list[str] = field(default_factory=list)
    pdf_url: str = ""
    source: str = "arxiv"
    # populated after processing
    full_text: str = ""
    analysis: dict = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)
    cluster_id: int = -1
    theme_name: str = ""

    # deduplication key
    @property
    def uid(self) -> str:
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id}"
        if self.doi:
            return f"doi:{self.doi}"
        return f"title:{normalize_title(self.title)}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["uid"] = self.uid
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Paper":
        data.pop("uid", None)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ArxivClient:
    """
    Fetches paper metadata from the arXiv API.

    Supports paginated retrieval, date range filtering, and
    category filtering.
    """

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "AI-SLR-Tool/1.0"})

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        max_results: int = 20,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        categories: Optional[list[str]] = None,
    ) -> list[Paper]:
        """
        Search arXiv and return up to max_results Paper objects.

        The query string supports arXiv advanced search syntax, e.g.
        'ti:machine learning AND abs:transformer'.
        Plain strings are automatically wrapped to search title + abstract.
        """
        search_query = self._build_query(query, categories)
        papers: list[Paper] = []
        fetched = 0
        start = 0

        while fetched < max_results:
            batch_size = min(ARXIV_MAX_RESULTS_PER_REQUEST, max_results - fetched)
            batch = self._fetch_batch(search_query, start=start, max_results=batch_size)
            if not batch:
                break
            papers.extend(batch)
            fetched += len(batch)
            start += len(batch)
            if len(batch) < batch_size:
                # exhausted all available results
                break
            time.sleep(ARXIV_REQUEST_DELAY_SECONDS)

        if start_year or end_year:
            papers = self._filter_by_year(papers, start_year, end_year)

        return papers

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_query(self, query: str, categories: Optional[list[str]]) -> str:
        """Construct an arXiv API query string."""
        # if query already uses field prefixes leave it alone
        if any(prefix in query for prefix in ("ti:", "abs:", "au:", "AND", "OR")):
            q = query
        else:
            q = f"ti:{query} OR abs:{query}"

        if categories:
            cat_filter = " OR ".join(f"cat:{c}" for c in categories)
            q = f"({q}) AND ({cat_filter})"

        return q

    def _fetch_batch(
        self, search_query: str, start: int, max_results: int
    ) -> list[Paper]:
        params = {
            "search_query": search_query,
            "start": start,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        url = f"{ARXIV_BASE_URL}?{urlencode(params)}"
        logger.debug("arXiv request: %s", url)

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error("arXiv API request failed: %s", exc)
            return []

        return self._parse_feed(response.text)

    def _parse_feed(self, xml_text: str) -> list[Paper]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.error("Failed to parse arXiv XML: %s", exc)
            return []

        papers = []
        for entry in root.findall("atom:entry", NS):
            paper = self._parse_entry(entry)
            if paper and paper.title:
                papers.append(paper)
        return papers

    def _parse_entry(self, entry: ET.Element) -> Optional[Paper]:
        try:
            # arXiv ID
            id_elem = entry.find("atom:id", NS)
            raw_id = id_elem.text.strip() if id_elem is not None else ""
            arxiv_id = self._extract_arxiv_id(raw_id)

            # title
            title_elem = entry.find("atom:title", NS)
            title = " ".join((title_elem.text or "").split()) if title_elem is not None else ""

            # abstract
            summary_elem = entry.find("atom:summary", NS)
            abstract = " ".join((summary_elem.text or "").split()) if summary_elem is not None else ""

            # published date
            pub_elem = entry.find("atom:published", NS)
            published = pub_elem.text.strip() if pub_elem is not None else ""
            year = parse_year(published)

            # authors
            authors = [
                a.find("atom:name", NS).text.strip()
                for a in entry.findall("atom:author", NS)
                if a.find("atom:name", NS) is not None
            ]

            # categories
            categories = [
                cat.get("term", "")
                for cat in entry.findall("atom:category", NS)
            ]

            # links — find the PDF link
            pdf_url = ""
            for link in entry.findall("atom:link", NS):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href", "").replace("http://", "https://")
                    if not pdf_url.endswith(".pdf"):
                        pdf_url += ".pdf"
                    break

            # DOI (optional field)
            doi_elem = entry.find("arxiv:doi", NS)
            doi = doi_elem.text.strip() if doi_elem is not None else ""

            return Paper(
                arxiv_id=arxiv_id,
                doi=doi,
                title=title,
                authors=authors,
                abstract=abstract,
                published=published,
                year=year,
                categories=categories,
                pdf_url=pdf_url,
                source="arxiv",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse arXiv entry: %s", exc)
            return None

    @staticmethod
    def _extract_arxiv_id(raw_id: str) -> str:
        """Extract the short arXiv ID (e.g. '2301.12345') from a full URL."""
        match = re.search(r"abs/(.+)$", raw_id)
        if match:
            return match.group(1).strip()
        return raw_id.strip()

    @staticmethod
    def _filter_by_year(
        papers: list[Paper],
        start_year: Optional[int],
        end_year: Optional[int],
    ) -> list[Paper]:
        filtered = []
        for p in papers:
            if p.year is None:
                filtered.append(p)
                continue
            if start_year and p.year < start_year:
                continue
            if end_year and p.year > end_year:
                continue
            filtered.append(p)
        return filtered


def deduplicate_papers(papers: list[Paper]) -> list[Paper]:
    """
    Remove duplicate papers based on arXiv ID, DOI, or normalized title.
    The first occurrence of each unique paper is kept.
    """
    seen: set[str] = set()
    unique: list[Paper] = []
    for paper in papers:
        key = paper.uid
        if key not in seen:
            seen.add(key)
            unique.append(paper)
        else:
            logger.debug("Duplicate removed: %s", paper.title)
    return unique
