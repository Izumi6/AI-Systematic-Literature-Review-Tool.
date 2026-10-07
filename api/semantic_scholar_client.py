"""
api/semantic_scholar_client.py
Optional Semantic Scholar API client (secondary source).

Only called when an API key is provided and the user enables this source.
Reference: https://api.semanticscholar.org/api-docs/
"""

import time
import logging
from typing import Optional

import requests

from config import (
    SEMANTIC_SCHOLAR_BASE_URL,
    SEMANTIC_SCHOLAR_FIELDS,
    SEMANTIC_SCHOLAR_DELAY_SECONDS,
    SEMANTIC_SCHOLAR_API_KEY,
)
from api.arxiv_client import Paper
from utils.helpers import parse_year

logger = logging.getLogger(__name__)


class SemanticScholarClient:
    """
    Searches Semantic Scholar for papers relevant to a query.
    Results are converted to the same Paper dataclass used by ArxivClient.
    """

    def __init__(self) -> None:
        self.session = requests.Session()
        headers = {"User-Agent": "AI-SLR-Tool/1.0"}
        if SEMANTIC_SCHOLAR_API_KEY:
            headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
        self.session.headers.update(headers)

    def search(
        self,
        query: str,
        max_results: int = 20,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> list[Paper]:
        """
        Query Semantic Scholar and return a list of Paper objects.
        """
        papers: list[Paper] = []
        offset = 0
        limit = min(max_results, 100)

        while len(papers) < max_results:
            batch = self._fetch_batch(
                query, offset=offset, limit=limit,
                start_year=start_year, end_year=end_year,
            )
            if not batch:
                break
            papers.extend(batch)
            offset += len(batch)
            if len(batch) < limit:
                break
            time.sleep(SEMANTIC_SCHOLAR_DELAY_SECONDS)

        return papers[:max_results]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_batch(
        self,
        query: str,
        offset: int,
        limit: int,
        start_year: Optional[int],
        end_year: Optional[int],
    ) -> list[Paper]:
        params: dict = {
            "query": query,
            "fields": SEMANTIC_SCHOLAR_FIELDS,
            "offset": offset,
            "limit": limit,
        }
        if start_year or end_year:
            y_start = start_year or 1900
            y_end = end_year or 2100
            params["year"] = f"{y_start}-{y_end}"

        url = f"{SEMANTIC_SCHOLAR_BASE_URL}/paper/search"
        try:
            response = self.session.get(url, params=params, timeout=30)
            if response.status_code == 429:
                logger.warning("Semantic Scholar rate limit hit; backing off 10 s.")
                time.sleep(10)
                return []
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            logger.error("Semantic Scholar request failed: %s", exc)
            return []

        return [
            self._to_paper(item)
            for item in data.get("data", [])
            if item.get("title")
        ]

    @staticmethod
    def _to_paper(item: dict) -> Paper:
        external = item.get("externalIds") or {}
        arxiv_id = external.get("ArXiv", "")
        doi = external.get("DOI", "")

        authors = [a.get("name", "") for a in item.get("authors", []) if a.get("name")]

        pdf_url = ""
        oap = item.get("openAccessPdf")
        if oap and isinstance(oap, dict):
            pdf_url = oap.get("url", "")

        year_val = item.get("year")
        year = int(year_val) if year_val else None

        categories = item.get("fieldsOfStudy") or []

        return Paper(
            arxiv_id=arxiv_id,
            doi=doi,
            title=item.get("title", "").strip(),
            authors=authors,
            abstract=item.get("abstract", "") or "",
            published=str(year) if year else "",
            year=year,
            categories=categories,
            pdf_url=pdf_url,
            source="semantic_scholar",
        )
