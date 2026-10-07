"""
analysis/references.py
Generates IEEE-formatted references from paper metadata.

Every reference is built strictly from the retrieved metadata.
No information is fabricated or altered.
"""

import re
from utils.helpers import authors_to_ieee


def format_ieee_reference(paper: dict, ref_number: int) -> str:
    """
    Format a single paper as an IEEE reference entry.

    Format:
        [N] A. Author, B. Author, and C. Author, "Title," Source, year.
        DOI: xxx  |  arXiv: xxx  |  URL: xxx
    """
    authors = paper.get("authors", [])
    title = paper.get("title", "Unknown title").strip()
    year = paper.get("year", "n.d.")
    source = paper.get("source", "").capitalize()
    arxiv_id = paper.get("arxiv_id", "")
    doi = paper.get("doi", "")
    url = paper.get("pdf_url", "")

    author_str = authors_to_ieee(authors) if authors else "Unknown Author"

    # build source/venue string
    if source == "Arxiv":
        venue = "arXiv preprint"
    elif source == "Semantic_scholar":
        venue = "Semantic Scholar"
    else:
        venue = source or "Unknown Venue"

    ref = f'[{ref_number}] {author_str}, "{title}," {venue}, {year}.'

    extras = []
    if doi:
        extras.append(f"DOI: {doi}")
    if arxiv_id:
        extras.append(f"arXiv: {arxiv_id}")
    if url and not doi:
        extras.append(f"URL: {url}")

    if extras:
        ref += " " + "  ".join(extras)

    return ref


def build_reference_list(papers: list[dict]) -> list[str]:
    """
    Return a numbered list of IEEE-formatted reference strings.
    """
    return [format_ieee_reference(p, i + 1) for i, p in enumerate(papers)]


def build_reference_map(papers: list[dict]) -> dict[str, int]:
    """
    Return a mapping of paper uid -> reference number for in-text citations.
    """
    mapping = {}
    for i, p in enumerate(papers):
        uid = p.get("uid") or p.get("arxiv_id") or p.get("title", "")
        mapping[uid] = i + 1
    return mapping
