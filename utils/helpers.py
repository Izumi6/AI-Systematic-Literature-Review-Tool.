"""
utils/helpers.py
General-purpose helper utilities used across the project.
"""

import re
import json
import hashlib
import logging
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_title(title: str) -> str:
    """Return a lowercase, punctuation-stripped title for deduplication."""
    return re.sub(r"[^a-z0-9\s]", "", title.lower().strip())


def clean_text(text: str) -> str:
    """
    Remove PDF artifacts and normalize whitespace.
    Keeps alphabetic, numeric, and common punctuation characters.
    """
    # replace non-breaking spaces and similar
    text = text.replace("\xa0", " ").replace("\x00", "")
    # collapse multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    # strip lines that are only numbers (page numbers / noise)
    lines = [ln for ln in text.splitlines() if not re.fullmatch(r"\s*\d+\s*", ln)]
    return "\n".join(lines).strip()


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, breaking on a word boundary where possible."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.8:
        truncated = truncated[:last_space]
    return truncated + " ..."


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def safe_json_loads(raw: str) -> Optional[dict]:
    """
    Parse JSON from a string that may be wrapped in markdown code fences.
    Returns None on failure.
    """
    # strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw.strip())
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # attempt to extract the first JSON object
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    logger.warning("Could not parse JSON response.")
    return None


# ---------------------------------------------------------------------------
# Identification / caching helpers
# ---------------------------------------------------------------------------

def paper_cache_key(paper_id: str) -> str:
    """Return a filesystem-safe cache key for a paper."""
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", paper_id)
    return safe


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def parse_year(date_string: str) -> Optional[int]:
    """
    Extract the 4-digit year from various date string formats.
    Returns None if the year cannot be determined.
    """
    match = re.search(r"\b(19|20)\d{2}\b", str(date_string))
    if match:
        return int(match.group())
    return None


def current_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Author formatting
# ---------------------------------------------------------------------------

def format_authors(authors: list[str], max_authors: int = 3) -> str:
    """
    Return a short author string, e.g. 'Smith, J., Jones, A., et al.'
    """
    if not authors:
        return "Unknown authors"
    if len(authors) <= max_authors:
        return ", ".join(authors)
    return ", ".join(authors[:max_authors]) + ", et al."


def authors_to_ieee(authors: list[str]) -> str:
    """
    Format an author list in IEEE style:
    'A. Smith, B. Jones, and C. Brown'
    Returns an empty string for an empty author list.
    """
    if not authors:
        return ""
    formatted = []
    for name in authors:
        parts = name.strip().split()
        if len(parts) >= 2:
            initials = " ".join(p[0] + "." for p in parts[:-1])
            formatted.append(f"{initials} {parts[-1]}")
        else:
            formatted.append(name)
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    return ", ".join(formatted[:-1]) + f", and {formatted[-1]}"
