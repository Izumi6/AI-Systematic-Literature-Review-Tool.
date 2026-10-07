"""
pdf_processing/extractor.py
Extracts and cleans text from PDF files using pdfminer.six.

Attempts to identify standard academic paper sections and returns
structured text suitable for LLM analysis.
"""

import logging
import re
from io import StringIO
from pathlib import Path
from typing import Optional

from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams

from config import SECTION_HEADERS
from utils.helpers import clean_text

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: Path) -> Optional[str]:
    """
    Extract raw text from a PDF file.

    Returns the extracted and cleaned text, or None on failure.
    """
    try:
        output = StringIO()
        with open(pdf_path, "rb") as fh:
            laparams = LAParams(
                line_margin=0.5,
                word_margin=0.1,
                char_margin=2.0,
                all_texts=False,
            )
            extract_text_to_fp(fh, output, laparams=laparams, output_type="text")
        raw_text = output.getvalue()
        return clean_text(raw_text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("pdfminer extraction failed for %s: %s", pdf_path, exc)
        return None


def extract_sections(text: str) -> dict[str, str]:
    """
    Attempt to split the extracted text into known academic sections.

    Returns a dict mapping lowercased section names to their content.
    If a section is not found, it is not included in the result.
    """
    if not text:
        return {}

    sections: dict[str, str] = {}
    lower_text = text.lower()

    # Build a list of (position, section_name) tuples
    positions = []
    for header in SECTION_HEADERS:
        # Match section headers that appear on their own line or after numbering
        pattern = rf"(?:^|\n)\s*(?:\d+[\.\s]+)?{re.escape(header)}\s*(?:\n|$)"
        for m in re.finditer(pattern, lower_text, re.MULTILINE):
            positions.append((m.start(), header))

    if not positions:
        # No recognisable headers — treat the whole text as the body
        return {"body": text}

    positions.sort(key=lambda x: x[0])

    for i, (pos, name) in enumerate(positions):
        # Find the original-case start
        start = text.lower().find(name, pos)
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        content = text[start:end].strip()
        # Strip the header line itself
        lines = content.splitlines()
        if lines:
            content = "\n".join(lines[1:]).strip()
        if content:
            sections[name] = content

    return sections


def build_paper_context(
    sections: dict[str, str], abstract: str = "", max_chars: int = 6000
) -> str:
    """
    Build a concise context string from the extracted sections.

    Prioritises the most informative sections and respects max_chars.
    If extraction failed, falls back to the abstract.
    """
    priority = [
        "abstract", "introduction", "methodology", "methods",
        "results", "conclusion", "limitations", "future work",
    ]

    if not sections:
        return abstract[:max_chars] if abstract else ""

    parts = []
    budget = max_chars

    # include abstract first if available separately
    if abstract and "abstract" not in sections:
        snippet = f"Abstract:\n{abstract[:1200]}\n\n"
        parts.append(snippet)
        budget -= len(snippet)

    for key in priority:
        if budget <= 0:
            break
        if key in sections:
            excerpt = sections[key][:max(200, budget // len(priority))]
            chunk = f"{key.title()}:\n{excerpt}\n\n"
            parts.append(chunk)
            budget -= len(chunk)

    # fill remaining with other sections
    for key, content in sections.items():
        if key in priority or budget <= 0:
            continue
        excerpt = content[:min(400, budget)]
        parts.append(f"{key.title()}:\n{excerpt}\n\n")
        budget -= len(excerpt) + len(key) + 10

    return "".join(parts).strip()
