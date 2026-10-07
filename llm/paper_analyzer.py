"""
llm/paper_analyzer.py
Structured per-paper analysis using an LLM.

Generates a structured JSON record for each paper describing:
  - Research Problem
  - Objective
  - Methodology
  - Dataset / Experimental Setup
  - Key Results
  - Main Findings
  - Limitations
  - Research Contribution
  - Future Work

Uses strict JSON output to ensure the application can reliably parse results.
Never invents information; marks missing fields as "Not reported in the paper."
"""

import json
import logging
from typing import Optional

from llm.openai_client import chat_completion
from config import LLM_MAX_TOKENS_ANALYSIS
from utils.helpers import safe_json_loads

logger = logging.getLogger(__name__)

ANALYSIS_FIELDS = [
    "research_problem",
    "objective",
    "methodology",
    "dataset_experimental_setup",
    "key_results",
    "main_findings",
    "limitations",
    "research_contribution",
    "future_work",
]

SYSTEM_PROMPT = """You are a systematic literature review assistant.
Analyze the provided academic paper text and extract structured information.
Respond ONLY with a valid JSON object — no prose, no markdown fences.
If a field cannot be determined from the provided text, set its value to
"Not reported in the paper." Do not fabricate or infer beyond what the
paper states."""

ANALYSIS_PROMPT_TEMPLATE = """Analyze the following academic paper and return a JSON object with
exactly these keys:
  "research_problem"          - The core problem or gap the paper addresses
  "objective"                 - The stated goal or research question
  "methodology"               - Methods, models, or techniques used
  "dataset_experimental_setup"- Datasets or experimental environment described
  "key_results"               - Quantitative or qualitative results reported
  "main_findings"             - Conclusions and insights the authors draw
  "limitations"               - Limitations acknowledged by the authors
  "research_contribution"     - Novel contribution to the field
  "future_work"               - Future directions suggested by the authors

---
Title: {title}
Authors: {authors}
Year: {year}

Paper Content:
{content}
---

Respond with ONLY the JSON object. Example:
{{
  "research_problem": "...",
  "objective": "...",
  "methodology": "...",
  "dataset_experimental_setup": "...",
  "key_results": "...",
  "main_findings": "...",
  "limitations": "...",
  "research_contribution": "...",
  "future_work": "..."
}}"""


def analyze_paper(
    title: str,
    authors: list[str],
    year: Optional[int],
    content: str,
) -> dict:
    """
    Send paper content to the LLM and return a structured analysis dict.

    Falls back to a dict of "Not reported" values if the LLM call fails
    or returns unparseable output.
    """
    author_str = ", ".join(authors[:5]) if authors else "Unknown"
    year_str = str(year) if year else "Unknown"

    prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        title=title,
        authors=author_str,
        year=year_str,
        content=content,
    )

    raw = chat_completion(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=LLM_MAX_TOKENS_ANALYSIS,
        temperature=0.1,
    )

    if raw:
        parsed = safe_json_loads(raw)
        if parsed and all(k in parsed for k in ANALYSIS_FIELDS):
            return parsed
        logger.warning("LLM returned incomplete JSON for paper: %s", title)

    return _default_analysis()


def _default_analysis() -> dict:
    return {field: "Not reported in the paper." for field in ANALYSIS_FIELDS}
