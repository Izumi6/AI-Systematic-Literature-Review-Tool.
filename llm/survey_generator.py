"""
llm/survey_generator.py
Generates the full literature survey text and identifies research gaps
using the LLM over the collection of analyzed papers.
"""

import logging
from typing import Optional

from llm.openai_client import chat_completion
from config import (
    LLM_MAX_TOKENS_SURVEY,
    LLM_MAX_TOKENS_GAPS,
    LLM_MAX_TOKENS_INTRO,
    LLM_MAX_TOKENS_THEME,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Theme naming
# ---------------------------------------------------------------------------

THEME_SYSTEM = (
    "You are an academic research assistant specializing in literature synthesis. "
    "Respond concisely and precisely."
)

THEME_PROMPT = """The following academic papers have been grouped into a cluster based on semantic similarity.
Provide a short (3–6 words), descriptive theme name that captures what these papers collectively address.
Respond with ONLY the theme name — no explanation, no punctuation at the end.

Papers in this cluster:
{paper_summaries}"""


def generate_theme_name(papers_in_cluster: list[dict]) -> str:
    """
    Ask the LLM to produce a short human-readable name for a paper cluster.
    Falls back to 'Cluster <n>' on failure.
    """
    if not papers_in_cluster:
        return "Uncategorized"

    summaries = []
    for p in papers_in_cluster[:8]:
        title = p.get("title", "")
        prob = p.get("analysis", {}).get("research_problem", "")[:120]
        summaries.append(f"- {title}: {prob}")

    prompt = THEME_PROMPT.format(paper_summaries="\n".join(summaries))
    name = chat_completion(
        prompt=prompt,
        system_prompt=THEME_SYSTEM,
        max_tokens=LLM_MAX_TOKENS_THEME,
        temperature=0.3,
    )
    return name.strip() if name else "Research Cluster"


# ---------------------------------------------------------------------------
# Research gap identification
# ---------------------------------------------------------------------------

GAPS_SYSTEM = (
    "You are an expert academic reviewer conducting a systematic literature review. "
    "Identify genuine research gaps based only on what the papers explicitly state. "
    "Do not invent limitations or gaps not supported by the provided text."
)

GAPS_PROMPT = """You are given structured summaries of {n_papers} research papers.
Identify the following from the literature:

1. Common limitations acknowledged across papers
2. Under-researched areas or missing perspectives
3. Conflicting or contradictory findings between papers
4. Missing datasets, benchmarks, or evaluation methods
5. Emerging research directions suggested by multiple papers

Clearly distinguish between:
  - Gaps explicitly stated by authors
  - Gaps inferred from reading across papers (mark these with "[Inferred]")

Do not present unsupported claims as established facts.

Paper summaries:
{paper_summaries}

Provide a clear, well-structured analysis. Use headers for each of the five categories above."""


def identify_research_gaps(papers: list[dict]) -> str:
    """
    Analyze all papers collectively and return a formatted research gaps section.
    """
    summaries = _build_paper_summaries(papers, max_papers=30)
    prompt = GAPS_PROMPT.format(n_papers=len(papers), paper_summaries=summaries)

    result = chat_completion(
        prompt=prompt,
        system_prompt=GAPS_SYSTEM,
        max_tokens=LLM_MAX_TOKENS_GAPS,
        temperature=0.2,
    )
    if result:
        return result
    return "Research gap analysis could not be completed due to an LLM error."


# ---------------------------------------------------------------------------
# Thematic survey generation
# ---------------------------------------------------------------------------

INTRO_SYSTEM = (
    "You are an academic writer composing a systematic literature review. "
    "Write in formal academic prose. Base all statements on the provided paper data."
)

INTRO_PROMPT = """Write an Introduction and Background section for a systematic literature review
on the topic: "{topic}".

The review covers {n_papers} papers spanning {year_range}.
Key themes identified: {themes}.

Write 3–4 paragraphs covering:
- Importance and context of the topic
- Why a systematic review is needed
- Scope of this review (types of papers, time period)
- Overview of what the review covers

Do not cite specific papers by name here — the themed sections will do that.
Write in formal, academic English."""


THEME_SECTION_SYSTEM = (
    "You are writing a section of a systematic literature review. "
    "Synthesize the given papers by theme rather than summarizing each individually. "
    "Use academic language. Reference papers by their shortened title in parentheses."
)

THEME_SECTION_PROMPT = """Write the '{theme_name}' section of a systematic literature review.
This section covers {n_papers} papers that share a common theme.

Synthesize these papers into a coherent narrative. Discuss:
- What problems these papers address
- Common methodologies used
- Key findings and how they relate
- How this body of work advances the field
- Important differences or contrasts between approaches

Paper details:
{paper_details}

Write 3–5 paragraphs. Cite papers by referencing their title in brackets, e.g., [Title].
Do not invent results, authors, or statistics not present in the paper details."""

CONCLUSION_PROMPT = """Write the Conclusion section of a systematic literature review on "{topic}".

Based on the following overview:
- Total papers reviewed: {n_papers}
- Themes covered: {themes}
- Key findings: {key_findings}

Write 2–3 paragraphs summarizing:
1. What the literature as a whole shows
2. The most significant contributions and patterns
3. Recommendations for researchers and practitioners
4. The overall direction of the field

Use formal academic English. Do not introduce new claims not supported by the data."""


def generate_introduction(
    topic: str,
    n_papers: int,
    year_range: str,
    themes: list[str],
) -> str:
    prompt = INTRO_PROMPT.format(
        topic=topic,
        n_papers=n_papers,
        year_range=year_range,
        themes=", ".join(themes),
    )
    result = chat_completion(
        prompt=prompt,
        system_prompt=INTRO_SYSTEM,
        max_tokens=LLM_MAX_TOKENS_INTRO,
        temperature=0.3,
    )
    return result or "Introduction generation failed."


def generate_theme_section(theme_name: str, papers: list[dict]) -> str:
    details = _build_paper_details(papers)
    prompt = THEME_SECTION_PROMPT.format(
        theme_name=theme_name,
        n_papers=len(papers),
        paper_details=details,
    )
    result = chat_completion(
        prompt=prompt,
        system_prompt=THEME_SECTION_SYSTEM,
        max_tokens=LLM_MAX_TOKENS_SURVEY,
        temperature=0.3,
    )
    return result or f"Section for theme '{theme_name}' could not be generated."


def generate_conclusion(
    topic: str,
    n_papers: int,
    themes: list[str],
    key_findings: str,
) -> str:
    prompt = CONCLUSION_PROMPT.format(
        topic=topic,
        n_papers=n_papers,
        themes=", ".join(themes),
        key_findings=key_findings[:1500],
    )
    result = chat_completion(
        prompt=prompt,
        system_prompt=INTRO_SYSTEM,
        max_tokens=LLM_MAX_TOKENS_INTRO,
        temperature=0.3,
    )
    return result or "Conclusion generation failed."


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_paper_summaries(papers: list[dict], max_papers: int = 30) -> str:
    lines = []
    for p in papers[:max_papers]:
        analysis = p.get("analysis", {})
        lines.append(
            f"Title: {p.get('title', 'Unknown')}\n"
            f"  Year: {p.get('year', 'N/A')}\n"
            f"  Problem: {analysis.get('research_problem', 'N/A')[:200]}\n"
            f"  Method: {analysis.get('methodology', 'N/A')[:150]}\n"
            f"  Results: {analysis.get('key_results', 'N/A')[:150]}\n"
            f"  Limitations: {analysis.get('limitations', 'N/A')[:150]}\n"
        )
    return "\n".join(lines)


def _build_paper_details(papers: list[dict]) -> str:
    lines = []
    for p in papers[:12]:
        analysis = p.get("analysis", {})
        authors = p.get("authors", [])
        author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
        lines.append(
            f"[{p.get('title', 'Unknown')}]\n"
            f"  Authors: {author_str} ({p.get('year', 'N/A')})\n"
            f"  Problem: {analysis.get('research_problem', 'N/A')[:250]}\n"
            f"  Methodology: {analysis.get('methodology', 'N/A')[:250]}\n"
            f"  Results: {analysis.get('key_results', 'N/A')[:200]}\n"
            f"  Contribution: {analysis.get('research_contribution', 'N/A')[:200]}\n"
        )
    return "\n\n".join(lines)
