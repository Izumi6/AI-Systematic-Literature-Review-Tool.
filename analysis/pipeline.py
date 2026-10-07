"""
analysis/pipeline.py
Orchestrates the full literature review pipeline.

This module ties together all other modules:
  1. Paper retrieval (arXiv + optional Semantic Scholar)
  2. Deduplication
  3. PDF download and text extraction
  4. LLM-based structured analysis
  5. Semantic embeddings
  6. Clustering and theme naming
  7. Cross-paper comparison
  8. Research gap identification
  9. Survey text generation
  10. Word report generation

Progress callbacks allow the Streamlit UI to display live status updates.
"""

import json
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from api.arxiv_client import ArxivClient, Paper, deduplicate_papers
from api.semantic_scholar_client import SemanticScholarClient
from pdf_processing.downloader import download_pdf
from pdf_processing.extractor import extract_text_from_pdf, extract_sections, build_paper_context
from llm.paper_analyzer import analyze_paper
from llm.survey_generator import (
    generate_theme_name,
    identify_research_gaps,
    generate_introduction,
    generate_theme_section,
    generate_conclusion,
)
from embeddings.embedder import embed_papers, save_embeddings, load_embeddings
from clustering.clusterer import cluster_papers, get_cluster_summary
from analysis.comparator import build_comparison_table, compute_statistics
from analysis.references import build_reference_list
from reporting.docx_writer import DocxWriter
from utils.cache import DiskCache
from utils.helpers import paper_cache_key, current_timestamp, truncate_text
from config import (
    METADATA_CACHE_DIR,
    PAPER_TEXT_CHAR_LIMIT,
    DEFAULT_EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float], None]

_metadata_cache = DiskCache(METADATA_CACHE_DIR)


# ---------------------------------------------------------------------------
# Step 1: Retrieve papers
# ---------------------------------------------------------------------------

def retrieve_papers(
    topic: str,
    keywords: str,
    max_papers: int,
    start_year: Optional[int],
    end_year: Optional[int],
    use_semantic_scholar: bool = False,
    progress: Optional[ProgressCallback] = None,
) -> list[dict]:
    """
    Retrieve papers from arXiv (and optionally Semantic Scholar),
    deduplicate, and return as a list of plain dicts.
    """
    _progress(progress, "Searching arXiv...", 0.1)

    query = topic
    if keywords:
        query = f"{topic} {keywords}"

    arxiv = ArxivClient()
    arxiv_papers: list[Paper] = arxiv.search(
        query=query,
        max_results=max_papers,
        start_year=start_year,
        end_year=end_year,
    )
    logger.info("arXiv returned %d papers.", len(arxiv_papers))

    all_papers = list(arxiv_papers)

    if use_semantic_scholar:
        _progress(progress, "Searching Semantic Scholar...", 0.2)
        ss = SemanticScholarClient()
        ss_papers = ss.search(
            query=query,
            max_results=max_papers // 2,
            start_year=start_year,
            end_year=end_year,
        )
        logger.info("Semantic Scholar returned %d papers.", len(ss_papers))
        all_papers.extend(ss_papers)

    _progress(progress, "Removing duplicates...", 0.3)
    unique = deduplicate_papers(all_papers)
    logger.info("After deduplication: %d papers.", len(unique))

    paper_dicts = [p.to_dict() for p in unique]
    return paper_dicts


# ---------------------------------------------------------------------------
# Step 2: Download PDFs and extract text
# ---------------------------------------------------------------------------

def process_pdfs(
    papers: list[dict],
    progress: Optional[ProgressCallback] = None,
) -> list[dict]:
    """
    For each paper: download the PDF (if available), extract text,
    and store the result in paper["full_text"].

    Falls back to the abstract if extraction fails.
    """
    n = len(papers)
    for i, paper in enumerate(papers):
        _progress(progress, f"Processing PDF {i+1}/{n}: {paper['title'][:50]}...", 0.0)

        # check if we already have text cached
        cache_key = f"text_{paper_cache_key(paper.get('arxiv_id', '') or paper['title'][:40])}"
        cached = _metadata_cache.get(cache_key)
        if cached:
            paper["full_text"] = cached
            continue

        pdf_path = None
        if paper.get("pdf_url"):
            pdf_path = download_pdf(
                paper.get("arxiv_id") or paper["title"][:40],
                paper["pdf_url"],
            )

        if pdf_path:
            raw_text = extract_text_from_pdf(pdf_path)
            if raw_text and len(raw_text) > 200:
                sections = extract_sections(raw_text)
                context = build_paper_context(
                    sections,
                    abstract=paper.get("abstract", ""),
                    max_chars=PAPER_TEXT_CHAR_LIMIT,
                )
                paper["full_text"] = context
                _metadata_cache.set(cache_key, context)
                continue

        # fallback to abstract
        paper["full_text"] = truncate_text(
            paper.get("abstract", "No abstract available."), PAPER_TEXT_CHAR_LIMIT
        )
        _metadata_cache.set(cache_key, paper["full_text"])

    return papers


# ---------------------------------------------------------------------------
# Step 3: LLM analysis
# ---------------------------------------------------------------------------

def analyze_papers(
    papers: list[dict],
    progress: Optional[ProgressCallback] = None,
) -> list[dict]:
    """
    Run structured LLM analysis on each paper.
    Caches results to avoid repeated API calls.
    """
    n = len(papers)
    for i, paper in enumerate(papers):
        _progress(
            progress,
            f"Analyzing paper {i+1}/{n}: {paper['title'][:50]}...",
            0.0,
        )

        cache_key = f"analysis_{paper_cache_key(paper.get('arxiv_id', '') or paper['title'][:40])}"
        cached = _metadata_cache.get(cache_key)
        if cached:
            paper["analysis"] = cached
            continue

        analysis = analyze_paper(
            title=paper.get("title", ""),
            authors=paper.get("authors", []),
            year=paper.get("year"),
            content=paper.get("full_text", paper.get("abstract", "")),
        )
        paper["analysis"] = analysis
        _metadata_cache.set(cache_key, analysis)

    return papers


# ---------------------------------------------------------------------------
# Step 4: Embeddings + Clustering
# ---------------------------------------------------------------------------

def embed_and_cluster(
    papers: list[dict],
    n_clusters: Optional[int] = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    clustering_method: str = "kmeans",
    progress: Optional[ProgressCallback] = None,
) -> tuple[list[dict], float]:
    """
    Generate embeddings, cluster papers, generate theme names,
    and annotate each paper with its cluster_id and theme_name.

    Returns the annotated list and the silhouette score.
    """
    _progress(progress, "Generating semantic embeddings...", 0.0)
    embeddings = embed_papers(papers, model_name=model_name)

    _progress(progress, "Clustering papers by topic...", 0.0)
    labels, silhouette = cluster_papers(
        embeddings,
        n_clusters=n_clusters,
        method=clustering_method,
    )

    clusters = get_cluster_summary(papers, labels)

    _progress(progress, "Generating theme names...", 0.0)
    theme_names: dict[int, str] = {}
    for cid, cluster_papers_list in clusters.items():
        name = generate_theme_name(cluster_papers_list)
        theme_names[cid] = name
        logger.info("Cluster %d -> '%s' (%d papers)", cid, name, len(cluster_papers_list))

    for paper, label in zip(papers, labels):
        paper["cluster_id"] = int(label)
        paper["theme_name"] = theme_names.get(int(label), f"Cluster {label}")

    return papers, silhouette


# ---------------------------------------------------------------------------
# Step 5: Full survey generation
# ---------------------------------------------------------------------------

def generate_survey(
    topic: str,
    papers: list[dict],
    progress: Optional[ProgressCallback] = None,
) -> dict:
    """
    Generate all text sections of the literature survey.

    Returns a dict with keys:
        introduction, theme_sections, research_gaps, conclusion, references,
        comparison_df, statistics
    """
    themes = sorted(set(p.get("theme_name", "Unassigned") for p in papers))
    years = [p["year"] for p in papers if p.get("year")]
    year_range = f"{min(years)}–{max(years)}" if years else "Unknown"

    _progress(progress, "Writing introduction...", 0.0)
    introduction = generate_introduction(topic, len(papers), year_range, themes)

    _progress(progress, "Generating thematic sections...", 0.0)
    theme_sections: dict[str, str] = {}
    for theme in themes:
        theme_papers = [p for p in papers if p.get("theme_name") == theme]
        theme_sections[theme] = generate_theme_section(theme, theme_papers)

    _progress(progress, "Identifying research gaps...", 0.0)
    research_gaps = identify_research_gaps(papers)

    # key findings for conclusion (brief summary of gaps)
    key_findings = research_gaps[:800] if research_gaps else ""

    _progress(progress, "Writing conclusion...", 0.0)
    conclusion = generate_conclusion(topic, len(papers), themes, key_findings)

    _progress(progress, "Building reference list...", 0.0)
    references = build_reference_list(papers)

    comparison_df = build_comparison_table(papers)
    statistics = compute_statistics(papers)

    return {
        "introduction": introduction,
        "theme_sections": theme_sections,
        "research_gaps": research_gaps,
        "conclusion": conclusion,
        "references": references,
        "comparison_df": comparison_df,
        "statistics": statistics,
        "themes": themes,
    }


# ---------------------------------------------------------------------------
# Step 6: Word report
# ---------------------------------------------------------------------------

def generate_report(
    topic: str,
    papers: list[dict],
    survey: dict,
    progress: Optional[ProgressCallback] = None,
) -> Path:
    """
    Write the complete .docx literature survey.
    Returns the path to the saved file.
    """
    _progress(progress, "Generating Word document...", 0.0)
    writer = DocxWriter()
    path = writer.build(
        topic=topic,
        papers=papers,
        themes=survey.get("themes", []),
        theme_sections=survey.get("theme_sections", {}),
        comparison_df=survey.get("comparison_df"),
        research_gaps=survey.get("research_gaps", ""),
        introduction=survey.get("introduction", ""),
        conclusion=survey.get("conclusion", ""),
        references=survey.get("references", []),
    )
    return path


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _progress(callback: Optional[ProgressCallback], message: str, value: float):
    if callback:
        callback(message, value)
