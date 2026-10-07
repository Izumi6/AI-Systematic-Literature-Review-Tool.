"""
embeddings/embedder.py
Generates and caches sentence-transformer embeddings for research papers.

Embeddings are stored as NumPy arrays in the data/embeddings directory
so they do not need to be recomputed on every run.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer

from config import DEFAULT_EMBEDDING_MODEL, EMBEDDINGS_CACHE_DIR

logger = logging.getLogger(__name__)

_model_cache: dict[str, SentenceTransformer] = {}


def get_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """Return a cached SentenceTransformer model instance."""
    if model_name not in _model_cache:
        logger.info("Loading embedding model: %s", model_name)
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def build_paper_text(paper: dict) -> str:
    """
    Build the text representation used for embedding a paper.
    Combines title, abstract, and key analysis fields for richer semantics.
    """
    parts = [paper.get("title", "")]

    abstract = paper.get("abstract", "")
    if abstract:
        parts.append(abstract[:500])

    analysis = paper.get("analysis", {})
    for field in ("research_problem", "methodology", "main_findings"):
        val = analysis.get(field, "")
        if val and val != "Not reported in the paper.":
            parts.append(val[:200])

    return " ".join(p.strip() for p in parts if p.strip())


def embed_papers(
    papers: list[dict],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> np.ndarray:
    """
    Generate embeddings for a list of paper dicts.

    Returns an (N, D) numpy array where N is the number of papers
    and D is the embedding dimension.
    """
    model = get_model(model_name)
    texts = [build_paper_text(p) for p in papers]
    logger.info("Embedding %d papers with model '%s'.", len(texts), model_name)
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings


def save_embeddings(key: str, embeddings: np.ndarray) -> None:
    """Persist embeddings to disk under data/embeddings/<key>.npy."""
    path = EMBEDDINGS_CACHE_DIR / f"{key}.npy"
    np.save(path, embeddings)
    logger.debug("Saved embeddings to %s", path)


def load_embeddings(key: str) -> Optional[np.ndarray]:
    """Load previously saved embeddings, or return None if not found."""
    path = EMBEDDINGS_CACHE_DIR / f"{key}.npy"
    if path.exists():
        return np.load(path)
    return None
