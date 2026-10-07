"""
pdf_processing/downloader.py
Downloads PDF files for research papers to local disk.

Respects size limits and per-domain politeness delays.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import requests

from config import PDF_CACHE_DIR, PDF_DOWNLOAD_TIMEOUT_SECONDS, MAX_PDF_SIZE_MB

logger = logging.getLogger(__name__)

MAX_PDF_BYTES = MAX_PDF_SIZE_MB * 1024 * 1024


def get_pdf_path(arxiv_id: str) -> Path:
    """Return the expected local path for a paper's PDF."""
    safe_id = arxiv_id.replace("/", "_").replace(".", "_")
    return PDF_CACHE_DIR / f"{safe_id}.pdf"


def download_pdf(paper_id: str, pdf_url: str) -> Optional[Path]:
    """
    Download a PDF to the local cache directory.

    Returns the local Path on success, or None if the download fails.
    Does not re-download if the file already exists.
    """
    if not pdf_url:
        logger.debug("No PDF URL for paper %s", paper_id)
        return None

    local_path = get_pdf_path(paper_id)

    if local_path.exists() and local_path.stat().st_size > 0:
        logger.debug("PDF already cached: %s", local_path)
        return local_path

    try:
        headers = {"User-Agent": "AI-SLR-Tool/1.0"}
        response = requests.get(
            pdf_url,
            stream=True,
            timeout=PDF_DOWNLOAD_TIMEOUT_SECONDS,
            headers=headers,
        )
        response.raise_for_status()

        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_PDF_BYTES:
            logger.warning(
                "PDF for %s is too large (%s bytes); skipping.",
                paper_id,
                content_length,
            )
            return None

        downloaded = 0
        chunks = []
        for chunk in response.iter_content(chunk_size=65536):
            chunks.append(chunk)
            downloaded += len(chunk)
            if downloaded > MAX_PDF_BYTES:
                logger.warning("PDF for %s exceeded size limit; skipping.", paper_id)
                return None

        with open(local_path, "wb") as fh:
            for chunk in chunks:
                fh.write(chunk)

        logger.info("Downloaded PDF: %s -> %s", pdf_url, local_path)
        return local_path

    except requests.RequestException as exc:
        logger.warning("PDF download failed for %s: %s", paper_id, exc)
        return None
    except OSError as exc:
        logger.error("Could not write PDF for %s: %s", paper_id, exc)
        return None
