"""
config.py
Central configuration for the AI Systematic Literature Review Tool.
All tuneable constants are defined here so other modules can import them.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
PDF_CACHE_DIR = DATA_DIR / "pdfs"
METADATA_CACHE_DIR = DATA_DIR / "metadata"
EMBEDDINGS_CACHE_DIR = DATA_DIR / "embeddings"

for _d in (DATA_DIR, OUTPUTS_DIR, PDF_CACHE_DIR, METADATA_CACHE_DIR, EMBEDDINGS_CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# API keys and endpoints (loaded from environment)
# ---------------------------------------------------------------------------
API_KEY: str = os.getenv("API_KEY", os.getenv("OPENAI_API_KEY", ""))
API_BASE_URL: str = os.getenv("API_BASE_URL", os.getenv("OPENAI_BASE_URL", ""))
OPENAI_API_KEY: str = API_KEY  # Backward compatibility alias
SEMANTIC_SCHOLAR_API_KEY: str = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

# ---------------------------------------------------------------------------
# arXiv API
# ---------------------------------------------------------------------------
ARXIV_BASE_URL = "http://export.arxiv.org/api/query"
ARXIV_MAX_RESULTS_PER_REQUEST = 100        # hard API limit per call
ARXIV_REQUEST_DELAY_SECONDS = 3.0          # politeness delay between calls

# ---------------------------------------------------------------------------
# Semantic Scholar API
# ---------------------------------------------------------------------------
SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1"
SEMANTIC_SCHOLAR_FIELDS = (
    "title,authors,year,abstract,externalIds,openAccessPdf,fieldsOfStudy"
)
SEMANTIC_SCHOLAR_DELAY_SECONDS = 1.0

# ---------------------------------------------------------------------------
# PDF processing
# ---------------------------------------------------------------------------
PDF_DOWNLOAD_TIMEOUT_SECONDS = 30
MAX_PDF_SIZE_MB = 20
SECTION_HEADERS = [
    "abstract", "introduction", "related work", "background",
    "methodology", "methods", "experimental setup", "experiments",
    "results", "discussion", "conclusion", "future work",
    "limitations", "references",
]

# ---------------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------------
LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
OPENAI_MODEL = LLM_MODEL  # Backward compatibility alias
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS_ANALYSIS = 1500
LLM_MAX_TOKENS_SURVEY = 4000
LLM_MAX_TOKENS_THEME = 300
LLM_MAX_TOKENS_GAPS = 2000
LLM_MAX_TOKENS_INTRO = 1500
PAPER_TEXT_CHAR_LIMIT = 6000   # chars sent to LLM per paper

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
)

# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
MIN_CLUSTERS = 2
MAX_CLUSTERS = 10
DEFAULT_CLUSTERS = 4
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Streamlit
# ---------------------------------------------------------------------------
APP_TITLE = "AI Systematic Literature Review Tool"
MAX_PAPERS_UI = 50    # upper limit shown in the UI slider
