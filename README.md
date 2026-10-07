# AI Systematic Literature Review Tool

An end-to-end academic literature review assistant that automatically retrieves research papers from arXiv, analyzes each paper with a large language model, clusters them by semantic theme, identifies research gaps, and generates a professionally formatted Word document literature survey.

Developed as a major internship project demonstrating the practical application of NLP, LLM APIs, semantic embeddings, and automated document generation in an academic research workflow.

---

## Problem Statement

Conducting a systematic literature review is one of the most time-consuming tasks in academic research. A researcher must:

- Identify relevant papers from multiple sources
- Read and summarize each paper individually
- Group papers by theme or methodology
- Identify gaps and conflicts in the literature
- Write a coherent, structured survey

This process can take weeks for even a moderately sized corpus. This tool automates the retrieval, reading, analysis, and synthesis steps while keeping the researcher in control of the final output.

---

## Objectives

1. Retrieve relevant research papers automatically from arXiv (primary) and Semantic Scholar (optional secondary source).
2. Extract full text from PDFs using pdfminer.six with graceful fallback to abstracts.
3. Analyze each paper with an LLM to extract structured information: problem, methodology, results, limitations, and contributions.
4. Cluster papers into thematic groups using sentence-transformer embeddings and K-Means.
5. Identify research gaps and emerging directions across the entire corpus.
6. Generate a complete, thematically organized literature survey with IEEE references.
7. Export the survey as a formatted Word (.docx) document.

---

## Features

- **Live paper retrieval** from arXiv with date range and category filtering
- **Semantic Scholar integration** as a configurable secondary source
- **PDF text extraction** with section detection (Abstract, Introduction, Methods, etc.)
- **Structured per-paper analysis** (9 fields) using LLM reasoning with JSON schema enforcement
- **Semantic embeddings** via sentence-transformers with disk caching
- **Automatic clustering** with optimal-k selection via Silhouette Score
- **LLM-generated theme names** for each cluster
- **Cross-paper comparison table** exportable as CSV
- **Research gap identification** distinguishing author-stated vs. inferred gaps
- **Complete survey generation** organized by theme, not by paper
- **IEEE reference list** built solely from retrieved metadata
- **Word document export** with cover page, TOC, tables, and academic formatting
- **Disk caching** for API results, PDF text, and LLM analyses to minimize redundant calls
- **9-step Streamlit wizard** with progress indicators and paper selection controls

---

## System Architecture

```
User Input (Streamlit UI)
        |
        v
   analysis/pipeline.py  <-- orchestrates all steps
        |
   +----+----+--------+----------+----------+----------+
   |         |        |          |          |          |
arXiv API  S2 API  PDF DL   PDF Text   LLM Analysis  Embeddings
   |         |      |            |          |          |
   +----+----+      pdfminer  extractor   llm_client   embedder
        |           .six                               |
   deduplicate                                    sentence-
        |                                        transformers
        v
   DiskCache (data/)
        |
        v
   Clustering (scikit-learn KMeans)
        |
        v
   Theme Naming (LLM)
        |
        v
   Survey Generation (LLM, section by section)
        |
        v
   Reference Builder (from metadata only)
        |
        v
   DocxWriter (python-docx)
        |
        v
   outputs/*.docx  (downloadable from UI)
```

---

## Technology Stack

| Component | Library / Service |
|---|---|
| Web UI | Streamlit |
| Paper Retrieval | arXiv Atom API (requests) |
| Secondary Source | Semantic Scholar REST API |
| PDF Extraction | pdfminer.six |
| LLM Analysis & Synthesis | OpenAI-compatible API (GPT-4o-mini, Groq, DeepSeek, etc.) |
| Embeddings | sentence-transformers |
| Clustering | scikit-learn (KMeans, Agglomerative) |
| Data Processing | pandas, numpy |
| Visualizations | Plotly |
| Word Export | python-docx |
| Caching | Custom DiskCache (JSON files) |
| Configuration | python-dotenv |

---

## Project Structure

```
AI Systematic Literature Review Tool/
├── app.py                        # Streamlit application (entry point)
├── config.py                     # Centralized configuration
├── requirements.txt
├── .env.example
├── README.md
│
├── api/
│   ├── __init__.py
│   ├── arxiv_client.py           # arXiv search and feed parsing
│   └── semantic_scholar_client.py# Optional secondary source
│
├── pdf_processing/
│   ├── __init__.py
│   ├── downloader.py             # PDF download with size limits and caching
│   └── extractor.py              # pdfminer text extraction and section detection
│
├── llm/
│   ├── __init__.py
│   ├── openai_client.py          # OpenAI wrapper with retry logic
│   ├── paper_analyzer.py         # Structured 9-field paper analysis
│   └── survey_generator.py       # Theme naming, gap analysis, survey text
│
├── embeddings/
│   ├── __init__.py
│   └── embedder.py               # Sentence-transformer embeddings with .npy caching
│
├── clustering/
│   ├── __init__.py
│   └── clusterer.py              # KMeans / Agglomerative with Silhouette Score
│
├── analysis/
│   ├── __init__.py
│   ├── pipeline.py               # Full pipeline orchestrator
│   ├── comparator.py             # Cross-paper comparison table and statistics
│   └── references.py             # IEEE reference formatter
│
├── reporting/
│   ├── __init__.py
│   └── docx_writer.py            # Word document generation (python-docx)
│
├── utils/
│   ├── __init__.py
│   ├── helpers.py                # Text cleaning, JSON parsing, date utils
│   └── cache.py                  # Disk-based JSON cache
│
├── data/
│   ├── pdfs/                     # Downloaded PDF files (auto-created)
│   ├── metadata/                 # Cached API results and LLM analyses
│   └── embeddings/               # Saved .npy embedding arrays
│
├── outputs/                      # Generated .docx reports
│
└── tests/
    ├── __init__.py
    ├── test_arxiv_client.py
    ├── test_pdf_extractor.py
    ├── test_clustering.py
    ├── test_references.py
    └── test_helpers.py
```

---

## Installation

### Prerequisites

- Python 3.10 or later
- pip
- An API key (OpenAI, Groq, DeepSeek, OpenRouter, or any OpenAI-compatible API)

### Steps

```bash
# 1. Clone or download the project
cd "AI Systematic Literature Review Tool"

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Open .env and set API_KEY=your_actual_key
```

---

## Environment Setup

Open `.env` and configure:

```env
API_KEY=your_api_key_here         # Required for LLM analysis & synthesis
LLM_MODEL=gpt-4o-mini             # Optional (e.g. gpt-4o-mini, llama-3.3-70b-versatile, deepseek-chat)
API_BASE_URL=                     # Optional custom endpoint (leave blank for standard OpenAI)
SEMANTIC_SCHOLAR_API_KEY=         # Optional, blank = free tier
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Optional
```

API keys are loaded via `python-dotenv` and are never hard-coded in the source.

---

## Running the Application

```bash
streamlit run app.py
```

The application opens in your browser at `http://localhost:8501`.

---

## Workflow

The application guides you through 9 steps:

| Step | Description |
|------|-------------|
| 1. Research Topic | Enter topic, keywords, paper count, and year range |
| 2. Paper Search | Retrieves papers from arXiv and downloads PDFs |
| 3. Retrieved Papers | Review papers and select which to include |
| 4. Paper Analysis | LLM analyzes each paper for 9 structured fields |
| 5. Theme Clustering | Embeds and clusters papers; generates theme names |
| 6. Comparative Analysis | Cross-paper comparison table with year chart |
| 7. Research Gaps | LLM identifies gaps across the full corpus |
| 8. Literature Survey | Generates the full thematic survey text |
| 9. Download Report | Download the formatted .docx Word document |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Individual test modules:

```bash
python -m pytest tests/test_arxiv_client.py -v
python -m pytest tests/test_clustering.py -v
python -m pytest tests/test_helpers.py -v
```

---

## Evaluation Metrics

The tool tracks and displays the following quality indicators:

| Metric | Description |
|--------|-------------|
| Retrieval Count | Number of papers retrieved per query |
| Deduplication | Duplicate detection using arXiv ID, DOI, or normalized title |
| Silhouette Score | Clustering quality (-1 to 1; higher is better separation) |
| LLM Output Validity | JSON structure completeness check for all 9 analysis fields |
| PDF Extraction Rate | Proportion of papers with successful full-text extraction |

---

## Limitations

This tool has the following known limitations, which should be disclosed when presenting the project:

1. **LLM costs**: Each paper analysis requires one API call. Analyzing 20+ papers with GPT-4 (non-mini) can be expensive. Use `gpt-4o-mini` for development.
2. **PDF extraction quality**: pdfminer.six works well on text-based PDFs but may produce garbled output for scanned, two-column, or heavily formatted papers. The fallback to abstract handles this gracefully.
3. **arXiv bias**: The primary data source is arXiv, which skews toward preprints in CS, physics, and mathematics. Published journal papers not posted to arXiv will not be retrieved.
4. **LLM hallucination risk**: Despite strict prompting and JSON enforcement, LLMs can occasionally produce inaccurate summaries. Users should verify key claims against the original papers.
5. **Clustering quality**: With small paper sets (fewer than 10), clustering results may not be meaningful. The Silhouette Score is provided to help the user judge this.
6. **No full PRISMA compliance**: This tool automates retrieval and synthesis but does not enforce the full PRISMA systematic review protocol (screening criteria, GRADE assessment, etc.).

---

## Future Enhancements

- Support for PubMed and IEEE Xplore APIs
- Citation network visualization (paper-to-paper relationships)
- PRISMA-compliant filtering workflow
- User-defined exclusion criteria
- Multi-language support
- Fine-tuned domain-specific analysis models
- Integration with reference managers (Zotero, Mendeley)
- Incremental updates when new papers are published

---

## Acknowledgments

- arXiv open-access repository for making research papers freely accessible
- Semantic Scholar for the open academic graph API
- Hugging Face for the sentence-transformers library
- Open-source AI and LLM communities for inference APIs
