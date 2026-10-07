"""
app.py
AI Systematic Literature Review Tool
Main Streamlit application.

Runs the complete pipeline through an interactive multi-step UI.
"""

import logging
import json
from pathlib import Path
from io import BytesIO

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# local modules
from config import APP_TITLE, MAX_PAPERS_UI, DEFAULT_EMBEDDING_MODEL, OPENAI_API_KEY
from analysis.pipeline import (
    retrieve_papers,
    process_pdfs,
    analyze_papers,
    embed_and_cluster,
    generate_survey,
    generate_report,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .main-header h1 {
        color: #e2e8f0;
        font-size: 1.9rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: #94a3b8;
        font-size: 0.92rem;
        margin: 0.4rem 0 0;
    }

    .step-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }

    .step-card h3 {
        color: #e2e8f0;
        font-size: 1rem;
        font-weight: 600;
        margin: 0 0 0.5rem;
    }

    .step-card p {
        color: #94a3b8;
        font-size: 0.85rem;
        margin: 0;
    }

    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }

    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #3b82f6;
    }

    .metric-card .label {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }

    .paper-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
        transition: border-color 0.2s;
    }

    .paper-card:hover {
        border-color: #3b82f6;
    }

    .paper-card .paper-title {
        color: #e2e8f0;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }

    .paper-card .paper-meta {
        color: #64748b;
        font-size: 0.8rem;
    }

    .theme-badge {
        display: inline-block;
        background: #1d4ed8;
        color: #bfdbfe;
        font-size: 0.75rem;
        font-weight: 500;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        margin-right: 0.4rem;
        margin-bottom: 0.3rem;
    }

    .sidebar-nav-item {
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        margin-bottom: 0.25rem;
        cursor: pointer;
        font-size: 0.88rem;
        color: #94a3b8;
        transition: all 0.2s;
    }

    .sidebar-nav-item.active {
        background: #1d4ed8;
        color: white;
    }

    .status-success {
        color: #4ade80;
        font-size: 0.85rem;
    }

    .status-pending {
        color: #fbbf24;
        font-size: 0.85rem;
    }

    div[data-testid="stExpander"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
    }

    .stButton > button {
        background: #2563eb;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 500;
        padding: 0.5rem 1.5rem;
        transition: background 0.2s;
    }

    .stButton > button:hover {
        background: #1d4ed8;
    }

    .stTextInput > div > div > input,
    .stTextArea textarea,
    .stSelectbox > div > div {
        background: #1e293b;
        color: #e2e8f0;
        border-color: #334155;
    }

    .survey-section {
        background: #1e293b;
        border-left: 3px solid #3b82f6;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
    }

    .gap-item {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.88rem;
        color: #cbd5e1;
    }

    hr {
        border-color: #334155;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "step": 1,
        "papers": [],
        "selected_paper_indices": [],
        "survey": None,
        "report_path": None,
        "topic": "",
        "keywords": "",
        "max_papers": 10,
        "start_year": 2018,
        "end_year": 2025,
        "use_ss": False,
        "n_clusters": None,
        "cluster_method": "kmeans",
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "silhouette_score": 0.0,
        "api_key_ok": bool(OPENAI_API_KEY),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("### AI Literature Review")
        st.markdown("---")

        steps = [
            (1, "Research Topic"),
            (2, "Paper Search"),
            (3, "Retrieved Papers"),
            (4, "Paper Analysis"),
            (5, "Theme Clustering"),
            (6, "Comparative Analysis"),
            (7, "Research Gaps"),
            (8, "Literature Survey"),
            (9, "Download Report"),
        ]

        for num, label in steps:
            is_done = _step_done(num)
            is_active = st.session_state.step == num
            icon = "Done" if is_done and not is_active else str(num)
            style = "background:#1d4ed8;color:white;" if is_active else (
                "background:#134e2a;color:#4ade80;" if is_done else ""
            )
            if st.sidebar.button(
                f"  {icon}  {label}",
                key=f"nav_{num}",
                use_container_width=True,
                disabled=not (is_done or num == 1 or num == st.session_state.step),
            ):
                st.session_state.step = num
                st.rerun()

        st.markdown("---")
        # API key status
        if st.session_state.api_key_ok:
            st.markdown('<span class="status-success">OpenAI key loaded</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-pending">No OpenAI key — set in .env</span>', unsafe_allow_html=True)
            new_key = st.text_input("Enter OpenAI key", type="password", key="runtime_key")
            if new_key:
                import openai
                openai.api_key = new_key
                import config
                config.OPENAI_API_KEY = new_key
                import llm.openai_client as _oc
                _oc._client = None
                st.session_state.api_key_ok = True
                st.rerun()

        st.markdown("---")
        st.caption("AI Systematic Literature Review Tool | Internship Project")


def _step_done(num: int) -> bool:
    if num == 1:
        return bool(st.session_state.topic)
    if num in (2, 3):
        return bool(st.session_state.papers)
    if num == 4:
        return bool(st.session_state.papers and st.session_state.papers[0].get("analysis"))
    if num == 5:
        return bool(st.session_state.papers and st.session_state.papers[0].get("theme_name"))
    if num in (6, 7, 8):
        return st.session_state.survey is not None
    if num == 9:
        return st.session_state.report_path is not None
    return False


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>AI Systematic Literature Review Tool</h1>
        <p>Automated academic literature discovery, analysis, and survey generation powered by arXiv and large language models.</p>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page 1 — Research Topic
# ---------------------------------------------------------------------------
def page_topic():
    st.markdown("### Define Your Research Topic")
    st.markdown("Provide your research topic and search parameters. The tool will retrieve relevant papers from arXiv.")

    col1, col2 = st.columns([2, 1])

    with col1:
        topic = st.text_input(
            "Research Topic",
            value=st.session_state.topic,
            placeholder="e.g. Large Language Models for Code Generation",
            help="The main subject of your literature review.",
        )
        keywords = st.text_input(
            "Additional Keywords (optional)",
            value=st.session_state.keywords,
            placeholder="e.g. transformer, BERT, fine-tuning",
            help="Comma-separated keywords to narrow the search.",
        )

    with col2:
        max_papers = st.slider(
            "Number of Papers",
            min_value=5,
            max_value=MAX_PAPERS_UI,
            value=st.session_state.max_papers,
            step=5,
        )
        year_range = st.slider(
            "Publication Year Range",
            min_value=2000,
            max_value=2025,
            value=(st.session_state.start_year, st.session_state.end_year),
        )
        use_ss = st.checkbox(
            "Also search Semantic Scholar",
            value=st.session_state.use_ss,
            help="Requires an optional Semantic Scholar API key for higher rate limits.",
        )

    st.markdown("---")

    col_a, col_b, col_c = st.columns([2, 1, 2])
    with col_b:
        if st.button("Start Search", use_container_width=True):
            if not topic.strip():
                st.error("Please enter a research topic.")
                return

            st.session_state.topic = topic.strip()
            st.session_state.keywords = keywords.strip()
            st.session_state.max_papers = max_papers
            st.session_state.start_year = year_range[0]
            st.session_state.end_year = year_range[1]
            st.session_state.use_ss = use_ss
            # clear downstream state
            st.session_state.papers = []
            st.session_state.survey = None
            st.session_state.report_path = None
            st.session_state.step = 2
            st.rerun()


# ---------------------------------------------------------------------------
# Page 2 — Paper Search (retrieval)
# ---------------------------------------------------------------------------
def page_search():
    st.markdown("### Retrieving Papers from arXiv")
    st.markdown(
        f"Searching for papers on **{st.session_state.topic}** "
        f"({st.session_state.start_year}–{st.session_state.end_year})"
    )

    if st.session_state.papers:
        st.info(f"{len(st.session_state.papers)} papers already retrieved. Proceed to step 3 or re-search below.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Continue to Retrieved Papers"):
                st.session_state.step = 3
                st.rerun()
        with col2:
            if st.button("Re-run Search"):
                st.session_state.papers = []
                st.rerun()
        return

    status_placeholder = st.empty()
    progress_bar = st.progress(0)

    def update_progress(msg: str, _val: float):
        status_placeholder.markdown(f"**{msg}**")

    try:
        update_progress("Connecting to arXiv API...", 0.05)
        progress_bar.progress(10)

        papers = retrieve_papers(
            topic=st.session_state.topic,
            keywords=st.session_state.keywords,
            max_papers=st.session_state.max_papers,
            start_year=st.session_state.start_year,
            end_year=st.session_state.end_year,
            use_semantic_scholar=st.session_state.use_ss,
            progress=update_progress,
        )

        progress_bar.progress(60)
        update_progress("Downloading and extracting PDF text...", 0.0)

        papers = process_pdfs(papers, progress=update_progress)

        progress_bar.progress(100)

        if not papers:
            st.warning("No papers found. Try a different topic or adjust the year range.")
            return

        st.session_state.papers = papers
        st.session_state.selected_paper_indices = list(range(len(papers)))
        status_placeholder.markdown(f"**Retrieved {len(papers)} papers successfully.**")
        st.session_state.step = 3
        st.rerun()

    except Exception as exc:
        st.error(f"An error occurred during retrieval: {exc}")
        logger.exception("Retrieval error")


# ---------------------------------------------------------------------------
# Page 3 — Retrieved Papers
# ---------------------------------------------------------------------------
def page_papers():
    papers = st.session_state.papers
    if not papers:
        st.warning("No papers retrieved yet.")
        return

    st.markdown(f"### Retrieved Papers ({len(papers)} total)")
    st.markdown("Select which papers to include in the analysis and final survey.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _metric(str(len(papers)), "Papers Retrieved")
    sources = {}
    for p in papers:
        s = p.get("source", "unknown")
        sources[s] = sources.get(s, 0) + 1
    with col2:
        _metric(str(sources.get("arxiv", 0)), "From arXiv")
    with col3:
        _metric(str(sources.get("semantic_scholar", 0)), "From Semantic Scholar")
    years = [p["year"] for p in papers if p.get("year")]
    with col4:
        yr = f"{min(years)}–{max(years)}" if years else "-"
        _metric(yr, "Year Range")

    st.markdown("---")

    # Paper selection
    selected = []
    for i, paper in enumerate(papers):
        with st.expander(f"{i+1}. {paper.get('title', 'Unknown')} ({paper.get('year', '-')})", expanded=False):
            c1, c2 = st.columns([3, 1])
            with c1:
                authors = paper.get("authors", [])
                author_str = ", ".join(authors[:4]) + (" et al." if len(authors) > 4 else "")
                st.markdown(f"**Authors:** {author_str or 'Unknown'}")
                st.markdown(f"**Source:** {paper.get('source', '-')} | **Categories:** {', '.join(paper.get('categories', [])[:3]) or '-'}")
                abstract = paper.get("abstract", "No abstract available.")
                st.markdown(f"**Abstract:** {abstract[:400]}{'...' if len(abstract) > 400 else ''}")
                if paper.get("pdf_url"):
                    st.markdown(f"[View PDF]({paper['pdf_url']})")
            with c2:
                include = st.checkbox(
                    "Include in survey",
                    value=(i in st.session_state.selected_paper_indices),
                    key=f"include_{i}",
                )
            if include:
                selected.append(i)

    st.session_state.selected_paper_indices = selected

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**{len(selected)} papers selected** for analysis.")
    with col_b:
        if st.button("Run LLM Analysis on Selected Papers", use_container_width=True):
            if not selected:
                st.error("Please select at least one paper.")
                return
            if not st.session_state.api_key_ok:
                st.error("OpenAI API key is required for analysis. Please enter it in the sidebar.")
                return
            st.session_state.step = 4
            st.rerun()


# ---------------------------------------------------------------------------
# Page 4 — Paper Analysis
# ---------------------------------------------------------------------------
def page_analysis():
    papers = st.session_state.papers
    selected_idx = st.session_state.selected_paper_indices

    if not papers or not selected_idx:
        st.warning("No papers selected for analysis.")
        return

    selected_papers = [papers[i] for i in selected_idx]

    # check if already done
    already_done = all(p.get("analysis") for p in selected_papers)

    st.markdown("### Structured Paper Analysis")
    st.markdown(
        f"Analyzing {len(selected_papers)} papers using {st.session_state.get('openai_model', 'GPT-4o-mini')}. "
        "Each paper is analyzed for its research problem, methodology, results, and contributions."
    )

    if not already_done:
        progress_bar = st.progress(0)
        status = st.empty()
        n = len(selected_papers)

        for i, paper in enumerate(selected_papers):
            status.markdown(f"**Analyzing {i+1}/{n}: {paper['title'][:60]}...**")
            progress_bar.progress(int((i + 1) / n * 100))

        selected_papers = analyze_papers(selected_papers)
        # write back
        for idx_orig, paper in zip(selected_idx, selected_papers):
            st.session_state.papers[idx_orig] = paper

        progress_bar.progress(100)
        status.markdown("**Analysis complete.**")

    st.markdown("---")

    # Display results
    for paper in selected_papers:
        analysis = paper.get("analysis", {})
        with st.expander(f"{paper.get('title', 'Unknown')} ({paper.get('year', '-')})", expanded=False):
            cols = st.columns(3)
            fields = [
                ("Research Problem", "research_problem"),
                ("Objective", "objective"),
                ("Methodology", "methodology"),
                ("Dataset / Experimental Setup", "dataset_experimental_setup"),
                ("Key Results", "key_results"),
                ("Main Findings", "main_findings"),
                ("Limitations", "limitations"),
                ("Research Contribution", "research_contribution"),
                ("Future Work", "future_work"),
            ]
            for j, (label, key) in enumerate(fields):
                col = cols[j % 3]
                val = analysis.get(key, "Not reported in the paper.")
                col.markdown(f"**{label}**")
                col.markdown(f"<div class='gap-item'>{val}</div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Run Clustering and Theme Generation", use_container_width=False):
        st.session_state.step = 5
        st.rerun()


# ---------------------------------------------------------------------------
# Page 5 — Theme Clustering
# ---------------------------------------------------------------------------
def page_clustering():
    papers = st.session_state.papers
    selected_idx = st.session_state.selected_paper_indices
    selected_papers = [papers[i] for i in selected_idx]

    st.markdown("### Theme Clustering")
    st.markdown(
        "Papers are embedded using sentence transformers and grouped into thematic "
        "clusters using K-Means. Each cluster is assigned a human-readable theme name."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        n_clusters = st.selectbox(
            "Number of Clusters",
            options=["Auto"] + list(range(2, min(11, len(selected_papers)))),
            index=0,
        )
    with col2:
        method = st.selectbox("Clustering Method", ["kmeans", "agglomerative"])
    with col3:
        embed_model = st.selectbox(
            "Embedding Model",
            ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L6-v2"],
            index=0,
        )

    already_clustered = all(p.get("theme_name") for p in selected_papers)

    if not already_clustered or st.button("Re-run Clustering"):
        with st.spinner("Generating embeddings and clustering papers..."):
            n_k = None if n_clusters == "Auto" else int(n_clusters)
            annotated, sil = embed_and_cluster(
                selected_papers,
                n_clusters=n_k,
                model_name=embed_model,
                clustering_method=method,
            )
            for idx_orig, paper in zip(selected_idx, annotated):
                st.session_state.papers[idx_orig] = paper

            st.session_state.silhouette_score = sil

    # Gather clustering results
    themes = {}
    for p in selected_papers:
        tn = p.get("theme_name", "Unassigned")
        themes.setdefault(tn, []).append(p)

    # Metrics
    sil = st.session_state.silhouette_score
    c1, c2, c3 = st.columns(3)
    _metric(str(len(themes)), "Themes Identified", c1)
    _metric(str(len(selected_papers)), "Papers Clustered", c2)
    _metric(f"{sil:.3f}", "Silhouette Score", c3)

    st.markdown("---")

    # Theme breakdown chart
    theme_counts = {k: len(v) for k, v in themes.items()}
    fig = px.bar(
        x=list(theme_counts.keys()),
        y=list(theme_counts.values()),
        labels={"x": "Theme", "y": "Number of Papers"},
        color=list(theme_counts.values()),
        color_continuous_scale="Blues",
        title="Papers per Theme",
    )
    fig.update_layout(
        plot_bgcolor="#1e293b",
        paper_bgcolor="#1e293b",
        font_color="#e2e8f0",
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Theme details
    for theme_name, theme_papers in themes.items():
        with st.expander(f"{theme_name} ({len(theme_papers)} papers)", expanded=False):
            for p in theme_papers:
                st.markdown(f"- **{p.get('title', 'Unknown')}** ({p.get('year', '-')})")

    st.markdown("---")
    if st.button("Generate Comparative Analysis"):
        st.session_state.step = 6
        st.rerun()


# ---------------------------------------------------------------------------
# Page 6 — Comparative Analysis
# ---------------------------------------------------------------------------
def page_comparison():
    papers = st.session_state.papers
    selected_idx = st.session_state.selected_paper_indices
    selected_papers = [papers[i] for i in selected_idx]

    st.markdown("### Comparative Analysis")
    st.markdown("Cross-paper comparison across problem, method, dataset, results, and limitations.")

    from analysis.comparator import build_comparison_table, compute_statistics
    df = build_comparison_table(selected_papers)
    stats = compute_statistics(selected_papers)

    # Year distribution
    if stats["by_year"]:
        year_df = pd.DataFrame(
            list(stats["by_year"].items()), columns=["Year", "Count"]
        ).sort_values("Year")
        fig = px.bar(
            year_df, x="Year", y="Count",
            title="Papers by Publication Year",
            color="Count", color_continuous_scale="Viridis",
        )
        fig.update_layout(
            plot_bgcolor="#1e293b", paper_bgcolor="#1e293b",
            font_color="#e2e8f0", showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Comparison table
    st.markdown("#### Cross-Paper Comparison Table")
    st.dataframe(
        df,
        use_container_width=True,
        height=min(400, len(df) * 60 + 50),
    )

    col_dl, _ = st.columns([1, 3])
    with col_dl:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download as CSV",
            data=csv,
            file_name="comparison_table.csv",
            mime="text/csv",
        )

    st.markdown("---")
    if st.button("Identify Research Gaps"):
        st.session_state.step = 7
        st.rerun()


# ---------------------------------------------------------------------------
# Page 7 — Research Gaps
# ---------------------------------------------------------------------------
def page_gaps():
    papers = st.session_state.papers
    selected_idx = st.session_state.selected_paper_indices
    selected_papers = [papers[i] for i in selected_idx]

    st.markdown("### Research Gaps and Open Challenges")

    if st.session_state.survey and st.session_state.survey.get("research_gaps"):
        gaps_text = st.session_state.survey["research_gaps"]
        st.markdown('<div class="survey-section">', unsafe_allow_html=True)
        st.markdown(gaps_text)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        with st.spinner("Analyzing research gaps across all papers..."):
            from llm.survey_generator import identify_research_gaps
            gaps = identify_research_gaps(selected_papers)
            if not st.session_state.survey:
                st.session_state.survey = {}
            st.session_state.survey["research_gaps"] = gaps
        st.markdown('<div class="survey-section">', unsafe_allow_html=True)
        st.markdown(gaps)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Generate Full Literature Survey"):
        st.session_state.step = 8
        st.rerun()


# ---------------------------------------------------------------------------
# Page 8 — Literature Survey
# ---------------------------------------------------------------------------
def page_survey():
    papers = st.session_state.papers
    selected_idx = st.session_state.selected_paper_indices
    selected_papers = [papers[i] for i in selected_idx]
    topic = st.session_state.topic

    st.markdown("### Literature Survey Generation")
    st.markdown(
        "The complete survey is generated section by section. "
        "This may take a few minutes depending on the number of papers."
    )

    if st.session_state.survey and st.session_state.survey.get("introduction"):
        _display_survey(st.session_state.survey, selected_papers)
    else:
        progress_bar = st.progress(0)
        status = st.empty()

        sections = [
            "Introduction",
            "Thematic Sections",
            "Research Gaps",
            "Conclusion",
            "References",
        ]

        def _prog(msg, _v):
            status.markdown(f"**{msg}**")

        try:
            with st.spinner("Generating survey..."):
                survey = generate_survey(
                    topic=topic,
                    papers=selected_papers,
                    progress=_prog,
                )
                # preserve existing gaps if already computed
                if st.session_state.survey and st.session_state.survey.get("research_gaps"):
                    survey["research_gaps"] = st.session_state.survey["research_gaps"]

                st.session_state.survey = survey
                progress_bar.progress(100)
                status.markdown("**Survey generation complete.**")

        except Exception as exc:
            st.error(f"Survey generation error: {exc}")
            logger.exception("Survey generation failed")
            return

        _display_survey(st.session_state.survey, selected_papers)

    st.markdown("---")
    if st.button("Proceed to Download Report"):
        st.session_state.step = 9
        st.rerun()


def _display_survey(survey: dict, papers: list[dict]):
    tabs = st.tabs([
        "Introduction", "Thematic Review",
        "Research Gaps", "Conclusion", "References",
    ])

    with tabs[0]:
        st.markdown('<div class="survey-section">', unsafe_allow_html=True)
        st.markdown(survey.get("introduction", "Not generated."))
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[1]:
        for theme, text in survey.get("theme_sections", {}).items():
            st.markdown(f"#### {theme}")
            st.markdown('<div class="survey-section">', unsafe_allow_html=True)
            st.markdown(text)
            st.markdown("</div>", unsafe_allow_html=True)

    with tabs[2]:
        st.markdown('<div class="survey-section">', unsafe_allow_html=True)
        st.markdown(survey.get("research_gaps", "Not generated."))
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[3]:
        st.markdown('<div class="survey-section">', unsafe_allow_html=True)
        st.markdown(survey.get("conclusion", "Not generated."))
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[4]:
        for ref in survey.get("references", []):
            st.markdown(f"<div class='gap-item'>{ref}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page 9 — Download Report
# ---------------------------------------------------------------------------
def page_download():
    st.markdown("### Download Literature Survey Report")

    if not st.session_state.survey:
        st.warning("Please complete the survey generation first.")
        return

    papers = st.session_state.papers
    selected_idx = st.session_state.selected_paper_indices
    selected_papers = [papers[i] for i in selected_idx]
    topic = st.session_state.topic
    survey = st.session_state.survey

    # Generate if not yet done
    if not st.session_state.report_path:
        with st.spinner("Building Word document..."):
            try:
                path = generate_report(topic, selected_papers, survey)
                st.session_state.report_path = path
            except Exception as exc:
                st.error(f"Report generation error: {exc}")
                logger.exception("Report generation failed")
                return

    report_path = Path(st.session_state.report_path)

    if report_path.exists():
        with open(report_path, "rb") as fh:
            docx_bytes = fh.read()

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("#### Report Ready")
            st.markdown(
                f"Your literature survey on **{topic}** has been generated and saved as a Word document (.docx)."
            )
            st.markdown(
                f"- **Papers included:** {len(selected_papers)}\n"
                f"- **Themes:** {len(survey.get('themes', []))}\n"
                f"- **File:** `{report_path.name}`"
            )
        with col2:
            st.download_button(
                label="Download .docx Report",
                data=docx_bytes,
                file_name=report_path.name,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown("#### Report Contents")
        sections = [
            "Cover Page and Title",
            "Table of Contents",
            "Introduction and Background",
            "Search Strategy and Methodology",
            "Overview of Selected Papers",
            "Thematic Literature Review",
            "Comparative Analysis Table",
            "Research Gaps and Open Challenges",
            "Future Research Directions",
            "Conclusion",
            "IEEE References",
        ]
        for section in sections:
            st.markdown(f"- {section}")
    else:
        st.error("Report file not found. Please regenerate.")
        st.session_state.report_path = None


# ---------------------------------------------------------------------------
# Helper components
# ---------------------------------------------------------------------------
def _metric(value: str, label: str, col=None):
    html = f"""
    <div class="metric-card">
        <div class="value">{value}</div>
        <div class="label">{label}</div>
    </div>
    """
    target = col if col else st
    target.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
def main():
    render_sidebar()
    render_header()

    step = st.session_state.step
    pages = {
        1: page_topic,
        2: page_search,
        3: page_papers,
        4: page_analysis,
        5: page_clustering,
        6: page_comparison,
        7: page_gaps,
        8: page_survey,
        9: page_download,
    }
    page_fn = pages.get(step, page_topic)
    page_fn()


if __name__ == "__main__":
    main()
