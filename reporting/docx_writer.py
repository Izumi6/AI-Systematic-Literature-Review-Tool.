"""
reporting/docx_writer.py
Generates a professionally formatted .docx literature survey using python-docx.

Includes:
  - Cover / title section
  - Table of contents (manual, linked)
  - Research methodology
  - Thematic literature review sections
  - Comparison table
  - Research gaps
  - Conclusion
  - References
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import pandas as pd

from config import OUTPUTS_DIR
from utils.helpers import current_timestamp, format_authors

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

def _set_heading_style(paragraph, level: int = 1):
    paragraph.style = f"Heading {level}"


def _set_cell_bg(cell, hex_color: str):
    """Set table cell background colour."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def _add_horizontal_rule(doc: Document):
    p = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "4")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "CCCCCC")
    pBdr.append(bottom)
    pPr.append(pBdr)


def _add_toc(doc: Document):
    """Insert a manual table of contents placeholder."""
    paragraph = doc.add_paragraph()
    run = paragraph.add_run("Table of Contents")
    run.bold = True
    run.font.size = Pt(13)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT

    toc_entries = [
        ("1. Introduction and Background", "3"),
        ("2. Search Strategy and Methodology", "4"),
        ("3. Overview of Selected Papers", "5"),
        ("4. Thematic Literature Review", "6"),
        ("5. Comparative Analysis", "8"),
        ("6. Research Gaps", "10"),
        ("7. Future Research Directions", "11"),
        ("8. Conclusion", "12"),
        ("9. References", "13"),
    ]
    for entry, page in toc_entries:
        p = doc.add_paragraph(style="Normal")
        run_text = p.add_run(f"  {entry}")
        run_text.font.size = Pt(10)
        p.paragraph_format.space_after = Pt(2)


# ---------------------------------------------------------------------------
# Main document builder
# ---------------------------------------------------------------------------

class DocxWriter:
    """
    Assembles and saves the complete literature survey as a .docx file.
    """

    def __init__(self) -> None:
        self.doc = Document()
        self._configure_styles()

    def _configure_styles(self):
        """Apply document-level defaults (margins, fonts)."""
        from docx.shared import Cm
        sections = self.doc.sections
        for section in sections:
            section.top_margin = Cm(2.5)
            section.bottom_margin = Cm(2.5)
            section.left_margin = Cm(3.0)
            section.right_margin = Cm(2.5)

        # set default body font
        style = self.doc.styles["Normal"]
        font = style.font
        font.name = "Cambria"
        font.size = Pt(11)

    def build(
        self,
        topic: str,
        papers: list[dict],
        themes: list[str],
        theme_sections: dict[str, str],
        comparison_df: pd.DataFrame,
        research_gaps: str,
        introduction: str,
        conclusion: str,
        references: list[str],
        search_strategy: str = "",
    ) -> Path:
        """
        Build the complete document and save it to the outputs directory.

        Returns the path to the saved .docx file.
        """
        doc = self.doc

        # ------------------------------------------------------------------
        # Cover page
        # ------------------------------------------------------------------
        cover_title = doc.add_paragraph()
        cover_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cover_title.add_run("AI Systematic Literature Review Tool")
        run.bold = True
        run.font.size = Pt(18)
        run.font.color.rgb = RGBColor(0x1A, 0x73, 0xE8)

        doc.add_paragraph()

        survey_title = doc.add_paragraph()
        survey_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        t_run = survey_title.add_run(f"Literature Survey on:\n{topic}")
        t_run.bold = True
        t_run.font.size = Pt(15)

        doc.add_paragraph()

        meta = doc.add_paragraph()
        meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
        meta.add_run(
            f"Generated: {datetime.now().strftime('%B %d, %Y')}\n"
            f"Papers Reviewed: {len(papers)}\n"
            f"Themes Identified: {len(themes)}"
        ).font.size = Pt(10)

        doc.add_page_break()

        # ------------------------------------------------------------------
        # Table of Contents
        # ------------------------------------------------------------------
        h = doc.add_heading("Table of Contents", level=1)
        _add_toc(doc)
        doc.add_page_break()

        # ------------------------------------------------------------------
        # 1. Introduction
        # ------------------------------------------------------------------
        doc.add_heading("1. Introduction and Background", level=1)
        self._add_body(introduction)
        _add_horizontal_rule(doc)
        doc.add_paragraph()

        # ------------------------------------------------------------------
        # 2. Search Strategy
        # ------------------------------------------------------------------
        doc.add_heading("2. Search Strategy and Methodology", level=1)
        if search_strategy:
            self._add_body(search_strategy)
        else:
            self._add_body(
                f"This systematic literature review was conducted using the arXiv open-access "
                f"repository as the primary source. Papers were retrieved using the query term "
                f'"{topic}" and filtered by relevance. A total of {len(papers)} papers were '
                f"selected after deduplication. PDF full-text was extracted where available; "
                f"otherwise, abstracts were used for analysis. Each paper was analyzed using "
                f"a large language model (GPT-4o-mini) to extract structured information. "
                f"Papers were subsequently clustered into {len(themes)} thematic groups using "
                f"K-Means clustering on sentence-transformer embeddings."
            )
        doc.add_paragraph()

        # ------------------------------------------------------------------
        # 3. Overview of Selected Papers
        # ------------------------------------------------------------------
        doc.add_heading("3. Overview of Selected Papers", level=1)
        self._add_overview_table(papers)
        doc.add_paragraph()

        # ------------------------------------------------------------------
        # 4. Thematic Literature Review
        # ------------------------------------------------------------------
        doc.add_heading("4. Thematic Literature Review", level=1)
        for i, (theme, section_text) in enumerate(theme_sections.items(), start=1):
            doc.add_heading(f"4.{i} {theme}", level=2)
            self._add_body(section_text)
            doc.add_paragraph()

        # ------------------------------------------------------------------
        # 5. Comparative Analysis
        # ------------------------------------------------------------------
        doc.add_heading("5. Comparative Analysis", level=1)
        self._add_comparison_table(comparison_df)
        doc.add_paragraph()

        # ------------------------------------------------------------------
        # 6. Research Gaps
        # ------------------------------------------------------------------
        doc.add_heading("6. Research Gaps and Open Challenges", level=1)
        self._add_body(research_gaps)
        doc.add_paragraph()

        # ------------------------------------------------------------------
        # 7. Future Research Directions
        # ------------------------------------------------------------------
        doc.add_heading("7. Future Research Directions", level=1)
        self._add_body(
            "Based on the research gaps identified, the following directions represent "
            "promising avenues for future investigation in the field of " + topic + ":\n\n"
            + "\n".join(f"- {t}" for t in themes)
        )
        doc.add_paragraph()

        # ------------------------------------------------------------------
        # 8. Conclusion
        # ------------------------------------------------------------------
        doc.add_heading("8. Conclusion", level=1)
        self._add_body(conclusion)
        doc.add_paragraph()

        # ------------------------------------------------------------------
        # 9. References
        # ------------------------------------------------------------------
        doc.add_heading("9. References", level=1)
        for ref in references:
            p = doc.add_paragraph(style="Normal")
            p.add_run(ref).font.size = Pt(9)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.first_line_indent = Inches(-0.3)

        # ------------------------------------------------------------------
        # Save
        # ------------------------------------------------------------------
        safe_topic = "".join(
            c if c.isalnum() or c in (" ", "_") else "_" for c in topic
        )[:50].strip()
        filename = f"Literature_Survey_{safe_topic}_{current_timestamp()}.docx"
        output_path = OUTPUTS_DIR / filename
        doc.save(output_path)
        logger.info("Saved literature survey: %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_body(self, text: str):
        """Add paragraphs of body text, splitting on double newlines."""
        for paragraph_text in text.split("\n\n"):
            stripped = paragraph_text.strip()
            if not stripped:
                continue
            if stripped.startswith(("- ", "* ")):
                # bullet list
                for item in stripped.splitlines():
                    item = item.lstrip("- *").strip()
                    if item:
                        p = self.doc.add_paragraph(style="List Bullet")
                        p.add_run(item).font.size = Pt(10.5)
            else:
                p = self.doc.add_paragraph(style="Normal")
                p.add_run(stripped).font.size = Pt(10.5)
                p.paragraph_format.space_after = Pt(6)

    def _add_overview_table(self, papers: list[dict]):
        """Add a compact overview table of all papers."""
        headers = ["#", "Title", "Authors", "Year", "Source"]
        table = self.doc.add_table(rows=1, cols=len(headers))
        table.style = "Table Grid"

        hdr_row = table.rows[0]
        for i, h in enumerate(headers):
            cell = hdr_row.cells[i]
            _set_cell_bg(cell, "1A73E8")
            run = cell.paragraphs[0].add_run(h)
            run.bold = True
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            run.font.size = Pt(9)

        for idx, p in enumerate(papers, start=1):
            row = table.add_row()
            values = [
                str(idx),
                p.get("title", "")[:60],
                format_authors(p.get("authors", []), max_authors=2),
                str(p.get("year", "-")),
                p.get("source", "-"),
            ]
            for i, val in enumerate(values):
                cell = row.cells[i]
                if idx % 2 == 0:
                    _set_cell_bg(cell, "F0F4FF")
                cell.paragraphs[0].add_run(val).font.size = Pt(9)

    def _add_comparison_table(self, df: pd.DataFrame):
        """Add the cross-paper comparison DataFrame as a Word table."""
        if df.empty:
            self.doc.add_paragraph("No comparison data available.")
            return

        cols = list(df.columns)
        table = self.doc.add_table(rows=1, cols=len(cols))
        table.style = "Table Grid"

        hdr_row = table.rows[0]
        for i, col in enumerate(cols):
            cell = hdr_row.cells[i]
            _set_cell_bg(cell, "2C3E50")
            run = cell.paragraphs[0].add_run(col)
            run.bold = True
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            run.font.size = Pt(8)

        for row_idx, row_data in df.iterrows():
            row = table.add_row()
            for col_idx, val in enumerate(row_data):
                cell = row.cells[col_idx]
                if row_idx % 2 == 0:
                    _set_cell_bg(cell, "F8F9FA")
                text = str(val)[:150] if val else "-"
                cell.paragraphs[0].add_run(text).font.size = Pt(8)
