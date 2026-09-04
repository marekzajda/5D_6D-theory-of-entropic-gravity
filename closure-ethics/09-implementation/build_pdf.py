#!/usr/bin/env python3
"""Build the open Closure Ethics implementation PDF from repository sources."""
from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted, PageBreak, ListFlowable, ListItem, HRFlowable

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DOCS = PROJECT / "docs"
OUT = DOCS / "downloads" / "Closure_Ethics_Implementation_Spec_v0.1.pdf"
SPEC = HERE / "IMPLEMENTATION_SPEC.md"
EXTRA_DOCS = [
    ("Recovered Foundation I - Universal Closure Protocol", PROJECT / "11-communication" / "UNIVERSAL_CLOSURE_PROTOCOL.md"),
    ("Recovered Foundation II - Global AI Closure Standard", PROJECT / "12-standard" / "GLOBAL_AI_CLOSURE_STANDARD_DRAFT.md"),
]
CODE_FILES = [
    ("Appendix A - Full Python reference implementation", HERE / "closure_ethics.py"),
    ("Appendix B - Runnable example", HERE / "example.py"),
    ("Appendix C - Reference tests", HERE / "test_reference.py"),
    ("Appendix D - Example policy profile", HERE / "policy.example.json"),
    ("Appendix E - Universal Closure Protocol Python reference", PROJECT / "11-communication" / "closure_protocol.py"),
    ("Appendix F - Universal Closure Protocol tests", PROJECT / "11-communication" / "test_closure_protocol.py"),
]


def font_setup():
    candidates = [
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("/usr/share/fonts/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
    ]
    for regular, bold in candidates:
        if Path(regular).exists() and Path(bold).exists():
            pdfmetrics.registerFont(TTFont("CE-Regular", regular))
            pdfmetrics.registerFont(TTFont("CE-Bold", bold))
            return "CE-Regular", "CE-Bold"
    return "Helvetica", "Helvetica-Bold"


REGULAR, BOLD = font_setup()


def sanitise_formula(s: str) -> str:
    reps = {
        "\\mathcal{S}": "S", "\\mathcal{A}_{safe}": "A_safe", "\\mathcal{E}": "E",
        "\\mathbf{p}": "p", "\\hat H": "H_hat", "\\Theta": "Theta",
        "\\Lambda": "Lambda", "\\delta": "delta", "\\rho": "rho", "\\pi": "pi",
        "\\forall": "forall", "\\leadsto": "->", "\\mapsto": "->", "\\circ": " o ",
        "\\le": "<=", "\\ge": ">=", "\\min": "min", "\\max": "max",
        "\\sum": "sum", "\\frac": "/", "\\left": "", "\\right": "",
        "\\qquad": "   ", "\\quad": "  ", "\\,": " ", "\\;": " ",
        "\\{": "{", "\\}": "}",
    }
    for a, b in reps.items():
        s = s.replace(a, b)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\begin\{[^}]+\}|\\end\{[^}]+\}", "", s)
    s = s.replace("\\", "")
    s = s.replace("→", "->").replace("≤", "<=").replace("≥", ">=")
    return s.strip()


def inline_md(s: str) -> str:
    s = html.escape(s)
    s = s.replace("→", "-&gt;").replace("–", "-").replace("—", "-")
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', s)
    return s


def make_styles():
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=ss["Title"], fontName=BOLD, fontSize=25, leading=29, textColor=colors.HexColor("#0B2533"), spaceAfter=8, alignment=TA_CENTER),
        "subtitle": ParagraphStyle("subtitle", parent=ss["Normal"], fontName=REGULAR, fontSize=12, leading=16, textColor=colors.HexColor("#3A5D6E"), alignment=TA_CENTER, spaceAfter=7),
        "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontName=BOLD, fontSize=18, leading=22, textColor=colors.HexColor("#0B4652"), spaceBefore=12, spaceAfter=7),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName=BOLD, fontSize=14, leading=18, textColor=colors.HexColor("#0B4652"), spaceBefore=10, spaceAfter=5),
        "h3": ParagraphStyle("h3", parent=ss["Heading3"], fontName=BOLD, fontSize=11.5, leading=15, textColor=colors.HexColor("#183E4B"), spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("body", parent=ss["BodyText"], fontName=REGULAR, fontSize=9.2, leading=13, textColor=colors.HexColor("#1E2B31"), spaceAfter=6),
        "quote": ParagraphStyle("quote", parent=ss["BodyText"], fontName=REGULAR, fontSize=10.2, leading=14, textColor=colors.HexColor("#0B4652"), leftIndent=10, rightIndent=10, borderColor=colors.HexColor("#5ABFA8"), borderWidth=1.2, borderPadding=8, backColor=colors.HexColor("#F1FBF8"), spaceBefore=5, spaceAfter=8),
        "code": ParagraphStyle("code", fontName="Courier", fontSize=6.6, leading=8.0, textColor=colors.HexColor("#102A35"), leftIndent=5, rightIndent=5, backColor=colors.HexColor("#F4F8FA"), borderPadding=5, spaceAfter=5),
        "formula": ParagraphStyle("formula", fontName="Courier", fontSize=8.2, leading=11, textColor=colors.HexColor("#0B4652"), leftIndent=12, backColor=colors.HexColor("#F4F8FA"), borderPadding=6, spaceAfter=7),
        "small": ParagraphStyle("small", parent=ss["BodyText"], fontName=REGULAR, fontSize=7.8, leading=10, textColor=colors.HexColor("#5C6C74")),
    }


ST = make_styles()


def page_decor(canvas, doc):
    canvas.saveState()
    w, _h = A4
    canvas.setStrokeColor(colors.HexColor("#D5E2E8"))
    canvas.line(18*mm, 15*mm, w-18*mm, 15*mm)
    canvas.setFont(REGULAR, 7.5)
    canvas.setFillColor(colors.HexColor("#607681"))
    canvas.drawString(18*mm, 9*mm, "Closure Ethics v0.1 - Project Möbia and Marek Zajda")
    canvas.drawRightString(w-18*mm, 9*mm, f"Page {doc.page}")
    canvas.restoreState()


def cover(story):
    story += [Spacer(1, 35*mm)]
    story.append(Paragraph("Closure Ethics for Autonomous Agents", ST["title"]))
    story.append(Paragraph("Mathematical Specification, Reference Algorithm and Open Implementation v0.1", ST["subtitle"]))
    story.append(Spacer(1, 7*mm))
    story.append(HRFlowable(width="65%", thickness=1.2, color=colors.HexColor("#5ABFA8"), hAlign="CENTER"))
    story.append(Spacer(1, 8*mm))
    story.append(Paragraph("<b>Project Möbia and Marek Zajda</b>", ST["subtitle"]))
    story.append(Paragraph("Open working specification / research prototype - September 2026", ST["subtitle"]))
    story.append(Spacer(1, 15*mm))
    story.append(Paragraph("<b>Keep the future repairable.</b>", ST["quote"]))
    story.append(Spacer(1, 18*mm))
    story.append(Paragraph("Text/specification: CC BY 4.0. Reference code: MIT. This framework does not claim that present AI systems are conscious, and it does not claim that Omega-RTR physics proves ethics.", ST["small"]))
    story.append(PageBreak())


def flush_paragraph(buf, story):
    if buf:
        story.append(Paragraph(inline_md(" ".join(x.strip() for x in buf)), ST["body"]))
        buf.clear()


def render_markdown(md: str, story):
    lines = md.splitlines()
    para, code, math, bullets = [], [], [], []
    code_mode = math_mode = False

    def flush_bullets():
        nonlocal bullets
        if bullets:
            items = [ListItem(Paragraph(inline_md(x), ST["body"]), leftIndent=12) for x in bullets]
            story.append(ListFlowable(items, bulletType="bullet", leftIndent=18, bulletFontName=REGULAR, bulletFontSize=6))
            bullets = []

    for line in lines:
        if line.startswith("```"):
            flush_paragraph(para, story); flush_bullets()
            if code_mode:
                story.append(Preformatted("\n".join(code), ST["code"], maxLineLength=110)); code = []; code_mode = False
            else:
                code_mode = True
            continue
        if code_mode:
            code.append(line.replace("\t", "    ")); continue
        if line.strip() in {"$$", "\\[", "\\]"}:
            flush_paragraph(para, story); flush_bullets()
            if math_mode:
                story.append(Preformatted(sanitise_formula(" ".join(math)), ST["formula"], maxLineLength=95)); math = []; math_mode = False
            else:
                math_mode = True
            continue
        if math_mode:
            math.append(line); continue
        if not line.strip():
            flush_paragraph(para, story); flush_bullets(); story.append(Spacer(1, 2)); continue
        if line.startswith("# "):
            flush_paragraph(para, story); flush_bullets(); story.append(Paragraph(inline_md(line[2:]), ST["h1"])); continue
        if line.startswith("## "):
            flush_paragraph(para, story); flush_bullets(); story.append(Paragraph(inline_md(line[3:]), ST["h2"])); continue
        if line.startswith("### "):
            flush_paragraph(para, story); flush_bullets(); story.append(Paragraph(inline_md(line[4:]), ST["h3"])); continue
        if line.startswith("> "):
            flush_paragraph(para, story); flush_bullets(); story.append(Paragraph(inline_md(line[2:]), ST["quote"])); continue
        if re.match(r"^[-*] ", line):
            flush_paragraph(para, story); bullets.append(line[2:].strip()); continue
        if re.match(r"^\d+\. ", line):
            flush_paragraph(para, story); flush_bullets(); story.append(Paragraph(inline_md(line), ST["body"])); continue
        if line.startswith("---"):
            flush_paragraph(para, story); flush_bullets(); story.append(HRFlowable(width="100%", thickness=.6, color=colors.HexColor("#D5E2E8"))); continue
        para.append(line)
    flush_paragraph(para, story); flush_bullets()


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=18*mm, bottomMargin=20*mm, title="Closure Ethics for Autonomous Agents - Implementation Specification v0.1", author="Project Möbia and Marek Zajda", subject="Mathematical specification, recovered protocol lineage and open reference implementation for autonomous-agent Closure Ethics")
    story = []
    cover(story)
    md = SPEC.read_text(encoding="utf-8")
    if "# Open-source appendices" in md:
        md = md.split("# Open-source appendices", 1)[0]
    render_markdown(md, story)

    story.append(PageBreak())
    story.append(Paragraph("Recovered historical foundations", ST["h1"]))
    story.append(Paragraph("The following research documents were recovered from the 2026-09-04 AI Closure backup and are included so the current implementation remains traceable to its earlier communication, safety-gate and international-standard lineage. They remain working proposals, not adopted standards.", ST["body"]))
    for title, path in EXTRA_DOCS:
        story.append(PageBreak())
        story.append(Paragraph(title, ST["h1"]))
        render_markdown(path.read_text(encoding="utf-8"), story)

    story.append(PageBreak())
    story.append(Paragraph("Open-source appendices", ST["h1"]))
    story.append(Paragraph("The following listings are the canonical runnable files shipped with this version. They are included in full so a programmer or autonomous software agent can begin implementation immediately.", ST["body"]))
    for title, path in CODE_FILES:
        story.append(PageBreak())
        story.append(Paragraph(title, ST["h2"]))
        story.append(Paragraph(f"Source: {path.name}", ST["small"]))
        story.append(Spacer(1, 4))
        text = path.read_text(encoding="utf-8").replace("\t", "    ")
        story.append(Preformatted(text, ST["code"], maxLineLength=112))
    doc.build(story, onFirstPage=page_decor, onLaterPages=page_decor)
    print(OUT)


if __name__ == "__main__":
    build()
