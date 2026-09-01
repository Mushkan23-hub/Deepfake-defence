"""
pdf_report.py — PDF forensic report generator using ReportLab
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)
from reportlab.lib.enums import TA_CENTER
from datetime import datetime
import io

NAVY  = colors.HexColor("#1a1a2e")
RED   = colors.HexColor("#c62828")
GREEN = colors.HexColor("#1b5e20")
GREY  = colors.HexColor("#f8fafc")
LGREY = colors.HexColor("#e2e8f0")
WHITE = colors.white

TS = lambda: TableStyle([
    ("BACKGROUND",    (0,0),(-1,0),  NAVY),
    ("TEXTCOLOR",     (0,0),(-1,0),  WHITE),
    ("FONTNAME",      (0,0),(-1,0),  "Helvetica-Bold"),
    ("FONTSIZE",      (0,0),(-1,-1), 9),
    ("ROWBACKGROUNDS",(0,1),(-1,-1), [GREY, WHITE]),
    ("GRID",          (0,0),(-1,-1), 0.5, LGREY),
    ("PADDING",       (0,0),(-1,-1), 6),
])


def generate_report(scan: dict, username: str) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm)

    title_s = ParagraphStyle("T", fontSize=22, fontName="Helvetica-Bold",
        textColor=NAVY, alignment=TA_CENTER, spaceAfter=4)
    sub_s   = ParagraphStyle("S", fontSize=10, textColor=colors.grey,
        alignment=TA_CENTER, spaceAfter=20)
    h2_s    = ParagraphStyle("H2", fontSize=13, fontName="Helvetica-Bold",
        textColor=NAVY, spaceBefore=16, spaceAfter=8)
    flag_s  = ParagraphStyle("F", fontSize=9, leading=13, leftIndent=10)
    foot_s  = ParagraphStyle("Ft", fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
    sh_s    = ParagraphStyle("SH", fontSize=10, fontName="Helvetica-Bold",
        textColor=NAVY, spaceBefore=8, spaceAfter=4)

    story = []

    story.append(Paragraph("DEEPFAKE DEFENCE", title_s))
    story.append(Paragraph("Forensic Analysis Report", sub_s))
    story.append(HRFlowable(width="100%", thickness=2, color=NAVY))
    story.append(Spacer(1, 0.4*cm))

    is_fake = scan.get("is_fake", False)
    verdict_text = "DEEPFAKE DETECTED" if is_fake else "AUTHENTIC"

    summary = [
        ["Field", "Value"],
        ["Verdict",     verdict_text],
        ["Confidence",  f"{scan.get('confidence','—')}%"],
        ["File",        scan.get("filename","Unknown")],
        ["Type",        scan.get("file_type","Unknown").upper()],
        ["Analysed by", username],
        ["Report Date", datetime.utcnow().strftime("%d %B %Y, %H:%M UTC")],
    ]
    t = Table(summary, colWidths=[5*cm, 12*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,0),  NAVY),
        ("TEXTCOLOR",    (0,0),(-1,0),  WHITE),
        ("FONTNAME",     (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0,0),(-1,-1), 9),
        ("BACKGROUND",   (0,1),(-1,1),  RED if is_fake else GREEN),
        ("TEXTCOLOR",    (0,1),(-1,1),  WHITE),
        ("FONTNAME",     (0,1),(-1,1),  "Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0,2),(-1,-1),[GREY, WHITE]),
        ("GRID",         (0,0),(-1,-1), 0.5, LGREY),
        ("PADDING",      (0,0),(-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5*cm))

    # AI Scores
    ai_scores = scan.get("ai_scores", {})
    if ai_scores:
        story.append(Paragraph("AI Detection Scores", h2_s))
        story.append(HRFlowable(width="100%", thickness=0.5, color=LGREY))
        rows = [["Signal", "Score", "Status"]]
        for k, v in ai_scores.items():
            pct = round(float(v)*100, 1)
            rows.append([k.replace("_"," ").title(), f"{pct}%",
                         "Suspicious" if pct > 50 else "Normal"])
        st = Table(rows, colWidths=[6*cm, 4*cm, 7*cm])
        st.setStyle(TS())
        story.append(st)
        story.append(Spacer(1, 0.4*cm))

    # Explanation
    story.append(Paragraph("Detection Explanation", h2_s))
    story.append(HRFlowable(width="100%", thickness=0.5, color=LGREY))
    for line in scan.get("explanation", []):
        story.append(Paragraph(str(line), flag_s))
        story.append(Spacer(1, 0.1*cm))
    story.append(Spacer(1, 0.3*cm))

    # Forensics sections
    forensics = scan.get("forensics", {})

    exif = forensics.get("exif", {})
    if exif and not exif.get("error"):
        story.append(Paragraph("Metadata Forensics (ExifTool)", h2_s))
        story.append(HRFlowable(width="100%", thickness=0.5, color=LGREY))
        for f in exif.get("flags", []):
            story.append(Paragraph(str(f), flag_s))
        meta = exif.get("metadata", {})
        if meta:
            rows = [["Property","Value"]] + [[k,str(v)] for k,v in meta.items()]
            mt = Table(rows, colWidths=[6*cm,11*cm])
            mt.setStyle(TS())
            story.append(mt)
        story.append(Spacer(1, 0.4*cm))

    ela = forensics.get("ela", {})
    if ela and not ela.get("error"):
        story.append(Paragraph("Error Level Analysis (ELA)", h2_s))
        story.append(HRFlowable(width="100%", thickness=0.5, color=LGREY))
        for f in ela.get("flags", []):
            story.append(Paragraph(str(f), flag_s))
        rows = [["Metric","Value"],
                ["Mean Error",  str(ela.get("mean_error","—"))],
                ["Max Error",   str(ela.get("max_error","—"))],
                ["Std Dev",     str(ela.get("std_error","—"))],
                ["Suspicious",  "Yes" if ela.get("suspicious") else "No"]]
        et = Table(rows, colWidths=[6*cm,11*cm])
        et.setStyle(TS())
        story.append(et)
        story.append(Spacer(1, 0.4*cm))

    ffp = forensics.get("ffprobe", {})
    if ffp and not ffp.get("error"):
        story.append(Paragraph("Video Stream Forensics (ffprobe)", h2_s))
        story.append(HRFlowable(width="100%", thickness=0.5, color=LGREY))
        for f in ffp.get("flags", []):
            story.append(Paragraph(str(f), flag_s))
        for sec, label in [("format","Container"),("video","Video Stream"),("audio","Audio Stream")]:
            data = ffp.get(sec, {})
            if not data: continue
            story.append(Paragraph(label, sh_s))
            rows = [["Property","Value"]] + [[k,str(v)] for k,v in data.items()]
            st = Table(rows, colWidths=[6*cm,11*cm])
            st.setStyle(TS())
            story.append(st)
        story.append(Spacer(1, 0.4*cm))

    audio_f = forensics.get("audio", {})
    if audio_f:
        story.append(Paragraph("Audio Forensics", h2_s))
        story.append(HRFlowable(width="100%", thickness=0.5, color=LGREY))
        rows = [["Metric","Value"],
                ["Duration",    f"{audio_f.get('duration_seconds','—')}s"],
                ["Sample Rate", f"{audio_f.get('sample_rate','—')} Hz"],
                ["Clone Score", f"{audio_f.get('clone_score','—')}%"],
                ["Edit Score",  f"{audio_f.get('edit_score','—')}%"],
                ["Edit Points", ", ".join([f"{t}s" for t in audio_f.get("edit_points_sec",[])]) or "None"]]
        at = Table(rows, colWidths=[6*cm,11*cm])
        at.setStyle(TS())
        story.append(at)

    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "Generated by Deepfake Defence · Mushkan Bhagat · VIT Bhopal University",
        foot_s))

    doc.build(story)
    return buf.getvalue()
