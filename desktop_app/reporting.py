from __future__ import annotations

from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _register_arabic_font() -> str:
    candidates = [
        Path("C:/Windows/Fonts/tahoma.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    ]
    for font_path in candidates:
        if not font_path.exists():
            continue
        try:
            pdfmetrics.registerFont(TTFont("ECGArabic", str(font_path)))
            return "ECGArabic"
        except Exception:
            continue
    return "Helvetica"


def _shape_arabic(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return raw
    try:
        import arabic_reshaper  # type: ignore
        from bidi.algorithm import get_display  # type: ignore

        return get_display(arabic_reshaper.reshape(raw))
    except Exception:
        return raw


def export_pdf_report(
    out_path: str | Path,
    session_id: str,
    badge: str,
    metrics: dict,
    findings: list[str],
    summary: str,
    *,
    ai_report_en: str = "",
    ai_report_ar: str = "",
    source_name: str = "",
    ai_provider: str = "",
) -> None:
    out_path = Path(out_path)

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    body_style = styles["BodyText"]
    arabic_font = _register_arabic_font()
    ar_style = ParagraphStyle(
        "arabic_style",
        parent=body_style,
        fontName=arabic_font,
        alignment=TA_RIGHT,
        leading=15,
    )

    small_style = ParagraphStyle(
        "small_style",
        parent=body_style,
        fontSize=9,
        leading=12,
        spaceAfter=6,
    )

    story = []

    patient_name = str(metrics.get("patient_name", "Unknown"))

    story.append(Paragraph("ECG Session Report", title_style))
    story.append(Spacer(1, 0.25 * cm))

    story.append(Paragraph(f"<b>Patient Name:</b> {escape(patient_name)}", body_style))
    story.append(Paragraph(f"<b>Session ID:</b> {escape(session_id)}", body_style))
    story.append(Paragraph(f"<b>Source:</b> {escape(source_name or 'Unknown')}", body_style))
    story.append(Paragraph(f"<b>Final Status:</b> {escape(badge)}", body_style))
    if ai_provider:
        story.append(Paragraph(f"<b>Narrative Provider:</b> {escape(ai_provider)}", body_style))
    story.append(Spacer(1, 0.3 * cm))

    table_data = [
     ["Metric", "Value"],
     ["Duration (sec)", str(metrics.get("duration_sec", ""))],
     ["Captured beats", str(metrics.get("n_beats", ""))],
     ["Normal %", str(metrics.get("pct_normal", ""))],
     ["Abnormal %", str(metrics.get("pct_abnormal", ""))],
     ["Unusable %", str(metrics.get("pct_unusable", ""))],
     ]

    def add_metric(label: str, key: str):
     value = str(metrics.get(key, "") or "").strip()
     if value:
         table_data.append([label, value])

    add_metric("Average BPM", "bpm_avg")
    add_metric("Min BPM", "bpm_min")
    add_metric("Max BPM", "bpm_max")
    add_metric("HRV SDNN (ms)", "hrv_sdnn_ms")
    add_metric("Longest Tachy >110 BPM (sec)", "tachy_longest_sec")
    add_metric("Sustained Tachycardia", "sustained_tachy")

    high_rate_total = str(metrics.get("high_rate_total_sec", "") or "").strip()
    high_rate_longest = str(metrics.get("high_rate_longest_sec", "") or "").strip()
    low_rate_total = str(metrics.get("low_rate_total_sec", "") or "").strip()
    low_rate_longest = str(metrics.get("low_rate_longest_sec", "") or "").strip()
    if high_rate_total:
        table_data.append(["Time >100 BPM (sec)", high_rate_total])
    if high_rate_longest:
        table_data.append(["Longest run >100 BPM (sec)", high_rate_longest])
    if low_rate_total:
        table_data.append(["Time <60 BPM (sec)", low_rate_total])
    if low_rate_longest:
        table_data.append(["Longest run <60 BPM (sec)", low_rate_longest])

    dominant_arrhythmia = str(metrics.get("dominant_arrhythmia", "") or "").strip()
    mitbih_basis = str(metrics.get("mitbih_basis", "") or "").strip()
    mitbih_distribution = str(metrics.get("mitbih_distribution", "") or "").strip()
    if dominant_arrhythmia:
        table_data.append(["Dominant MIT-BIH Type", dominant_arrhythmia])
    if mitbih_basis:
        table_data.append(["MIT-BIH Basis", mitbih_basis])
    if mitbih_distribution:
        table_data.append(["MIT-BIH Distribution (% of abnormal beats)", mitbih_distribution])

    table = Table(table_data, colWidths=[6.5 * cm, 8.5 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#eef2f7")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.35 * cm))

    story.append(Paragraph("Session Report Summary", heading_style))
    if ai_report_en.strip():
        story.append(Paragraph(escape(ai_report_en.strip()), body_style))
    elif summary.strip():
        story.append(Paragraph(escape(summary.strip()), body_style))
    else:
        story.append(Paragraph("No narrative available.", body_style))
    story.append(Spacer(1, 0.35 * cm))

    story.append(Paragraph("Key Findings", heading_style))
    if findings:
        for item in findings:
            story.append(Paragraph(f"- {escape(item)}", body_style))
    else:
        story.append(Paragraph("No additional findings.", body_style))

    story.append(Spacer(1, 0.4 * cm))
    story.append(
        Paragraph(
            "This report is generated for project demonstration and software evaluation purposes only. "
            "It is not a final medical diagnosis.",
            small_style,
        )
    )

    doc.build(story)
