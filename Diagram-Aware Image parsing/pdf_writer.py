"""PDF generation"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer


def build_pdf(output_pdf, final_sections, diagram_map):
    """PDF builder"""
    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
    )

    # Use built-in ReportLab styles and tweak only the essentials.
    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    body.fontSize = 11
    body.leading = 15
    title = styles["Heading3"]

    story = []
    for idx, section in enumerate(final_sections):
        # Each section corresponds to one input image/page.
        image_name = str(section.get("image", ""))
        text = str(section.get("text", "")).strip()

        # Section header for traceability back to source image.
        story.append(Paragraph(image_name or f"Page {idx + 1}", title))
        story.append(Spacer(1, 0.2 * cm))

        # Main cleaned text body.
        if text:
            story.append(Paragraph(text.replace("\n", "<br/>"), body))
            story.append(Spacer(1, 0.35 * cm))

        # Append all detected diagram crops for this image.
        image_diagrams = diagram_map.get(image_name, [])
        for path in image_diagrams:
            pic = RLImage(path)
            # Scale down wide diagrams to fit printable area.
            max_w = doc.width
            if pic.drawWidth > max_w:
                scale = max_w / float(pic.drawWidth)
                pic.drawWidth = max_w
                pic.drawHeight = pic.drawHeight * scale
            story.append(pic)
            story.append(Spacer(1, 0.25 * cm))

        # Keep one source image per PDF page for clean separation.
        if idx < len(final_sections) - 1:
            story.append(PageBreak())

    # Fallback so PDF is still valid even if extraction returned nothing.
    if not story:
        story.append(Paragraph("No content extracted.", body))

    doc.build(story)
