"""
Main pipeline for diagram-aware image parsing and PDF generation.
"""
import argparse
import os
import tempfile

from PIL import Image

from diagram_detector import detect_diagrams, load_model
from pdf_writer import build_pdf
from vlm_text import extract_text, finalize_document


def validate_paths(input_folder, output_pdf, diagram_model):
    """Initial Validation"""
    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    if not os.path.exists(diagram_model):
        raise FileNotFoundError(f"Diagram model not found: {diagram_model}")
    if os.path.splitext(output_pdf)[1].lower() != ".pdf":
        raise ValueError("Output must be a .pdf file")


def list_images(input_folder):
    """List image files, sorted by filename."""
    names =[n for n in os.listdir(input_folder) if os.path.splitext(n)[1].lower() in {".jpg", ".jpeg", ".png"}]
    names.sort()
    if not names:
        raise RuntimeError("No .jpg/.jpeg/.png images found in input folder.")
    return [os.path.join(input_folder, n) for n in names]


def save_diagram_crops(image, detections, tmp_dir, image_stem):
    """Save cropped out diagrams to a temp dir"""
    paths = []
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        crop = image.crop((x1, y1, x2, y2))
        if crop.width <= 0 or crop.height <= 0:
            continue
        path = os.path.join(tmp_dir, f"{image_stem}_diagram_{idx}.png")
        crop.save(path)
        paths.append(path)
    return paths


def process_all_images(args, model, tmp_dir):
    """Process each image: detect diagrams, extract text, and prepare for final cleanup."""
    pages = []
    diagram_map = {}

    for image_path in list_images(args.input_folder):
        image_name = os.path.basename(image_path)
        image_stem = os.path.splitext(image_name)[0]
        image = Image.open(image_path).convert("RGB")

        detections = detect_diagrams(model, image, conf=args.diagram_conf)
        diagram_map[image_name] = save_diagram_crops(image, detections, tmp_dir, image_stem)

        pages.append(
            {
                "image": image_name,
                "raw_text": extract_text(image, base_url=args.base_url, model=args.vlm_model),
            }
        )

    return pages, diagram_map


def sanitize_sections(sections, pages):
    """After final cleanup, ensure we only keep sections for images we processed."""
    allowed = {p["image"] for p in pages}
    cleaned = []
    for section in sections:
        image = str(section.get("image", "")).strip()
        if image not in allowed:
            continue
        text = str(section.get("text", "")).strip()
        cleaned.append({"image": image, "text": text})

    if cleaned:
        return cleaned
    return [{"image": p["image"], "text": p["raw_text"]} for p in pages]


def run_pipeline(args):
    """Main pipeline orchestration."""
    validate_paths(args.input_folder, args.output_pdf, args.diagram_model)

    model = load_model(args.diagram_model) #Preloading

    with tempfile.TemporaryDirectory(prefix="note_pipeline_") as tmp_dir:
        pages, diagram_map = process_all_images(args, model, tmp_dir)

        try:
            # second VLM to cleanup text
            sections = finalize_document(pages, base_url=args.base_url, model=args.vlm_model)
        except RuntimeError:
            # fallback
            sections = []

        sections = sanitize_sections(sections, pages)
        build_pdf(args.output_pdf, sections, diagram_map)


def parse_args():
    """Args parsing"""
    parser = argparse.ArgumentParser(description="Process a folder of note images into one PDF")
    parser.add_argument("--input-folder", required=True, help="Folder containing .jpg/.jpeg/.png files")
    parser.add_argument("--output-pdf", required=True, help="Output PDF path")
    parser.add_argument("--diagram-model", required=True, help="Path to YOLO diagram model (.pt)")
    parser.add_argument("--diagram-conf", type=float, default=0.25, help="Diagram detection confidence")
    parser.add_argument("--base-url", default="http://localhost:1234/v1", help="LM Studio base URL")
    parser.add_argument("--vlm-model", required=True, help="LM Studio model name")
    return parser.parse_args()


def main():
    """Program entry point"""
    args = parse_args()
    run_pipeline(args)
    print(f"Saved PDF: {args.output_pdf}")


if __name__ == "__main__":
    main()
