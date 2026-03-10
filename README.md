# Diagram-Aware Notes to PDF Pipeline

A local **Document AI** pipeline that converts handwritten note images into a clean, structured PDF. It combines a **YOLO-based diagram detection model** with a **vision-capable VLM** for OCR transcription and text cleanup — preserving both written content and visual diagrams in a single consolidated output. Written mostly in PEP8 Standard.

---

## How It Works

The pipeline runs in three stages:

**Stage 1 — Per-image extraction**
Each image is processed independently. YOLO detects and crops diagram regions, and the full image is sent to a VLM for first-pass text transcription.

**Stage 2 — Global cleanup**
All first-pass transcriptions are bundled and sent back to the VLM with a strict JSON schema. The model cleans OCR artifacts, normalizes formatting, and returns coherent section ordering.

**Stage 3 — PDF composition**
Cleaned text and cropped diagrams are assembled into a single final PDF, section by section.

### Pipeline Steps

1. Load all note images from the input folder (`.jpg`, `.jpeg`, `.png`)
2. Run YOLO diagram detection on each image
3. Merge overlapping detection boxes
4. Crop diagram regions to temporary files
5. Send each full image to the VLM for first-pass transcription
6. Send all transcriptions back to the VLM for global cleanup and reordering
7. Write cleaned text + diagram crops to the final PDF

### Intermediate Data

| Variable | Contents |
|---|---|
| `pages` | List of first-pass payloads: `{ image, raw_text }` |
| `diagram_map` | Map of image name → cropped diagram paths |

---

## Requirements

### Diagram Detection Model

Any compatible **Ultralytics YOLO** `.pt` weights file. Passed via `--diagram-model`.

Options:
- Your own trained model
- A public model from Hugging Face (fine-tune later if needed)

### VLM / LLM Model

The vision model **must**:
1. Support image/vision input (multimodal)
2. **Not** be run in thinking/reasoning mode
3. Be accessible via an LM Studio OpenAI-compatible endpoint

Tested with: **Qwen 3 VL 8B**
Default API base: `http://localhost:1234/v1`

---

## Installation

**1. Create and activate a virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Start LM Studio** and load a vision model (tested: Qwen 3 VL 8B)

**4. Run the pipeline**
```bash
python main.py --input-folder "notes" --output-pdf "output/notes_final.pdf" --diagram-model "diag_parser.pt" --diagram-conf 0.10 --vlm-model "qwen/qwen3-vl-8b"
```

---

## CLI Reference

| Flag | Required | Default | Description |
|---|---|---|---|
| `--input-folder` | ✅ | — | Folder containing input images |
| `--output-pdf` | ✅ | — | Output PDF path |
| `--diagram-model` | ✅ | — | Path to YOLO `.pt` weights |
| `--diagram-conf` | ❌ | `0.25` | Diagram detection confidence threshold |
| `--base-url` | ❌ | `http://localhost:1234/v1` | LM Studio API base URL |
| `--vlm-model` | ❌ | — | Vision model name loaded in LM Studio |

---

## Project Structure

| File | Role |
|---|---|
| `main.py` | Orchestrates the full pipeline |
| `diagram_detector.py` | Model loading, detection, and overlap merging |
| `vlm_text.py` | First-pass transcription and global cleanup |
| `pdf_writer.py` | Assembles text and diagrams into the final PDF |

---

## Troubleshooting

| Error | Fix |
|---|---|
| `Input folder not found` | Check the `--input-folder` path |
| `Diagram model not found` | Check the `--diagram-model` file path |
| LM Studio connection error | Ensure the local server is running at `--base-url` |
| Weak or missing text output | Confirm the model supports vision input and is not in reasoning mode |

## Example
![alt text](./Diagram-Aware%20Image%20parsing/Demo%20Files/Demo.png)
![alt text](./Diagram-Aware%20Image%20parsing/Demo%20Files/image.png)
