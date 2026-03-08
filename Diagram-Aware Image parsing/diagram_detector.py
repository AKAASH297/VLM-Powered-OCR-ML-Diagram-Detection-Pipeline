"""
Diagram detection package"""
from ultralytics import YOLO  # type: ignore


def coords_to_bounds(box, width, height):
    """Convert raw box coordinates to valid integer bounds within image dimensions."""
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _touching(a, b, gap=4):
    """Check if two boxes are touching or within a small gap."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 <= bx2 + gap and ax2 + gap >= bx1 and ay1 <= by2 + gap and ay2 + gap >= by1


def _to_numpy(x):
    """Convert tensor to numpy array if needed."""
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return x


def _merge_boxes(detections):
    """Merge overlapping boxes into single bounding boxes."""
    merged = []
    for det in detections:
        hit = False
        for cur in merged:
            if _touching(cur["bbox"], det["bbox"]):
                x1, y1, x2, y2 = cur["bbox"]
                a1, b1, a2, b2 = det["bbox"]
                cur["bbox"] = [min(x1, a1), min(y1, b1), max(x2, a2), max(y2, b2)]
                cur["score"] = max(cur["score"], det["score"])
                hit = True
                break
        if not hit:
            merged.append(dict(det))
    merged.sort(key=lambda d: d["score"], reverse=True)
    return merged


def _filter_boxes(detections, width, height):
    """Remove boxes that are too large (full-page) or too tiny (noise)."""
    image_area = float(width * height)
    out = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        area = float((x2 - x1) * (y2 - y1))
        ratio = area / image_area if image_area else 0.0
        if ratio > 0.70:
            continue
        if ratio < 0.0008:
            continue
        out.append(det)
    return out


def _expand_boxes(detections, width, height, pad_ratio=0.08, min_pad=16):
    """Expand each detection to capture nearby labels like axis names.

    pad_ratio expands based on box size, min_pad guarantees a baseline margin.
    """
    expanded = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        bw = x2 - x1
        bh = y2 - y1
        pad_x = max(min_pad, int(bw * pad_ratio))
        pad_y = max(min_pad, int(bh * pad_ratio))

        bigger = [x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y]
        clipped = coords_to_bounds(bigger, width, height)
        if clipped:
            expanded.append({"bbox": clipped, "score": det["score"]})
    return expanded


def load_model(model_path):
    """Model loading"""
    return YOLO(model_path)


def detect_diagrams(model, image, conf=0.25):
    """Detect diagrams with confidence backoff and simple box filtering."""
    width, height = image.size
    thresholds = [conf, 0.10, 0.05, 0.01]
    tried = set()

    for threshold in thresholds:
        if threshold in tried:
            continue
        tried.add(threshold)

        result = model.predict(source=image, conf=threshold, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue

        detections = []
        boxes = _to_numpy(result.boxes.xyxy).tolist()
        scores = _to_numpy(result.boxes.conf).tolist()
        for box, score in zip(boxes, scores):
            bbox = coords_to_bounds(box, width, height)
            if bbox:
                detections.append({"bbox": bbox, "score": float(score)})

        detections = _merge_boxes(detections)
        detections = _filter_boxes(detections, width, height)
        detections = _expand_boxes(detections, width, height)
        if detections:
            return detections

    return []
