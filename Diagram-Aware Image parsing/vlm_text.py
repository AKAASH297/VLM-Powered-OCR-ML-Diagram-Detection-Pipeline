"""Functions for Vision Language Model (VLM) Operations"""
import base64
import io
import json
import re
import urllib.error
import urllib.request


def _image_to_data_url(image):
    """convert python image to base64 data url for embedding in VLM prompts"""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _extract_text(content):
    """Fixing formatting issues like triple qoutes"""
    blocktexts = re.search(r"```(?:text)?\s*(.*?)\s*```", content, flags=re.DOTALL)
    if blocktexts:
        return blocktexts.group(1).strip()
    return content.strip()


def _extract_json(content):
    """extract JSON blob from VLM response"""
    blocktexts = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, flags=re.DOTALL)
    if blocktexts:
        return blocktexts.group(1)
    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        return content[start : end + 1]
    return None


def _chat_completion(payload, base_url):
    """main function to call LM Studio"""
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LM Studio HTTP error {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not connect to LM Studio: {e}") from e

    return json.loads(raw)["choices"][0]["message"]["content"]


def extract_text(image, base_url, model):
    # First pass: per-image transcription.
    # Goal here is recall, not perfect formatting.
    prompt = (
        "Transcribe this handwritten note into plain text. "
        "Only include main note content (titles, bullets, sentences, equations in body text). "
        "Do NOT transcribe text that belongs inside diagrams/charts/graphs/axes/tables/state nodes/legends. "
        "Ignore isolated labels such as axis labels, node labels, x/y marks, and annotation letters near drawings. "
        "Keep line breaks and bullet points when possible. "
        "Return only plain text, no markdown, no explanation."
    )

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(image)}},
                ],
            }
        ],
    }

    content = _chat_completion(payload, base_url=base_url)
    return _extract_text(content)


def finalize_document(page_payloads, base_url, model):
    # Second pass: global cleanup and ordering across all pages.
    # Returns strict JSON sections [{image, text}] for deterministic downstream use.
    prompt = (
        "You are finalizing OCR text from multiple note images. "
        "Reorder and clean the text into a coherent final document. "
        "Keep facts unchanged and do not invent content. "
        "Remove leftover diagram-only labels/noise (example: lone axis letters, node IDs, marker letters). "
        "Keep only body-note text. "
        "Return only valid JSON with this exact schema: "
        "{\"sections\":[{\"image\":\"filename\",\"text\":\"clean text\"}]}. "
        f"Input JSON: {json.dumps(page_payloads, ensure_ascii=True)}"
    )

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }
    content = _chat_completion(payload, base_url=base_url)
    blob = _extract_json(content)
    if not blob:
        raise RuntimeError("Final VLM stage did not return valid JSON.")

    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Could not parse final VLM JSON: {e}") from e

    sections = parsed.get("sections", [])
    if not isinstance(sections, list):
        raise RuntimeError("Final VLM output missing sections list.")
    return sections
