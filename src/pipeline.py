import base64
import logging

import cv2
import numpy as np
from PIL import Image

from src.preprocessing  import preprocess
from src.detection      import detect_words, crop_word
from src.ocr            import get_trocr, trocr_predict
from src.postprocessing import postprocess, speak_nepali

logger = logging.getLogger(__name__)


# ============================================================
# VISUALISATION
# ============================================================

def _render_detection_viz(img_bgr: np.ndarray,
                          line_groups: list,
                          binary: np.ndarray = None) -> str:
    if binary is not None:
        vis = cv2.cvtColor(cv2.bitwise_not(binary), cv2.COLOR_GRAY2BGR)
    else:
        vis = img_bgr.copy()

    COLORS = [
        (60, 200, 100), (60, 160, 220), (220, 80, 60),
        (220, 160, 40), (180, 60, 220), (40, 200, 200),
    ]
    for li, line in enumerate(line_groups):
        col = COLORS[li % len(COLORS)]
        for wi, box in enumerate(line):
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            cv2.rectangle(vis, (x, y), (x+w, y+h), col, 2)
            cv2.putText(vis, f"L{li+1}W{wi+1}", (x, max(y-4, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf.tobytes()).decode()


# ============================================================
# FULL PIPELINE
# ============================================================

def run_pipeline(pil_image: Image.Image, speak: bool = False) -> dict:
    logger.info("=== Pipeline start ===")

    img_bgr, binary = preprocess(pil_image)
    boxes, line_groups = detect_words(img_bgr, binary=binary)

    viz_b64    = _render_detection_viz(img_bgr, line_groups, binary=binary)
    line_count = len(line_groups)
    word_count = len(boxes)

    # Encode binary for "Preprocessed" tab (lossless PNG, inverted for display)
    binary_display = cv2.bitwise_not(binary)
    _, bin_buf  = cv2.imencode(".png", binary_display)
    binary_b64  = base64.b64encode(bin_buf.tobytes()).decode()

    if not boxes:
        return {
            "raw_text":   "", "final_text": "", "corrections": [],
            "word_count": 0,  "regions":    0,  "audio_url":   None,
            "lines":      [], "method":     "none",
            "viz_b64":    viz_b64, "binary_b64": binary_b64, "line_count": 0,
        }

    proc, _ = get_trocr()
    rec_lines = []
    for line in line_groups:
        words = []
        for box in line:
            w = trocr_predict(crop_word(img_bgr, box)) if proc else ""
            if w:
                words.append(w)
        if words:
            rec_lines.append(words)

    raw   = "\n".join(" ".join(wl) for wl in rec_lines)
    final, corr = postprocess(rec_lines)

    audio = None
    if speak and final:
        p = speak_nepali(final)
        if p:
            audio = "/audio"

    method = "cc+vpp+trocr" if proc else "cc+vpp+boxes"
    logger.info(f"=== Done: {word_count} words / {line_count} lines ===")

    return {
        "raw_text":   raw,
        "final_text": final,
        "corrections": corr,
        "word_count": len(final.split()),
        "regions":    word_count,
        "lines":      [" ".join(wl) for wl in rec_lines],
        "audio_url":  audio,
        "method":     method,
        "viz_b64":    viz_b64,
        "binary_b64": binary_b64,
        "line_count": line_count,
    }