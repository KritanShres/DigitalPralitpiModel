"""
src/detection.py — Line and word detection for Devanagari handwriting

Line detection : Connected Component Y-centroid clustering  (v2)
Word detection : VPP (Vertical Projection Profile) per line band  (v3)

Detection v2/v3 fixes:
  Bug A — Fixed LINE_GAP=25 too small for multi-line documents.
    Fix: dynamic LINE_GAP = max(20, int(median_char_h * 0.55)).
    Bands shorter than 40% of median_char_h are discarded (diacritic-only).

  Bug B — Band padding eats into adjacent lines.
    Fix: fixed 3px pad; adjacent bands clamped to their midpoint.

  Bug C — Box height equals full band height → overlaps adjacent lines.
    Fix: tight per-word ink extent from actual ink rows in column span.

  Bug D — Broken shirorekha splits one word into two.
    Fix: merge gap 10px; shirorekha suppression zone top 1/4.

  Bug E — Conjunct consonants (द्व, त्य, न्य …) split a word.
    Fix: Pass 2 conjunct-aware merge with dynamic ceiling and min_gap guard.
"""

import logging
import numpy as np
import cv2
from PIL import Image

from src.preprocessing import (
    _to_grayscale,
    _retinex_normalise, _apply_clahe,
    _smart_binarize, _correct_skew,
)

logger = logging.getLogger(__name__)

_easyocr_reader = None


# ============================================================
# EASYOCR (lazy loader, emergency fallback only)
# ============================================================

def get_easyocr():
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            _easyocr_reader = easyocr.Reader(
                ["ne", "en"], gpu=False, recognizer=False, verbose=False)
            logger.info("EasyOCR loaded.")
        except Exception as e:
            logger.error(f"EasyOCR: {e}")
    return _easyocr_reader


# ============================================================
# LINE DETECTION — CC Y-centroid clustering
# ============================================================

def _find_line_bands(binary: np.ndarray) -> list[tuple[int, int]]:
    """
    Detect text line bands by clustering CC blobs on their Y centroid.

    Steps:
      1. Collect character-sized blobs (area 30–0.5% of image,
         height 5px–30% of image height).
      2. Estimate median character height from accepted blobs.
      3. Set dynamic LINE_GAP = max(20, int(median_char_h * 0.55)).
      4. Sort blobs by Y centroid; cluster with LINE_GAP.
      5. Discard bands whose height < 40% of median_char_h
         (diacritic-only bands).
      6. Clamp adjacent bands to their midpoint to prevent overlap.
      7. Apply a conservative fixed pad of 3px.

    Returns sorted list of (y_start, y_end).
    """
    img_h, img_w = binary.shape
    n, _, stats_cc, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8)

    max_blob = img_h * img_w * 0.005
    blobs = []
    for i in range(1, n):
        area = int(stats_cc[i, cv2.CC_STAT_AREA])
        bh   = int(stats_cc[i, cv2.CC_STAT_HEIGHT])
        top  = int(stats_cc[i, cv2.CC_STAT_TOP])
        left = int(stats_cc[i, cv2.CC_STAT_LEFT])
        bw   = int(stats_cc[i, cv2.CC_STAT_WIDTH])
        cy   = float(centroids[i][1])
        if 30 < area < max_blob and bh > 5 and bh < img_h * 0.3:
            blobs.append({"top": top, "bot": top + bh,
                          "left": left, "right": left + bw,
                          "bh": bh, "cy": cy})

    if not blobs:
        logger.warning("CC: no character blobs found")
        return []

    # Median character height → dynamic LINE_GAP
    median_char_h = float(np.median([b["bh"] for b in blobs]))
    LINE_GAP      = max(20, int(median_char_h * 0.55))
    MIN_BAND_H    = max(8,  int(median_char_h * 0.40))
    logger.info(f"CC: median_char_h={median_char_h:.1f}  "
                f"LINE_GAP={LINE_GAP}  MIN_BAND_H={MIN_BAND_H}")

    blobs.sort(key=lambda b: b["cy"])
    clusters, cur = [], [blobs[0]]
    for i in range(1, len(blobs)):
        if blobs[i]["cy"] - blobs[i-1]["cy"] > LINE_GAP:
            clusters.append(cur); cur = [blobs[i]]
        else:
            cur.append(blobs[i])
    if cur:
        clusters.append(cur)

    # Raw bands — tight ink extents of each cluster
    raw_bands = []
    for cluster in clusters:
        y1 = min(b["top"] for b in cluster)
        y2 = max(b["bot"] for b in cluster)
        if y2 < img_h * 0.05:    continue   # top-border noise
        if y2 - y1 < MIN_BAND_H: continue   # diacritic-only band
        raw_bands.append([y1, y2])
    raw_bands.sort(key=lambda b: b[0])

    # Prevent overlap: clamp each band's bottom to midpoint before next band
    for i in range(len(raw_bands) - 1):
        gap_mid = (raw_bands[i][1] + raw_bands[i+1][0]) // 2
        if raw_bands[i][1] > raw_bands[i+1][0]:
            raw_bands[i][1]   = gap_mid
            raw_bands[i+1][0] = gap_mid + 1

    # Apply fixed 3px pad
    PAD = 3
    bands = [(max(0, y1 - PAD), min(img_h, y2 + PAD))
             for y1, y2 in raw_bands]

    logger.info(f"CC line detection: {len(bands)} bands from "
                f"{len(blobs)} blobs / {len(clusters)} clusters")
    return bands


# ============================================================
# WORD DETECTION — VPP per line band
# ============================================================

def _find_words_in_band(binary: np.ndarray,
                        y1: int, y2: int,
                        img_w: int) -> list[dict]:
    """
    Find word bounding boxes within a line band using Vertical Projection Profile.

    Pipeline:
      1. Extract full strip (for tight height & conjunct detection).
      2. Shirorekha-suppressed copy: blank top 1/4 of strip.
      3. VPP on suppressed strip → column gap map.
      4. PASS 1 — merge gaps <= 10px (minor broken shirorekha).
      5. PASS 2 — conjunct-aware merge (gaps 11–14px with dynamic ceiling):
           B) gap interior completely empty in full strip
           C) sub-shirorekha ink in single column left of gap
           D) sub-shirorekha ink in single column right of gap
           E) gap narrower than 70% of line's minimum gap (conjunct guard)
      6. Filter spans: width 15px–75% of image, area >= 200px².
      7. Tight per-word ink height from the FULL strip + 3px pad.
    """
    img_h, _ = binary.shape
    band_h   = y2 - y1
    if band_h < 4:
        return []

    # Full strip (for tight ink-height and conjunct detection)
    full_strip    = binary[y1:y2, :]
    suppress_rows = max(1, band_h // 4)

    # Shirorekha-suppressed copy
    sup_strip = full_strip.copy()
    sup_strip[:suppress_rows, :] = 0

    # VPP on suppressed strip
    vpp    = np.sum(sup_strip, axis=0) // 255
    is_gap = vpp == 0

    # Find raw column spans
    spans, in_s, cs = [], False, 0
    for c in range(img_w):
        if not is_gap[c] and not in_s:
            in_s, cs = True, c
        elif is_gap[c] and in_s:
            in_s = False
            spans.append([cs, c])
    if in_s:
        spans.append([cs, img_w])

    # ── PASS 1: merge gaps <= 10px (minor broken strokes) ────────────────
    MERGE_GAP_1 = 10
    merged = []
    for s in spans:
        if merged and s[0] - merged[-1][1] <= MERGE_GAP_1:
            merged[-1][1] = s[1]
        else:
            merged.append(s[:])

    # ── PASS 2: conjunct-aware merge ─────────────────────────────────────
    # A gap is a conjunct break (not a word boundary) when ALL of:
    #   A) gap_w is in [11, dynamic_ceil] px
    #   B) gap interior completely empty in full strip
    #   C) sub-shirorekha ink in single column immediately left of gap
    #   D) sub-shirorekha ink in single column immediately right of gap
    #   E) gap_w < 70% of line's minimum gap (conjunct gaps are narrowest)
    #
    # dynamic_ceil = min(14, median_gap * 0.45), clamped >= 11.
    # This prevents over-merging on cursive text where all inter-word
    # gaps are uniformly narrow (condition E rejects them).

    body_strip = full_strip[suppress_rows:, :]   # below shirorekha zone

    all_gaps = [merged[k+1][0] - merged[k][1] for k in range(len(merged)-1)]
    if len(all_gaps) >= 2:
        median_gap   = float(np.median(all_gaps))
        min_gap      = float(min(all_gaps))
        dynamic_ceil = min(14, int(median_gap * 0.45))
    else:
        median_gap   = 999.0
        min_gap      = 999.0
        dynamic_ceil = 14
    dynamic_ceil = max(dynamic_ceil, 11)

    i = 0
    while i < len(merged) - 1:
        ga    = merged[i][1]
        gb    = merged[i+1][0]
        gap_w = gb - ga

        if 10 < gap_w <= dynamic_ceil:

            # E — must be clearly narrower than the line's minimum gap
            if gap_w >= min_gap * 0.70:
                i += 1
                continue

            # B — gap interior completely empty in full strip
            gap_ink = int(np.sum(full_strip[:, ga:gb]))
            if gap_ink == 0:

                # C — sub-shirorekha ink immediately left of gap
                left_col  = max(0, ga - 1)
                left_body = bool(np.any(body_strip[:, left_col] > 0))

                # D — sub-shirorekha ink immediately right of gap
                right_col  = min(img_w - 1, gb)
                right_body = bool(np.any(body_strip[:, right_col] > 0))

                if left_body and right_body:
                    merged[i][1] = merged[i+1][1]
                    merged.pop(i+1)
                    continue   # re-examine same index (chain merges)

        i += 1

    # ── Build boxes with tight ink height ────────────────────────────────
    PAD_H = 3
    boxes = []
    for x1, x2 in merged:
        bw = x2 - x1
        if bw < 15:            continue
        if bw > img_w * 0.75:  continue
        if x1 < 3:             continue

        col_slice = full_strip[:, x1:x2]
        row_ink   = np.any(col_slice > 0, axis=1)
        ink_rows  = np.where(row_ink)[0]

        if len(ink_rows) == 0:
            continue

        ink_top = int(ink_rows[0])
        ink_bot = int(ink_rows[-1])
        tight_h = ink_bot - ink_top + 1

        if bw * tight_h < 200:
            continue

        box_y = max(0,     y1 + ink_top - PAD_H)
        box_b = min(img_h, y1 + ink_bot + PAD_H + 1)
        box_h = box_b - box_y

        boxes.append({
            "x": int(x1),
            "y": int(box_y),
            "w": int(bw),
            "h": int(box_h),
        })

    return boxes


# ============================================================
# EASYOCR FALLBACK
# ============================================================

def _easyocr_fallback(img_bgr: np.ndarray,
                      binary: np.ndarray,
                      min_area: int = 300) -> tuple[list, list]:
    reader = get_easyocr()
    if reader is None:
        return [], []
    img_h, img_w = img_bgr.shape[:2]
    try:
        clean   = cv2.cvtColor(cv2.bitwise_not(binary), cv2.COLOR_GRAY2BGR)
        results = reader.detect(clean)
        raw     = results[0][0] if results and results[0] else []
    except Exception as e:
        logger.warning(f"EasyOCR: {e}")
        return [], []
    boxes = []
    for b in raw:
        x1 = max(0, int(b[0])); x2 = min(img_w, int(b[1]))
        y1 = max(0, int(b[2])); y2 = min(img_h, int(b[3]))
        w, h = x2 - x1, y2 - y1
        if y2 > img_h * 0.85 or y1 < img_h * 0.05 or w * h < min_area:
            continue
        boxes.append({"x": x1, "y": y1, "w": w, "h": h})
    if not boxes:
        return [], []
    boxes.sort(key=lambda b: b["y"])
    lines, cur, cy = [], [], None
    for b in boxes:
        my = b["y"] + b["h"] // 2
        if cy is None or abs(my - cy) <= 30:
            cur.append(b); cy = my
        else:
            lines.append(sorted(cur, key=lambda b: b["x"]))
            cur = [b]; cy = my
    if cur:
        lines.append(sorted(cur, key=lambda b: b["x"]))
    return [b for ln in lines for b in ln], lines


# ============================================================
# MAIN DETECTION
# ============================================================

def detect_words(img_bgr: np.ndarray,
                 binary: np.ndarray = None,
                 min_area: int = 300) -> tuple[list, list]:
    """
    Full detection pipeline:
      1. CC Y-centroid clustering → line bands
      2. VPP per band             → word boxes with tight heights
      3. EasyOCR emergency fallback if nothing found

    Returns (flat_boxes, line_groups).
    """
    if binary is None:
        # Use the same unconditional pipeline as preprocess()
        # so detection is consistent regardless of which code path calls it.
        gray   = _to_grayscale(img_bgr)
        gray   = _retinex_normalise(gray)
        gray   = _apply_clahe(gray)
        binary = _smart_binarize(gray)
        binary, _, _, _, _ = _correct_skew(binary)

    img_h, img_w = binary.shape
    line_bands   = _find_line_bands(binary)

    all_boxes, line_groups = [], []
    for (y1, y2) in line_bands:
        word_boxes = _find_words_in_band(binary, y1, y2, img_w)
        if word_boxes:
            word_boxes = sorted(word_boxes, key=lambda b: b["x"])
            line_groups.append(word_boxes)
            all_boxes.extend(word_boxes)

    if all_boxes:
        logger.info(f"detect_words: {len(all_boxes)} words / "
                    f"{len(line_groups)} lines  [CC+VPP v3]")
        return all_boxes, line_groups

    logger.warning("CC+VPP found nothing — EasyOCR fallback")
    return _easyocr_fallback(img_bgr, binary, min_area)


# ============================================================
# WORD CROP (for TrOCR input)
# ============================================================

def crop_word(img_bgr: np.ndarray, box: dict, padding: int = 8) -> Image.Image:
    """
    Crop a word region from the colour image with padding and enhance for TrOCR:
      - Resize to 64px height (keeps aspect ratio)
      - CLAHE contrast normalisation in LAB colour space
      - Return as RGB PIL image
    """
    h, w = img_bgr.shape[:2]
    x1   = max(0, box["x"] - padding)
    y1   = max(0, box["y"] - padding)
    x2   = min(w, box["x"] + box["w"] + padding)
    y2   = min(h, box["y"] + box["h"] + padding)
    crop = img_bgr[y1:y2, x1:x2]

    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return Image.new("RGB", (128, 64), color=(255, 255, 255))

    tgt_h = 64
    scale = tgt_h / crop.shape[0]
    tgt_w = max(32, int(crop.shape[1] * scale))
    crop  = cv2.resize(crop, (tgt_w, tgt_h), interpolation=cv2.INTER_CUBIC)

    lab      = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b_ = cv2.split(lab)
    l        = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(l)
    crop     = cv2.cvtColor(cv2.merge([l, a, b_]), cv2.COLOR_LAB2BGR)

    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))