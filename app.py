"""
app.py -- Digital Pratilipi: Nepali Handwritten OCR
Kantipur Engineering College -- CT 755 Major Project  v12

Architecture:
  PREPROCESSING   : Adaptive Sharpening → Grayscale → Gamma Lift
                    → CLAHE → Smart Binarization → Skew Correction
  LINE DETECTION  : Connected Component Y-centroid clustering  (v2)
  WORD DETECTION  : VPP (Vertical Projection Profile) per line band  (v2)
  SPELL CORRECT   : Large vocabulary from nepali-bhasa/nepali-spell
                    with BK-tree for fast nearest-neighbour lookup.

v12 — Variable-illumination robust preprocessing
v12.1 — Multi-line robust bounding box detection
══════════════════════════════════════════════════════════════════

  Bug 1 — Wrong Otsu polarity on dark-background images  [_smart_binarize]
  ─────────────────────────────────────────────────────────────────────────
  THRESH_BINARY_INV hardcodes: "background is light, ink is dark."
  On any dark-background image (dim room, dark paper, shadow across the
  page, night-time phone photo) this is wrong: the majority of pixels are
  dark (background), so INV makes the entire background white and ink
  disappears.  The resulting binary is ~100% white; CC finds no blobs; the
  display-inversion step then produces a near-black image.

  Fix (_smart_binarize):
    1. Run plain THRESH_BINARY + THRESH_OTSU (no INV yet).
    2. If Otsu T < 15 or T > 240 → bimodal split failed → fall back to
       adaptiveThreshold (GAUSSIAN_C, block-local, always works).
    3. Check white pixel fraction:
         > 0.50 → majority white = background is white → invert (INK=WHITE).
         ≤ 0.50 → majority dark  = background is dark  → already INK=WHITE.
         But also catch extremes: > 0.90 or < 0.10 → always invert.
    4. Border sanity: sample 10px border (always background).  If border
       is mostly white in the result → background ended up white → flip.
    5. 2×2 morphological opening strips noise amplified by CLAHE.

  Bug 2 — Gamma correction threshold too narrow  [_adaptive_gamma]
  ──────────────────────────────────────────────────────────────────
  Gamma lift previously fired only below mean=80, missing the most common
  failure: dim phone photos with mean 80–140.  These have a compressed
  histogram where ink and paper peaks sit close together.  Gamma < 1
  lifts mid-tones non-linearly, spreading the histogram before CLAHE and
  Otsu run on it.

  Fix: extend trigger to mean < 150 with graduated per-tier strengths:
    mean < 50  → γ = 0.40   mean < 80  → γ = 0.55
    mean < 110 → γ = 0.68   mean < 130 → γ = 0.78
    mean < 150 → γ = 0.88   mean ≥ 150 → skip

  Bug 3 — OR-logic over-sharpens bright images with faint ink  [_adaptive_sharpen]
  ──────────────────────────────────────────────────────────────────────────────────
  `mean < 80 OR std < 35 → α=1.0` fired at full strength for bright images
  with faint ink (high mean, low std — light pencil on white paper).
  Full sharpening halos the faint strokes; Otsu classifies halos as ink.

  Fix: derive alpha independently from mean AND from std, take min():
    α_mean: mean < 80 → 1.0 … mean ≥ 210 → 0.05
    α_std:  std  < 20 → 0.4 … std  ≥ 75  → 0.15
    α_final = min(α_mean, α_std)
  Faint-on-bright (high mean, low std): min(0.05, 0.7) = 0.05 ✓
  Dark/faint     (low  mean, low std):  min(1.0,  0.4) = 0.40 ✓

  CLAHE improvement:
    clipLimit now scales with darkness severity (4.0 for mean<60 down to
    2.0 for flat-but-bright) instead of fixed 2.5.

  Detection v2 — Multi-line robust bounding boxes
  ─────────────────────────────────────────────────
  Four bugs fixed in _find_line_bands and _find_words_in_band:

  Bug A — Fixed LINE_GAP=25 too small for multi-line documents.
    Diacritics (matras, anusvara, chandrabindu) have Y-centroids close
    to characters on adjacent lines, creating spurious micro-bands.
    Fix: dynamic LINE_GAP = max(20, int(median_char_h * 0.55)).
    Bands shorter than 40% of median_char_h are discarded as
    diacritic-only bands.

  Bug B — Band padding eats into adjacent lines.
    pad = max(4, (y2-y1)//8) grows with line height, causing padded
    bands to overlap on tightly-spaced documents.
    Fix: fixed 3px pad; adjacent bands clamped to their midpoint.

  Bug C — Box height equals full band height → overlaps adjacent lines.
    Fix: tight per-word ink extent from actual ink rows in the column
    span, with a fixed 3px vertical pad.

  Bug D — Broken shirorekha splits one word into two.
    Fix: merge gap raised from 5px to 10px; shirorekha suppression
    zone increased from top 1/5 to top 1/4 of the strip.

Why CC for lines (not HPP):
  CC blob Y-centroids have a clear gap between text lines even when
  descenders physically touch ascenders on adjacent lines.

Why VPP for words (not CC):
  Devanagari shirorekha connects all characters in a word horizontally.
  VPP after shirorekha suppression finds clean inter-word gaps.

Vocabulary / spell-correction:
  Full nepali-bhasa/nepali-spell vocabulary (~75k forms) + corpus.
  BK-tree nearest-neighbour search, Levenshtein on Unicode code-points.
"""

import os, base64, logging, re, unicodedata
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TROCR_MODEL  = os.getenv("TROCR_MODEL", "aayushpuri01/TrOCR-Devanagari")

_DEFAULT_VOCAB_DIR    = PROJECT_ROOT / "data"
VOCAB_DIR             = Path(os.getenv("NEPALI_VOCAB_DIR", str(_DEFAULT_VOCAB_DIR)))
VOCAB_DICTIONARY_FILE = VOCAB_DIR / "vocabulary-dictionary"
VOCAB_CORPUS_FILE     = VOCAB_DIR / "vocabulary-corpus"


# ============================================================
# PREPROCESSING  (pipeline v12 — variable-illumination robust)
# ============================================================

def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR/RGB image to grayscale. Returns unchanged if already 1-ch."""
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        return image.squeeze()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _analyze_image(gray: np.ndarray) -> dict:
    """
    Measure global brightness, contrast, and spatial illumination uniformity.

    Uniformity is measured by dividing the image into a 4×4 grid of 16
    blocks, computing each block's mean, then taking the std of those
    16 means.  A uniformly lit image has similar block means → low std.
    An image with a lighting gradient → high std.
    """
    mean = float(np.mean(gray))
    std  = float(np.std(gray))

    h, w    = gray.shape
    bh, bw  = max(1, h // 4), max(1, w // 4)
    block_means = [
        float(np.mean(gray[r*bh:(r+1)*bh, c*bw:(c+1)*bw]))
        for r in range(4) for c in range(4)
    ]
    nonuniformity = float(np.std(block_means))
    bmin, bmax    = min(block_means), max(block_means)
    minmax_ratio  = bmin / (bmax + 1e-6)
    is_nonuniform = (nonuniformity > 15) or (minmax_ratio < 0.65)

    return {
        "mean":          mean,
        "std":           std,
        "is_dark":       mean < 160,
        "is_very_dark":  mean < 80,
        "is_flat":       std  < 50,
        "nonuniformity": nonuniformity,
        "minmax_ratio":  minmax_ratio,
        "is_nonuniform": is_nonuniform,
    }


def _normalize_illumination(gray: np.ndarray, stats: dict) -> np.ndarray:
    """
    Remove slowly-varying background illumination via large-kernel Gaussian.
      normalised = clip(128 + gray − background, 0, 255)
    Skipped for uniform images (is_nonuniform=False).
    """
    if not stats["is_nonuniform"]:
        return gray

    h, w = gray.shape
    k    = max(81, min(h, w) // 5)
    k    = k if k % 2 == 1 else k + 1

    bg   = cv2.GaussianBlur(gray, (k, k), sigmaX=0)
    norm = np.clip(
        128.0 + gray.astype(np.float32) - bg.astype(np.float32),
        0, 255
    ).astype(np.uint8)

    logger.info(
        f"illum_normalize: ksize={k}  "
        f"mean {stats['mean']:.1f}→{float(np.mean(norm)):.1f}  "
        f"std {stats['std']:.1f}→{float(np.std(norm)):.1f}  "
        f"nonuniformity={stats['nonuniformity']:.1f}"
    )
    return norm


def _adaptive_sharpen(image: np.ndarray, stats: dict) -> np.ndarray:
    """
    Unsharp-mask sharpening, α = min(α_from_mean, α_from_std).

    α from mean:  <80→1.0, <130→0.7, <170→0.4, <210→0.2, else→0.05
    α from std:   <20→0.4, <45→0.7, <75→0.5, else→0.15
    Kernel: [0,−α,0 / −α,1+4α,−α / 0,−α,0]
    """
    mean    = stats["mean"]
    std     = stats["std"]
    alpha_m = (1.0  if mean < 80  else 0.7  if mean < 130 else
               0.4  if mean < 170 else 0.2  if mean < 210 else 0.05)
    alpha_s = (0.4  if std  < 20  else 0.7  if std  < 45  else
               0.5  if std  < 75  else 0.15)
    alpha   = min(alpha_m, alpha_s)
    logger.info(
        f"adaptive_sharpen: mean={mean:.1f} std={std:.1f} "
        f"α_m={alpha_m} α_s={alpha_s} → α={alpha:.2f}"
    )
    k = alpha
    kernel = np.array([[0,-k,0],[-k,1+4*k,-k],[0,-k,0]], dtype=np.float32)
    return cv2.filter2D(image, ddepth=-1, kernel=kernel)


def _adaptive_gamma(gray: np.ndarray, stats: dict) -> np.ndarray:
    """
    LUT-based gamma lift for dark images (mean < 150).
    Skipped for non-uniform images (already normalised to ~128).

    Gamma tiers:
      mean <50→γ=0.40, <80→0.55, <110→0.68, <130→0.78, <150→0.88
    """
    if stats["is_nonuniform"]:
        return gray
    mean = stats["mean"]
    if   mean < 50:  gamma = 0.40
    elif mean < 80:  gamma = 0.55
    elif mean < 110: gamma = 0.68
    elif mean < 130: gamma = 0.78
    elif mean < 150: gamma = 0.88
    else:
        return gray
    lut = np.array(
        [min(255, int((i/255.0)**(1.0/gamma)*255)) for i in range(256)],
        dtype=np.uint8,
    )
    logger.info(f"adaptive_gamma: mean={mean:.1f} → γ={gamma:.2f}")
    return cv2.LUT(gray, lut)


def _apply_clahe(gray: np.ndarray, stats: dict) -> np.ndarray:
    """
    CLAHE with tile count scaled to image height.

    tile_n = max(8, min(16, h // 80))
    Applied when: is_dark OR is_flat OR is_nonuniform.
    clipLimit by mean: <60→4.0, <100→3.5, <130→3.0, <160→2.5, else→2.0
    """
    if not (stats["is_dark"] or stats["is_flat"] or stats["is_nonuniform"]):
        return gray

    mean   = stats["mean"]
    clip   = (4.0 if mean < 60  else 3.5 if mean < 100 else
              3.0 if mean < 130 else 2.5 if mean < 160 else 2.0)
    h      = gray.shape[0]
    tile_n = max(8, min(16, h // 80))
    clahe  = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile_n, tile_n))
    result = clahe.apply(gray)
    logger.info(
        f"CLAHE: clip={clip} tiles=({tile_n},{tile_n})  "
        f"mean {stats['mean']:.1f}→{float(np.mean(result)):.1f}  "
        f"std {stats['std']:.1f}→{float(np.std(result)):.1f}"
    )
    return result


def _fix_polarity(binary: np.ndarray) -> np.ndarray:
    """
    Ensure INK=WHITE (255), background=BLACK (0).

    Check 1+2 — white fraction:
      >0.90 or <0.10 → clearly wrong polarity → invert.
      >0.50           → background is white    → invert.
    Check 3 — border strip (almost always background):
      border >75% white → background mapped to white → invert.
    """
    white_frac = float(np.count_nonzero(binary)) / binary.size

    if white_frac > 0.90 or white_frac < 0.10:
        logger.info(f"fix_polarity step1: white_frac={white_frac:.2f} extreme → invert")
        binary     = cv2.bitwise_not(binary)
        white_frac = 1.0 - white_frac

    if white_frac > 0.50:
        logger.info(f"fix_polarity step2: white_frac={white_frac:.2f} majority-white=bg → invert")
        binary     = cv2.bitwise_not(binary)
        white_frac = 1.0 - white_frac

    h, w   = binary.shape
    pad    = max(10, min(h, w) // 20)
    border = np.concatenate([
        binary[:pad, :].ravel(),  binary[-pad:, :].ravel(),
        binary[:, :pad].ravel(),  binary[:, -pad:].ravel(),
    ])
    b_white = float(np.count_nonzero(border)) / border.size
    if b_white > 0.75:
        logger.info(f"fix_polarity step3: border_white={b_white:.2f} > 0.75 → invert")
        binary = cv2.bitwise_not(binary)
    return binary


def _morphological_cleanup(binary: np.ndarray) -> np.ndarray:
    """
    3×3 morphological opening to remove noise blobs after CLAHE/normalisation.
    Devanagari matras and nuktas are typically 4+ pixels wide at 960px
    resolution, so they survive the 3×3 erosion safely.
    """
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)


def _smart_binarize(gray: np.ndarray, stats: dict) -> np.ndarray:
    """
    Binarize with method routing based on image characteristics.

    Routing:
      is_nonuniform  → adaptiveThreshold always
      otherwise      → Otsu with adaptiveThreshold fallback if T extreme

    Convention on exit: INK=WHITE (255), background=BLACK (0).
    """
    assert gray.ndim == 2
    need_cleanup = stats["is_dark"] or stats["is_flat"] or stats["is_nonuniform"]

    h      = gray.shape[0]
    block  = max(11, h // 20)
    block  = block if block % 2 == 1 else block + 1
    block  = min(block, 61)
    C_ADAPTIVE = 15

    if stats["is_nonuniform"]:
        gray_smooth = cv2.GaussianBlur(gray, (3, 3), sigmaX=0.8)
        logger.info(
            f"smart_binarize: non-uniform → adaptiveThreshold "
            f"(blockSize={block}, C={C_ADAPTIVE})"
        )
        binary = cv2.adaptiveThreshold(
            gray_smooth, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=block, C=C_ADAPTIVE,
        )
        binary = _fix_polarity(binary)
        if need_cleanup:
            binary = _morphological_cleanup(binary)
        return binary

    T, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    logger.info(f"smart_binarize: Otsu T={T:.0f}")

    if T < 15 or T > 240:
        gray_smooth = cv2.GaussianBlur(gray, (3, 3), sigmaX=0.8)
        logger.warning(
            f"Otsu T={T:.0f} extreme → adaptiveThreshold "
            f"(blockSize={block}, C={C_ADAPTIVE})"
        )
        binary = cv2.adaptiveThreshold(
            gray_smooth, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=block, C=C_ADAPTIVE,
        )
        binary = _fix_polarity(binary)
        if need_cleanup:
            binary = _morphological_cleanup(binary)
        return binary

    binary = _fix_polarity(binary)
    if need_cleanup:
        binary = _morphological_cleanup(binary)
    return binary


def _correct_skew(binary: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Detect and correct skew via HoughLinesP.
    Returns (corrected_binary, skew_angle_degrees).
    """
    assert binary.ndim == 2
    inverted = cv2.bitwise_not(binary)
    edges    = cv2.Canny(inverted, 50, 150, apertureSize=3)
    lines    = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=80,
        minLineLength=50, maxLineGap=10,
    )

    skew_angle = 0.0
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = x2 - x1, y2 - y1
            if dx == 0: continue
            a = np.degrees(np.arctan2(dy, dx))
            if -45 <= a <= 45: angles.append(a)
        if angles:
            skew_angle = float(np.median(angles))

    h, w   = binary.shape
    center = (w / 2.0, h / 2.0)
    M      = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
    corrected = cv2.warpAffine(
        binary, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )
    return corrected, skew_angle


def preprocess(pil_image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline (v12 — variable-illumination robust):

      1. Upscale to >= 960px wide
      2. Grayscale + measure stats
      3. Adaptive sharpening       (α = min(α_mean, α_std))
      4. Re-grayscale
      5. Illumination normalisation (large Gaussian BG subtraction)
      6. Adaptive gamma lift        (dark uniform images only)
      7. CLAHE                      (size-proportional tiles)
      8. Smart binarization         (adaptiveThreshold or Otsu + polarity fix)
      9. Skew correction

    Returns (colour_bgr, binary_ink_white).
    """
    img_bgr = cv2.cvtColor(
        np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR
    )

    # Step 1 — Upscale
    h, w = img_bgr.shape[:2]
    if w < 960:
        scale   = 960 / w
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_CUBIC)

    # Step 2 — Grayscale + stats
    gray_raw = _to_grayscale(img_bgr)
    stats    = _analyze_image(gray_raw)
    logger.info(
        f"preprocess: {img_bgr.shape[1]}×{img_bgr.shape[0]}px  "
        f"mean={stats['mean']:.1f}  std={stats['std']:.1f}  "
        f"nonuniformity={stats['nonuniformity']:.1f}  "
        f"is_nonuniform={stats['is_nonuniform']}"
    )

    # Step 3 — Adaptive sharpening
    sharpened = _adaptive_sharpen(img_bgr, stats)

    # Step 4 — Re-grayscale
    gray = _to_grayscale(sharpened)

    # Step 5 — Illumination normalisation
    gray = _normalize_illumination(gray, stats)

    # Step 6 — Gamma lift (dark uniform images only)
    gray = _adaptive_gamma(gray, stats)

    # Step 7 — CLAHE
    gray = _apply_clahe(gray, stats)

    # Step 8 — Smart binarization
    binary = _smart_binarize(gray, stats)

    # Step 9 — Skew correction
    binary_corrected, skew_angle = _correct_skew(binary)

    if abs(skew_angle) > 0.5:
        h2, w2 = sharpened.shape[:2]
        M = cv2.getRotationMatrix2D((w2 / 2.0, h2 / 2.0), skew_angle, 1.0)
        img_bgr = cv2.warpAffine(
            sharpened, M, (w2, h2),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
        )
    else:
        img_bgr = sharpened

    logger.info(
        f"preprocess done: skew={skew_angle:.2f}°  "
        f"dark={stats['is_dark']}  flat={stats['is_flat']}  "
        f"nonuniform={stats['is_nonuniform']}"
    )
    return img_bgr, binary_corrected


# ============================================================
# LINE DETECTION — CC Y-centroid clustering  (v2 — multi-line robust)
# ============================================================
#
# Bug A — Fixed LINE_GAP=25 too small for multi-line documents
# ─────────────────────────────────────────────────────────────
# On a multi-line document, diacritics (matras, anusvara, chandrabindu)
# sitting above or below a line have Y-centroids very close to the
# characters of the adjacent line.  With LINE_GAP=25 these diacritics
# often land in a different cluster from their own line, creating a
# spurious narrow band that gets a bounding box of its own.
#
# Fix: estimate the median character height from all accepted blobs,
# then set LINE_GAP = max(20, int(median_char_height * 0.55)).
# After clustering, bands shorter than 40% of median_char_height are
# discarded as diacritic-only bands.
#
# Bug B — Band padding eats into adjacent lines
# ──────────────────────────────────────────────
# pad = max(4, (y2-y1)//8) grows with line height.  On a tightly-spaced
# document the padded band of line N overlaps line N+1.
#
# Fix: fixed 3px pad; adjacent bands clamped to their midpoint if they
# would overlap.

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
# WORD DETECTION — VPP per line band  (v3 — conjunct-aware merge)
# ============================================================
#
# Bug C — Box height equals full band height → overlaps adjacent lines.
#   Fix: tight per-word ink height from actual ink rows in the column
#   span.  Fixed 3px vertical pad.
#
# Bug D — Broken shirorekha splits one word into two.
#   Fix: merge gap raised to 10px; suppression zone top 1/4.
#
# Bug E — Conjunct consonants (द्व, त्य, न्य, ञ्च …) split a word.
#   A halant-conjunct glyph drops below the shirorekha.  In the
#   suppressed VPP the columns over the hanging body are zero (the
#   headstroke was removed; the body starts below the suppression
#   boundary), creating a false gap of up to ~18px that Pass 1
#   (10px) cannot bridge.
#   Fix: Pass 2 examines remaining gaps of 11–20px.  A gap is a
#   conjunct gap when the sub-shirorekha body zone has ink immediately
#   adjacent to BOTH edges of the gap.  A real inter-word space is
#   truly empty on at least one side.

def _find_words_in_band(binary: np.ndarray,
                        y1: int, y2: int,
                        img_w: int) -> list[dict]:
    """
    Find word bounding boxes within a line band using VPP.

    Pipeline:
      1. Extract full strip (used for tight-height calculation).
      2. Shirorekha-suppressed copy: blank top 1/4 of strip.
      3. VPP on suppressed strip → column gap map.
      4. PASS 1 — merge gaps <= 10px (handles minor broken shirorekha).
      5. PASS 2 — conjunct-aware merge (gaps up to 20px that have
                  sub-shirorekha ink continuity on BOTH sides are bridges,
                  not real inter-word spaces).
      6. Filter spans: width 15px–75% of image, area >= 200px².
      7. Tight per-word ink height from the FULL strip + 3px pad.

    Pass 2 rationale — conjunct consonants (e.g. द्व, त्य, न्य, ञ्च):
      A conjunct glyph drops below the shirorekha line.  In the
      shirorekha-suppressed VPP the columns over the conjunct are zero
      (the suppressed zone removed the headstroke ink and the body
      hangs lower), producing a false gap of up to ~18px.
      A REAL inter-word gap has zero ink in the ENTIRE column range of
      the gap in the FULL strip.  A conjunct gap has ink below the
      shirorekha zone on at least one side of the gap.

      For each remaining gap [ga, gb] between adjacent spans:
        left_body  = full_strip[suppress_rows:, ga-2 : ga+1]   (3px left edge)
        right_body = full_strip[suppress_rows:, gb-1 : gb+2]   (3px right edge)
        If BOTH sides have sub-shirorekha ink → conjunct gap → merge.
        Gap width ceiling: 20px (safe; real inter-word >= 25px at 960px).
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

    # ── PASS 2: conjunct-aware merge (gaps 11–20px) ───────────────────────
    #
    # A gap is a conjunct break (not a word boundary) when ALL of:
    #
    #   A) gap_w is in [11, 20] px
    #
    #   B) The gap interior [ga..gb] is COMPLETELY EMPTY in the full strip
    #      (zero ink in every column of the gap).  Both conjunct breaks and
    #      real inter-word spaces satisfy this, but it rejects gaps where
    #      a matra/nukta pixel bleeds into the space.
    #
    #   C) sub-shirorekha ink exists within 1px of the LEFT edge of the gap
    #      (column ga-1): the conjunct body ends immediately at the gap.
    #
    #   D) sub-shirorekha ink exists within 1px of the RIGHT edge of the gap
    #      (column gb): the next character body starts immediately at the gap.
    #
    #   E) gap_w < dynamic_ceiling, where dynamic_ceiling is derived from
    #      the MEDIAN of all current inter-span gaps in the line.
    #      Conjunct gaps are narrow outliers; real inter-word gaps cluster
    #      around the median spacing.  If there is only one gap (two spans),
    #      the ceiling defaults to 16px.
    #      ceiling = min(16, median_gap * 0.55)  — never exceeds 16px.
    #
    # Condition E is the decisive safety net: if the gap being examined is
    # close to the line's typical inter-word spacing, it cannot be a
    # conjunct break (which is always much narrower than word spacing).
    # This prevents the "बसेर पढ्छु" class of errors where a normal
    # inter-word gap has incidental body ink on both sides.

    body_strip = full_strip[suppress_rows:, :]   # below shirorekha zone

    # Compute dynamic ceiling from median of all inter-span gaps
    all_gaps = [merged[k+1][0] - merged[k][1] for k in range(len(merged)-1)]
    if len(all_gaps) >= 2:
        median_gap     = float(np.median(all_gaps))
        dynamic_ceil   = min(16, int(median_gap * 0.55))
    else:
        dynamic_ceil   = 16   # only one gap in the line, use safe default

    # Ensure ceiling is at least 11 (otherwise Pass 2 is never triggered)
    dynamic_ceil = max(dynamic_ceil, 11)

    i = 0
    while i < len(merged) - 1:
        ga    = merged[i][1]       # right edge of left span
        gb    = merged[i+1][0]     # left edge of right span
        gap_w = gb - ga

        if 10 < gap_w <= dynamic_ceil:

            # B — gap interior completely empty in the full strip
            gap_ink = int(np.sum(full_strip[:, ga:gb]))
            if gap_ink == 0:

                # C — sub-shirorekha ink in the single column left of gap
                left_col  = max(0, ga - 1)
                left_body = bool(np.any(body_strip[:, left_col] > 0))

                # D — sub-shirorekha ink in the single column right of gap
                right_col  = min(img_w - 1, gb)
                right_body = bool(np.any(body_strip[:, right_col] > 0))

                if left_body and right_body:
                    # All conditions met → conjunct gap → merge
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

        # Tight vertical extent from full strip
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
# MAIN DETECTION
# ============================================================

_easyocr_reader = None

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


def detect_words(img_bgr: np.ndarray,
                 binary: np.ndarray = None,
                 min_area: int = 300) -> tuple[list, list]:
    """
    Full detection pipeline:
      1. CC Y-centroid clustering (v2) → line bands
      2. VPP per band (v2)            → word boxes with tight heights
      3. EasyOCR emergency fallback if nothing found

    Returns (flat_boxes, line_groups).
    """
    if binary is None:
        gray   = _to_grayscale(img_bgr)
        stats  = _analyze_image(gray)
        gray   = _adaptive_gamma(gray, stats)
        gray   = _apply_clahe(gray, stats)
        binary = _smart_binarize(gray, stats)
        binary, _ = _correct_skew(binary)

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
                    f"{len(line_groups)} lines  [CC+VPP v2]")
        return all_boxes, line_groups

    logger.warning("CC+VPP found nothing — EasyOCR fallback")
    return _easyocr_fallback(img_bgr, binary, min_area)


def _easyocr_fallback(img_bgr, binary, min_area=300):
    reader = get_easyocr()
    if reader is None: return [], []
    img_h, img_w = img_bgr.shape[:2]
    try:
        clean   = cv2.cvtColor(cv2.bitwise_not(binary), cv2.COLOR_GRAY2BGR)
        results = reader.detect(clean)
        raw     = results[0][0] if results and results[0] else []
    except Exception as e:
        logger.warning(f"EasyOCR: {e}"); return [], []
    boxes = []
    for b in raw:
        x1=max(0,int(b[0])); x2=min(img_w,int(b[1]))
        y1=max(0,int(b[2])); y2=min(img_h,int(b[3]))
        w, h = x2-x1, y2-y1
        if y2>img_h*0.85 or y1<img_h*0.05 or w*h<min_area: continue
        boxes.append({"x":x1,"y":y1,"w":w,"h":h})
    if not boxes: return [], []
    boxes.sort(key=lambda b: b["y"])
    lines, cur, cy = [], [], None
    for b in boxes:
        my = b["y"]+b["h"]//2
        if cy is None or abs(my-cy)<=30: cur.append(b); cy=my
        else: lines.append(sorted(cur,key=lambda b:b["x"])); cur=[b]; cy=my
    if cur: lines.append(sorted(cur,key=lambda b:b["x"]))
    return [b for ln in lines for b in ln], lines


# ============================================================
# WORD CROP
# ============================================================

def crop_word(img_bgr: np.ndarray, box: dict, padding: int = 8) -> Image.Image:
    """
    Crop a word region from the colour image with padding and enhance for TrOCR:
      - Resize to 64px height (keeps aspect ratio)
      - CLAHE contrast normalisation in LAB colour space
      - Return as RGB PIL image
    """
    h, w = img_bgr.shape[:2]
    x1   = max(0, box["x"]-padding)
    y1   = max(0, box["y"]-padding)
    x2   = min(w, box["x"]+box["w"]+padding)
    y2   = min(h, box["y"]+box["h"]+padding)
    crop = img_bgr[y1:y2, x1:x2]

    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return Image.new("RGB", (128, 64), color=(255, 255, 255))

    tgt_h = 64
    scale = tgt_h / crop.shape[0]
    tgt_w = max(32, int(crop.shape[1]*scale))
    crop  = cv2.resize(crop, (tgt_w, tgt_h), interpolation=cv2.INTER_CUBIC)

    lab      = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b_ = cv2.split(lab)
    l        = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4)).apply(l)
    crop     = cv2.cvtColor(cv2.merge([l, a, b_]), cv2.COLOR_LAB2BGR)

    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


# ============================================================
# TROCR
# ============================================================

_processor = None
_model     = None


def get_trocr():
    global _processor, _model
    if _processor is None:
        try:
            from transformers import (
                VisionEncoderDecoderModel, AutoTokenizer,
                ViTImageProcessor, TrOCRProcessor,
            )
            logger.info(f"Loading TrOCR from Hub: {TROCR_MODEL} …")
            feature_extractor = ViTImageProcessor.from_pretrained(TROCR_MODEL)
            tokenizer         = AutoTokenizer.from_pretrained(TROCR_MODEL)
            _processor        = TrOCRProcessor(
                image_processor=feature_extractor, tokenizer=tokenizer)
            _model            = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL)
            _model.eval()
            logger.info("TrOCR loaded successfully.")
        except Exception as e:
            logger.error(f"TrOCR load failed: {e}")
            _processor = _model = None
    return _processor, _model


def trocr_predict(word_img: Image.Image) -> str:
    """Run a single word image through TrOCR and return the decoded string."""
    import torch
    proc, model = get_trocr()
    if not proc: return ""
    try:
        pixel_values = proc(
            images=word_img.convert("RGB"), return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values, max_new_tokens=48, num_beams=5,
                early_stopping=True, repetition_penalty=1.2, length_penalty=1.0,
            )
        return proc.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    except Exception as e:
        logger.warning(f"TrOCR predict error: {e}"); return ""


# ============================================================
# VOCABULARY
# ============================================================

def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _load_vocab_file(path: Path) -> set[str]:
    words: set[str] = set()
    if not path.exists():
        logger.warning(f"Vocabulary file not found: {path}"); return words
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                w = _nfc(raw.strip())
                if w and not w.startswith("#"): words.add(w)
        logger.info(f"Loaded {len(words):,} words from {path.name}")
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
    return words


_VOCAB:   set[str] | None = None
_BK_TREE: object          = None


def _get_vocab() -> set[str]:
    global _VOCAB
    if _VOCAB is None:
        logger.info("Loading Nepali vocabulary …")
        _VOCAB = _load_vocab_file(VOCAB_DICTIONARY_FILE) | \
                 _load_vocab_file(VOCAB_CORPUS_FILE)
        if not _VOCAB:
            logger.warning("No external vocabulary files — using built-in seed.")
            _VOCAB = {_nfc(w) for w in {
                "म","मेरो","हाम्रो","तिमी","तिम्रो","उ","उसको","यो","त्यो",
                "हामी","तपाई","आफ्नो","छ","छन्","हो","हुन्","गर्छ","गयो",
                "आयो","भयो","गर्नु","आउनु","जानु","खानु","पिउनु","पढ्नु",
                "लेख्नु","हेर्नु","बोल्नु","सुन्नु","भन्नु","हुन्छ","गर्छु",
                "जान्छु","आउँछु","पर्छ","नाम","घर","देश","नेपाल","मान्छे",
                "आमा","बाबा","दाजु","भाइ","दिदी","बहिनी","साथी","स्कुल",
                "किताब","कलम","पानी","खाना","दूध","चिया","काम","पैसा",
                "समय","दिन","रात","बिहान","साँझ","सहर","गाउँ","जीवन",
                "संसार","मन","राम्रो","नराम्रो","ठूलो","सानो","धेरै",
                "थोरै","नया","खुसी","दुखी","सुन्दर","रमाइलो",
                "मलाई","हामीलाई","तिमीलाई","उसलाई",
                "मा","को","का","की","लाई","बाट","देखि","सम्म","साथ",
                "पनि","नै","र","तर","वा","भने","भनेर","छु","छौ","छन",
                "थियो","थिए","थिएन","छैन","गरे","गरेको","भएको",
                "।","?","!",",",".",
            }}
        logger.info(f"Total vocabulary size: {len(_VOCAB):,} words")
    return _VOCAB


# ============================================================
# BK-TREE
# ============================================================

class _BKTree:
    """
    Minimal BK-tree over Unicode strings using Levenshtein distance.
    Supports efficient nearest-neighbour queries in sub-linear time.
    """
    __slots__ = ("word", "children")

    def __init__(self, word: str):
        self.word = word
        self.children: dict[int, "_BKTree"] = {}

    @staticmethod
    def _lev(a: str, b: str) -> int:
        if a == b:  return 0
        if not a:   return len(b)
        if not b:   return len(a)
        if len(a) < len(b): a, b = b, a
        prev = list(range(len(b)+1))
        for ca in a:
            curr = [prev[0]+1]
            for j, cb in enumerate(b):
                curr.append(min(prev[j+1]+1, curr[j]+1,
                                prev[j]+(0 if ca==cb else 1)))
            prev = curr
        return prev[-1]

    def insert(self, word: str) -> None:
        node = self
        while True:
            d = self._lev(node.word, word)
            if d == 0: return
            if d not in node.children:
                node.children[d] = _BKTree(word); return
            node = node.children[d]

    def search(self, query: str, max_dist: int) -> list[tuple[int, str]]:
        results, stack = [], [self]
        while stack:
            node = stack.pop()
            d    = self._lev(node.word, query)
            if d <= max_dist: results.append((d, node.word))
            lo, hi = d-max_dist, d+max_dist
            for k, child in node.children.items():
                if lo <= k <= hi: stack.append(child)
        return results


def _get_bktree() -> _BKTree | None:
    global _BK_TREE
    if _BK_TREE is None:
        vocab = _get_vocab()
        if not vocab: return None
        logger.info(f"Building BK-tree for {len(vocab):,} words …")
        words = list(vocab)
        root  = _BKTree(words[0])
        for w in words[1:]: root.insert(w)
        _BK_TREE = root
        logger.info("BK-tree ready.")
    return _BK_TREE


# ============================================================
# SPELL CORRECTION
# ============================================================

_RE_NON_WORD   = re.compile(r'^[०-९0-9।,.!?\s\u0964\u0965]+$')
_RE_DEVANAGARI = re.compile(r'[\u0900-\u097F]')


def spell_correct(word: str, max_dist: int = 1) -> tuple[str, bool]:
    """
    Return (corrected_word, was_changed).

    Strategy:
      1. NFC-normalise.
      2. Skip short, non-Devanagari, or punctuation tokens.
      3. O(1) set-membership → already correct.
      4. BK-tree search (max_dist+1 for words len>=6, capped at 2).
      5. Prefer dictionary words over corpus-only; tie-break alphabetically.
    """
    word = _nfc(word)
    if not word or len(word) < 3:         return word, False
    if _RE_NON_WORD.match(word):          return word, False
    if not _RE_DEVANAGARI.search(word):   return word, False

    vocab = _get_vocab()
    if word in vocab: return word, False

    effective_dist = min(max_dist + (1 if len(word) >= 6 else 0), 2)
    tree = _get_bktree()
    if tree is None: return word, False

    candidates = tree.search(word, effective_dist)
    if not candidates: return word, False

    min_d = min(c[0] for c in candidates)
    best  = sorted(
        [w for d, w in candidates if d == min_d],
        key=lambda w: (0 if w in vocab else 1, w)
    )
    corrected = best[0]
    return (corrected, True) if corrected != word else (word, False)


# ============================================================
# POSTPROCESSING
# ============================================================

def postprocess(lines: list[list[str]]) -> tuple[str, list[dict]]:
    """NFC-normalise, spell-correct, and join lines."""
    corrections: list[dict] = []
    result_lines: list[str] = []
    for line in lines:
        out_words: list[str] = []
        for word in line:
            word = _nfc(word)
            corrected, changed = spell_correct(word)
            if changed:
                corrections.append({"original": word, "corrected": corrected})
            out_words.append(corrected)
        result_lines.append(" ".join(out_words))
    return "\n".join(result_lines), corrections


# ============================================================
# TTS
# ============================================================

def speak_nepali(text: str, path: str = "output_audio.mp3") -> str | None:
    if not text.strip(): return None
    try:
        from gtts import gTTS
        gTTS(text=text, lang="ne", slow=False).save(path)
        return path
    except Exception as e:
        logger.error(f"TTS: {e}"); return None


# ============================================================
# PIPELINE
# ============================================================

def _render_detection_viz(img_bgr: np.ndarray,
                          line_groups: list,
                          binary: np.ndarray = None) -> str:
    """
    Draw coloured bounding boxes on the preprocessed binary image
    (inverted to black-ink-on-white) and return as base64 JPEG.
    """
    if binary is not None:
        vis = cv2.cvtColor(cv2.bitwise_not(binary), cv2.COLOR_GRAY2BGR)
    else:
        vis = img_bgr.copy()

    COLORS = [(60,200,100),(60,160,220),(220,80,60),
              (220,160,40),(180,60,220),(40,200,200)]
    for li, line in enumerate(line_groups):
        col = COLORS[li % len(COLORS)]
        for wi, box in enumerate(line):
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            cv2.rectangle(vis, (x,y), (x+w,y+h), col, 2)
            cv2.putText(vis, f"L{li+1}W{wi+1}", (x, max(y-4, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf.tobytes()).decode()


def run_pipeline(pil_image: Image.Image, speak: bool = False) -> dict:
    logger.info("=== Pipeline start ===")
    img_bgr, binary = preprocess(pil_image)
    boxes, line_groups = detect_words(img_bgr, binary=binary)

    viz_b64    = _render_detection_viz(img_bgr, line_groups, binary=binary)
    line_count = len(line_groups)
    word_count = len(boxes)

    binary_display = cv2.bitwise_not(binary)
    _, bin_buf = cv2.imencode(".png", binary_display)
    binary_b64 = base64.b64encode(bin_buf.tobytes()).decode()

    if not boxes:
        return {
            "raw_text": "", "final_text": "", "corrections": [],
            "word_count": 0, "regions": 0, "audio_url": None,
            "lines": [], "method": "none",
            "viz_b64": viz_b64, "binary_b64": binary_b64, "line_count": 0,
        }

    proc, _ = get_trocr()
    rec_lines = []
    for line in line_groups:
        words = []
        for box in line:
            w = trocr_predict(crop_word(img_bgr, box)) if proc else ""
            if w: words.append(w)
        if words: rec_lines.append(words)

    raw   = "\n".join(" ".join(wl) for wl in rec_lines)
    final, corr = postprocess(rec_lines)

    audio = None
    if speak and final:
        p = speak_nepali(final)
        if p: audio = "/audio"

    method = "cc+vpp+trocr" if proc else "cc+vpp+boxes"
    logger.info(f"=== Done: {word_count} words / {line_count} lines ===")
    return {
        "raw_text": raw, "final_text": final, "corrections": corr,
        "word_count": len(final.split()), "regions": word_count,
        "lines": [" ".join(wl) for wl in rec_lines],
        "audio_url": audio, "method": method,
        "viz_b64": viz_b64, "binary_b64": binary_b64, "line_count": line_count,
    }


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route("/")
def serve_frontend():
    from flask import send_from_directory
    return send_from_directory(".", "index.html")


@app.route("/health")
def health():
    p, _  = get_trocr()
    vocab = _get_vocab()
    return jsonify({
        "status":       "ok",
        "trocr":        p is not None,
        "easyocr":      _easyocr_reader is not None,
        "vocab_size":   len(vocab),
        "bktree_ready": _BK_TREE is not None,
    })


@app.route("/ocr",      methods=["POST"])
@app.route("/ocr_full", methods=["POST"])
def ocr_endpoint():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image"}), 400
        img   = Image.open(request.files["image"].stream).convert("RGB")
        speak = request.form.get("speak", "0") == "1"
        return jsonify(run_pipeline(img, speak=speak))
    except Exception as e:
        logger.exception("Pipeline error")
        return jsonify({"error": str(e)}), 500


@app.route("/tts", methods=["POST"])
def tts_endpoint():
    try:
        text = request.get_json(force=True).get("text", "")
        if not text: return jsonify({"error": "No text"}), 400
        path = speak_nepali(text)
        if not path: return jsonify({"error": "TTS failed"}), 500
        return send_file(path, mimetype="audio/mpeg", download_name="speech.mp3")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/audio")
def get_audio():
    if not os.path.exists("output_audio.mp3"):
        return jsonify({"error": "No audio"}), 404
    return send_file("output_audio.mp3", mimetype="audio/mpeg",
                     download_name="speech.mp3")


@app.route("/detect_viz", methods=["POST"])
def detect_viz():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image"}), 400
        pil = Image.open(request.files["image"].stream).convert("RGB")
        img_bgr, binary = preprocess(pil)
        boxes, line_groups = detect_words(img_bgr, binary=binary)
        image_b64 = _render_detection_viz(img_bgr, line_groups, binary=binary)
        return jsonify({
            "image_b64":  image_b64,
            "word_count": len(boxes),
            "line_count": len(line_groups),
            "boxes": [{"x": int(b["x"]), "y": int(b["y"]),
                       "w": int(b["w"]), "h": int(b["h"])}
                      for b in boxes],
        })
    except Exception as e:
        logger.exception("detect_viz error")
        return jsonify({"error": str(e)}), 500


@app.route("/vocab_check", methods=["POST"])
def vocab_check():
    """Debug endpoint: POST {"word":"…"} → membership + BK-tree neighbours."""
    try:
        data     = request.get_json(force=True)
        word     = _nfc(data.get("word", "").strip())
        if not word: return jsonify({"error": "No word"}), 400
        vocab    = _get_vocab()
        in_vocab = word in vocab
        tree     = _get_bktree()
        neighbours = []
        if tree:
            hits = tree.search(word, max_dist=2)
            hits.sort()
            neighbours = [{"dist": d, "word": w} for d, w in hits[:10]]
        return jsonify({"word": word, "in_vocab": in_vocab,
                        "neighbours": neighbours})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Digital Pratilipi OCR  v12.1  [Multi-line robust detection]")
    print("="*60)
    print(f"  TROCR_MODEL : {TROCR_MODEL}")
    print(f"  VOCAB_DIR   : {VOCAB_DIR}")
    print(f"  URL         : http://localhost:5000")
    print("="*60 + "\n")
    get_trocr()
    get_easyocr()
    _get_vocab()
    _get_bktree()
    app.run(host="0.0.0.0", port=5000, debug=False)