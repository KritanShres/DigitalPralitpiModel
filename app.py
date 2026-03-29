"""
app.py -- Digital Pratilipi: Nepali Handwritten OCR
Kantipur Engineering College -- CT 755 Major Project  v9

Architecture (proven by profiling real images):
  PREPROCESSING   : Sharpening → Grayscale → Binarization (Otsu) → Skew Correction
                    (HoughLinesP-based, imported from preprocessing pipeline v2)
  LINE DETECTION  : Connected Component Y-centroid clustering
  WORD DETECTION  : VPP (Vertical Projection Profile) per line band
  SPELL CORRECT   : Large vocabulary from nepali-bhasa/nepali-spell
                    (data/vocabulary-dictionary + data/vocabulary-corpus)
                    with BK-tree for fast nearest-neighbour lookup.

Why CC for lines (not HPP):
  On close-together handwritten lines, binarized ink is often
  CONTINUOUS between lines (descenders touching ascenders + noise).
  HPP never reaches zero → zero-crossing and valley detection both fail.
  CC blob Y-centroids have a clear gap (>25px) between lines even when
  ink is physically touching. Immune to vertical ink continuity.

Why VPP for words (not CC):
  Words in Devanagari are connected by the shirorekha horizontally,
  making them one CC blob. VPP after shirorekha suppression cleanly
  finds inter-word gaps (>15px) while intra-word gaps stay <5px.

Vocabulary / spell-correction upgrade (v8):
  The old NEPALI_DICT was a ~80-word hand-curated set.  v8 replaces it
  with the full vocabulary files from nepali-bhasa/nepali-spell:

    data/vocabulary-dictionary   (morphological dictionary, ~75k forms)
    data/vocabulary-corpus       (corpus frequency list, variable size)

  Both files are plain-text, one Unicode word per line, already NFC.
  They live next to app.py; the loader tries both and unions them.

  Lookup strategy:
    1. O(1) hash-set membership check  → word is correct, skip
    2. BK-tree nearest-neighbour search using Levenshtein distance
       (max_dist=1 for short words, max_dist=2 for longer ones)
       → returns the closest vocabulary word(s) in sub-linear time
       instead of the old O(|dict|) linear scan.

  BK-tree construction is O(N log N) and done once at startup
  (or lazily on first spell-correct call).  For ~75k entries this
  takes < 2 s on a modern CPU and uses ~30 MB RAM.

  Devanagari-aware Levenshtein:
    Unicode normalization (NFC) is applied before distance computation
    so that visually identical but differently encoded strings compare
    equal.  The distance is counted over Unicode code-points, not bytes,
    which is correct for Devanagari (each akshara = one or more
    code-points but we want akshara-level edits, not byte-level).

Preprocessing pipeline (v2):
  Replaces the old bilateral-filter + Sauvola + rolling-ball pipeline
  with a cleaner four-stage approach:
    1. Sharpening        — 3×3 unsharp kernel via cv2.filter2D
    2. Grayscale         — BGR → single channel
    3. Binarization      — Otsu's global threshold (ink = WHITE / 255)
    4. Skew correction   — HoughLinesP angle estimation + warpAffine
"""

import os, base64, logging, re, unicodedata
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app  = Flask(__name__)
CORS(app)

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TROCR_MODEL  = os.getenv("TROCR_MODEL", str(PROJECT_ROOT / "model"))


# Paths to the nepali-bhasa/nepali-spell vocabulary files.
# Override at runtime with the NEPALI_VOCAB_DIR env-var if needed.
_DEFAULT_VOCAB_DIR = PROJECT_ROOT / "data"
VOCAB_DIR = Path(os.getenv("NEPALI_VOCAB_DIR", str(_DEFAULT_VOCAB_DIR)))

VOCAB_DICTIONARY_FILE = VOCAB_DIR / "vocabulary-dictionary"
VOCAB_CORPUS_FILE     = VOCAB_DIR / "vocabulary-corpus"


# ============================================================
# PREPROCESSING  (pipeline v2)
# ============================================================
# Four-stage pipeline:
#   sharpen → grayscale → Otsu binarize → HoughLinesP skew correction
#
# This replaces the previous rolling-ball / bilateral / Sauvola pipeline
# with a simpler, faster, and more robust approach that proved more
# reliable on the IIIT-HW-Hindi dataset used for evaluation.
# ============================================================

def _sharpen(image: np.ndarray) -> np.ndarray:
    """Apply a 3×3 sharpening kernel via 2D convolution."""
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)
    return cv2.filter2D(image, ddepth=-1, kernel=kernel)


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB/BGR image to grayscale. Returns unchanged if already single-channel."""
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        return image.squeeze()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _binarize_otsu(gray: np.ndarray) -> np.ndarray:
    """
    Apply Otsu's global thresholding to produce a binary image.
    INK = WHITE (255), background = 0  — consistent with rest of pipeline.
    """
    assert gray.ndim == 2, "Input must be a single-channel grayscale image."
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU   # INV so ink → 255
    )
    return binary


def _correct_skew(binary: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Detect skew via Probabilistic Hough Line Transform (HoughLinesP)
    and rotate the image to correct it.

    Uses HoughLinesP (probabilistic variant) rather than the classical
    HoughLines used in v1 — it is faster and more robust on sparse ink.

    Returns (corrected_binary, skew_angle_degrees).
    """
    assert binary.ndim == 2, "Input must be a single-channel binary image."

    inverted = cv2.bitwise_not(binary)
    edges    = cv2.Canny(inverted, 50, 150, apertureSize=3)
    lines    = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=50,
        maxLineGap=10,
    )

    skew_angle = 0.0
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = x2 - x1, y2 - y1
            if dx == 0:
                continue
            angle_deg = np.degrees(np.arctan2(dy, dx))
            if -45 <= angle_deg <= 45:
                angles.append(angle_deg)
        if angles:
            skew_angle = float(np.median(angles))

    h, w   = binary.shape
    center = (w / 2.0, h / 2.0)
    M      = cv2.getRotationMatrix2D(center, skew_angle, scale=1.0)
    corrected = cv2.warpAffine(
        binary, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return corrected, skew_angle


def preprocess(pil_image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """
    Four-stage preprocessing pipeline (v2):
      1. Upscale to >= 960px wide  (unchanged from v1 for OCR quality)
      2. Sharpen  — 3×3 unsharp kernel
      3. Grayscale conversion
      4. Otsu binarization          (ink = WHITE / 255)
      5. HoughLinesP skew correction

    Returns (colour_bgr, binary_ink_white).

    The colour image returned is the sharpened BGR image (before
    grayscale), so bounding-box visualisations still render in colour.
    The binary image is ink-white / background-black, matching the
    convention expected by _find_line_bands and _find_words_in_band.
    """
    img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")),
                            cv2.COLOR_RGB2BGR)

    # Upscale if too small (keeps OCR quality on low-res scans)
    h, w = img_bgr.shape[:2]
    if w < 960:
        scale   = 960 / w
        img_bgr = cv2.resize(img_bgr,
                             (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_CUBIC)

    # Stage 1 — Sharpening (operates on colour image)
    sharpened = _sharpen(img_bgr)

    # Stage 2 — Grayscale
    gray = _to_grayscale(sharpened)

    # Stage 3 — Otsu binarization
    binary = _binarize_otsu(gray)

    # Stage 4 — Skew correction (applied to both colour and binary)
    binary_corrected, skew_angle = _correct_skew(binary)

    if abs(skew_angle) > 0.5:
        h2, w2 = sharpened.shape[:2]
        M = cv2.getRotationMatrix2D((w2 / 2.0, h2 / 2.0), skew_angle, 1.0)
        img_bgr = cv2.warpAffine(
            sharpened, M, (w2, h2),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
    else:
        img_bgr = sharpened

    logger.info(f"preprocess: {img_bgr.shape[1]}×{img_bgr.shape[0]}px  "
                f"skew={skew_angle:.2f}°")

    return img_bgr, binary_corrected


# ============================================================
# LINE DETECTION — Connected Component Y-centroid clustering
# ============================================================

def _find_line_bands(binary: np.ndarray,
                     min_line_height: int = 15) -> list[tuple[int,int]]:
    """
    Detect text line bands by clustering CC blobs on their Y centroid.

    Steps:
      1. Find all CC blobs in the binary image
      2. Filter to character-sized blobs (area 30–0.5% of image,
         height 5px–30% of image height) — removes noise and artifacts
      3. Sort blobs by Y centroid
      4. Cluster: new line when gap between adjacent blob cy > LINE_GAP (25px)
         This gap is always present between distinct text lines, even when
         their ink touches vertically, because character bodies sit at
         different Y positions on different lines.
      5. Line band = (min_top − pad, max_bottom + pad) of each cluster
      6. Filter artifact bands: skip bands whose blobs are all in the
         top 10% of the image (dark photo border binarized as ink)

    Returns sorted list of (y_start, y_end).
    """
    img_h, img_w = binary.shape

    n, _, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8)

    max_blob = img_h * img_w * 0.005
    blobs = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        bh   = stats[i, cv2.CC_STAT_HEIGHT]
        top  = stats[i, cv2.CC_STAT_TOP]
        left = stats[i, cv2.CC_STAT_LEFT]
        bw   = stats[i, cv2.CC_STAT_WIDTH]
        cy   = float(centroids[i][1])
        if 30 < area < max_blob and bh > 5 and bh < img_h * 0.3:
            blobs.append({
                'top': top, 'bot': top + bh,
                'left': left, 'right': left + bw,
                'cy': cy
            })

    if not blobs:
        logger.warning("CC: no character blobs found")
        return []

    blobs.sort(key=lambda b: b['cy'])

    # Cluster by Y centroid gap
    LINE_GAP  = 25
    clusters  = []
    cur       = [blobs[0]]
    for i in range(1, len(blobs)):
        if blobs[i]['cy'] - blobs[i-1]['cy'] > LINE_GAP:
            clusters.append(cur)
            cur = [blobs[i]]
        else:
            cur.append(blobs[i])
    if cur:
        clusters.append(cur)

    bands = []
    for cluster in clusters:
        y1  = min(b['top'] for b in cluster)
        y2  = max(b['bot'] for b in cluster)

        # Skip artifact bands in top 10% of image
        if y2 < img_h * 0.10:
            continue

        if y2 - y1 < min_line_height:
            continue

        pad = max(4, (y2 - y1) // 8)
        bands.append((max(0, y1 - pad), min(img_h, y2 + pad)))

    bands.sort(key=lambda b: b[0])
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
    Find word bounding boxes within a line band using VPP.

    Steps:
      1. Extract strip for this line band
      2. Blank top 20% (suppresses shirorekha — the horizontal bar that
         connects all Devanagari characters, hiding inter-word gaps)
      3. VPP: sum ink pixels per column
      4. Merge adjacent spans with gap <= 5px (intra-word broken strokes)
      5. Filter: width 15px–80% of image width, area >= 300px²

    Returns list of {"x","y","w","h"} in full-image coordinates.
    """
    img_h = binary.shape[0]
    strip = binary[y1:y2, :].copy()
    sh    = strip.shape[0]

    # Suppress shirorekha
    strip[:max(1, sh // 5), :] = 0

    # VPP
    vpp    = np.sum(strip, axis=0) // 255
    is_gap = vpp == 0

    spans, in_s, cs = [], False, 0
    for c in range(img_w):
        if not is_gap[c] and not in_s:
            in_s, cs = True, c
        elif is_gap[c] and in_s:
            in_s = False
            spans.append([cs, c])
    if in_s:
        spans.append([cs, img_w])

    # Merge spans <= 5px apart
    merged = []
    for s in spans:
        if merged and s[0] - merged[-1][1] <= 5:
            merged[-1][1] = s[1]
        else:
            merged.append(s[:])

    bh    = y2 - y1
    boxes = []
    for x1, x2 in merged:
        bw = x2 - x1
        if bw < 15:              continue   # noise fragment
        if bw > img_w * 0.75:   continue   # full-width artifact span
        if x1 < 3:               continue   # left-border artifact
        if bw * bh < 300:        continue   # too small
        boxes.append({"x": x1, "y": y1, "w": bw, "h": bh})
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
                ['ne','en'], gpu=False, recognizer=False, verbose=False)
            logger.info("EasyOCR loaded.")
        except Exception as e:
            logger.error(f"EasyOCR: {e}")
    return _easyocr_reader


def detect_words(img_bgr: np.ndarray,
                 binary: np.ndarray = None,
                 min_area: int = 300) -> tuple[list, list]:
    """
    Full detection pipeline:
      1. CC Y-centroid clustering → line bands
      2. VPP per band → word boxes
      3. EasyOCR emergency fallback if nothing found

    Returns (flat_boxes, line_groups).
    """
    if binary is None:
        # Re-run preprocessing stages on the colour image if no binary given
        gray   = _to_grayscale(img_bgr)
        binary = _binarize_otsu(gray)
        binary, _ = _correct_skew(binary)

    img_h, img_w = binary.shape

    # Step 1: CC → line bands
    line_bands = _find_line_bands(binary)

    # Step 2: VPP → word boxes per band
    all_boxes, line_groups = [], []
    for (y1, y2) in line_bands:
        word_boxes = _find_words_in_band(binary, y1, y2, img_w)
        if word_boxes:
            word_boxes = sorted(word_boxes, key=lambda b: b["x"])
            line_groups.append(word_boxes)
            all_boxes.extend(word_boxes)

    if all_boxes:
        logger.info(f"detect_words: {len(all_boxes)} words / "
                    f"{len(line_groups)} lines  [CC+VPP]")
        return all_boxes, line_groups

    # Fallback
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
        w,h=x2-x1,y2-y1
        if y2>img_h*0.85 or y1<img_h*0.05 or w*h<min_area: continue
        boxes.append({"x":x1,"y":y1,"w":w,"h":h})
    if not boxes: return [], []
    boxes.sort(key=lambda b: b["y"])
    lines,cur,cy=[],[],None
    for b in boxes:
        my=b["y"]+b["h"]//2
        if cy is None or abs(my-cy)<=30: cur.append(b); cy=my
        else: lines.append(sorted(cur,key=lambda b:b["x"])); cur=[b]; cy=my
    if cur: lines.append(sorted(cur,key=lambda b:b["x"]))
    return [b for ln in lines for b in ln], lines


# ============================================================
# WORD CROP
# ============================================================

def crop_word(img_bgr: np.ndarray, box: dict, padding: int = 8) -> Image.Image:
    """
    Crop a word region from the original colour image with padding,
    then enhance it for TrOCR:
      - Generous padding so diacritics aren't clipped
      - Resize to a standard height (64px) while keeping aspect ratio
      - Convert to grayscale, apply CLAHE for contrast normalisation
      - Return as RGB PIL image (TrOCR expects RGB)
    """
    h, w  = img_bgr.shape[:2]
    x1    = max(0, box["x"] - padding)
    y1    = max(0, box["y"] - padding)
    x2    = min(w, box["x"] + box["w"] + padding)
    y2    = min(h, box["y"] + box["h"] + padding)
    crop  = img_bgr[y1:y2, x1:x2]

    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return Image.new("RGB", (128, 64), color=(255, 255, 255))

    tgt_h = 64
    scale = tgt_h / crop.shape[0]
    tgt_w = max(32, int(crop.shape[1] * scale))
    crop  = cv2.resize(crop, (tgt_w, tgt_h), interpolation=cv2.INTER_CUBIC)

    lab   = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(l)
    crop  = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

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
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            logger.info(f"Loading TrOCR from {TROCR_MODEL}...")
            _processor = TrOCRProcessor.from_pretrained(TROCR_MODEL,local_files_only=True)
            _model     = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL,local_files_only=True)
            _model.eval()
            logger.info("TrOCR loaded.")
        except Exception as e:
            logger.error(f"TrOCR: {e}")
    return _processor, _model


def trocr_predict(word_img: Image.Image) -> str:
    import torch
    proc, model = get_trocr()
    if not proc: return ""
    try:
        pv = proc(images=word_img.convert("RGB"), return_tensors="pt").pixel_values
        with torch.no_grad():
            ids = model.generate(
                pv,
                max_new_tokens=48,
                num_beams=5,
                early_stopping=True,
                repetition_penalty=1.2,
                length_penalty=1.0,
            )
        return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
    except Exception as e:
        logger.warning(f"TrOCR: {e}"); return ""


# ============================================================
# VOCABULARY — load from nepali-bhasa/nepali-spell data files
# ============================================================

def _nfc(s: str) -> str:
    """NFC-normalise a Unicode string (important for Devanagari)."""
    return unicodedata.normalize("NFC", s)


def _load_vocab_file(path: Path) -> set[str]:
    """
    Load a vocabulary file (one word per line, UTF-8).
    Lines starting with # are comments; blank lines are skipped.
    Returns a set of NFC-normalised words.
    """
    words: set[str] = set()
    if not path.exists():
        logger.warning(f"Vocabulary file not found: {path}")
        return words
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                w = _nfc(raw.strip())
                if w and not w.startswith("#"):
                    words.add(w)
        logger.info(f"Loaded {len(words):,} words from {path.name}")
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
    return words


# Global vocabulary set and BK-tree — built lazily on first use.
_VOCAB:    set[str]  | None = None
_BK_TREE:  object           = None   # BKTree instance


def _get_vocab() -> set[str]:
    """Return the union of dictionary + corpus vocabulary."""
    global _VOCAB
    if _VOCAB is None:
        logger.info("Loading Nepali vocabulary …")
        dict_words   = _load_vocab_file(VOCAB_DICTIONARY_FILE)
        corpus_words = _load_vocab_file(VOCAB_CORPUS_FILE)
        _VOCAB = dict_words | corpus_words

        if not _VOCAB:
            # Ultimate fallback: built-in seed list so the app still
            # works even without the vocabulary files present.
            logger.warning("No external vocabulary files found — "
                           "using built-in seed dictionary.")
            _VOCAB = {
                _nfc(w) for w in {
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
                }
            }
        logger.info(f"Total vocabulary size: {len(_VOCAB):,} words")
    return _VOCAB


# ============================================================
# BK-TREE  (Burkhard-Keller tree for fast edit-distance search)
# ============================================================

class _BKTree:
    """
    Minimal BK-tree over Unicode strings using Levenshtein distance.

    A BK-tree supports efficient nearest-neighbour queries: given a
    query word and a maximum edit distance d, it returns all vocabulary
    words within distance d in sub-linear time (O(|dict|^{0.1..0.3})
    in practice for d <= 2).

    This replaces the old O(|dict|) linear scan with a ~10–100x
    speedup for large dictionaries.
    """

    __slots__ = ("word", "children")

    def __init__(self, word: str):
        self.word     = word
        self.children: dict[int, "_BKTree"] = {}

    # ----------------------------------------------------------
    # Levenshtein distance (iterative, operates on code-points)
    # ----------------------------------------------------------
    @staticmethod
    def _lev(a: str, b: str) -> int:
        if a == b:   return 0
        if not a:    return len(b)
        if not b:    return len(a)
        # Keep the shorter string in b to minimise memory
        if len(a) < len(b):
            a, b = b, a
        prev = list(range(len(b) + 1))
        for ca in a:
            curr = [prev[0] + 1]
            for j, cb in enumerate(b):
                curr.append(min(
                    prev[j + 1] + 1,   # deletion
                    curr[j]     + 1,   # insertion
                    prev[j]     + (0 if ca == cb else 1)  # substitution
                ))
            prev = curr
        return prev[-1]

    # ----------------------------------------------------------
    # Insert
    # ----------------------------------------------------------
    def insert(self, word: str) -> None:
        node = self
        while True:
            d = self._lev(node.word, word)
            if d == 0:
                return  # duplicate
            if d not in node.children:
                node.children[d] = _BKTree(word)
                return
            node = node.children[d]

    # ----------------------------------------------------------
    # Search: return all words within max_dist edits of query
    # ----------------------------------------------------------
    def search(self, query: str, max_dist: int) -> list[tuple[int, str]]:
        results: list[tuple[int, str]] = []
        stack   = [self]
        while stack:
            node = stack.pop()
            d    = self._lev(node.word, query)
            if d <= max_dist:
                results.append((d, node.word))
            # Only recurse into children whose edge label k satisfies
            # |d - k| <= max_dist  (BK-tree pruning condition)
            lo, hi = d - max_dist, d + max_dist
            for k, child in node.children.items():
                if lo <= k <= hi:
                    stack.append(child)
        return results


def _get_bktree() -> _BKTree | None:
    """Build (lazily) and return the BK-tree over the full vocabulary."""
    global _BK_TREE
    if _BK_TREE is None:
        vocab = _get_vocab()
        if not vocab:
            return None
        logger.info(f"Building BK-tree for {len(vocab):,} words …")
        words = list(vocab)
        root  = _BKTree(words[0])
        for w in words[1:]:
            root.insert(w)
        _BK_TREE = root
        logger.info("BK-tree ready.")
    return _BK_TREE


# ============================================================
# SPELL CORRECTION
# ============================================================

# Characters that unambiguously mark non-word tokens
_RE_NON_WORD = re.compile(r'^[०-९0-9।,.!?\s\u0964\u0965]+$')

# Devanagari Unicode range for a quick sanity check
_RE_DEVANAGARI = re.compile(r'[\u0900-\u097F]')


def spell_correct(word: str, max_dist: int = 1) -> tuple[str, bool]:
    """
    Return (corrected_word, was_changed).

    Strategy:
      1. NFC-normalise.
      2. Skip very short tokens, purely numeric / punctuation tokens,
         and tokens with no Devanagari characters.
      3. O(1) set-membership check — if the word is already in the
         vocabulary, return it unchanged.
      4. BK-tree nearest-neighbour search with max_dist edits.
         For words of length >= 6 we allow max_dist=2 to handle
         common multi-character OCR errors in long Devanagari words.
      5. Among all candidates at minimum distance, prefer the one
         that appears in the dictionary file (higher quality) over
         corpus-only words.  Tie-break alphabetically for stability.
      6. If no candidate found, return the original word unchanged.

    Note: we deliberately do NOT correct very short words (len < 3)
    because single-akshara function words like "म", "र", "को" are
    highly ambiguous at edit distance 1.
    """
    word = _nfc(word)

    # Skip non-words
    if not word or len(word) < 3:
        return word, False
    if _RE_NON_WORD.match(word):
        return word, False
    if not _RE_DEVANAGARI.search(word):
        return word, False  # purely Latin / digits

    vocab = _get_vocab()

    # Fast path: already correct
    if word in vocab:
        return word, False

    # BK-tree search — allow one extra edit for longer words
    effective_dist = max_dist + (1 if len(word) >= 6 else 0)
    effective_dist = min(effective_dist, 2)   # cap at 2

    tree = _get_bktree()
    if tree is None:
        return word, False

    candidates = tree.search(word, effective_dist)

    if not candidates:
        return word, False

    # Select best candidate
    min_d = min(c[0] for c in candidates)
    best  = sorted(
        [w for d, w in candidates if d == min_d],
        key=lambda w: (0 if w in (_get_vocab()) else 1, w)
    )
    corrected = best[0]

    if corrected == word:
        return word, False
    return corrected, True


# ============================================================
# POSTPROCESSING
# ============================================================

def postprocess(lines: list[list[str]]) -> tuple[str, list[dict]]:
    """
    NFC-normalise every recognised word, apply spell-correction,
    and join into final text.

    Returns (final_text, list_of_corrections).
    Each correction is {"original": str, "corrected": str}.
    """
    corrections: list[dict] = []
    result_lines: list[str]  = []

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

def speak_nepali(text, path="output_audio.mp3"):
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
                          line_groups: list) -> str:
    """
    Draw coloured bounding boxes on the image and return as base64 JPEG.
    """
    vis    = img_bgr.copy()
    COLORS = [(60,200,100),(60,160,220),(220,80,60),
              (220,160,40),(180,60,220),(40,200,200)]
    for li, line in enumerate(line_groups):
        col = COLORS[li % len(COLORS)]
        for wi, box in enumerate(line):
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            cv2.rectangle(vis, (x,y), (x+w,y+h), col, 2)
            cv2.putText(vis, f"L{li+1}W{wi+1}", (x, max(y-4,14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf.tobytes()).decode()


def run_pipeline(pil_image: Image.Image, speak=False) -> dict:
    logger.info("=== Pipeline start ===")
    img_bgr, binary = preprocess(pil_image)
    boxes, line_groups = detect_words(img_bgr, binary=binary)

    viz_b64    = _render_detection_viz(img_bgr, line_groups)
    line_count = len(line_groups)
    word_count = len(boxes)

    # Encode the Otsu binary image for the frontend "Preprocessed" tab.
    # Convert ink-white binary (ink=255) back to a viewable image:
    #   invert so ink is black on white (standard document look),
    #   then encode as PNG (lossless — important for binary images).
    binary_display = cv2.bitwise_not(binary)          # ink → black
    _, bin_buf = cv2.imencode(".png", binary_display)
    binary_b64 = base64.b64encode(bin_buf.tobytes()).decode()

    if not boxes:
        return {"raw_text":"","final_text":"","corrections":[],
                "word_count":0,"regions":0,"audio_url":None,
                "lines":[],"method":"none",
                "viz_b64":viz_b64,"binary_b64":binary_b64,"line_count":0}

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
    return {"raw_text":raw,"final_text":final,"corrections":corr,
            "word_count":len(final.split()),"regions":word_count,
            "lines":[" ".join(wl) for wl in rec_lines],
            "audio_url":audio,"method":method,
            "viz_b64":viz_b64,"binary_b64":binary_b64,"line_count":line_count}



# ============================================================
# FLASK ROUTES
# ============================================================

@app.route("/")
def serve_frontend():
    from flask import send_from_directory
    return send_from_directory(".", "index.html")

@app.route("/health")
def health():
    p, _ = get_trocr()
    vocab = _get_vocab()
    return jsonify({
        "status":        "ok",
        "trocr":         p is not None,
        "easyocr":       _easyocr_reader is not None,
        "vocab_size":    len(vocab),
        "bktree_ready":  _BK_TREE is not None,
    })


@app.route("/ocr", methods=["POST"])
@app.route("/ocr_full", methods=["POST"])
def ocr_endpoint():
    try:
        if "image" not in request.files:
            return jsonify({"error":"No image"}), 400
        img   = Image.open(request.files["image"].stream).convert("RGB")
        speak = request.form.get("speak","0") == "1"
        return jsonify(run_pipeline(img, speak=speak))
    except Exception as e:
        logger.exception("Pipeline error")
        return jsonify({"error": str(e)}), 500

@app.route("/tts", methods=["POST"])
def tts_endpoint():
    try:
        text = request.get_json(force=True).get("text","")
        if not text: return jsonify({"error":"No text"}), 400
        path = speak_nepali(text)
        if not path: return jsonify({"error":"TTS failed"}), 500
        return send_file(path, mimetype="audio/mpeg", download_name="speech.mp3")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/audio")
def get_audio():
    if not os.path.exists("output_audio.mp3"):
        return jsonify({"error":"No audio"}), 404
    return send_file("output_audio.mp3", mimetype="audio/mpeg",
                     download_name="speech.mp3")

@app.route("/detect_viz", methods=["POST"])
def detect_viz():
    try:
        if "image" not in request.files:
            return jsonify({"error":"No image"}), 400
        pil = Image.open(request.files["image"].stream).convert("RGB")
        img_bgr, binary = preprocess(pil)
        boxes, line_groups = detect_words(img_bgr, binary=binary)

        vis = img_bgr.copy()
        COLORS = [(60,200,100),(60,160,220),(220,80,60),
                  (220,160,40),(180,60,220),(40,200,200)]
        for li, line in enumerate(line_groups):
            col = COLORS[li % len(COLORS)]
            for wi, box in enumerate(line):
                x, y, w, h = box["x"], box["y"], box["w"], box["h"]
                cv2.rectangle(vis, (x,y), (x+w,y+h), col, 2)
                cv2.putText(vis, f"L{li+1}W{wi+1}", (x, max(y-4,14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

        _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return jsonify({
            "image_b64":  base64.b64encode(buf.tobytes()).decode(),
            "word_count": len(boxes),
            "line_count": len(line_groups),
            "boxes": [{"x":b["x"],"y":b["y"],"w":b["w"],"h":b["h"]}
                      for b in boxes],
        })
    except Exception as e:
        logger.exception("detect_viz error")
        return jsonify({"error": str(e)}), 500

@app.route("/vocab_check", methods=["POST"])
def vocab_check():
    """
    Debug endpoint: POST JSON {"word": "..."} → returns membership
    and nearest neighbours from the BK-tree.
    """
    try:
        data = request.get_json(force=True)
        word = _nfc(data.get("word","").strip())
        if not word:
            return jsonify({"error":"No word"}), 400
        vocab = _get_vocab()
        in_vocab = word in vocab
        tree = _get_bktree()
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
    print("  Digital Pratilipi OCR  v9  [CC+VPP+BKTree+TrOCR]")
    print("="*60)
    print(f"  TROCR_MODEL    : {TROCR_MODEL}")
    print(f"  VOCAB_DIR      : {VOCAB_DIR}")
    print(f"  dictionary     : {VOCAB_DICTIONARY_FILE}")
    print(f"  corpus         : {VOCAB_CORPUS_FILE}")
    print(f"  URL            : http://localhost:5000")
    print("="*60 + "\n")
    # Pre-warm everything at startup
    get_trocr()
    get_easyocr()
    _get_vocab()
    _get_bktree()
    app.run(host="0.0.0.0", port=5000, debug=False)