"""
Indic OCR Dataset Preprocessing Pipeline
=========================================
Preprocessing steps: Sharpening → Grayscale → Binarization → Skew Correction

Supports the IIIT-HW-Hindi dataset with arbitrary numeric folder nesting.
Mirrors the original folder structure under a new output root.

Usage:
    python preprocess_indic_ocr.py

Or import and call process_split() directly for custom usage.
"""

import os
import sys
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

TRAIN_INPUT  = r"C:\Users\ASUS\Desktop\IIIT-HW-Hindi_v1\HindiSeg\HindiSeg\train"
VAL_INPUT    = r"C:\Users\ASUS\Desktop\IIIT-HW-Hindi_v1\HindiSeg\HindiSeg\val"

OUTPUT_ROOT  = r"C:\Users\ASUS\Desktop\IIIT-HW-Hindi_v1\HindiSeg\HindiSeg\preprocessed"

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
SKIP_EXTENSIONS             = {".txt", ".py", ".json", ".xml", ".csv", ".md"}

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(OUTPUT_ROOT, "preprocessing.log") 
                            if os.path.isdir(OUTPUT_ROOT) else "preprocessing.log",
                            encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Step 1 – Sharpening
# ──────────────────────────────────────────────

def sharpen(image: np.ndarray) -> np.ndarray:
    """
    Unsharp-mask sharpening:
    result = original + strength * (original − blurred)
    Enhances fine stroke details before any resolution-reducing step.
    """
    blurred  = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened


# ──────────────────────────────────────────────
# Step 2 – Grayscale Conversion
# ──────────────────────────────────────────────

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR/RGB image to single-channel grayscale."""
    if len(image.shape) == 2:
        return image                         # already grayscale
    if image.shape[2] == 4:                  # BGRA → BGR first
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ──────────────────────────────────────────────
# Step 3 – Binarization
# ──────────────────────────────────────────────

def binarize(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive Gaussian thresholding.
    Better than global Otsu for handwritten documents with uneven lighting.
    Block size 31 and C=10 work well for typical Hindi handwriting scans;
    tune if your images are very small or very noisy.
    """
    binary = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )
    return binary


# ──────────────────────────────────────────────
# Step 4 – Skew Correction
# ──────────────────────────────────────────────

def correct_skew(binary: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
    """
    Detect and correct document skew using the Hough line transform on
    Canny edges. Rotates only when a reliable skew angle is found.

    Args:
        binary:    Binarised (white text on black or black text on white) image.
        max_angle: Cap rotation to ±max_angle degrees to avoid wild flips.

    Returns:
        De-skewed image (same dtype/shape).
    """
    # Invert so text is white on black for edge detection
    inv = cv2.bitwise_not(binary)

    edges = cv2.Canny(inv, threshold1=50, threshold2=150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=50,
        maxLineGap=10,
    )

    if lines is None:
        return binary   # no lines detected – return as-is

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue                    # vertical line – skip
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) <= max_angle:
            angles.append(angle)

    if not angles:
        return binary

    skew_angle = float(np.median(angles))

    # Skip trivially small skew (< 0.3°) to avoid unnecessary interpolation
    if abs(skew_angle) < 0.3:
        return binary

    h, w = binary.shape
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), skew_angle, 1.0)
    corrected = cv2.warpAffine(
        binary, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return corrected


# ──────────────────────────────────────────────
# Full Pipeline
# ──────────────────────────────────────────────

def preprocess_image(image_path: Path) -> np.ndarray:
    """
    Run the complete preprocessing pipeline on a single image.

    Pipeline:
        1. Load (BGR via OpenCV)
        2. Sharpen
        3. Grayscale
        4. Binarize  (adaptive threshold)
        5. Skew-correct
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img        = sharpen(img)
    gray       = to_grayscale(img)
    binary     = binarize(gray)
    corrected  = correct_skew(binary)

    return corrected


# ──────────────────────────────────────────────
# Recursive folder walker
# ──────────────────────────────────────────────

def is_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def process_split(input_root: str, output_root: str, split_name: str) -> None:
    """
    Recursively walks `input_root / split_name`, finds every image regardless
    of how deep the numeric sub-folders go, preprocesses it, and saves the
    result under `output_root / split_name` mirroring the original tree.

    Non-image files (.txt, .py, …) are silently skipped.
    """
    src_root = Path(input_root)
    dst_root = Path(output_root) / split_name

    if not src_root.exists():
        log.error("Input path does not exist: %s", src_root)
        return

    log.info("=" * 60)
    log.info("Processing split : %s", split_name)
    log.info("Source           : %s", src_root)
    log.info("Destination      : %s", dst_root)
    log.info("=" * 60)

    processed = skipped = errors = 0

    # os.walk yields (dirpath, [subdirs], [files]) for every directory in the tree
    for dirpath, _subdirs, filenames in os.walk(src_root):
        dir_path = Path(dirpath)

        for filename in filenames:
            file_path = dir_path / filename

            # Skip non-image files explicitly
            if file_path.suffix.lower() in SKIP_EXTENSIONS:
                log.debug("Skipping non-image file: %s", file_path)
                skipped += 1
                continue

            if not is_image(file_path):
                log.debug("Skipping unsupported file: %s", file_path)
                skipped += 1
                continue

            # Mirror directory structure under destination
            relative     = file_path.relative_to(src_root)
            output_path  = dst_root / relative

            # Create parent directories as needed
            output_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                result = preprocess_image(file_path)

                # Save as PNG to preserve binary quality (lossless)
                save_path = output_path.with_suffix(".png")
                cv2.imwrite(str(save_path), result)

                log.info("✔  %s → %s", relative, save_path.relative_to(dst_root))
                processed += 1

            except Exception as exc:
                log.error("✘  %s  —  %s", relative, exc)
                errors += 1

    log.info("-" * 60)
    log.info("Split '%s' done.  Processed: %d  |  Skipped: %d  |  Errors: %d",
             split_name, processed, skipped, errors)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main() -> None:
    # Ensure output root exists before logging setup tries to write the log file
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

    # Re-add file handler now that OUTPUT_ROOT exists
    fh = logging.FileHandler(
        os.path.join(OUTPUT_ROOT, "preprocessing.log"), encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                                      datefmt="%H:%M:%S"))
    log.addHandler(fh)

    log.info("Output root      : %s", OUTPUT_ROOT)

    process_split(TRAIN_INPUT, OUTPUT_ROOT, "train")
    process_split(VAL_INPUT,   OUTPUT_ROOT, "val")

    log.info("All done.")


if __name__ == "__main__":
    main()