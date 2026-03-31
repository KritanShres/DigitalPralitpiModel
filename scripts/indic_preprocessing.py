"""
Indic OCR Dataset Preprocessing Pipeline
=========================================
Preprocessing steps: Sharpening → Grayscale → Binarization (Otsu) → Skew Correction

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


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

TEST_INPUT = r"C:\Users\ASUS\Desktop\IIIT-HW-Hindi_v1\HindiSeg\HindiSeg\test"
# VAL_INPUT   = r"C:\Users\ASUS\Desktop\IIIT-HW-Hindi_v1\HindiSeg\HindiSeg\val"
OUTPUT_ROOT = r"C:\Users\ASUS\Desktop\IIIT-HW-Hindi_v1\HindiSeg\HindiSeg\preprocessed"

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
SKIP_EXTENSIONS             = {".txt", ".py", ".json", ".xml", ".csv", ".md"}


# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Step 1 – Sharpening
# ──────────────────────────────────────────────

def sharpen(image: np.ndarray) -> np.ndarray:
    """Apply a 3×3 sharpening kernel via 2D convolution."""
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0],
    ], dtype=np.float32)
    return cv2.filter2D(image, ddepth=-1, kernel=kernel)


# ──────────────────────────────────────────────
# Step 2 – Grayscale Conversion
# ──────────────────────────────────────────────

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR/BGRA image to single-channel grayscale."""
    if image.ndim == 2:
        return image                                  # already grayscale
    if image.ndim == 3 and image.shape[2] == 1:
        return image.squeeze()                        # single-channel wrapped
    if image.shape[2] == 4:                           # BGRA → BGR first
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ──────────────────────────────────────────────
# Step 3 – Binarization (Otsu's Method)
# ──────────────────────────────────────────────

def binarize(gray: np.ndarray) -> np.ndarray:
    """Apply Otsu's global thresholding to produce a binary image."""
    assert gray.ndim == 2, "Input must be a single-channel grayscale image."
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return binary


# ──────────────────────────────────────────────
# Step 4 – Skew Correction
# ──────────────────────────────────────────────

def correct_skew(binary: np.ndarray, max_angle: float = 45.0) -> np.ndarray:
    """
    Detect skew via Hough Line Transform and rotate the image to correct it.
    Skips rotation when the detected angle is negligibly small (< 0.3°).

    Args:
        binary:    Binarised single-channel image.
        max_angle: Only consider line angles within ±max_angle degrees.

    Returns:
        De-skewed binary image (same dtype/shape).
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

    if lines is None:
        return binary   # no lines detected – return as-is

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        if dx == 0:
            continue
        angle_deg = np.degrees(np.arctan2(dy, dx))
        if abs(angle_deg) <= max_angle:
            angles.append(angle_deg)

    if not angles:
        return binary

    skew_angle = float(np.median(angles))

    # Skip trivially small skew to avoid unnecessary interpolation artefacts
    if abs(skew_angle) < 0.3:
        return binary

    h, w   = binary.shape
    center = (w / 2.0, h / 2.0)
    M      = cv2.getRotationMatrix2D(center, skew_angle, scale=1.0)
    corrected = cv2.warpAffine(
        binary, M, (w, h),
        flags=cv2.INTER_CUBIC,
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
        1. Load  (BGR / BGRA via OpenCV)
        2. Sharpen          (3×3 convolution kernel)
        3. Grayscale        (BGR → single-channel)
        4. Binarize         (Otsu's global threshold)
        5. Skew-correct     (Hough line transform)
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img       = sharpen(img)
    gray      = to_grayscale(img)
    binary    = binarize(gray)
    corrected = correct_skew(binary)

    return corrected


# ──────────────────────────────────────────────
# Recursive Folder Walker
# ──────────────────────────────────────────────

def is_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def process_split(input_root: str, output_root: str, split_name: str) -> None:
    """
    Recursively walks `input_root`, finds every image regardless of how deep
    the numeric sub-folders go, preprocesses it, and saves the result under
    `output_root / split_name` mirroring the original directory tree.

    Non-image files (.txt, .py, …) are silently skipped.
    Preprocessed images are saved as lossless PNGs.
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

    for dirpath, _subdirs, filenames in os.walk(src_root):
        dir_path = Path(dirpath)

        for filename in filenames:
            file_path = dir_path / filename

            if file_path.suffix.lower() in SKIP_EXTENSIONS:
                log.debug("Skipping non-image file: %s", file_path)
                skipped += 1
                continue

            if not is_image(file_path):
                log.debug("Skipping unsupported file: %s", file_path)
                skipped += 1
                continue

            # Mirror directory structure under destination
            relative    = file_path.relative_to(src_root)
            output_path = dst_root / relative
            output_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                result    = preprocess_image(file_path)
                save_path = output_path.with_suffix(".png")   # lossless output
                cv2.imwrite(str(save_path), result)

                log.info("✔  %s → %s", relative, save_path.relative_to(dst_root))
                processed += 1

            except Exception as exc:
                log.error("✘  %s  —  %s", relative, exc)
                errors += 1

    log.info("-" * 60)
    log.info(
        "Split '%s' done.  Processed: %d  |  Skipped: %d  |  Errors: %d",
        split_name, processed, skipped, errors,
    )


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

def main() -> None:
    # Ensure output root exists before attaching the file log handler
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(
        os.path.join(OUTPUT_ROOT, "preprocessing.log"), encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
    ))
    log.addHandler(fh)

    log.info("Output root : %s", OUTPUT_ROOT)

    process_split(TEST_INPUT, OUTPUT_ROOT, "train")
    # process_split(VAL_INPUT,   OUTPUT_ROOT, "val")

    log.info("All done.")


if __name__ == "__main__":
    main()