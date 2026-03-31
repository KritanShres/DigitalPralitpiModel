"""
Image Preprocessing Pipeline for OCR
Implements: Sharpening → Grayscale → Binarization → Skew Correction
Outputs a 2×2 grid PNG of the four pipeline stages.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


# ─────────────────────────────────────────────
# 1. SHARPENING
# ─────────────────────────────────────────────
def sharpen(image: np.ndarray) -> np.ndarray:
    """Apply a 3×3 sharpening kernel via 2D convolution."""
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)
    return cv2.filter2D(image, ddepth=-1, kernel=kernel)


# ─────────────────────────────────────────────
# 2. GRAYSCALE CONVERSION
# ─────────────────────────────────────────────
def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB/BGR image to grayscale. Returns unchanged if already single-channel."""
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        return image.squeeze()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ─────────────────────────────────────────────
# 3. BINARIZATION (Otsu's Method)
# ─────────────────────────────────────────────
def binarize(gray: np.ndarray) -> np.ndarray:
    """Apply Otsu's global thresholding to produce a binary image."""
    assert gray.ndim == 2, "Input must be a single-channel grayscale image."
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary


# ─────────────────────────────────────────────
# 4. SKEW CORRECTION
# ─────────────────────────────────────────────
def correct_skew(binary: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Detect skew via Hough Line Transform and rotate the image to correct it.
    Returns (corrected_image, skew_angle_degrees).
    """
    assert binary.ndim == 2, "Input must be a single-channel binary image."

    inverted = cv2.bitwise_not(binary)
    edges    = cv2.Canny(inverted, 50, 150, apertureSize=3)
    lines    = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                                threshold=80, minLineLength=50, maxLineGap=10)

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
    corrected = cv2.warpAffine(binary, M, (w, h),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
    return corrected, skew_angle


# ─────────────────────────────────────────────
# PIPELINE RUNNER
# ─────────────────────────────────────────────
def run_pipeline(image_path: str, output_path: str = None) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if output_path is None:
        output_path = os.path.join(script_dir, "preprocessing_result.png")

    # Load & run stages
    original_bgr          = cv2.imread(image_path)
    if original_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    sharpened             = sharpen(original_bgr)
    gray                  = to_grayscale(sharpened)
    binary                = binarize(gray)
    corrected, skew_angle = correct_skew(binary)

    # Four pipeline stages (no original)
    stages = [
        ("1. Sharpened",                     cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB), False),
        ("2. Grayscale",                     gray,                                        True),
        ("3. Binarized (Otsu)",              binary,                                      True),
        (f"4. Skew Corrected ({skew_angle:.2f}°)", corrected,                            True),
    ]

    # ── 2×2 grid ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 12), facecolor="white")
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            wspace=0.08, hspace=0.18,
                            left=0.02, right=0.98,
                            top=0.96, bottom=0.06)

    axes = []
    for i, (title, img, is_gray) in enumerate(stages):
        row, col = divmod(i, 2)
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img, cmap="gray" if is_gray else None, aspect="auto")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#bbbbbb")
            spine.set_linewidth(1.0)
        axes.append((ax, title))

    # Add captions after layout is finalised
    fig.canvas.draw()
    for ax, title in axes:
        pos = ax.get_position()
        fig.text(
            pos.x0 + pos.width / 2,
            pos.y0 - 0.015,
            title,
            ha="center", va="top",
            color="black", fontsize=16, fontweight="bold"
        )

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"✓ Saved 2×2 grid → {output_path}")
    print(f"  Detected skew angle: {skew_angle:.2f}°")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    input_path  = r"C:\Users\ASUS\Desktop\IIIT-HW-Hindi_v1\HindiSeg\HindiSeg\test\6\132\7.jpg"
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "preprocessing_result.png")
    run_pipeline(input_path, output_path)