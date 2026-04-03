import logging
import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger(__name__)


# ============================================================
# GRAYSCALE
# ============================================================

def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR/RGB image to grayscale. Returns unchanged if already 1-ch."""
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        return image.squeeze()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ============================================================
# ADAPTIVE SHARPENING  (unchanged from v12)
# ============================================================

def _adaptive_sharpen(image: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Unsharp-mask sharpening, α = min(α_from_mean, α_from_std).

    α from mean:  <80→1.0, <130→0.7, <170→0.4, <210→0.2, else→0.05
    α from std:   <20→0.4, <45→0.7, <75→0.5, else→0.15
    Kernel: [0,−α,0 / −α,1+4α,−α / 0,−α,0]
    """
    alpha_m = (1.0  if mean < 80  else 0.7  if mean < 130 else
               0.4  if mean < 170 else 0.2  if mean < 210 else 0.05)
    alpha_s = (0.4  if std  < 20  else 0.7  if std  < 45  else
               0.5  if std  < 75  else 0.15)
    alpha   = min(alpha_m, alpha_s)
    logger.info(f"sharpen: mean={mean:.1f} std={std:.1f} α_m={alpha_m} α_s={alpha_s} → α={alpha:.2f}")
    k      = alpha
    kernel = np.array([[0,-k,0],[-k,1+4*k,-k],[0,-k,0]], dtype=np.float32)
    return cv2.filter2D(image, ddepth=-1, kernel=kernel)


# ============================================================
# STEP A — RETINEX BACKGROUND NORMALISATION  (unconditional)
# ============================================================

def _retinex_normalise(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    k    = max(101, min(h, w) // 4)
    k    = k if k % 2 == 1 else k + 1

    bg   = cv2.GaussianBlur(gray.astype(np.float32), (k, k), sigmaX=0)
    norm = np.clip(
        gray.astype(np.float32) / (bg + 1e-6) * 200.0,
        0, 255
    ).astype(np.uint8)

    logger.info(
        f"retinex: ksize={k}  "
        f"mean {float(np.mean(gray)):.1f}→{float(np.mean(norm)):.1f}  "
        f"std {float(np.std(gray)):.1f}→{float(np.std(norm)):.1f}"
    )
    return norm


# ============================================================
# STEP B — CLAHE  (unconditional, gentle fixed settings)
# ============================================================

def _apply_clahe(gray: np.ndarray) -> np.ndarray:
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    result = clahe.apply(gray)
    logger.info(
        f"CLAHE: mean {float(np.mean(gray)):.1f}→{float(np.mean(result)):.1f}  "
        f"std {float(np.std(gray)):.1f}→{float(np.std(result)):.1f}"
    )
    return result


# ============================================================
# BINARIZATION
# ============================================================

def _fix_polarity(binary: np.ndarray) -> np.ndarray:
    white_frac = float(np.count_nonzero(binary)) / binary.size

    if white_frac > 0.90 or white_frac < 0.10:
        logger.info(f"fix_polarity: white_frac={white_frac:.2f} extreme → invert")
        binary     = cv2.bitwise_not(binary)
        white_frac = 1.0 - white_frac

    if white_frac > 0.50:
        logger.info(f"fix_polarity: white_frac={white_frac:.2f} majority-white → invert")
        binary     = cv2.bitwise_not(binary)
        white_frac = 1.0 - white_frac

    h, w = binary.shape
    pad  = max(10, min(h, w) // 20)
    border = np.concatenate([
        binary[:pad, :].ravel(),  binary[-pad:, :].ravel(),
        binary[:, :pad].ravel(),  binary[:, -pad:].ravel(),
    ])
    if float(np.count_nonzero(border)) / border.size > 0.75:
        logger.info("fix_polarity: border majority-white → invert")
        binary = cv2.bitwise_not(binary)
    return binary


def _morphological_cleanup(binary: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)


def _smart_binarize(gray: np.ndarray) -> np.ndarray:
    assert gray.ndim == 2

    h      = gray.shape[0]
    block  = max(11, h // 20)
    block  = block if block % 2 == 1 else block + 1
    block  = min(block, 61)
    C_ADAPTIVE = 15

    T, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    logger.info(f"smart_binarize: Otsu T={T:.0f}")

    if T < 15 or T > 240:
        logger.warning(f"Otsu T={T:.0f} extreme → adaptiveThreshold fallback")
        gray_smooth = cv2.GaussianBlur(gray, (3, 3), sigmaX=0.8)
        binary = cv2.adaptiveThreshold(
            gray_smooth, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=block, C=C_ADAPTIVE,
        )

    binary = _fix_polarity(binary)
    binary = _morphological_cleanup(binary)
    return binary


# ============================================================
# SKEW CORRECTION — expanded canvas (no content clipping)
# ============================================================

def _rotation_matrix_expanded(h: int, w: int,
                               angle_deg: float) -> tuple[np.ndarray, int, int]:
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    corners   = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
    ones      = np.ones((4, 1), dtype=np.float32)
    rotated   = (M @ np.hstack([corners, ones]).T).T

    x_min, y_min = rotated[:, 0].min(), rotated[:, 1].min()
    x_max, y_max = rotated[:, 0].max(), rotated[:, 1].max()

    new_w = int(np.ceil(x_max - x_min))
    new_h = int(np.ceil(y_max - y_min))

    # Translate so nothing goes out of frame
    M[0, 2] -= x_min
    M[1, 2] -= y_min

    return M, new_w, new_h


def _correct_skew(binary: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, int, int]:
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
            if dx == 0:
                continue
            a = np.degrees(np.arctan2(dy, dx))
            if -45 <= a <= 45:
                angles.append(a)
        if angles:
            skew_angle = float(np.median(angles))

    h, w = binary.shape
    M, new_w, new_h = _rotation_matrix_expanded(h, w, skew_angle)

    corrected = cv2.warpAffine(
        binary, M, (new_w, new_h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )
    return corrected, skew_angle, M, new_w, new_h


# ============================================================
# FULL PIPELINE  (v13 — consistent)
# ============================================================

def preprocess(pil_image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    img_bgr = cv2.cvtColor(
        np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR
    )

    # Step 1 — Upscale
    h, w = img_bgr.shape[:2]
    if w < 960:
        scale   = 960 / w
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_CUBIC)

    # Step 2 — Grayscale (measure stats for sharpening only)
    gray_raw = _to_grayscale(img_bgr)
    mean_raw = float(np.mean(gray_raw))
    std_raw  = float(np.std(gray_raw))
    logger.info(
        f"preprocess: {img_bgr.shape[1]}×{img_bgr.shape[0]}px  "
        f"mean={mean_raw:.1f}  std={std_raw:.1f}"
    )

    # Step 3 — Adaptive sharpening (on colour image, before grayscale pipeline)
    sharpened = _adaptive_sharpen(img_bgr, mean_raw, std_raw)

    # Step 4 — Re-grayscale (single-channel from here on)
    gray = _to_grayscale(sharpened)

    # Step 5 — Retinex background normalisation (unconditional)
    gray = _retinex_normalise(gray)

    # Step 6 — CLAHE (unconditional, gentle)
    # Spreads the ink histogram slightly after retinex standardisation.
    gray = _apply_clahe(gray)

    # Step 7 — Smart binarization (Otsu on consistently normalised input)
    binary = _smart_binarize(gray)

    # Step 8 — Skew correction with expanded canvas
    binary_corrected, skew_angle, M_exp, new_w, new_h = _correct_skew(binary)

    if abs(skew_angle) > 0.5:
        h2, w2 = sharpened.shape[:2]
        M_colour, nw_c, nh_c = _rotation_matrix_expanded(h2, w2, skew_angle)
        img_bgr = cv2.warpAffine(
            sharpened, M_colour, (nw_c, nh_c),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
        )
    else:
        img_bgr = sharpened

    logger.info(f"preprocess done: skew={skew_angle:.2f}°")
    return img_bgr, binary_corrected