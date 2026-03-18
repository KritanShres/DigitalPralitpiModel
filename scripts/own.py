import os
import re
import torch
import urllib.request
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
from PIL import Image
from pathlib import Path
from transformers import (
    RobertaTokenizer,
    ViTImageProcessor,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

# ===========================
# DEVANAGARI FONT SETUP
# ===========================
def setup_devanagari_font() -> str | None:
    """
    Returns a font name that supports Devanagari.
    Priority:
      1. Already installed on system (Nirmala UI, Noto Sans Devanagari, etc.)
      2. Download Noto Sans Devanagari into ./fonts/ and register it.
    """
    # 1 — Check system fonts first
    candidates = ["Nirmala UI", "Noto Sans Devanagari", "Lohit Devanagari", "FreeSerif"]
    available  = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            print(f"✓ Using system font: {name}")
            return name

    # 2 — Download Noto Sans Devanagari
    font_dir  = Path("fonts")
    font_path = font_dir / "NotoSansDevanagari-Regular.ttf"
    font_url  = (
        "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/"
        "NotoSansDevanagari/NotoSansDevanagari-Regular.ttf"
    )

    if not font_path.exists():
        font_dir.mkdir(exist_ok=True)
        print("Downloading Noto Sans Devanagari font...")
        try:
            urllib.request.urlretrieve(font_url, font_path)
            print(f"✓ Font downloaded to {font_path}")
        except Exception as e:
            print(f"⚠ Font download failed: {e}")
            print("  Devanagari text may not render correctly, but the script will still run.")
            return None

    fm.fontManager.addfont(str(font_path))
    font_name = fm.FontProperties(fname=str(font_path)).get_name()
    print(f"✓ Registered font: {font_name}")
    return font_name


DEVA_FONT = setup_devanagari_font()

# Apply globally — suppress the repeated 'findfont' warnings
if DEVA_FONT:
    matplotlib.rcParams['font.family'] = DEVA_FONT
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ===========================
# CONFIGURATION
# ===========================
MODEL_DIR  = "model/"
IMAGE_DIR  = "mock-up-dataset/"
OUTPUT_PNG = "mock_up_results.png"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SUPPORTED  = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

# ===========================
# LOAD PROCESSOR & MODEL
# ===========================
print("Loading model...")

encode = 'google/vit-base-patch16-224-in21k'
decode = 'flax-community/roberta-hindi'

feature_extractor = ViTImageProcessor.from_pretrained(encode)
tokenizer         = RobertaTokenizer.from_pretrained(decode)
processor         = TrOCRProcessor(image_processor=feature_extractor, tokenizer=tokenizer)

model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()
print(f"✓ Model ready on {DEVICE}\n")

# ===========================
# COLLECT IMAGES
# ===========================
image_paths = sorted([
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if Path(f).suffix.lower() in SUPPORTED
])

if not image_paths:
    raise FileNotFoundError(f"No images found in '{IMAGE_DIR}'. Supported: {SUPPORTED}")

print(f"Found {len(image_paths)} image(s) in '{IMAGE_DIR}'\n")

# ===========================
# PREDICT
# ===========================
def predict(image_input) -> str:
    if isinstance(image_input, (str, Path)):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input.convert("RGB")

    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_length=64,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=2.0,
        )
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


results = []
for i, path in enumerate(image_paths):
    pred = predict(path)
    results.append((path, pred))
    print(f"[{i+1}/{len(image_paths)}]  {os.path.basename(path):<30}  →  {pred}")

# ===========================
# VISUALIZE  (one row per image)
# ===========================
def make_font_props(size: int) -> dict:
    """Return fontdict with Devanagari font if available, else plain size."""
    if DEVA_FONT:
        return {"fontproperties": fm.FontProperties(family=DEVA_FONT, size=size)}
    return {"fontsize": size}

n   = len(results)
fig = plt.figure(figsize=(13, 4.0 * n))
fig.patch.set_facecolor("#f8f8f8")

gs = fig.add_gridspec(n, 2, width_ratios=[1, 1.6], hspace=0.5, wspace=0.08)

for i, (img_path, pred) in enumerate(results):
    image = Image.open(img_path).convert("RGB")

    # ── Image panel ──────────────────────────────
    ax_img = fig.add_subplot(gs[i, 0])
    ax_img.imshow(image, cmap="gray", aspect="auto")
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    for spine in ax_img.spines.values():
        spine.set_edgecolor("#cccccc")
        spine.set_linewidth(1.2)

    # ── Text panel ───────────────────────────────
    ax_txt = fig.add_subplot(gs[i, 1])
    ax_txt.set_xlim(0, 1)
    ax_txt.set_ylim(0, 1)
    ax_txt.axis("off")

    # Filename
    ax_txt.text(0.0, 0.92,
        f"#{i+1}  {os.path.basename(img_path)}",
        fontsize=9, color="#888888", fontstyle="italic", va="top")

    # "Predicted" label
    ax_txt.text(0.0, 0.72, "Predicted",
        fontsize=9, color="#444444", fontweight="bold", va="top")

    # Prediction box — use registered Devanagari font explicitly
    ax_txt.text(0.0, 0.55,
        pred if pred.strip() else "(no output)",
        va="top",
        color="#1a1a2e",
        bbox=dict(boxstyle="round,pad=0.45",
                  facecolor="#eef2ff",
                  edgecolor="#9fa8da",
                  linewidth=1.2),
        **make_font_props(14)
    )

plt.suptitle(
    f"Mock-up Dataset  —  {n} image(s)  |  Model: ViT + RoBERTa Hindi",
    fontsize=13, fontweight="bold", y=1.002, color="#1a1a2e"
)

plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print(f"\n✓ Figure saved → {OUTPUT_PNG}")