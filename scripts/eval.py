import os
import torch
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from pathlib import Path
from transformers import (
    RobertaTokenizer,
    ViTImageProcessor,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

matplotlib.rcParams['font.family'] = ['Mangal', 'Nirmala UI', 'Arial Unicode MS', 'DejaVu Sans']

# ===========================
# CONFIGURATION
# ===========================
MODEL_DIR = "model/"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# ===========================
# LOAD PROCESSOR & MODEL
# ===========================
print("Loading processor and model...")

encode = 'google/vit-base-patch16-224-in21k'
decode = 'flax-community/roberta-hindi'

feature_extractor = ViTImageProcessor.from_pretrained(encode)
tokenizer         = RobertaTokenizer.from_pretrained(decode)
processor         = TrOCRProcessor(image_processor=feature_extractor, tokenizer=tokenizer)

model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()

print(f"✓ Model loaded from '{MODEL_DIR}' on {DEVICE}\n")

# ===========================
# CORE PREDICT FUNCTION
# ===========================
def predict(image_input) -> str:
    """
    Accepts a file path (str/Path) or a PIL Image.
    Returns the predicted Devanagari text string.
    """
    if isinstance(image_input, (str, Path)):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise TypeError(f"Unsupported input type: {type(image_input)}")

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


# ===========================
# VISUALIZE SINGLE IMAGE
# ===========================
def visualize_single(image_path: str, ground_truth: str = None):
    """
    Display one image with its predicted (and optionally ground truth) text.
    """
    image     = Image.open(image_path).convert("RGB")
    predicted = predict(image)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.imshow(image, cmap='gray')
    ax.axis("off")

    title = f"Predicted:  {predicted}"
    if ground_truth:
        title += f"\nGround Truth: {ground_truth}"

    ax.set_title(title, fontsize=14, pad=12, loc='center')
    plt.tight_layout()
    plt.show()

    print(f"Predicted   : {predicted}")
    if ground_truth:
        print(f"Ground Truth: {ground_truth}")

    return predicted


# ===========================
# VISUALIZE BATCH (GRID)
# ===========================
def visualize_batch(image_paths: list, ground_truths: list = None, cols: int = 3):
    """
    Display a grid of images, each with its predicted text below.

    Args:
        image_paths  : List of image file paths.
        ground_truths: Optional list of ground truth labels (same length).
        cols         : Number of columns in the grid.
    """
    n    = len(image_paths)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten() if n > 1 else [axes]

    for i, image_path in enumerate(image_paths):
        image     = Image.open(image_path).convert("RGB")
        predicted = predict(image)

        axes[i].imshow(image, cmap='gray')
        axes[i].axis("off")

        label = f"Pred: {predicted}"
        if ground_truths and i < len(ground_truths):
            label += f"\nTrue: {ground_truths[i]}"

        axes[i].set_title(label, fontsize=11, pad=8)
        print(f"[{i+1}/{n}] {os.path.basename(image_path)} → {predicted}")

    # Hide any unused subplot slots
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Devanagari Handwritten OCR Results", fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()


# ===========================
# EVALUATE FROM VAL/TEST FILE
# ===========================
def evaluate_from_txt(txt_file: str, root_dir: str, num_samples: int = 12):
    """
    Read a dataset .txt file (same format used in training),
    run inference, and display a grid with predictions vs ground truth.

    txt_file format expected:  <relative_image_path> <label>
    """
    import pandas as pd
    from evaluate import load as eval_load

    rows = []
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                rows.append({"image": parts[0], "label": parts[1]})

    df = pd.DataFrame(rows).head(num_samples)

    image_paths   = [os.path.join(root_dir, row["image"]) for _, row in df.iterrows()]
    ground_truths = df["label"].tolist()

    # Filter out missing files
    valid = [(p, g) for p, g in zip(image_paths, ground_truths) if os.path.isfile(p)]
    if not valid:
        print("No valid image files found. Check root_dir and txt_file paths.")
        return

    image_paths, ground_truths = zip(*valid)

    # Collect predictions
    predictions = [predict(p) for p in image_paths]

    # Compute CER
    cer_metric = eval_load("cer")
    cer_score  = cer_metric.compute(predictions=list(predictions), references=list(ground_truths))
    print(f"\n{'='*40}")
    print(f"  Samples evaluated : {len(predictions)}")
    print(f"  CER               : {cer_score:.4f}  ({cer_score*100:.2f}%)")
    print(f"{'='*40}\n")

    # Display grid
    n    = len(image_paths)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten() if n > 1 else [axes]

    for i, (img_path, pred, gt) in enumerate(zip(image_paths, predictions, ground_truths)):
        image = Image.open(img_path).convert("RGB")
        axes[i].imshow(image, cmap='gray')
        axes[i].axis("off")

        color = "green" if pred.strip() == gt.strip() else "red"
        axes[i].set_title(
            f"Pred: {pred}\nTrue: {gt}",
            fontsize=10,
            color=color,
            pad=8
        )

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        f"OCR Results  |  CER: {cer_score*100:.2f}%",
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.show()

    return list(zip(image_paths, predictions, ground_truths))


# ===========================
# ENTRY POINT
# ===========================
if __name__ == "__main__":

    # --- Option 1: Single image ---
    # visualize_single("path/to/your/sample.png", ground_truth="आपका लेबल")

    # --- Option 2: A few images in a grid ---
    # visualize_batch(
    #     image_paths=["img1.png", "img2.png", "img3.png"],
    #     ground_truths=["लेबल 1", "लेबल 2", "लेबल 3"]
    # )

    # --- Option 3: Evaluate directly from your val/test .txt file ---
    evaluate_from_txt(
        txt_file  = r"C:\Users\ASUS\Desktop\IIIT-HW-Hindi_v1\val.txt",
        root_dir  = r"C:\Users\ASUS\Desktop\IIIT-HW-Hindi_v1\HindiSeg",
        num_samples = 12   # how many samples to visualize
    )