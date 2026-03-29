"""
TrOCR Evaluation Script
========================
Evaluates the trained TrOCR model on:
  (a) IIIT-HW-Dev test split  — the dataset the model was trained on
  (b) Real-world Nepali set   — your manually collected Nepali handwriting

Outputs:
  - CER for both splits
  - Word Error Rate (WER) for both splits
  - Per-sample prediction log (CSV)
  - Console summary ready to cite in the research paper

Usage:
    python evaluate_model.py

Requirements:
    pip install transformers evaluate jiwer pandas pillow tqdm tabulate
"""

import os
import torch
import evaluate
import pandas as pd
from PIL import Image
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from transformers import (
    TrOCRProcessor,
    ViTImageProcessor,
    RobertaTokenizer,
    VisionEncoderDecoderModel,
)

# =============================================================================
# >>>  FILL THESE IN  <<<
# =============================================================================

# Path to your saved model folder (the one named 'model' in project root)
MODEL_DIR = "./model"

# --- (a) IIIT-HW-Dev test split ---
IIIT_TEST_TXT  = r"C:\Users\ASUS\Desktop\IIIT-HW-Hindi_v1\test.txt"
IIIT_IMAGE_DIR = r"C:\Users\ASUS\Desktop\IIIT-HW-Hindi_v1\HindiSeg"

# --- (b) Real-world Nepali test set ---
# Same format as the IIIT txt files:  image_filename SPACE ground_truth_word
# e.g.   nepali_001.png नमस्ते
NEPALI_TEST_TXT  = ""   # e.g. r"C:\Users\ASUS\Desktop\NepaliRealWorld\test.txt"
NEPALI_IMAGE_DIR = ""   # e.g. r"C:\Users\ASUS\Desktop\NepaliRealWorld\images"

# Encoder / decoder identifiers (must match training)
ENCODE = "google/vit-base-patch16-224-in21k"
DECODE = "flax-community/roberta-hindi"

# Inference batch size — lower if you get OOM errors
BATCH_SIZE = 16

# Beam search settings (must match generation_config used in training)
NUM_BEAMS   = 4
MAX_LENGTH  = 64

# Output folder for result CSVs and the summary log
OUTPUT_DIR  = "./evaluation_results"

# =============================================================================


# ── helpers ───────────────────────────────────────────────────────────────────

def dataset_generator(data_path: str) -> pd.DataFrame:
    """Identical to the dataloader used during training."""
    with open(data_path, encoding="utf-8") as f:
        lines = f.readlines()
    rows = []
    for line in lines:
        parts = line.strip().split(" ", 1)
        if len(parts) == 2:
            rows.append({"file_name": parts[0].strip(), "text": parts[1].strip()})
    return pd.DataFrame(rows)


def load_image(image_dir: str, filename: str) -> Image.Image:
    path = os.path.join(image_dir, filename)
    return Image.open(path).convert("RGB")


def run_evaluation(
    model,
    processor,
    df: pd.DataFrame,
    image_dir: str,
    split_name: str,
    device,
    cer_metric,
    wer_metric,
) -> dict:
    """
    Runs inference on every sample in df, computes CER and WER,
    and returns a result dict plus a per-sample DataFrame.
    """
    model.eval()

    all_preds   = []
    all_labels  = []
    all_files   = []
    errors      = []

    print(f"\n{'='*60}")
    print(f"  Evaluating: {split_name}  ({len(df):,} samples)")
    print(f"{'='*60}")

    # Process in batches
    for start in tqdm(range(0, len(df), BATCH_SIZE), desc=split_name):
        batch_df = df.iloc[start : start + BATCH_SIZE]
        images, labels, files = [], [], []

        for _, row in batch_df.iterrows():
            try:
                img = load_image(image_dir, row["file_name"])
                images.append(img)
                labels.append(str(row["text"]).strip())
                files.append(row["file_name"])
            except Exception as e:
                errors.append({"file": row["file_name"], "error": str(e)})
                continue

        if not images:
            continue

        pixel_values = processor(
            images=images, return_tensors="pt"
        ).pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                num_beams=NUM_BEAMS,
                max_length=MAX_LENGTH,
            )

        preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
        preds = [p.strip() for p in preds]

        all_preds.extend(preds)
        all_labels.extend(labels)
        all_files.extend(files)

    # ── metrics ───────────────────────────────────────────────────────────────
    cer = cer_metric.compute(predictions=all_preds, references=all_labels)
    wer = wer_metric.compute(predictions=all_preds, references=all_labels)

    # ── per-sample correctness ────────────────────────────────────────────────
    exact_matches = sum(p == l for p, l in zip(all_preds, all_labels))
    word_accuracy = exact_matches / len(all_labels) if all_labels else 0.0

    # ── per-sample DataFrame ──────────────────────────────────────────────────
    sample_df = pd.DataFrame({
        "file_name":    all_files,
        "ground_truth": all_labels,
        "prediction":   all_preds,
        "exact_match":  [p == l for p, l in zip(all_preds, all_labels)],
    })

    # flag errors
    if errors:
        err_df = pd.DataFrame(errors)
        print(f"\n  [warning] {len(errors)} images could not be loaded:")
        print(err_df.to_string(index=False))

    return {
        "split":         split_name,
        "total_samples": len(all_labels),
        "CER":           round(cer, 4),
        "WER":           round(wer, 4),
        "exact_matches": exact_matches,
        "word_accuracy": round(word_accuracy * 100, 2),
        "load_errors":   len(errors),
        "sample_df":     sample_df,
    }


def print_summary(results: list[dict]):
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY — RESEARCH PAPER NUMBERS")
    print("=" * 60)

    table_rows = []
    for r in results:
        table_rows.append([
            r["split"],
            f"{r['total_samples']:,}",
            f"{r['CER']:.4f}",
            f"{r['WER']:.4f}",
            f"{r['word_accuracy']}%",
            f"{r['exact_matches']:,}",
            f"{r['load_errors']}",
        ])

    headers = [
        "Split", "Samples", "CER", "WER",
        "Word Accuracy", "Exact Matches", "Load Errors"
    ]
    print(tabulate(table_rows, headers=headers, tablefmt="rounded_outline"))

    print("\n── Report-ready sentences ──\n")
    for r in results:
        print(
            f"  {r['split']}: CER = {r['CER']:.4f} ({r['CER']*100:.2f}%), "
            f"WER = {r['WER']:.4f} ({r['WER']*100:.2f}%), "
            f"Word-level accuracy = {r['word_accuracy']}% "
            f"({r['exact_matches']:,} / {r['total_samples']:,} exact matches)"
        )

    # CER gap analysis if both splits evaluated
    if len(results) == 2:
        cer_gap = results[1]["CER"] - results[0]["CER"]
        wer_gap = results[1]["WER"] - results[0]["WER"]
        direction = "higher" if cer_gap > 0 else "lower"
        print(f"\n  CER gap (Nepali - IIIT): {cer_gap:+.4f} "
              f"({direction} on real-world Nepali)")
        print(f"  WER gap (Nepali - IIIT): {wer_gap:+.4f}")

        if cer_gap > 0.05:
            print(
                "\n  Interpretation: The model generalizes notably less well to "
                "real-world Nepali handwriting than to the IIIT-HW-Dev test split. "
                "This is consistent with the vocabulary overlap finding that 78%% of "
                "IIIT-HW-Dev labels are Hindi-dominant, confirming the need for a "
                "dedicated Nepali handwriting dataset."
            )
        elif cer_gap > 0:
            print(
                "\n  Interpretation: A modest CER increase on real-world Nepali "
                "handwriting suggests partial generalization. The gap is likely "
                "attributable to Hindi-dominant training data and varying "
                "handwriting styles not seen during training."
            )
        else:
            print(
                "\n  Interpretation: The model generalizes well to real-world "
                "Nepali handwriting, suggesting that shared Devanagari script "
                "features compensate for vocabulary differences between the "
                "training (Hindi-dominant) and test (Nepali) distributions."
            )

    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] Device: {device}")

    # ── load model ────────────────────────────────────────────────────────────
    print(f"[setup] Loading model from: {MODEL_DIR}")
    feature_extractor = ViTImageProcessor.from_pretrained(ENCODE)
    tokenizer         = RobertaTokenizer.from_pretrained(DECODE)
    processor         = TrOCRProcessor(
                            image_processor=feature_extractor,
                            tokenizer=tokenizer
                        )
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    print(f"[setup] Model loaded. Parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    # ── metrics ───────────────────────────────────────────────────────────────
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")

    # ── evaluate splits ───────────────────────────────────────────────────────
    all_results = []

    # (a) IIIT-HW-Dev test split
    if IIIT_TEST_TXT and IIIT_IMAGE_DIR:
        iiit_df = dataset_generator(IIIT_TEST_TXT)
        result_iiit = run_evaluation(
            model, processor, iiit_df, IIIT_IMAGE_DIR,
            split_name="IIIT-HW-Dev (Hindi dominant)",
            device=device,
            cer_metric=cer_metric,
            wer_metric=wer_metric,
        )
        all_results.append(result_iiit)

        out_path = os.path.join(OUTPUT_DIR, "iiit_predictions.csv")
        result_iiit["sample_df"].to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n[output] IIIT predictions saved to: {out_path}")
    else:
        print("[skip] IIIT_TEST_TXT or IIIT_IMAGE_DIR not set — skipping IIIT split.")

    # (b) Real-world Nepali test set
    if NEPALI_TEST_TXT and NEPALI_IMAGE_DIR:
        nepali_df = dataset_generator(NEPALI_TEST_TXT)
        result_nepali = run_evaluation(
            model, processor, nepali_df, NEPALI_IMAGE_DIR,
            split_name="Real-world Nepali",
            device=device,
            cer_metric=cer_metric,
            wer_metric=wer_metric,
        )
        all_results.append(result_nepali)

        out_path = os.path.join(OUTPUT_DIR, "nepali_predictions.csv")
        result_nepali["sample_df"].to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[output] Nepali predictions saved to: {out_path}")
    else:
        print("[skip] NEPALI_TEST_TXT or NEPALI_IMAGE_DIR not set — skipping Nepali split.")

    if not all_results:
        print("[error] No splits were evaluated. Check that your file paths are set correctly.")
        return

    # ── summary ───────────────────────────────────────────────────────────────
    print_summary(all_results)

    # ── save summary CSV ──────────────────────────────────────────────────────
    summary_rows = [
        {k: v for k, v in r.items() if k != "sample_df"}
        for r in all_results
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "evaluation_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[output] Summary saved to: {summary_path}\n")


if __name__ == "__main__":
    main()