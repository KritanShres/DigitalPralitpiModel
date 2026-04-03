import os
import re
import json
import torch
import evaluate
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

from transformers import (
    RobertaTokenizer,
    TrOCRProcessor,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
    default_data_collator,
)

from dataloader import IAMDataset, dataset_generator

# ===========================
# CONFIGURATION
# ===========================
MODEL_PATH   = "./model"          
ENCODE       = "google/vit-base-patch16-224-in21k"
DECODE       = "flax-community/roberta-hindi"

# update these to your local IIIT-HW-Dev location
test_text_file = r"C:\Users\ASUS\Desktop\IIIT-HW-Hindi_v1\test.txt"
root_dir       = r"C:\Users\ASUS\Desktop\IIIT-HW-Hindi_v1\HindiSeg"

BATCH_SIZE     = 16
MAX_LENGTH     = 64
NUM_BEAMS      = 4
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR    = "./test_results"

# ===========================
# SETUP
# ===========================
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Using device : {DEVICE}")
print(f"Loading model from : {MODEL_PATH}")

# ===========================
# LOAD PROCESSOR & MODEL
# ===========================
feature_extractor = ViTImageProcessor.from_pretrained(ENCODE)
tokenizer         = RobertaTokenizer.from_pretrained(DECODE)
processor         = TrOCRProcessor(image_processor=feature_extractor, tokenizer=tokenizer)

model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
model.eval()
model.to(DEVICE)

print(f"Model parameters : {sum(p.numel() for p in model.parameters()):,}")

# ===========================
# LOAD TEST DATASET
# ===========================
test_df = dataset_generator(test_text_file)
print(f"Test samples : {len(test_df)}")

test_dataset = IAMDataset(root_dir=root_dir, df=test_df, processor=processor)
test_loader  = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=default_data_collator,
    num_workers=0,
    pin_memory=(DEVICE == "cuda"),
)

# ===========================
# METRICS
# ===========================
cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

# ===========================
# INFERENCE LOOP
# ===========================
all_predictions = []
all_references  = []

print("\nRunning inference on test set...")
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating", unit="batch"):
        pixel_values = batch["pixel_values"].to(DEVICE)
        labels       = batch["labels"]

        # Generate predictions
        generated_ids = model.generate(
            pixel_values,
            max_length=MAX_LENGTH,
            num_beams=NUM_BEAMS,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=2.0,
        )

        # Decode predictions
        pred_str = processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Decode labels (replace -100 padding with pad_token_id)
        labels[labels == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels, skip_special_tokens=True)

        all_predictions.extend(pred_str)
        all_references.extend(label_str)

# ===========================
# COMPUTE METRICS
# ===========================
cer = cer_metric.compute(predictions=all_predictions, references=all_references)
wer = wer_metric.compute(predictions=all_predictions, references=all_references)

print("\n" + "=" * 50)
print("         TEST SET RESULTS")
print("=" * 50)
print(f"  Total samples  : {len(all_predictions)}")
print(f"  CER            : {cer:.4f}  ({cer * 100:.2f}%)")
print(f"  WER            : {wer:.4f}  ({wer * 100:.2f}%)")
print("=" * 50)

# ===========================
# PER-SAMPLE RESULTS TABLE
# ===========================
results_df = pd.DataFrame({
    "reference":  all_references,
    "prediction": all_predictions,
    "match":      [p.strip() == r.strip() for p, r in zip(all_predictions, all_references)],
})

# Per-sample CER
def _per_sample_cer(pred: str, ref: str) -> float:
    if not ref:
        return 0.0
    return cer_metric.compute(predictions=[pred], references=[ref])

results_df["sample_cer"] = [
    _per_sample_cer(p, r)
    for p, r in zip(results_df["prediction"], results_df["reference"])
]

exact_match_pct = results_df["match"].mean() * 100
print(f"  Exact match    : {exact_match_pct:.2f}%")
print("=" * 50)

# ===========================
# SAVE RESULTS
# ===========================
csv_path = os.path.join(RESULTS_DIR, "predictions.csv")
results_df.to_csv(csv_path, index=False, encoding="utf-8")
print(f"\nPer-sample predictions saved to : {csv_path}")

# Save summary metrics to JSON
summary = {
    "model_path":        MODEL_PATH,
    "total_samples":     len(all_predictions),
    "cer":               round(cer, 6),
    "wer":               round(wer, 6),
    "exact_match_pct":   round(exact_match_pct, 4),
    "num_beams":         NUM_BEAMS,
    "max_length":        MAX_LENGTH,
    "device":            DEVICE,
}
summary_path = os.path.join(RESULTS_DIR, "summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"Summary metrics saved to        : {summary_path}")

# ===========================
# WORST PREDICTIONS (top-20 highest CER)
# ===========================
worst = results_df.nlargest(20, "sample_cer")[["reference", "prediction", "sample_cer"]]
worst_path = os.path.join(RESULTS_DIR, "worst_predictions.csv")
worst.to_csv(worst_path, index=False, encoding="utf-8")
print(f"Top-20 worst predictions saved  : {worst_path}")
