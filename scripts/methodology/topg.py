"""
Character-Level Confusion Matrix & Error Analysis — TOP RECOGNIZED CHARACTERS
===============================================================================
Same as the original confusion_matrix_analysis.py, with ONE key change:

  plot_confusion_heatmap_recognized()
  ────────────────────────────────────
  Instead of selecting the characters with the MOST substitution errors,
  this version selects the TOP-N characters by CORRECT recognition count.

  The matrix shows — for each well-recognized character (row = ground truth) —
  how often the model predicted it correctly vs. confused it with something else
  (column = predicted).  Diagonal = correct; off-diagonal = errors.

All other analysis (category breakdown, CER by word length, CSVs) is unchanged.
"""

import os
import unicodedata
import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import Levenshtein
from collections import defaultdict
from tqdm import tqdm
from tabulate import tabulate
from PIL import Image
from transformers import (
    TrOCRProcessor,
    ViTImageProcessor,
    RobertaTokenizer,
    VisionEncoderDecoderModel,
)

# =============================================================================
# >>>  FILL THESE IN  <<<
# =============================================================================

MODEL_DIR  = "./model"
TEST_TXT   = r"C:\Users\ASUS\Desktop\IIIT-HW-Hindi_v1\test.txt"
IMAGE_DIR  = r"C:\Users\ASUS\Desktop\IIIT-HW-Hindi_v1\HindiSeg"

ENCODE     = "google/vit-base-patch16-224-in21k"
DECODE     = "flax-community/roberta-hindi"

BATCH_SIZE  = 16
NUM_BEAMS   = 4
MAX_LENGTH  = 64

# None = run full test split; set e.g. 1000 for a quick preview
SAMPLE_SIZE = None

# Top N most-recognized characters to include in the heatmap
TOP_N = 25

OUTPUT_DIR = "./evaluation_results/confusion"

# Optional: path to a Devanagari-capable font for matplotlib
DEVANAGARI_FONT_PATH = ""

# =============================================================================


# ── Devanagari character categories ──────────────────────────────────────────

MATRAS             = set("ािीुूृेैोौंःँॅॉ\u094D")
HALANT             = "\u094D"
DEVANAGARI_DIGITS  = set("०१२३४५६७८९")
SHIROREKHA_ARTIFACTS = {"\u094D", "\u200C", "\u200D", "\u0902", "\u0903"}
VOWELS             = set("अआइईउऊऋएऐओऔ")
CONSONANTS         = set("कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह")


def char_category(ch: str) -> str:
    if ch in DEVANAGARI_DIGITS:       return "numeral"
    if ch in MATRAS:                  return "matra"
    if ch == HALANT:                  return "halant/conjunct"
    if ch in SHIROREKHA_ARTIFACTS:    return "shirorekha artifact"
    if ch in VOWELS:                  return "base vowel"
    if ch in CONSONANTS:              return "base consonant"
    cp = ord(ch)
    if 0x0900 <= cp <= 0x097F:        return "other Devanagari"
    return "non-Devanagari"


def is_devanagari(ch: str) -> bool:
    return 0x0900 <= ord(ch) <= 0x097F


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


# ── Font setup ────────────────────────────────────────────────────────────────

def setup_font():
    import requests, warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    font_filename = "NotoSansDevanagari-Regular.ttf"
    font_path     = os.path.join(OUTPUT_DIR, font_filename)

    if not os.path.isfile(font_path):
        url = (
            "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/"
            "NotoSansDevanagari/NotoSansDevanagari-Regular.ttf"
        )
        print("[font]  Downloading Noto Sans Devanagari ...")
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(font_path, "wb") as f:
                f.write(r.content)
            print(f"[font]  Saved to: {font_path}")
        except Exception as e:
            print(f"[font]  Download failed: {e}")
            return None

    fm.fontManager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = prop.get_name()
    print(f"[font]  Registered: {prop.get_name()}")
    return prop


# ── Data loader ───────────────────────────────────────────────────────────────

def dataset_generator(data_path: str) -> pd.DataFrame:
    with open(data_path, encoding="utf-8") as f:
        lines = f.readlines()
    rows = []
    for line in lines:
        parts = line.strip().split(" ", 1)
        if len(parts) == 2:
            rows.append({"file_name": parts[0].strip(), "text": parts[1].strip()})
    return pd.DataFrame(rows)


# ── Character-level alignment ─────────────────────────────────────────────────

def align_chars(ref: str, hyp: str):
    ops     = Levenshtein.editops(ref, hyp)
    result  = []
    ref_pos = hyp_pos = op_ptr = 0

    while ref_pos < len(ref) or hyp_pos < len(hyp):
        if op_ptr < len(ops):
            op, r_i, h_i = ops[op_ptr]
            if op == "insert" and h_i == hyp_pos:
                result.append(("insert", "", hyp[hyp_pos]))
                hyp_pos += 1; op_ptr += 1; continue
            if op == "delete" and r_i == ref_pos:
                result.append(("delete", ref[ref_pos], ""))
                ref_pos += 1; op_ptr += 1; continue
            if op == "replace" and r_i == ref_pos and h_i == hyp_pos:
                result.append(("replace", ref[ref_pos], hyp[hyp_pos]))
                ref_pos += 1; hyp_pos += 1; op_ptr += 1; continue

        if ref_pos < len(ref) and hyp_pos < len(hyp):
            result.append(("equal", ref[ref_pos], hyp[hyp_pos]))
            ref_pos += 1; hyp_pos += 1
        elif ref_pos < len(ref):
            result.append(("delete", ref[ref_pos], ""))
            ref_pos += 1
        else:
            result.append(("insert", "", hyp[hyp_pos]))
            hyp_pos += 1

    return result


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, processor, df, device):
    model.eval()
    all_preds, all_labels = [], []

    for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Inference"):
        batch = df.iloc[start: start + BATCH_SIZE]
        images, labels = [], []
        for _, row in batch.iterrows():
            try:
                img = Image.open(
                    os.path.join(IMAGE_DIR, row["file_name"])
                ).convert("RGB")
                images.append(img)
                labels.append(normalize(str(row["text"])))
            except Exception:
                continue
        if not images:
            continue

        pixel_values = processor(
            images=images, return_tensors="pt"
        ).pixel_values.to(device)

        with torch.no_grad():
            ids = model.generate(
                pixel_values, num_beams=NUM_BEAMS, max_length=MAX_LENGTH
            )

        preds = processor.batch_decode(ids, skip_special_tokens=True)
        all_preds.extend([normalize(p) for p in preds])
        all_labels.extend(labels)

    return all_labels, all_preds


# ── Confusion data builder ────────────────────────────────────────────────────

def build_confusion_data(refs, hyps):
    substitution_counts = defaultdict(lambda: defaultdict(int))
    category_errors     = defaultdict(lambda: defaultdict(int))
    word_len_errors     = defaultdict(list)
    deletion_counts     = defaultdict(int)
    insertion_counts    = defaultdict(int)
    correct_counts      = defaultdict(int)

    for ref, hyp in zip(refs, hyps):
        alignment  = align_chars(ref, hyp)
        word_errs  = sum(1 for op, _, _ in alignment if op != "equal")
        word_chars = len(ref)
        if word_chars > 0:
            word_len_errors[word_chars].append(word_errs / word_chars)

        for op, r_ch, h_ch in alignment:
            cat = char_category(r_ch) if r_ch else char_category(h_ch)

            if op == "equal":
                correct_counts[r_ch]             += 1
                category_errors[cat]["correct"]  += 1
            elif op == "replace":
                substitution_counts[r_ch][h_ch]      += 1
                category_errors[cat]["substitution"] += 1
            elif op == "delete":
                deletion_counts[r_ch]              += 1
                category_errors[cat]["deletion"]   += 1
            elif op == "insert":
                insertion_counts[h_ch]             += 1
                category_errors[cat]["insertion"]  += 1

    return (substitution_counts, correct_counts,
            deletion_counts, insertion_counts,
            category_errors, word_len_errors)


# ── ★ NEW: Confusion matrix for TOP MOST RECOGNIZED characters ────────────────

def plot_confusion_heatmap_recognized(
        substitution_counts, correct_counts,
        deletion_counts, font_prop, top_n=TOP_N):
    """
    Builds a confusion matrix whose rows/columns are the TOP-N characters
    by CORRECT recognition count (i.e. those the model handles best).

    Each cell (row=gt, col=pred) shows:
      • Diagonal  (gt == pred) : correct recognition count  ← dominant, green
      • Off-diag  (gt != pred) : substitution count         ← errors, red tones

    This lets you see BOTH how well each character is recognized AND
    what it gets mixed up with on the rare occasions it is wrong.

    Two variants are saved:
      recognized_confusion_heatmap.png         — raw counts
      recognized_confusion_heatmap_normed.png  — row-normalized (recognition rate)
    """

    # ── 1. Select top-N characters by correct count ───────────────────────────
    top_chars = [
        ch for ch, _ in
        sorted(correct_counts.items(), key=lambda x: -x[1])[:top_n]
    ]
    print(f"\n[info]  Top {top_n} most-recognized characters (by correct count):")
    for i, ch in enumerate(top_chars):
        print(f"        {i+1:2d}. '{ch}'  correct={correct_counts[ch]:,}")

    # ── 2. Build matrix  (rows = ground truth, cols = predicted) ─────────────
    #    • Diagonal  = correct_counts[ch]
    #    • Off-diag  = substitution_counts[gt_ch][pred_ch]   (if pred in top_chars)
    #    • Deletions are NOT shown as a column (they have no predicted char);
    #      add them to a separate summary line printed below.

    matrix = pd.DataFrame(0, index=top_chars, columns=top_chars, dtype=int)

    # Fill diagonal
    for ch in top_chars:
        matrix.loc[ch, ch] = correct_counts[ch]

    # Fill off-diagonal substitutions (both ref and hyp must be in top_chars)
    for r_ch, hyp_dict in substitution_counts.items():
        if r_ch not in top_chars:
            continue
        for h_ch, cnt in hyp_dict.items():
            if h_ch in top_chars:
                matrix.loc[r_ch, h_ch] += cnt

    # ── 3a. Raw-count heatmap ─────────────────────────────────────────────────
    _save_heatmap(
        matrix, top_chars, font_prop,
        title=(
            f"Confusion matrix — Top {top_n} most correctly recognised characters\n"
            "(diagonal = correct; off-diagonal = substitution count)"
        ),
        filename="recognized_confusion_heatmap.png",
        normalize_rows=False,
        cmap="RdYlGn",          # green = high correct, red = errors
    )

    # ── 3b. Row-normalised heatmap (recognition rate) ─────────────────────────
    # Each row is divided by its total (correct + substitutions within top_chars)
    # so the diagonal shows the % of times that character was recognised correctly.
    _save_heatmap(
        matrix, top_chars, font_prop,
        title=(
            f"Row-normalised confusion matrix — Top {top_n} most recognised characters\n"
            "(diagonal = recognition rate; off-diagonal = substitution rate)"
        ),
        filename="recognized_confusion_heatmap_normed.png",
        normalize_rows=True,
        cmap="RdYlGn",
    )

    # ── 4. Print recognition rate table ──────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RECOGNITION RATE FOR TOP CHARACTERS")
    print("=" * 65)
    rows_tbl = []
    for ch in top_chars:
        row_sum   = matrix.loc[ch].sum()
        diag_val  = matrix.loc[ch, ch]
        rec_rate  = diag_val / row_sum * 100 if row_sum > 0 else 0
        del_cnt   = deletion_counts.get(ch, 0)
        rows_tbl.append([
            ch,
            char_category(ch),
            int(diag_val),
            int(row_sum),
            f"{rec_rate:.1f}%",
            del_cnt,
        ])
    print(tabulate(
        rows_tbl,
        headers=["Char", "Category", "Correct", "Total seen", "Rec. rate", "Deletions"],
        tablefmt="rounded_outline",
    ))

    return matrix


def _save_heatmap(matrix, top_chars, font_prop,
                  title, filename, normalize_rows, cmap):
    """Internal helper: renders and saves one heatmap variant."""
    plot_matrix = matrix.copy().astype(float)

    if normalize_rows:
        row_sums = plot_matrix.sum(axis=1).replace(0, 1)
        plot_matrix = plot_matrix.div(row_sums, axis=0) * 100   # → percentage

    fig_size = max(14, len(top_chars) * 0.6)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size * 0.9))
    font_kw  = {"fontproperties": font_prop} if font_prop else {}

    sns.heatmap(
        plot_matrix,
        ax=ax,
        cmap=cmap,
        linewidths=0.4,
        linecolor="lightgray",
        annot=False,
        cbar=True,
        cbar_kws={
            "label": "Recognition rate (%)" if normalize_rows else "Count",
            "shrink": 0.6,
        },
        vmin=0,
        vmax=100 if normalize_rows else None,
    )

    # Manual cell annotations
    max_val = plot_matrix.values.max() if plot_matrix.values.max() > 0 else 1
    for i, r_ch in enumerate(top_chars):
        for j, h_ch in enumerate(top_chars):
            val = plot_matrix.loc[r_ch, h_ch]
            if val == 0:
                continue
            text_color = "white" if val > max_val * 0.65 else "black"
            label = f"{val:.0f}{'%' if normalize_rows else ''}"
            ax.text(
                j + 0.5, i + 0.5, label,
                ha="center", va="center",
                fontsize=7, color=text_color, fontweight="bold",
            )

    ax.set_xticks([i + 0.5 for i in range(len(top_chars))])
    ax.set_yticks([i + 0.5 for i in range(len(top_chars))])
    ax.set_xticklabels(top_chars, fontsize=12, rotation=0, **font_kw)
    ax.set_yticklabels(top_chars, fontsize=12, rotation=0, **font_kw)
    ax.set_xlabel("Predicted character", fontsize=13, labelpad=10)
    ax.set_ylabel("Ground truth character", fontsize=13, labelpad=10)
    ax.set_title(title, fontsize=13, pad=14)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[plot]  Saved → {out}")


# ── Existing helper plots (unchanged) ────────────────────────────────────────

def plot_wordlen_cer(word_len_errors):
    lengths  = sorted(word_len_errors.keys())
    avg_cer  = [np.mean(word_len_errors[l]) * 100 for l in lengths]
    counts   = [len(word_len_errors[l]) for l in lengths]
    buckets  = list(range(1, 16)) + ["16+"]
    bucket_cer, bucket_count = [], []
    for b in buckets:
        if b == "16+":
            vals = [v for l, v in zip(lengths, avg_cer) if l >= 16]
            cnt  = sum(c for l, c in zip(lengths, counts) if l >= 16)
        else:
            vals = [v for l, v in zip(lengths, avg_cer) if l == b]
            cnt  = sum(c for l, c in zip(lengths, counts) if l == b)
        bucket_cer.append(np.mean(vals) if vals else 0)
        bucket_count.append(cnt)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar([str(b) for b in buckets], bucket_cer,
            color="#4C72B0", alpha=0.85, label="Avg CER (%)")
    ax1.set_xlabel("Ground-truth word length (characters)", fontsize=12)
    ax1.set_ylabel("Average CER (%)", fontsize=12, color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")
    ax2 = ax1.twinx()
    ax2.plot([str(b) for b in buckets], bucket_count,
             color="#DD8452", marker="o", linewidth=2, label="Sample count")
    ax2.set_ylabel("Number of samples", fontsize=12, color="#DD8452")
    ax2.tick_params(axis="y", labelcolor="#DD8452")
    ax1.set_title("Average CER by word length", fontsize=14)
    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "cer_by_word_length.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot]  Word-length CER chart saved → {out}")


def plot_category_errors(category_errors):
    cats   = list(category_errors.keys())
    ops    = ["correct", "substitution", "deletion", "insertion"]
    colors = ["#2ca02c", "#d62728", "#ff7f0e", "#9467bd"]
    data   = {op: [category_errors[c].get(op, 0) for c in cats] for op in ops}
    totals = [sum(category_errors[c].values()) for c in cats]
    data_pct = {
        op: [v / t * 100 if t > 0 else 0 for v, t in zip(data[op], totals)]
        for op in ops
    }
    fig, ax = plt.subplots(figsize=(11, 5))
    bottoms = np.zeros(len(cats))
    for op, color in zip(ops, colors):
        ax.bar(cats, data_pct[op], bottom=bottoms,
               label=op.capitalize(), color=color, alpha=0.88)
        bottoms += np.array(data_pct[op])
    ax.set_xlabel("Character category", fontsize=12)
    ax.set_ylabel("Percentage of operations (%)", fontsize=12)
    ax.set_title("Error type distribution by Devanagari character category", fontsize=13)
    ax.legend(loc="lower right")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "error_by_category.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot]  Category error chart saved → {out}")


# ── Console summary ───────────────────────────────────────────────────────────

def print_summary(substitution_counts, correct_counts,
                  deletion_counts, insertion_counts,
                  category_errors, refs, hyps):
    pairs = []
    for r_ch, hyp_dict in substitution_counts.items():
        for h_ch, cnt in hyp_dict.items():
            total_ref = (correct_counts[r_ch]
                         + sum(substitution_counts[r_ch].values())
                         + deletion_counts[r_ch])
            rate = cnt / total_ref * 100 if total_ref > 0 else 0
            pairs.append((r_ch, h_ch, cnt, round(rate, 1), char_category(r_ch)))
    pairs.sort(key=lambda x: -x[2])

    print("\n" + "=" * 65)
    print("  TOP 20 CHARACTER SUBSTITUTION ERRORS")
    print("=" * 65)
    print(tabulate(
        [(r, h, cnt, f"{rate}%", cat) for r, h, cnt, rate, cat in pairs[:20]],
        headers=["Ground truth", "Predicted", "Count", "Rate", "Category"],
        tablefmt="rounded_outline",
    ))

    print("\n" + "=" * 65)
    print("  ERROR BREAKDOWN BY CHARACTER CATEGORY")
    print("=" * 65)
    cat_rows = []
    for cat, ops in sorted(category_errors.items()):
        total = sum(ops.values())
        subs  = ops.get("substitution", 0)
        dels  = ops.get("deletion", 0)
        ins   = ops.get("insertion", 0)
        cat_rows.append([cat, total, ops.get("correct", 0),
                         subs, dels, ins,
                         f"{(subs+dels+ins)/total*100:.1f}%" if total else "0%"])
    cat_rows.sort(key=lambda x: -x[1])
    print(tabulate(cat_rows,
                   headers=["Category", "Total ops", "Correct",
                             "Subst.", "Del.", "Ins.", "Error rate"],
                   tablefmt="rounded_outline"))

    total_ops = sum(sum(ops.values()) for ops in category_errors.values())
    total_err = sum(
        ops.get("substitution", 0) + ops.get("deletion", 0)
        + ops.get("insertion", 0)
        for ops in category_errors.values()
    )
    cer = total_err / total_ops * 100 if total_ops else 0
    print(f"\n  Total character operations : {total_ops:,}")
    print(f"  Total character errors     : {total_err:,}")
    print(f"  Overall CER (from matrix)  : {cer:.2f}%")
    return pairs


# ── Save CSVs ─────────────────────────────────────────────────────────────────

def save_csvs(pairs, category_errors, word_len_errors,
              recognized_matrix):

    pd.DataFrame(pairs, columns=[
        "ground_truth", "predicted", "count", "rate_pct", "category"
    ]).to_csv(os.path.join(OUTPUT_DIR, "substitution_errors.csv"),
              index=False, encoding="utf-8-sig")

    cat_rows = []
    for cat, ops in category_errors.items():
        total = sum(ops.values())
        cat_rows.append({
            "category":     cat,
            "total":        total,
            "correct":      ops.get("correct", 0),
            "substitution": ops.get("substitution", 0),
            "deletion":     ops.get("deletion", 0),
            "insertion":    ops.get("insertion", 0),
            "error_rate":   round(
                (ops.get("substitution",0)+ops.get("deletion",0)
                 +ops.get("insertion",0))/total*100 if total else 0, 2),
        })
    pd.DataFrame(cat_rows).to_csv(
        os.path.join(OUTPUT_DIR, "category_errors.csv"),
        index=False, encoding="utf-8-sig")

    pd.DataFrame([
        {"word_length": l,
         "avg_cer": round(np.mean(v)*100, 2),
         "sample_count": len(v)}
        for l, v in sorted(word_len_errors.items())
    ]).to_csv(os.path.join(OUTPUT_DIR, "wordlength_cer.csv"),
              index=False, encoding="utf-8-sig")

    # ★ NEW: Save the recognized-character matrix as CSV too
    recognized_matrix.to_csv(
        os.path.join(OUTPUT_DIR, "recognized_confusion_matrix.csv"),
        encoding="utf-8-sig")

    print(f"[output] CSVs saved to: {OUTPUT_DIR}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    font_prop = setup_font()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] Device : {device}")

    print(f"[setup] Loading model from: {MODEL_DIR}")
    feature_extractor = ViTImageProcessor.from_pretrained(ENCODE)
    tokenizer         = RobertaTokenizer.from_pretrained(DECODE)
    processor         = TrOCRProcessor(
                            image_processor=feature_extractor,
                            tokenizer=tokenizer)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
    model.to(device)
    print("[setup] Model loaded.")

    print(f"[data]  Loading: {TEST_TXT}")
    df = dataset_generator(TEST_TXT)
    if SAMPLE_SIZE and SAMPLE_SIZE < len(df):
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        print(f"[data]  Sampled {SAMPLE_SIZE} samples.")
    else:
        print(f"[data]  Full test split: {len(df):,} samples.")

    refs, hyps = run_inference(model, processor, df, device)
    print(f"[eval]  Inference complete. {len(refs):,} pairs collected.")

    print("[eval]  Aligning characters ...")
    (substitution_counts, correct_counts,
     deletion_counts, insertion_counts,
     category_errors, word_len_errors) = build_confusion_data(refs, hyps)

    print("[plot]  Generating plots ...")

    # ★ NEW: confusion matrix for TOP MOST RECOGNIZED characters
    recognized_matrix = plot_confusion_heatmap_recognized(
        substitution_counts, correct_counts,
        deletion_counts, font_prop, top_n=TOP_N
    )

    plot_wordlen_cer(word_len_errors)
    plot_category_errors(category_errors)

    pairs = print_summary(
        substitution_counts, correct_counts,
        deletion_counts, insertion_counts,
        category_errors, refs, hyps
    )

    save_csvs(pairs, category_errors, word_len_errors, recognized_matrix)

    print(f"\n[done]  All outputs saved to: {OUTPUT_DIR}/")
    print("        Files produced:")
    print("          recognized_confusion_heatmap.png        ← ★ NEW (raw counts)")
    print("          recognized_confusion_heatmap_normed.png ← ★ NEW (recognition %)")
    print("          recognized_confusion_matrix.csv         ← ★ NEW (source data)")
    print("          cer_by_word_length.png")
    print("          error_by_category.png")
    print("          substitution_errors.csv")
    print("          category_errors.csv")
    print("          wordlength_cer.csv")


if __name__ == "__main__":
    main()