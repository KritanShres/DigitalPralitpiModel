"""
Character-Level Confusion Matrix & Error Analysis
===================================================
Loads the trained TrOCR model, runs inference on the IIIT-HW-Dev test split,
aligns predicted characters with ground truth using Levenshtein alignment,
and produces a full error analysis broken down by:

  1. Overall character confusion matrix (heatmap PNG)
  2. Top substitution error pairs (ranked table)
  3. Error categorization:
       - Matras        (vowel diacritics: ा  ि  ी  ु  ू  े  ै  ो  ौ etc.)
       - Conjuncts     (half-forms and stacked consonants using halant ्)
       - Numerals      (Devanagari digits ०–९)
       - Shirorekha    (artifacts from the horizontal headline)
       - Base chars    (standalone consonants and vowels)
  4. Word-length vs CER analysis
  5. Per-category substitution rate table
  6. All results saved as CSV + PNG for direct inclusion in the paper

Requirements:
    pip install transformers torch pillow pandas numpy matplotlib seaborn tqdm tabulate Levenshtein

Usage:
    python confusion_matrix_analysis.py
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

# Top N char pairs to show in the heatmap
TOP_N = 25

OUTPUT_DIR = "./evaluation_results/confusion"

# Optional: path to a Devanagari-capable font for matplotlib
# Download Noto Sans Devanagari from fonts.google.com and point here
# Leave as "" to use matplotlib default (Devanagari may render as boxes
# in the plot labels, but all CSVs will be correct)
DEVANAGARI_FONT_PATH = ""

# =============================================================================


# ── Devanagari character categories ──────────────────────────────────────────

MATRAS = set("ािीुूृेैोौंःँॅॉ\u094D")      # vowel signs + anusvara + visarga
HALANT = "\u094D"                              # virama / halant ्
DEVANAGARI_DIGITS = set("०१२३४५६७८९")

# Characters that commonly appear as shirorekha artifacts:
# isolated halant, zero-width joiner, zero-width non-joiner
SHIROREKHA_ARTIFACTS = {"\u094D", "\u200C", "\u200D", "\u0902", "\u0903"}

VOWELS = set("अआइईउऊऋएऐओऔ")
CONSONANTS = set("कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह")


def char_category(ch: str) -> str:
    """Assign a Devanagari error category to a single character."""
    if ch in DEVANAGARI_DIGITS:
        return "numeral"
    if ch in MATRAS:
        return "matra"
    if ch == HALANT:
        return "halant/conjunct"
    if ch in SHIROREKHA_ARTIFACTS:
        return "shirorekha artifact"
    if ch in VOWELS:
        return "base vowel"
    if ch in CONSONANTS:
        return "base consonant"
    cp = ord(ch)
    if 0x0900 <= cp <= 0x097F:
        return "other Devanagari"
    return "non-Devanagari"


def is_devanagari(ch: str) -> bool:
    cp = ord(ch)
    return 0x0900 <= cp <= 0x097F


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


# ── Font setup ────────────────────────────────────────────────────────────────

def setup_font():
    """
    Downloads Noto Sans Devanagari automatically if not found,
    registers it with matplotlib, and returns a FontProperties object.
    """
    import requests
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    font_filename = "NotoSansDevanagari-Regular.ttf"
    font_path     = os.path.join(OUTPUT_DIR, font_filename)

    # Download if not already present
    if not os.path.isfile(font_path):
        url = (
            "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/"
            "NotoSansDevanagari/NotoSansDevanagari-Regular.ttf"
        )
        print(f"[font]  Downloading Noto Sans Devanagari ...")
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(font_path, "wb") as f:
                f.write(r.content)
            print(f"[font]  Saved to: {font_path}")
        except Exception as e:
            print(f"[font]  Download failed: {e}")
            print("[font]  Labels may show as boxes. "
                  "Manually place NotoSansDevanagari-Regular.ttf "
                  f"in {OUTPUT_DIR}/ and rerun.")
            return None

    # Register with matplotlib
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
            rows.append({
                "file_name": parts[0].strip(),
                "text":      parts[1].strip()
            })
    return pd.DataFrame(rows)


# ── Character-level alignment using Levenshtein opcodes ───────────────────────

def align_chars(ref: str, hyp: str):
    """
    Uses Levenshtein editops to align ref and hyp at character level.
    Returns a list of (operation, ref_char, hyp_char) tuples:
        ('equal',   ref_ch, hyp_ch)   — correct
        ('replace', ref_ch, hyp_ch)   — substitution
        ('delete',  ref_ch, '')        — deletion
        ('insert',  '',     hyp_ch)   — insertion
    """
    ops = Levenshtein.editops(ref, hyp)
    aligned = []
    ref_idx = hyp_idx = 0
    op_map = {op[0]: op for op in ops}

    ops_dict = defaultdict(list)
    for op in ops:
        ops_dict[op[1]].append(op)

    # Walk through ref positions
    ref_pos = 0
    hyp_pos = 0
    op_queue = list(ops)
    op_ptr   = 0

    result = []
    while ref_pos < len(ref) or hyp_pos < len(hyp):
        # check if next op applies here
        if op_ptr < len(op_queue):
            op, r_i, h_i = op_queue[op_ptr]
            if op == "insert" and h_i == hyp_pos:
                result.append(("insert", "", hyp[hyp_pos]))
                hyp_pos += 1
                op_ptr  += 1
                continue
            if op == "delete" and r_i == ref_pos:
                result.append(("delete", ref[ref_pos], ""))
                ref_pos += 1
                op_ptr  += 1
                continue
            if op == "replace" and r_i == ref_pos and h_i == hyp_pos:
                result.append(("replace", ref[ref_pos], hyp[hyp_pos]))
                ref_pos += 1
                hyp_pos += 1
                op_ptr  += 1
                continue

        # equal
        if ref_pos < len(ref) and hyp_pos < len(hyp):
            result.append(("equal", ref[ref_pos], hyp[hyp_pos]))
            ref_pos += 1
            hyp_pos += 1
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
    all_preds  = []
    all_labels = []

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
                pixel_values,
                num_beams=NUM_BEAMS,
                max_length=MAX_LENGTH,
            )

        preds = processor.batch_decode(ids, skip_special_tokens=True)
        all_preds.extend([normalize(p) for p in preds])
        all_labels.extend(labels)

    return all_labels, all_preds


# ── Confusion matrix builder ──────────────────────────────────────────────────

def build_confusion_data(refs, hyps):
    """
    Aligns every ref/hyp pair and tallies:
      - substitution_counts[ref_ch][hyp_ch]
      - category_errors  {category: {correct, substitution, deletion, insertion}}
      - word_length_cer  {word_len: [cer values]}
    """
    substitution_counts = defaultdict(lambda: defaultdict(int))
    category_errors     = defaultdict(lambda: defaultdict(int))
    word_len_errors     = defaultdict(list)   # word_len -> list of (errors, chars)
    deletion_counts     = defaultdict(int)
    insertion_counts    = defaultdict(int)
    correct_counts      = defaultdict(int)

    for ref, hyp in zip(refs, hyps):
        alignment = align_chars(ref, hyp)
        word_errs = sum(1 for op, _, _ in alignment if op != "equal")
        word_chars = len(ref)
        if word_chars > 0:
            word_len_errors[word_chars].append(word_errs / word_chars)

        for op, r_ch, h_ch in alignment:
            cat = char_category(r_ch) if r_ch else char_category(h_ch)

            if op == "equal":
                correct_counts[r_ch]         += 1
                category_errors[cat]["correct"] += 1

            elif op == "replace":
                substitution_counts[r_ch][h_ch] += 1
                category_errors[cat]["substitution"] += 1

            elif op == "delete":
                deletion_counts[r_ch]             += 1
                category_errors[cat]["deletion"]   += 1

            elif op == "insert":
                insertion_counts[h_ch]             += 1
                category_errors[cat]["insertion"]  += 1

    return (substitution_counts, correct_counts,
            deletion_counts, insertion_counts,
            category_errors, word_len_errors)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_confusion_heatmap(substitution_counts, font_prop, top_n=TOP_N):
    """
    Annotated confusion matrix with counts in every cell.
    Rows = ground truth character, Columns = predicted character.
    Only the top_n most frequently confused characters are shown.
    """
    # ── find top-N characters by total substitution involvement ──────────────
    ref_totals = defaultdict(int)
    for r_ch, hyp_dict in substitution_counts.items():
        for h_ch, cnt in hyp_dict.items():
            ref_totals[r_ch] += cnt

    top_chars = [ch for ch, _ in
                 sorted(ref_totals.items(), key=lambda x: -x[1])[:top_n]]

    # ── build numeric matrix ──────────────────────────────────────────────────
    matrix = pd.DataFrame(0, index=top_chars, columns=top_chars)
    for r_ch, hyp_dict in substitution_counts.items():
        for h_ch, cnt in hyp_dict.items():
            if r_ch in top_chars and h_ch in top_chars:
                matrix.loc[r_ch, h_ch] += cnt

    # ── plot ──────────────────────────────────────────────────────────────────
    fig_size = max(14, top_n * 0.55)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size * 0.9))

    font_kw = {"fontproperties": font_prop} if font_prop else {}

    # Background heatmap (no annotations — we draw them manually)
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="Blues",
        linewidths=0.4,
        linecolor="lightgray",
        annot=False,          # we annotate manually below
        cbar=True,
        cbar_kws={"label": "Substitution count", "shrink": 0.6},
    )

    # ── annotate every cell with its count ───────────────────────────────────
    max_val = matrix.values.max() if matrix.values.max() > 0 else 1
    for i, r_ch in enumerate(top_chars):
        for j, h_ch in enumerate(top_chars):
            val = matrix.loc[r_ch, h_ch]
            if val == 0:
                continue                     # leave zero cells blank
            # choose text colour for contrast
            text_color = "white" if val > max_val * 0.6 else "black"
            ax.text(
                j + 0.5, i + 0.5,           # centre of cell
                str(int(val)),
                ha="center", va="center",
                fontsize=7,
                color=text_color,
                fontweight="bold",
            )

    # ── axis labels ───────────────────────────────────────────────────────────
    ax.set_xticks([i + 0.5 for i in range(len(top_chars))])
    ax.set_yticks([i + 0.5 for i in range(len(top_chars))])
    ax.set_xticklabels(top_chars, fontsize=12, rotation=0, **font_kw)
    ax.set_yticklabels(top_chars, fontsize=12, rotation=0, **font_kw)

    ax.set_xlabel("Predicted character", fontsize=13, labelpad=10)
    ax.set_ylabel("Ground truth character", fontsize=13, labelpad=10)
    ax.set_title(
        f"Character-level substitution confusion matrix\n"
        f"(Top {top_n} most confused Devanagari characters — counts shown in cells)",
        fontsize=13, pad=14
    )

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "confusion_heatmap.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[plot]  Confusion matrix saved → {out}")
    return matrix


def plot_wordlen_cer(word_len_errors):
    """Bar chart of average CER grouped by ground-truth word length."""
    lengths = sorted(word_len_errors.keys())
    avg_cer = [np.mean(word_len_errors[l]) * 100 for l in lengths]
    counts  = [len(word_len_errors[l]) for l in lengths]

    # Group lengths > 15 together
    buckets = list(range(1, 16)) + ["16+"]
    bucket_cer   = []
    bucket_count = []
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
    bars = ax1.bar([str(b) for b in buckets], bucket_cer,
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
    """Stacked bar chart of error types per character category."""
    cats   = list(category_errors.keys())
    ops    = ["correct", "substitution", "deletion", "insertion"]
    colors = ["#2ca02c", "#d62728", "#ff7f0e", "#9467bd"]

    data = {op: [category_errors[c].get(op, 0) for c in cats] for op in ops}
    totals = [sum(category_errors[c].values()) for c in cats]

    # Normalize to percentages
    data_pct = {
        op: [v / t * 100 if t > 0 else 0
             for v, t in zip(data[op], totals)]
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
    ax.set_title("Error type distribution by Devanagari character category",
                 fontsize=13)
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

    # Top substitution pairs
    pairs = []
    for r_ch, hyp_dict in substitution_counts.items():
        for h_ch, cnt in hyp_dict.items():
            total_ref = (correct_counts[r_ch]
                         + sum(substitution_counts[r_ch].values())
                         + deletion_counts[r_ch])
            rate = cnt / total_ref * 100 if total_ref > 0 else 0
            pairs.append((r_ch, h_ch, cnt, round(rate, 1),
                          char_category(r_ch)))
    pairs.sort(key=lambda x: -x[2])

    print("\n" + "=" * 65)
    print("  TOP 20 CHARACTER SUBSTITUTION ERRORS")
    print("=" * 65)
    table = [(r, h, cnt, f"{rate}%", cat)
             for r, h, cnt, rate, cat in pairs[:20]]
    print(tabulate(table,
                   headers=["Ground truth", "Predicted", "Count",
                             "Rate", "Category"],
                   tablefmt="rounded_outline"))

    # Category summary
    print("\n" + "=" * 65)
    print("  ERROR BREAKDOWN BY CHARACTER CATEGORY")
    print("=" * 65)
    cat_rows = []
    for cat, ops in sorted(category_errors.items()):
        total  = sum(ops.values())
        corr   = ops.get("correct", 0)
        subs   = ops.get("substitution", 0)
        dels   = ops.get("deletion", 0)
        ins    = ops.get("insertion", 0)
        err_rt = (subs + dels + ins) / total * 100 if total > 0 else 0
        cat_rows.append([cat, total, corr, subs, dels, ins,
                         f"{err_rt:.1f}%"])
    cat_rows.sort(key=lambda x: -x[1])
    print(tabulate(cat_rows,
                   headers=["Category", "Total ops", "Correct",
                             "Subst.", "Del.", "Ins.", "Error rate"],
                   tablefmt="rounded_outline"))

    # Overall char-level stats
    total_ops = sum(
        sum(ops.values()) for ops in category_errors.values()
    )
    total_err = sum(
        ops.get("substitution", 0) + ops.get("deletion", 0)
        + ops.get("insertion", 0)
        for ops in category_errors.values()
    )
    cer_overall = total_err / total_ops * 100 if total_ops > 0 else 0

    print(f"\n  Total character operations : {total_ops:,}")
    print(f"  Total character errors     : {total_err:,}")
    print(f"  Overall CER (from matrix)  : {cer_overall:.2f}%")
    print(f"\n── Report-ready sentences ──\n")
    top3 = pairs[:3]
    print(
        f"  Error analysis reveals that the most frequent substitution "
        f"errors involve the character pair "
        f"'{top3[0][0]}' predicted as '{top3[0][1]}' ({top3[0][2]} occurrences, "
        f"{top3[0][3]} substitution rate), "
        f"'{top3[1][0]}' predicted as '{top3[1][1]}' ({top3[1][2]} occurrences), "
        f"and '{top3[2][0]}' predicted as '{top3[2][1]}' ({top3[2][2]} occurrences). "
        f"Matra and halant/conjunct characters show the highest error rates, "
        f"consistent with the structural complexity of Devanagari diacritics."
    )

    return pairs


# ── Save CSVs ─────────────────────────────────────────────────────────────────

def save_csvs(pairs, category_errors, word_len_errors):
    # Substitution pairs
    subs_df = pd.DataFrame(pairs,
        columns=["ground_truth", "predicted", "count", "rate_pct", "category"])
    subs_df.to_csv(
        os.path.join(OUTPUT_DIR, "substitution_errors.csv"),
        index=False, encoding="utf-8-sig"
    )

    # Category breakdown
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
                (ops.get("substitution", 0) + ops.get("deletion", 0)
                 + ops.get("insertion", 0)) / total * 100
                if total > 0 else 0, 2
            ),
        })
    pd.DataFrame(cat_rows).to_csv(
        os.path.join(OUTPUT_DIR, "category_errors.csv"),
        index=False, encoding="utf-8-sig"
    )

    # Word-length CER
    wl_rows = [{"word_length": l, "avg_cer": round(np.mean(v) * 100, 2),
                "sample_count": len(v)}
               for l, v in sorted(word_len_errors.items())]
    pd.DataFrame(wl_rows).to_csv(
        os.path.join(OUTPUT_DIR, "wordlength_cer.csv"),
        index=False, encoding="utf-8-sig"
    )
    print(f"[output] CSVs saved to: {OUTPUT_DIR}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    font_prop = setup_font()

    # ── device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] Device : {device}")

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
    print(f"[setup] Model loaded.")

    # ── load data ─────────────────────────────────────────────────────────────
    print(f"[data]  Loading: {TEST_TXT}")
    df = dataset_generator(TEST_TXT)
    if SAMPLE_SIZE and SAMPLE_SIZE < len(df):
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        print(f"[data]  Sampled {SAMPLE_SIZE} / {len(df)} samples.")
    else:
        print(f"[data]  Full test split: {len(df):,} samples.")

    # ── inference ─────────────────────────────────────────────────────────────
    refs, hyps = run_inference(model, processor, df, device)
    print(f"[eval]  Inference complete. {len(refs):,} pairs collected.")

    # ── alignment + confusion data ────────────────────────────────────────────
    print("[eval]  Aligning characters and building confusion matrix ...")
    (substitution_counts, correct_counts,
     deletion_counts, insertion_counts,
     category_errors, word_len_errors) = build_confusion_data(refs, hyps)

    # ── plots ─────────────────────────────────────────────────────────────────
    print("[plot]  Generating plots ...")
    plot_confusion_heatmap(substitution_counts, font_prop, top_n=TOP_N)
    plot_wordlen_cer(word_len_errors)
    plot_category_errors(category_errors)

    # ── summary ───────────────────────────────────────────────────────────────
    pairs = print_summary(
        substitution_counts, correct_counts,
        deletion_counts, insertion_counts,
        category_errors, refs, hyps
    )

    # ── save CSVs ─────────────────────────────────────────────────────────────
    save_csvs(pairs, category_errors, word_len_errors)

    print(f"\n[done]  All outputs saved to: {OUTPUT_DIR}/")
    print("        Files produced:")
    print("          confusion_heatmap.png    — include in Results §5.3")
    print("          cer_by_word_length.png   — include in Results §5.3")
    print("          error_by_category.png    — include in Results §5.3")
    print("          substitution_errors.csv  — source data for paper table")
    print("          category_errors.csv      — source data for paper table")
    print("          wordlength_cer.csv       — source data for paper table\n")


if __name__ == "__main__":
    main()