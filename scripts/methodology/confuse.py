"""
Plot Confusion Heatmaps from Pre-computed CSVs
================================================
Reads the CSVs produced by confusion_matrix_recognized.py and regenerates
all plots — no model, no inference, no dataset required.

CSVs needed (all in CSV_DIR):
    recognized_confusion_matrix.csv   — NxN character confusion matrix
    substitution_errors.csv           — ranked substitution pairs
    category_errors.csv               — per-category error breakdown
    wordlength_cer.csv                — avg CER per word length

Usage:
    python plot_from_csv.py
"""

import os
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from tabulate import tabulate

# =============================================================================
CSV_DIR    = "./evaluation_results/confusion"
OUTPUT_DIR = "./evaluation_results/confusion"
# =============================================================================


def setup_font():
    font_path = os.path.join(OUTPUT_DIR, "NotoSansDevanagari-Regular.ttf")
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
        except Exception as e:
            print(f"[font]  Download failed: {e}")
            return None
    fm.fontManager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    # Apply per-label, NOT via rcParams, so Devanagari glyphs render correctly
    print(f"[font]  Registered: {prop.get_name()}")
    return prop


def plot_recognized_heatmaps(matrix_df, font_prop):
    top_chars = list(matrix_df.index)
    n         = len(top_chars)

    def _save(plot_matrix, title, filename, normalize_rows):
        fig_size = max(14, n * 0.6)
        fig, ax  = plt.subplots(figsize=(fig_size, fig_size * 0.9))

        sns.heatmap(
            plot_matrix,
            ax=ax,
            cmap="Blues",
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
            xticklabels=False,   # suppress seaborn's labels; we set manually
            yticklabels=False,
        )

        # Cell annotations
        max_val = plot_matrix.values.max() if plot_matrix.values.max() > 0 else 1
        for i, r_ch in enumerate(top_chars):
            for j, h_ch in enumerate(top_chars):
                val = plot_matrix.loc[r_ch, h_ch]
                if val == 0:
                    continue
                text_color = "white" if val > max_val * 0.65 else "black"
                label = f"{val:.0f}{'%' if normalize_rows else ''}"
                ax.text(j + 0.5, i + 0.5, label,
                        ha="center", va="center",
                        fontsize=7, color=text_color, fontweight="bold")

        # Set tick positions then labels with Devanagari font per label object
        ax.set_xticks([i + 0.5 for i in range(n)])
        ax.set_yticks([i + 0.5 for i in range(n)])
        x_lbls = ax.set_xticklabels(top_chars, fontsize=11, rotation=0)
        y_lbls = ax.set_yticklabels(top_chars, fontsize=11, rotation=0)
        if font_prop:
            for lbl in x_lbls:
                lbl.set_fontproperties(font_prop)
                lbl.set_fontsize(11)
            for lbl in y_lbls:
                lbl.set_fontproperties(font_prop)
                lbl.set_fontsize(11)

        ax.tick_params(axis="both", which="both",
                       bottom=False, top=False, left=False, right=False)
        ax.set_xlabel("Predicted Character", fontsize=14,
                      labelpad=12, fontweight="bold")
        ax.set_ylabel("Ground Truth Character", fontsize=14,
                      labelpad=12, fontweight="bold")
        ax.set_title(title, fontsize=13, pad=14)

        plt.tight_layout()
        out = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(out, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"[plot]  Saved → {out}")

    # Raw counts
    _save(
        matrix_df.astype(float),
        title=(
            f"Confusion Matrix — Top {n} Most Correctly Recognised Characters\n"
            "(diagonal = correct count; off-diagonal = substitution count)"
        ),
        filename="recognized_confusion_heatmap.png",
        normalize_rows=False,
    )

    # Row-normalised
    normed   = matrix_df.astype(float)
    row_sums = normed.sum(axis=1).replace(0, 1)
    normed   = normed.div(row_sums, axis=0) * 100
    _save(
        normed,
        title=(
            f"Row-Normalised Confusion Matrix — Top {n} Most Recognised Characters\n"
            "(diagonal = recognition rate %; off-diagonal = substitution rate %)"
        ),
        filename="recognized_confusion_heatmap_normed.png",
        normalize_rows=True,
    )


def plot_wordlen_cer(wl_df):
    buckets      = list(range(1, 16)) + ["16+"]
    bucket_cer, bucket_count = [], []
    for b in buckets:
        sub = wl_df[wl_df["word_length"] >= 16] if b == "16+" \
              else wl_df[wl_df["word_length"] == b]
        bucket_cer.append(sub["avg_cer"].mean() if len(sub) else 0)
        bucket_count.append(sub["sample_count"].sum())

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
    ax1.set_title("Average CER by Word Length", fontsize=14)
    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "cer_by_word_length.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot]  Saved → {out}")


def plot_category_errors(cat_df):
    cats   = cat_df["category"].tolist()
    ops    = ["correct", "substitution", "deletion", "insertion"]
    colors = ["#2ca02c", "#d62728", "#ff7f0e", "#9467bd"]
    totals = cat_df["total"].values
    fig, ax = plt.subplots(figsize=(11, 5))
    bottoms = np.zeros(len(cats))
    for op, color in zip(ops, colors):
        vals = cat_df[op].values / np.where(totals > 0, totals, 1) * 100
        ax.bar(cats, vals, bottom=bottoms,
               label=op.capitalize(), color=color, alpha=0.88)
        bottoms += vals
    ax.set_xlabel("Character category", fontsize=12)
    ax.set_ylabel("Percentage of operations (%)", fontsize=12)
    ax.set_title("Error Type Distribution by Devanagari Character Category", fontsize=13)
    ax.legend(loc="lower right")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "error_by_category.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot]  Saved → {out}")


def print_summary(subs_df, cat_df):
    print("\n" + "=" * 65)
    print("  TOP 20 CHARACTER SUBSTITUTION ERRORS")
    print("=" * 65)
    print(tabulate(
        subs_df.head(20)[["ground_truth", "predicted",
                           "count", "rate_pct", "category"]].values.tolist(),
        headers=["Ground truth", "Predicted", "Count", "Rate", "Category"],
        tablefmt="rounded_outline",
    ))
    print("\n" + "=" * 65)
    print("  ERROR BREAKDOWN BY CHARACTER CATEGORY")
    print("=" * 65)
    print(tabulate(
        cat_df.sort_values("total", ascending=False)[
            ["category", "total", "correct",
             "substitution", "deletion", "insertion", "error_rate"]
        ].values.tolist(),
        headers=["Category", "Total ops", "Correct",
                 "Subst.", "Del.", "Ins.", "Error rate %"],
        tablefmt="rounded_outline",
    ))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    font_prop = setup_font()

    print("[load]  Reading CSVs ...")
    matrix_df = pd.read_csv(os.path.join(CSV_DIR, "recognized_confusion_matrix.csv"),
                             index_col=0, encoding="utf-8-sig")
    subs_df   = pd.read_csv(os.path.join(CSV_DIR, "substitution_errors.csv"),
                             encoding="utf-8-sig")
    cat_df    = pd.read_csv(os.path.join(CSV_DIR, "category_errors.csv"),
                             encoding="utf-8-sig")
    wl_df     = pd.read_csv(os.path.join(CSV_DIR, "wordlength_cer.csv"),
                             encoding="utf-8-sig")
    print(f"        Matrix : {matrix_df.shape}  |  Subs : {len(subs_df):,}  |  "
          f"Categories : {len(cat_df)}  |  Word-length rows : {len(wl_df)}")

    print("\n[plot]  Generating plots ...")
    plot_recognized_heatmaps(matrix_df, font_prop)
    plot_wordlen_cer(wl_df)
    plot_category_errors(cat_df)
    print_summary(subs_df, cat_df)

    print(f"\n[done]  Saved to: {OUTPUT_DIR}/")
    print("          recognized_confusion_heatmap.png")
    print("          recognized_confusion_heatmap_normed.png")
    print("          cer_by_word_length.png")
    print("          error_by_category.png")


if __name__ == "__main__":
    main()