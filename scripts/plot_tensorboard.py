"""
TensorBoard Log Visualizer — Handwritten Devanagari Text Detection
Encoder-Decoder Transformer (Word-Level)

Expects a flat log directory with event files like:
    logs/events.out.tfevents.1771738064.DESKTOP-KHBAVIJ.5392

Usage:
    python plot_tensorboard_logs.py
    python plot_tensorboard_logs.py --logdir ./logs
    python plot_tensorboard_logs.py --logdir ./logs --output_dir ./charts --dpi 150
"""

import os
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    raise ImportError(
        "\nCould not import TensorBoard. Install it with:\n"
        "    pip install tensorboard\n"
    )

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — tweak these if your tag names differ
# ─────────────────────────────────────────────────────────────────────────────

# Keywords to classify a tag as "eval/validation"
EVAL_KEYWORDS = ["eval", "val", "validation", "test", "dev"]

# Metric groups: (display title, [keywords that match tag names])
METRIC_GROUPS = [
    ("Loss",           ["loss", "nll", "cross_entropy", "ce"]),
    ("Accuracy",       ["acc", "accuracy"]),
    ("CER / WER",      ["cer", "wer", "edit_distance", "edit"]),
    ("Learning Rate",  ["lr", "learning_rate"]),
    ("Gradient Norm",  ["grad_norm", "gradient_norm", "global_norm"]),
    ("CTC Loss",       ["ctc"]),
    ("Perplexity",     ["ppl", "perplexity"]),
    ("Attention",      ["attn", "attention"]),
]

COLORS = {
    "train": "#4C72B0",
    "eval":  "#DD8452",
}
SMOOTH_WEIGHT = 0.85   # 0 = no smoothing, 0.99 = very heavy


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def ema_smooth(values: np.ndarray, weight: float = SMOOTH_WEIGHT) -> np.ndarray:
    smoothed, last = [], float(values[0])
    for v in values:
        last = weight * last + (1 - weight) * float(v)
        smoothed.append(last)
    return np.array(smoothed)


def is_eval_tag(tag: str) -> bool:
    tl = tag.lower()
    return any(k in tl for k in EVAL_KEYWORDS)


def group_of(tag: str) -> str:
    tl = tag.lower()
    for group_name, keywords in METRIC_GROUPS:
        if any(k in tl for k in keywords):
            return group_name
    return "Other"


def short_name(tag: str) -> str:
    """Strip common prefixes like 'train/', 'eval/' for cleaner labels."""
    for prefix in ("train/", "eval/", "val/", "test/", "training/", "validation/"):
        if tag.lower().startswith(prefix):
            return tag[len(prefix):]
    return tag


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

def load_all_scalars(logdir: str) -> dict:
    """
    Read every scalar from all event files in logdir (flat, no subdirs needed).
    Returns: { tag: { "steps": np.array, "values": np.array } }
    """
    print(f"  Loading events from: {logdir}")
    ea = EventAccumulator(logdir, size_guidance={"scalars": 0})  # 0 = load all
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    if not tags:
        raise ValueError(
            f"No scalar data found in '{logdir}'.\n"
            "Make sure the path contains 'events.out.tfevents.*' files."
        )

    print(f"  Found {len(tags)} scalar tag(s):")
    data = {}
    for tag in sorted(tags):
        events = ea.Scalars(tag)
        steps  = np.array([e.step  for e in events])
        values = np.array([e.value for e in events])
        data[tag] = {"steps": steps, "values": values}
        print(f"    [{len(steps):>5} pts]  {tag}")

    return data


def split_train_eval(data: dict):
    """Split scalar dict into (train_dict, eval_dict) by tag name."""
    train, eval_ = {}, {}
    for tag, vals in data.items():
        if is_eval_tag(tag):
            eval_[tag] = vals
        else:
            train[tag] = vals
    return train, eval_


def build_groups(train: dict, eval_: dict) -> dict:
    """
    Returns:
        { group_name: { short_tag: { "train": vals_or_None,
                                     "eval":  vals_or_None } } }
    """
    all_short = {}   # short_tag -> {"train": ..., "eval": ...}

    for tag, vals in train.items():
        s = short_name(tag)
        all_short.setdefault(s, {"train": None, "eval": None})
        all_short[s]["train"] = vals

    for tag, vals in eval_.items():
        s = short_name(tag)
        all_short.setdefault(s, {"train": None, "eval": None})
        all_short[s]["eval"] = vals

    groups = defaultdict(dict)
    for short, tv in all_short.items():
        groups[group_of(short)][short] = tv

    return dict(groups)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _style_ax(ax, title):
    ax.set_facecolor("#F7F8FA")
    ax.grid(True, color="#DEDEDE", linewidth=0.7, linestyle="--", zorder=0)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=7)
    ax.set_xlabel("Step", fontsize=9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, framealpha=0.75, loc="best")
    ax.spines[["top", "right"]].set_visible(False)


def plot_single_metric(ax, short_tag, tv: dict):
    """Plot one metric (train + eval if available) on the given Axes."""
    for split in ("train", "eval"):
        vals = tv[split]
        if vals is None:
            continue
        steps  = vals["steps"]
        values = vals["values"]
        color  = COLORS[split]
        label  = f"{split.capitalize()} — {short_tag}"

        # Raw (faint)
        ax.plot(steps, values, color=color, alpha=0.25, lw=0.9, zorder=2)

        # Smoothed (bold)
        if len(values) >= 4:
            ax.plot(steps, ema_smooth(values), color=color,
                    lw=2.2, label=label, zorder=3)
        else:
            ax.plot(steps, values, color=color,
                    lw=2.2, label=label, zorder=3)

    _style_ax(ax, short_tag)


def save_group_figure(group_name: str, metrics: dict, output_dir: str, dpi: int):
    n     = len(metrics)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7.5 * ncols, 4.5 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    fig.suptitle(
        f"Devanagari HTR  |  {group_name}",
        fontsize=13, fontweight="bold", y=1.01
    )

    for i, (short_tag, tv) in enumerate(metrics.items()):
        plot_single_metric(axes_flat[i], short_tag, tv)

    # Hide spare axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    safe = group_name.lower().replace(" ", "_").replace("/", "_")
    path = os.path.join(output_dir, f"{safe}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


def save_overview(groups: dict, output_dir: str, dpi: int):
    """
    One-page summary: pick the single most representative metric
    from each group and plot them together.
    """
    chosen = {}
    priority_order = [g for g, _ in METRIC_GROUPS] + ["Other"]
    for gname in priority_order:
        if gname in groups:
            first_tag = next(iter(groups[gname]))
            chosen[f"{gname}\n({first_tag})"] = groups[gname][first_tag]

    if not chosen:
        return

    n     = len(chosen)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6.5 * ncols, 4.2 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    fig.suptitle(
        "Devanagari HTR — Training Overview\nEncoder-Decoder Transformer · Word-Level",
        fontsize=13, fontweight="bold", y=1.02
    )

    for i, (label, tv) in enumerate(chosen.items()):
        plot_single_metric(axes_flat[i], label, tv)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "00_overview.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot TensorBoard scalars for Devanagari HTR model"
    )
    parser.add_argument("--logdir",     default="./logs",
                        help="Directory containing events.out.tfevents.* files")
    parser.add_argument("--output_dir", default="./charts",
                        help="Where to save PNG charts (created if absent)")
    parser.add_argument("--dpi",        type=int, default=130,
                        help="Resolution of output PNGs (default 130)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "─" * 55)
    print("  Devanagari HTR — TensorBoard Log Plotter")
    print("─" * 55)
    print(f"  logdir     : {os.path.abspath(args.logdir)}")
    print(f"  output_dir : {os.path.abspath(args.output_dir)}")
    print(f"  dpi        : {args.dpi}")
    print("─" * 55 + "\n")

    # 1. Load
    data         = load_all_scalars(args.logdir)
    train, eval_ = split_train_eval(data)

    print(f"\n  Train tags : {len(train)}")
    print(f"  Eval  tags : {len(eval_)}\n")

    # 2. Group
    groups = build_groups(train, eval_)

    # 3. Overview
    print("Generating overview…")
    save_overview(groups, args.output_dir, args.dpi)

    # 4. Per-group charts
    print(f"Generating {len(groups)} group chart(s)…")
    for group_name, metrics in groups.items():
        save_group_figure(group_name, metrics, args.output_dir, args.dpi)

    print(f"\n✓ Done. All charts saved to: {os.path.abspath(args.output_dir)}\n")


if __name__ == "__main__":
    main()