"""Generate final publication-quality plots for the report."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

plt.rcParams.update({
    'font.size': 12,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def plot_layer_wise_mae():
    """Layer-wise MAE comparison between French, English, and Digits."""
    with open(RESULTS_DIR / "probing_results.json") as f:
        data = json.load(f)

    layers = [0, 8, 16, 24, 32]
    fr_mae = [data["French"][str(l)]["mae_mean"] for l in layers]
    en_mae = [data["English"][str(l)]["mae_mean"] for l in layers]
    dg_mae = [data["Digits"][str(l)]["mae_mean"] for l in layers]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, fr_mae, 'b-o', label='French words', linewidth=2, markersize=8)
    ax.plot(layers, en_mae, 'g-s', label='English words', linewidth=2, markersize=8)
    ax.plot(layers, dg_mae, 'r-^', label='Digit strings', linewidth=2, markersize=8)

    ax.set_xlabel("Mistral-7B Layer")
    ax.set_ylabel("Probe MAE (Mean Absolute Error)")
    ax.set_title("Number Probing Accuracy by Layer: French vs English vs Digits")
    ax.legend()
    ax.set_xticks(layers)

    # Add ratio annotations
    for i, l in enumerate(layers):
        if en_mae[i] > 0 and l > 0:
            ratio = fr_mae[i] / en_mae[i]
            ax.annotate(f'{ratio:.1f}x', (l, fr_mae[i]),
                       textcoords="offset points", xytext=(10, 5),
                       fontsize=9, color='blue', alpha=0.7)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "layer_wise_mae.png")
    plt.close()
    print("Saved: layer_wise_mae.png")


def plot_per_category_comparison():
    """Per-category MAE comparison at best layer."""
    with open(RESULTS_DIR / "probing_results.json") as f:
        data = json.load(f)

    best_layer = "32"
    cat_order = ["units", "teens", "decimal_tens", "vigesimal_70s", "vigesimal_80s", "vigesimal_90s", "hundreds"]
    cat_labels = ["0-9\n(units)", "10-19\n(teens)", "20-69\n(decimal)", "70-79\n(vig 70s)", "80-89\n(vig 80s)", "90-99\n(vig 90s)", "100-999\n(hundreds)"]

    fr_cats = data["French"][best_layer]["per_category"]
    en_cats = data["English"][best_layer]["per_category"]

    fr_mae = [fr_cats[c]["mean_mae"] for c in cat_order]
    en_mae = [en_cats[c]["mean_mae"] for c in cat_order]

    x = np.arange(len(cat_order))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, fr_mae, width, label='French', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, en_mae, width, label='English', color='seagreen', alpha=0.8)

    # Highlight vigesimal categories
    ax.axvspan(2.5, 5.5, alpha=0.08, color='red', label='Vigesimal zone')

    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=10)
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title(f"Probe Error by Number Category (Layer {best_layer})")
    ax.legend()

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 5:
                ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                       f'{h:.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "per_category_mae.png")
    plt.close()
    print("Saved: per_category_mae.png")


def plot_cross_lingual_similarity():
    """Cross-lingual similarity by category."""
    with open(RESULTS_DIR / "geometry_results.json") as f:
        data = json.load(f)

    cats = data["cross_lingual_similarity"]
    cat_order = ["units", "teens", "decimal_tens", "vigesimal_70s", "vigesimal_80s", "vigesimal_90s"]
    cat_labels = ["0-9", "10-19", "20-69", "70-79", "80-89", "90-99"]

    means = [cats[c]["mean"] for c in cat_order]
    stds = [cats[c]["std"] for c in cat_order]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['steelblue'] * 3 + ['indianred'] * 3
    bars = ax.bar(range(len(cat_order)), means, yerr=stds, capsize=5,
                  color=colors, alpha=0.8)

    ax.set_xticks(range(len(cat_order)))
    ax.set_xticklabels(cat_labels)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Cross-Lingual Embedding Similarity (French ↔ English for Same Number)")
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, label='Random similarity baseline')
    ax.legend()

    # Add value labels
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.01, f'{m:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cross_lingual_similarity.png")
    plt.close()
    print("Saved: cross_lingual_similarity.png")


def plot_decade_proximity():
    """French 70s proximity: are they closer to 60s or 80s?"""
    with open(RESULTS_DIR / "geometry_results.json") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, lang, key in [(axes[0], "French", "french_70s_proximity"),
                          (axes[1], "English", "english_70s_proximity")]:
        to_60s = data[key]["to_60s"]
        to_80s = data[key]["to_80s"]

        bars = ax.bar(["70s ↔ 60s", "70s ↔ 80s"], [to_60s, to_80s],
                      color=['orange', 'purple'], alpha=0.8)
        ax.set_ylabel("Cosine Similarity")
        ax.set_title(f"{lang}: Where do the 70s live?")
        ax.set_ylim(0.7, 1.0)

        # Annotate
        for bar, val in zip(bars, [to_60s, to_80s]):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                   f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')

        # Add arrow showing the "pull"
        diff = to_60s - to_80s
        ax.text(0.5, 0.72, f"Δ = {diff:.4f}",
               ha='center', transform=ax.transAxes, fontsize=11,
               bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle("Do 70-79 cluster with 60s (linguistic) or 80s (numeric)?", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "decade_proximity.png")
    plt.close()
    print("Saved: decade_proximity.png")


def plot_within_decade_cohesion():
    """Compare within-decade cohesion between French and English."""
    with open(RESULTS_DIR / "geometry_results.json") as f:
        data = json.load(f)

    decades = list(range(10))
    fr_coh = [data["fr_decade_cohesion"][str(d)] for d in decades]
    en_coh = [data["en_decade_cohesion"][str(d)] for d in decades]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(10)
    width = 0.35

    ax.bar(x - width/2, fr_coh, width, label='French', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, en_coh, width, label='English', color='seagreen', alpha=0.8)

    ax.axvspan(6.5, 9.5, alpha=0.08, color='red', label='Vigesimal zone')

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d*10}s" for d in decades])
    ax.set_ylabel("Avg Cosine Similarity (within decade)")
    ax.set_title("Within-Decade Cohesion: How Tightly Do Numbers Cluster by Decade?")
    ax.legend()
    ax.set_ylim(0.6, 1.0)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "within_decade_cohesion.png")
    plt.close()
    print("Saved: within_decade_cohesion.png")


def main():
    plot_layer_wise_mae()
    plot_per_category_comparison()
    plot_cross_lingual_similarity()
    plot_decade_proximity()
    plot_within_decade_cohesion()
    print("\nAll final plots generated!")


if __name__ == "__main__":
    main()
