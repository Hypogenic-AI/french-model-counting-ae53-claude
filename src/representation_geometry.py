"""
Experiment 3: Representation geometry analysis.

Visualizes how French vs English number words are organized in embedding space:
1. PCA visualization of number embeddings (French vs English)
2. Cosine similarity heatmaps between consecutive numbers
3. Decade clustering analysis
4. "Number line" smoothness analysis
5. Per-number probe error visualization
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

SEED = 42
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
EMBEDDINGS_DIR = RESULTS_DIR / "embeddings"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})


def load_number_data():
    data_path = PROJECT_ROOT / "datasets" / "french_numbers" / "french_numbers_0_999.jsonl"
    records = []
    with open(data_path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_embeddings(lang, layer):
    path = EMBEDDINGS_DIR / f"{lang}_layer{layer}.npy"
    return np.load(path)


def categorize_number(n):
    if n < 10: return "units"
    elif n < 20: return "teens"
    elif n < 70: return "decimal_tens"
    elif n < 80: return "vigesimal_70s"
    elif n < 90: return "vigesimal_80s"
    elif n < 100: return "vigesimal_90s"
    else: return "hundreds"


def plot_pca_number_line(fr_emb, en_emb, numbers, title_suffix=""):
    """PCA visualization comparing French and English number embeddings (0-99 only)."""
    mask = np.array(numbers) < 100
    fr_sub = fr_emb[mask]
    en_sub = en_emb[mask]
    nums_sub = np.array(numbers)[mask]

    # Fit PCA on combined embeddings
    combined = np.vstack([fr_sub, en_sub])
    pca = PCA(n_components=2)
    combined_pca = pca.fit_transform(combined)
    fr_pca = combined_pca[:len(fr_sub)]
    en_pca = combined_pca[len(fr_sub):]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Color by number value
    for ax, pca_data, lang in [(axes[0], fr_pca, "French"), (axes[1], en_pca, "English")]:
        scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1],
                           c=nums_sub, cmap='viridis', s=20, alpha=0.7)

        # Highlight vigesimal range
        vig_mask = (nums_sub >= 70) & (nums_sub < 100)
        ax.scatter(pca_data[vig_mask, 0], pca_data[vig_mask, 1],
                  c='red', s=50, marker='x', alpha=0.8, label='Vigesimal (70-99)')

        # Label decade boundaries
        for n in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
            idx = np.where(nums_sub == n)[0]
            if len(idx) > 0:
                ax.annotate(str(n), (pca_data[idx[0], 0], pca_data[idx[0], 1]),
                          fontsize=9, fontweight='bold', color='black',
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        ax.set_title(f"{lang} Number Words (0-99){title_suffix}")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.legend()
        plt.colorbar(scatter, ax=ax, label="Number value")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pca_number_line.png")
    plt.close()
    print(f"  Saved: pca_number_line.png (variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%)")


def plot_decade_clustering(fr_emb, en_emb, numbers):
    """Show how numbers cluster by decade in French vs English (0-99)."""
    mask = np.array(numbers) < 100
    fr_sub = fr_emb[mask]
    en_sub = en_emb[mask]
    nums_sub = np.array(numbers)[mask]

    # Assign decade labels
    decades = nums_sub // 10

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    colors = cm.tab10(np.linspace(0, 1, 10))
    decade_labels = ['0s', '10s', '20s', '30s', '40s', '50s', '60s', '70s', '80s', '90s']

    for ax, emb, lang in [(axes[0], fr_sub, "French"), (axes[1], en_sub, "English")]:
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(emb)

        for d in range(10):
            dmask = decades == d
            ax.scatter(pca_data[dmask, 0], pca_data[dmask, 1],
                      c=[colors[d]], label=decade_labels[d],
                      s=30, alpha=0.7)
            # Draw convex hull or just centroid
            centroid = pca_data[dmask].mean(axis=0)
            ax.annotate(decade_labels[d], centroid, fontsize=10, fontweight='bold',
                       ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor=colors[d], alpha=0.3))

        ax.set_title(f"{lang}: Decade Clustering (PCA)")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "decade_clustering.png")
    plt.close()
    print("  Saved: decade_clustering.png")


def plot_cosine_similarity_heatmap(fr_emb, en_emb, numbers):
    """Cosine similarity between consecutive numbers in French vs English (0-99)."""
    mask = np.array(numbers) < 100
    fr_sub = fr_emb[mask]
    en_sub = en_emb[mask]

    fr_sim = cosine_similarity(fr_sub)
    en_sim = cosine_similarity(en_sub)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    vmin, vmax = min(fr_sim.min(), en_sim.min()), 1.0

    im1 = axes[0].imshow(fr_sim, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0].set_title("French: Cosine Similarity")
    axes[0].set_xlabel("Number")
    axes[0].set_ylabel("Number")
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(en_sim, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
    axes[1].set_title("English: Cosine Similarity")
    axes[1].set_xlabel("Number")
    plt.colorbar(im2, ax=axes[1])

    # Difference
    diff = fr_sim - en_sim
    im3 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-0.3, vmax=0.3, aspect='auto')
    axes[2].set_title("Difference (French - English)")
    axes[2].set_xlabel("Number")
    plt.colorbar(im3, ax=axes[2])

    # Add decade grid lines
    for ax in axes:
        for d in range(0, 100, 10):
            ax.axhline(y=d - 0.5, color='white', linewidth=0.5, alpha=0.5)
            ax.axvline(x=d - 0.5, color='white', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cosine_similarity_heatmap.png")
    plt.close()
    print("  Saved: cosine_similarity_heatmap.png")


def plot_consecutive_similarity(fr_emb, en_emb, numbers):
    """Plot cosine similarity between consecutive numbers — shows 'smoothness' of number line."""
    mask = np.array(numbers) < 100
    fr_sub = fr_emb[mask]
    en_sub = en_emb[mask]
    nums_sub = np.array(numbers)[mask]

    fr_consec_sim = []
    en_consec_sim = []
    for i in range(len(nums_sub) - 1):
        fr_consec_sim.append(1 - cosine(fr_sub[i], fr_sub[i+1]))
        en_consec_sim.append(1 - cosine(en_sub[i], en_sub[i+1]))

    fig, ax = plt.subplots(figsize=(14, 5))
    x = nums_sub[:-1]
    ax.plot(x, fr_consec_sim, 'b-', alpha=0.7, label='French', linewidth=1.5)
    ax.plot(x, en_consec_sim, 'g-', alpha=0.7, label='English', linewidth=1.5)

    # Highlight vigesimal boundaries
    for boundary in [69, 79, 89]:
        ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvspan(69, 99, alpha=0.1, color='red', label='Vigesimal zone')

    ax.set_xlabel("Number n")
    ax.set_ylabel("Cosine similarity (n, n+1)")
    ax.set_title("Consecutive Number Similarity: French vs English")
    ax.legend()
    ax.set_xlim(0, 99)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "consecutive_similarity.png")
    plt.close()
    print("  Saved: consecutive_similarity.png")

    return fr_consec_sim, en_consec_sim


def plot_probe_errors(numbers, categories):
    """Visualize per-number probe errors from Experiment 1."""
    results_path = RESULTS_DIR / "probing_results.json"
    if not results_path.exists():
        print("  Skipping probe error plot (probing_results.json not found)")
        return

    with open(results_path) as f:
        results = json.load(f)

    if "per_number_errors" not in results:
        print("  Skipping probe error plot (no per_number_errors)")
        return

    fr_errors = np.array(results["per_number_errors"]["french"])
    en_errors = np.array(results["per_number_errors"]["english"])

    # Plot errors for 0-99
    mask = np.array(numbers) < 100
    fr_err_sub = fr_errors[mask]
    en_err_sub = en_errors[mask]
    nums_sub = np.array(numbers)[mask]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(nums_sub - 0.2, fr_err_sub, width=0.4, alpha=0.7, label='French', color='blue')
    ax.bar(nums_sub + 0.2, en_err_sub, width=0.4, alpha=0.7, label='English', color='green')

    ax.axvspan(69.5, 99.5, alpha=0.1, color='red', label='Vigesimal zone')

    ax.set_xlabel("Number")
    ax.set_ylabel("Absolute Prediction Error")
    ax.set_title("Linear Probe Error by Number: French vs English")
    ax.legend()
    ax.set_xlim(-1, 100)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "probe_errors_by_number.png")
    plt.close()
    print("  Saved: probe_errors_by_number.png")

    # Also make a summary by category
    cats = np.array(categories)
    cat_order = ["units", "teens", "decimal_tens", "vigesimal_70s", "vigesimal_80s", "vigesimal_90s", "hundreds"]
    cat_labels = ["0-9", "10-19", "20-69", "70-79", "80-89", "90-99", "100-999"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(cat_order))
    fr_means = [fr_errors[cats == c].mean() for c in cat_order]
    en_means = [en_errors[cats == c].mean() for c in cat_order]
    fr_stds = [fr_errors[cats == c].std() for c in cat_order]
    en_stds = [en_errors[cats == c].std() for c in cat_order]

    ax.bar(x_pos - 0.2, fr_means, 0.4, yerr=fr_stds, label='French', color='blue', alpha=0.7, capsize=3)
    ax.bar(x_pos + 0.2, en_means, 0.4, yerr=en_stds, label='English', color='green', alpha=0.7, capsize=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(cat_labels, rotation=45)
    ax.set_xlabel("Number Category")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("Probe Error by Number Category")
    ax.legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "probe_errors_by_category.png")
    plt.close()
    print("  Saved: probe_errors_by_category.png")


def plot_behavioral_results():
    """Visualize behavioral test results from Experiment 2."""
    results_path = RESULTS_DIR / "behavioral_results.json"
    if not results_path.exists():
        print("  Skipping behavioral plot (behavioral_results.json not found)")
        return

    with open(results_path) as f:
        results = json.load(f)

    # Conversion accuracy by category
    conversion = results.get("conversion", [])
    if conversion:
        cat_order = ["units", "teens", "decimal_tens", "vigesimal_70s", "vigesimal_80s", "vigesimal_90s", "hundreds"]
        cat_labels = ["0-9", "10-19", "20-69", "70-79", "80-89", "90-99", "100-999"]

        accuracies = []
        counts = []
        for cat in cat_order:
            cat_items = [r for r in conversion if r["category"] == cat]
            if cat_items:
                acc = sum(1 for r in cat_items if r["correct"]) / len(cat_items)
                accuracies.append(acc * 100)
                counts.append(len(cat_items))
            else:
                accuracies.append(0)
                counts.append(0)

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(range(len(cat_order)), accuracies,
                      color=['steelblue'] * 3 + ['indianred'] * 3 + ['steelblue'],
                      alpha=0.8)

        # Add count labels on bars
        for i, (acc, cnt) in enumerate(zip(accuracies, counts)):
            ax.text(i, acc + 1, f"n={cnt}", ha='center', fontsize=9)

        ax.set_xticks(range(len(cat_order)))
        ax.set_xticklabels(cat_labels, rotation=45)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"French Number→Digit Conversion Accuracy ({results.get('model', 'GPT-4.1')})")
        ax.set_ylim(0, 105)
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "behavioral_conversion_accuracy.png")
        plt.close()
        print("  Saved: behavioral_conversion_accuracy.png")

    # Comparison accuracy by pair type
    comparison = results.get("comparison", [])
    if comparison:
        pair_types = ["both_decimal", "mixed", "both_vigesimal"]
        pair_labels = ["Both Decimal", "Mixed", "Both Vigesimal"]

        accs = []
        ns = []
        for pt in pair_types:
            items = [r for r in comparison if r["pair_type"] == pt]
            if items:
                acc = sum(1 for r in items if r["correct"]) / len(items)
                accs.append(acc * 100)
                ns.append(len(items))
            else:
                accs.append(0)
                ns.append(0)

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(range(len(pair_types)), accs,
                      color=['steelblue', 'goldenrod', 'indianred'], alpha=0.8)
        for i, (acc, n) in enumerate(zip(accs, ns)):
            ax.text(i, acc + 1, f"n={n}", ha='center', fontsize=9)

        ax.set_xticks(range(len(pair_types)))
        ax.set_xticklabels(pair_labels)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"French Number Comparison Accuracy by Pair Type ({results.get('model', 'GPT-4.1')})")
        ax.set_ylim(0, 105)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "behavioral_comparison_accuracy.png")
        plt.close()
        print("  Saved: behavioral_comparison_accuracy.png")


def compute_representation_statistics(fr_emb, en_emb, numbers):
    """Compute quantitative metrics about representation geometry."""
    mask = np.array(numbers) < 100
    fr_sub = fr_emb[mask]
    en_sub = en_emb[mask]
    nums_sub = np.array(numbers)[mask]

    # 1. Cross-lingual similarity: how similar are French and English embeddings for the same number?
    cross_sim = []
    for i in range(len(nums_sub)):
        sim = 1 - cosine(fr_sub[i], en_sub[i])
        cross_sim.append(sim)
    cross_sim = np.array(cross_sim)

    # By category
    cats = np.array([categorize_number(n) for n in nums_sub])
    print("\nCross-lingual similarity (French↔English for same number):")
    for cat in ["units", "teens", "decimal_tens", "vigesimal_70s", "vigesimal_80s", "vigesimal_90s"]:
        mask_cat = cats == cat
        if mask_cat.sum() > 0:
            print(f"  {cat}: {cross_sim[mask_cat].mean():.4f} ± {cross_sim[mask_cat].std():.4f}")

    # 2. Within-decade cohesion (avg similarity within a decade)
    print("\nWithin-decade cohesion (avg cosine similarity):")
    fr_decade_cohesion = {}
    en_decade_cohesion = {}
    for d in range(10):
        dmask = (nums_sub // 10) == d
        fr_d = fr_sub[dmask]
        en_d = en_sub[dmask]
        fr_sim_d = cosine_similarity(fr_d).mean()
        en_sim_d = cosine_similarity(en_d).mean()
        fr_decade_cohesion[d] = float(fr_sim_d)
        en_decade_cohesion[d] = float(en_sim_d)
        print(f"  Decade {d*10}s: French={fr_sim_d:.4f}, English={en_sim_d:.4f}")

    # 3. Is 70-79 closer to 60-69 than to 80-89 in French?
    fr_60s = fr_sub[(nums_sub >= 60) & (nums_sub < 70)]
    fr_70s = fr_sub[(nums_sub >= 70) & (nums_sub < 80)]
    fr_80s = fr_sub[(nums_sub >= 80) & (nums_sub < 90)]

    sim_70_60 = cosine_similarity(fr_70s, fr_60s).mean()
    sim_70_80 = cosine_similarity(fr_70s, fr_80s).mean()
    print(f"\nFrench 70s proximity:")
    print(f"  70s ↔ 60s similarity: {sim_70_60:.4f}")
    print(f"  70s ↔ 80s similarity: {sim_70_80:.4f}")
    print(f"  → 70s are {'closer to 60s' if sim_70_60 > sim_70_80 else 'closer to 80s'}")

    # Same for English (expected: 70s should be equidistant or closer to 80s)
    en_60s = en_sub[(nums_sub >= 60) & (nums_sub < 70)]
    en_70s = en_sub[(nums_sub >= 70) & (nums_sub < 80)]
    en_80s = en_sub[(nums_sub >= 80) & (nums_sub < 90)]

    en_sim_70_60 = cosine_similarity(en_70s, en_60s).mean()
    en_sim_70_80 = cosine_similarity(en_70s, en_80s).mean()
    print(f"\nEnglish 70s proximity:")
    print(f"  70s ↔ 60s similarity: {en_sim_70_60:.4f}")
    print(f"  70s ↔ 80s similarity: {en_sim_70_80:.4f}")
    print(f"  → 70s are {'closer to 60s' if en_sim_70_60 > en_sim_70_80 else 'closer to 80s'}")

    return {
        "cross_lingual_similarity": {cat: {"mean": float(cross_sim[cats == cat].mean()),
                                           "std": float(cross_sim[cats == cat].std())}
                                     for cat in set(cats) if (cats == cat).sum() > 0},
        "fr_decade_cohesion": fr_decade_cohesion,
        "en_decade_cohesion": en_decade_cohesion,
        "french_70s_proximity": {
            "to_60s": float(sim_70_60),
            "to_80s": float(sim_70_80),
        },
        "english_70s_proximity": {
            "to_60s": float(en_sim_70_60),
            "to_80s": float(en_sim_70_80),
        },
    }


def main():
    print("=" * 60)
    print("Experiment 3: Representation Geometry Analysis")
    print("=" * 60)

    records = load_number_data()
    numbers = [r["number"] for r in records]
    categories = [categorize_number(n) for n in numbers]

    # Find available layers
    available_files = list(EMBEDDINGS_DIR.glob("french_layer*.npy"))
    if not available_files:
        print("ERROR: No embedding files found. Run extract_embeddings.py first.")
        return

    available_layers = sorted(set(int(f.stem.split("layer")[1]) for f in available_files))
    print(f"Available layers: {available_layers}")

    # Use the last layer (typically best for semantic content)
    best_layer = available_layers[-1]
    print(f"Using layer {best_layer} for geometric analysis")

    fr_emb = load_embeddings("french", best_layer)
    en_emb = load_embeddings("english", best_layer)
    digit_emb = load_embeddings("digits", best_layer)

    print(f"Embedding shapes: French={fr_emb.shape}, English={en_emb.shape}, Digits={digit_emb.shape}")

    # Generate all plots
    print("\nGenerating visualizations...")

    print("\n1. PCA Number Line")
    plot_pca_number_line(fr_emb, en_emb, numbers)

    print("\n2. Decade Clustering")
    plot_decade_clustering(fr_emb, en_emb, numbers)

    print("\n3. Cosine Similarity Heatmap")
    plot_cosine_similarity_heatmap(fr_emb, en_emb, numbers)

    print("\n4. Consecutive Similarity")
    fr_consec, en_consec = plot_consecutive_similarity(fr_emb, en_emb, numbers)

    print("\n5. Probe Error Visualization")
    plot_probe_errors(numbers, categories)

    print("\n6. Behavioral Results Visualization")
    plot_behavioral_results()

    print("\n7. Representation Statistics")
    geom_stats = compute_representation_statistics(fr_emb, en_emb, numbers)

    # Consecutive similarity stats for vigesimal vs decimal zones
    nums_sub = np.array(numbers[:100])
    fr_consec = np.array(fr_consec[:99])
    en_consec = np.array(en_consec[:99])

    vig_zone = (nums_sub[:-1] >= 69) & (nums_sub[:-1] < 99)
    dec_zone = (nums_sub[:-1] >= 19) & (nums_sub[:-1] < 69)

    print(f"\nConsecutive similarity stats:")
    print(f"  French vigesimal zone: {fr_consec[vig_zone].mean():.4f} ± {fr_consec[vig_zone].std():.4f}")
    print(f"  French decimal zone: {fr_consec[dec_zone].mean():.4f} ± {fr_consec[dec_zone].std():.4f}")
    print(f"  English vigesimal zone: {en_consec[vig_zone].mean():.4f} ± {en_consec[vig_zone].std():.4f}")
    print(f"  English decimal zone: {en_consec[dec_zone].mean():.4f} ± {en_consec[dec_zone].std():.4f}")

    u_stat, p_val = stats.mannwhitneyu(fr_consec[vig_zone], fr_consec[dec_zone])
    print(f"  Mann-Whitney (Fr vig vs dec): U={u_stat:.1f}, p={p_val:.6f}")

    geom_stats["consecutive_similarity"] = {
        "french_vigesimal_mean": float(fr_consec[vig_zone].mean()),
        "french_decimal_mean": float(fr_consec[dec_zone].mean()),
        "english_vigesimal_mean": float(en_consec[vig_zone].mean()),
        "english_decimal_mean": float(en_consec[dec_zone].mean()),
        "mann_whitney_p": float(p_val),
    }

    # Save all geometry stats
    with open(RESULTS_DIR / "geometry_results.json", "w") as f:
        json.dump(geom_stats, f, indent=2)
    print(f"\nGeometry results saved to {RESULTS_DIR / 'geometry_results.json'}")

    print("\nExperiment 3 complete!")


if __name__ == "__main__":
    main()
