"""
Comprehensive analysis script that:
1. Re-runs proper statistical tests on probing results
2. Fixes and runs counting sequences behavioral test
3. Generates all visualizations
4. Computes all statistics needed for REPORT.md
"""

import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
EMBEDDINGS_DIR = RESULTS_DIR / "embeddings"


def load_number_data():
    data_path = PROJECT_ROOT / "datasets" / "french_numbers" / "french_numbers_0_999.jsonl"
    records = []
    with open(data_path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def categorize_number(n):
    if n < 10: return "units"
    elif n < 20: return "teens"
    elif n < 70: return "decimal_tens"
    elif n < 80: return "vigesimal_70s"
    elif n < 90: return "vigesimal_80s"
    elif n < 100: return "vigesimal_90s"
    else: return "hundreds"


def run_proper_statistical_tests():
    """Re-run statistical tests using the correct layer and per-number errors."""
    print("=" * 60)
    print("Proper Statistical Tests on Probing Results")
    print("=" * 60)

    records = load_number_data()
    numbers = np.array([r["number"] for r in records])
    categories = np.array([categorize_number(n) for n in numbers])

    # Load embeddings for all languages at the best layer
    # Use layer with lowest MAE for French, which appears to be layer 32
    # But also compare at the layer where R² peaks for French
    with open(RESULTS_DIR / "probing_results.json") as f:
        probe_results = json.load(f)

    # Find best layer for French by lowest MAE
    fr_layers = probe_results["French"]
    best_layer_fr = min(fr_layers.keys(), key=lambda l: fr_layers[l]["mae_mean"])
    print(f"Best French layer (lowest MAE): {best_layer_fr} (MAE={fr_layers[best_layer_fr]['mae_mean']:.2f})")

    # Also check English
    en_layers = probe_results["English"]
    best_layer_en = min(en_layers.keys(), key=lambda l: en_layers[l]["mae_mean"])
    print(f"Best English layer (lowest MAE): {best_layer_en} (MAE={en_layers[best_layer_en]['mae_mean']:.2f})")

    # Use the last available layer for fair comparison
    available_layers = sorted([int(l) for l in fr_layers.keys()])
    comparison_layer = available_layers[-1]  # Last layer
    print(f"\nUsing layer {comparison_layer} for cross-lingual comparison")

    # Load embeddings
    fr_emb = np.load(EMBEDDINGS_DIR / f"french_layer{comparison_layer}.npy")
    en_emb = np.load(EMBEDDINGS_DIR / f"english_layer{comparison_layer}.npy")
    digit_emb = np.load(EMBEDDINGS_DIR / f"digits_layer{comparison_layer}.npy")

    # Get per-number cross-val predictions
    results = {}
    for lang, emb in [("French", fr_emb), ("English", en_emb), ("Digits", digit_emb)]:
        probe = Ridge(alpha=1.0)
        y_pred = cross_val_predict(probe, emb, numbers, cv=5)
        abs_errors = np.abs(numbers - y_pred)
        results[lang] = {
            "predictions": y_pred,
            "abs_errors": abs_errors,
        }
        print(f"\n{lang} (Layer {comparison_layer}):")
        print(f"  Overall MAE: {abs_errors.mean():.2f} ± {abs_errors.std():.2f}")

    # Per-category analysis (the key result)
    cat_order = ["units", "teens", "decimal_tens", "vigesimal_70s", "vigesimal_80s", "vigesimal_90s", "hundreds"]
    cat_labels = ["0-9", "10-19", "20-69", "70-79", "80-89", "90-99", "100-999"]

    print("\n" + "-" * 60)
    print("Per-Category MAE Comparison")
    print("-" * 60)
    print(f"{'Category':>15} {'French':>12} {'English':>12} {'Digits':>12} {'Fr/En Ratio':>12}")
    print("-" * 63)

    category_stats = {}
    for cat, label in zip(cat_order, cat_labels):
        mask = categories == cat
        fr_mae = results["French"]["abs_errors"][mask].mean()
        en_mae = results["English"]["abs_errors"][mask].mean()
        dg_mae = results["Digits"]["abs_errors"][mask].mean()
        ratio = fr_mae / en_mae if en_mae > 0 else float('inf')
        print(f"{label:>15} {fr_mae:>10.2f}   {en_mae:>10.2f}   {dg_mae:>10.2f}   {ratio:>10.2f}x")
        category_stats[cat] = {
            "french_mae": float(fr_mae),
            "english_mae": float(en_mae),
            "digits_mae": float(dg_mae),
            "ratio": float(ratio),
            "n": int(mask.sum()),
        }

    # Statistical tests
    print("\n" + "-" * 60)
    print("Statistical Tests")
    print("-" * 60)

    # H1: Vigesimal errors > Decimal errors (French)
    vig_mask = np.isin(categories, ["vigesimal_70s", "vigesimal_80s", "vigesimal_90s"])
    dec_mask = categories == "decimal_tens"

    fr_vig = results["French"]["abs_errors"][vig_mask]
    fr_dec = results["French"]["abs_errors"][dec_mask]
    en_vig = results["English"]["abs_errors"][vig_mask]
    en_dec = results["English"]["abs_errors"][dec_mask]

    print("\nH1: French vigesimal errors > French decimal errors")
    u1, p1 = stats.mannwhitneyu(fr_vig, fr_dec, alternative="greater")
    print(f"  French Vigesimal MAE: {fr_vig.mean():.2f} ± {fr_vig.std():.2f} (n={len(fr_vig)})")
    print(f"  French Decimal MAE: {fr_dec.mean():.2f} ± {fr_dec.std():.2f} (n={len(fr_dec)})")
    print(f"  Mann-Whitney U: {u1:.1f}, p = {p1:.6f}")
    cohens_d1 = (fr_vig.mean() - fr_dec.mean()) / np.sqrt((fr_vig.std()**2 + fr_dec.std()**2) / 2)
    print(f"  Cohen's d: {cohens_d1:.4f}")

    print("\nControl: English vigesimal errors vs decimal errors")
    u1e, p1e = stats.mannwhitneyu(en_vig, en_dec, alternative="greater")
    print(f"  English Vigesimal MAE: {en_vig.mean():.2f} ± {en_vig.std():.2f}")
    print(f"  English Decimal MAE: {en_dec.mean():.2f} ± {en_dec.std():.2f}")
    print(f"  Mann-Whitney U: {u1e:.1f}, p = {p1e:.6f}")

    # H2: French errors > English errors in vigesimal range
    print("\nFrench vs English on vigesimal range (70-99)")
    u2, p2 = stats.mannwhitneyu(fr_vig, en_vig, alternative="greater")
    print(f"  French vigesimal MAE: {fr_vig.mean():.2f}")
    print(f"  English vigesimal MAE: {en_vig.mean():.2f}")
    print(f"  Mann-Whitney U: {u2:.1f}, p = {p2:.6f}")
    cohens_d2 = (fr_vig.mean() - en_vig.mean()) / np.sqrt((fr_vig.std()**2 + en_vig.std()**2) / 2)
    print(f"  Cohen's d: {cohens_d2:.4f}")

    # French vs English overall
    print("\nFrench vs English overall")
    fr_all = results["French"]["abs_errors"]
    en_all = results["English"]["abs_errors"]
    u3, p3 = stats.mannwhitneyu(fr_all, en_all, alternative="greater")
    print(f"  French overall MAE: {fr_all.mean():.2f}")
    print(f"  English overall MAE: {en_all.mean():.2f}")
    print(f"  Mann-Whitney U: {u3:.1f}, p = {p3:.6f}")

    # Layer-wise analysis
    print("\n" + "-" * 60)
    print("Layer-wise MAE Comparison")
    print("-" * 60)
    layer_comparison = {}
    for layer_str in sorted(fr_layers.keys(), key=int):
        layer = int(layer_str)
        fr_mae = fr_layers[layer_str]["mae_mean"]
        en_mae = en_layers[layer_str]["mae_mean"]
        dg_mae = probe_results["Digits"][layer_str]["mae_mean"]
        ratio = fr_mae / en_mae if en_mae > 0 else 0
        print(f"  Layer {layer:2d}: French={fr_mae:7.2f}, English={en_mae:7.2f}, Digits={dg_mae:7.2f}, Fr/En={ratio:.2f}x")
        layer_comparison[layer] = {"french": fr_mae, "english": en_mae, "digits": dg_mae, "ratio": ratio}

    # Correlation: linguistic complexity vs probe error
    print("\n" + "-" * 60)
    print("Complexity-Error Correlation")
    print("-" * 60)
    word_counts = np.array([r["fr_word_count"] for r in records])
    fr_errors = results["French"]["abs_errors"]
    rho, p_rho = stats.spearmanr(word_counts, fr_errors)
    print(f"  Spearman correlation (word count vs error): rho={rho:.4f}, p={p_rho:.6f}")

    # Save updated results
    updated_stats = {
        "layer_used": comparison_layer,
        "category_stats": category_stats,
        "h1_vigesimal_vs_decimal_french": {
            "vigesimal_mae": float(fr_vig.mean()),
            "vigesimal_std": float(fr_vig.std()),
            "decimal_mae": float(fr_dec.mean()),
            "decimal_std": float(fr_dec.std()),
            "mann_whitney_u": float(u1),
            "p_value": float(p1),
            "cohens_d": float(cohens_d1),
        },
        "control_english_vig_vs_dec": {
            "vigesimal_mae": float(en_vig.mean()),
            "decimal_mae": float(en_dec.mean()),
            "p_value": float(p1e),
        },
        "french_vs_english_vigesimal": {
            "french_mae": float(fr_vig.mean()),
            "english_mae": float(en_vig.mean()),
            "mann_whitney_u": float(u2),
            "p_value": float(p2),
            "cohens_d": float(cohens_d2),
        },
        "french_vs_english_overall": {
            "french_mae": float(fr_all.mean()),
            "english_mae": float(en_all.mean()),
            "p_value": float(p3),
        },
        "complexity_correlation": {
            "spearman_rho": float(rho),
            "p_value": float(p_rho),
        },
        "layer_comparison": layer_comparison,
        "per_number_errors": {
            "french": fr_errors.tolist(),
            "english": en_all.tolist(),
            "numbers": numbers.tolist(),
            "categories": categories.tolist(),
        },
    }

    with open(RESULTS_DIR / "statistical_analysis.json", "w") as f:
        json.dump(updated_stats, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'statistical_analysis.json'}")

    return updated_stats


def run_counting_sequences_test():
    """Run the counting sequences behavioral test (fixing the field name issue)."""
    print("\n" + "=" * 60)
    print("Running Counting Sequences Test")
    print("=" * 60)

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    MODEL = "gpt-4.1"

    sequences_path = PROJECT_ROOT / "datasets" / "french_numbers" / "counting_sequences.jsonl"
    sequences = []
    with open(sequences_path) as f:
        for line in f:
            sequences.append(json.loads(line))

    results = []
    for seq in sequences:
        fr_words = seq["french_sequence"]
        start = seq["start"]

        # Give first 5, ask for next 5
        given = ", ".join(fr_words[:5])
        expected = fr_words[5:] if len(fr_words) > 5 else []

        if not expected:
            continue

        prompt = f"Continue this French counting sequence with the next {len(expected)} numbers: {given}\nReply with just the French number words separated by commas."

        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=200,
                )
                resp_text = response.choices[0].message.content.strip()
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    resp_text = f"ERROR: {e}"

        # Parse response
        predicted_words = [w.strip().strip('"').strip("'") for w in resp_text.split(",")]

        correct_count = 0
        for i, exp in enumerate(expected):
            if i < len(predicted_words):
                if predicted_words[i].lower().strip() == exp.lower().strip():
                    correct_count += 1

        results.append({
            "start": start,
            "given": given,
            "expected": expected,
            "response": resp_text,
            "predicted_words": predicted_words,
            "correct_count": correct_count,
            "total": len(expected),
            "crosses_vigesimal": seq.get("crosses_vigesimal", False),
        })
        print(f"  Seq starting at {start}: {correct_count}/{len(expected)} correct, crosses_vig={seq.get('crosses_vigesimal', False)}")

    # Summary
    overall_acc = np.mean([r["correct_count"] / r["total"] for r in results])
    vig_seqs = [r for r in results if r["crosses_vigesimal"]]
    non_vig_seqs = [r for r in results if not r["crosses_vigesimal"]]

    print(f"\n  Overall: {overall_acc*100:.1f}%")
    if vig_seqs:
        vig_acc = np.mean([r["correct_count"] / r["total"] for r in vig_seqs])
        print(f"  Crosses vigesimal: {vig_acc*100:.1f}% ({len(vig_seqs)} sequences)")
    if non_vig_seqs:
        non_vig_acc = np.mean([r["correct_count"] / r["total"] for r in non_vig_seqs])
        print(f"  No vigesimal crossing: {non_vig_acc*100:.1f}% ({len(non_vig_seqs)} sequences)")

    return results


def update_behavioral_results(counting_results):
    """Merge counting results into the behavioral results file."""
    results_path = RESULTS_DIR / "behavioral_results.json"
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
    else:
        data = {}

    data["counting"] = counting_results

    # Recompute counting summary
    if counting_results:
        count_mean = np.mean([r["correct_count"] / r["total"] for r in counting_results])
        data.setdefault("summary", {})["counting_accuracy"] = float(count_mean)

    with open(results_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Updated {results_path}")


def main():
    # 1. Statistical analysis of probing results
    stats_results = run_proper_statistical_tests()

    # 2. Run counting sequences test
    counting_results = run_counting_sequences_test()
    update_behavioral_results(counting_results)

    # 3. Print behavioral summary
    results_path = RESULTS_DIR / "behavioral_results.json"
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)

        print("\n" + "=" * 60)
        print("Behavioral Test Summary (GPT-4.1)")
        print("=" * 60)

        summary = data.get("summary", {})
        for key, val in summary.items():
            print(f"  {key}: {val*100:.1f}%" if isinstance(val, float) else f"  {key}: {val}")

    print("\n\nAll analyses complete!")


if __name__ == "__main__":
    main()
