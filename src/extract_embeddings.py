"""
Experiment 1: Extract hidden-state embeddings for French and English number words
from a multilingual LLM (Mistral 7B), then train linear probes to predict
numerical value from these embeddings.

This script:
1. Loads a multilingual model
2. Extracts embeddings for numbers 0-999 in French and English
3. Trains linear probes (ridge regression) to predict number value
4. Compares accuracy across vigesimal vs decimal subranges
5. Performs layer-wise analysis
"""

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from scipy import stats
from tqdm import tqdm

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
EMBEDDINGS_DIR = RESULTS_DIR / "embeddings"

# Model to use - Mistral 7B is multilingual and well-studied
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"


def load_number_data():
    """Load the pre-generated French numbers dataset."""
    data_path = PROJECT_ROOT / "datasets" / "french_numbers" / "french_numbers_0_999.jsonl"
    records = []
    with open(data_path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} number records")
    return records


def categorize_number(n):
    """Assign a category to a number for subrange analysis."""
    if n < 10:
        return "units"
    elif n < 20:
        return "teens"
    elif n < 70:
        return "decimal_tens"
    elif n < 80:
        return "vigesimal_70s"
    elif n < 90:
        return "vigesimal_80s"
    elif n < 100:
        return "vigesimal_90s"
    else:
        return "hundreds"


def extract_embeddings(model, tokenizer, texts, batch_size=32, layers_to_extract=None):
    """
    Extract hidden-state embeddings for a list of texts.
    Uses mean pooling over tokens for each text.
    Returns dict mapping layer_idx -> np.array of shape (len(texts), hidden_dim).
    """
    if layers_to_extract is None:
        layers_to_extract = list(range(0, model.config.num_hidden_layers + 1, 4))
        # Include last layer
        if model.config.num_hidden_layers not in layers_to_extract:
            layers_to_extract.append(model.config.num_hidden_layers)

    all_embeddings = {layer: [] for layer in layers_to_extract}

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64
            ).to(DEVICE)

            outputs = model(
                **inputs,
                output_hidden_states=True
            )

            # outputs.hidden_states is a tuple of (num_layers+1) tensors
            # Each tensor is (batch, seq_len, hidden_dim)
            attention_mask = inputs["attention_mask"]

            for layer_idx in layers_to_extract:
                hidden = outputs.hidden_states[layer_idx]  # (batch, seq, dim)
                # Mean pooling over non-padding tokens
                mask = attention_mask.unsqueeze(-1).float()  # (batch, seq, 1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (batch, dim)
                all_embeddings[layer_idx].append(pooled.cpu().numpy())

    # Concatenate all batches
    for layer_idx in layers_to_extract:
        all_embeddings[layer_idx] = np.concatenate(all_embeddings[layer_idx], axis=0)

    return all_embeddings


def train_probes(embeddings, values, categories):
    """
    Train linear probes (Ridge regression) to predict numerical value from embeddings.
    Uses 5-fold cross-validation. Returns per-fold and per-category metrics.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    values = np.array(values)
    categories = np.array(categories)

    fold_results = []
    per_category_errors = {cat: [] for cat in set(categories)}

    for fold, (train_idx, test_idx) in enumerate(kf.split(embeddings)):
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = values[train_idx], values[test_idx]
        cats_test = categories[test_idx]

        # Ridge regression
        probe = Ridge(alpha=1.0)
        probe.fit(X_train, y_train)
        y_pred = probe.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        fold_results.append({"fold": fold, "mae": mae, "r2": r2})

        # Per-category analysis
        for cat in set(categories):
            mask = cats_test == cat
            if mask.sum() > 0:
                cat_mae = mean_absolute_error(y_test[mask], y_pred[mask])
                per_category_errors[cat].append(cat_mae)

    return fold_results, per_category_errors


def main():
    print("=" * 60)
    print("Experiment 1: Embedding Extraction & Linear Probing")
    print("=" * 60)

    # Load data
    records = load_number_data()
    numbers = [r["number"] for r in records]
    french_words = [r["french"] for r in records]
    english_words = [r["english"] for r in records]
    categories = [categorize_number(n) for n in numbers]

    print(f"\nCategory distribution:")
    for cat in sorted(set(categories)):
        count = categories.count(cat)
        print(f"  {cat}: {count}")

    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map={"": DEVICE},
        output_hidden_states=True,
    )
    print(f"Model loaded. Layers: {model.config.num_hidden_layers}, Hidden dim: {model.config.hidden_size}")

    # Define layers to probe
    n_layers = model.config.num_hidden_layers
    layers_to_extract = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]
    print(f"Extracting embeddings from layers: {layers_to_extract}")

    # Extract embeddings for French and English
    print("\n--- Extracting French embeddings ---")
    fr_embeddings = extract_embeddings(model, tokenizer, french_words, batch_size=64, layers_to_extract=layers_to_extract)

    print("\n--- Extracting English embeddings ---")
    en_embeddings = extract_embeddings(model, tokenizer, english_words, batch_size=64, layers_to_extract=layers_to_extract)

    # Also extract digit-form embeddings as ceiling baseline
    digit_strings = [str(n) for n in numbers]
    print("\n--- Extracting digit embeddings ---")
    digit_embeddings = extract_embeddings(model, tokenizer, digit_strings, batch_size=64, layers_to_extract=layers_to_extract)

    # Save embeddings
    print("\nSaving embeddings...")
    for lang, embs in [("french", fr_embeddings), ("english", en_embeddings), ("digits", digit_embeddings)]:
        for layer_idx, emb in embs.items():
            np.save(EMBEDDINGS_DIR / f"{lang}_layer{layer_idx}.npy", emb)

    # Train probes and collect results
    print("\n" + "=" * 60)
    print("Training linear probes (Ridge regression, 5-fold CV)")
    print("=" * 60)

    all_results = {}
    for lang, embs in [("French", fr_embeddings), ("English", en_embeddings), ("Digits", digit_embeddings)]:
        print(f"\n--- {lang} ---")
        lang_results = {}
        for layer_idx in layers_to_extract:
            fold_results, cat_errors = train_probes(embs[layer_idx], numbers, categories)

            mean_mae = np.mean([r["mae"] for r in fold_results])
            std_mae = np.std([r["mae"] for r in fold_results])
            mean_r2 = np.mean([r["r2"] for r in fold_results])
            std_r2 = np.std([r["r2"] for r in fold_results])

            print(f"  Layer {layer_idx:2d}: MAE = {mean_mae:.2f} ± {std_mae:.2f}, R² = {mean_r2:.4f} ± {std_r2:.4f}")

            cat_summary = {}
            for cat, errors in cat_errors.items():
                if errors:
                    cat_summary[cat] = {"mean_mae": float(np.mean(errors)), "std_mae": float(np.std(errors))}

            lang_results[layer_idx] = {
                "mae_mean": float(mean_mae),
                "mae_std": float(std_mae),
                "r2_mean": float(mean_r2),
                "r2_std": float(std_r2),
                "per_category": cat_summary,
                "fold_results": fold_results
            }

        all_results[lang] = lang_results

    # Statistical tests: vigesimal vs decimal
    print("\n" + "=" * 60)
    print("Statistical comparison: Vigesimal vs Decimal (French)")
    print("=" * 60)

    best_layer = max(layers_to_extract, key=lambda l: all_results["French"][l]["r2_mean"])
    print(f"Using best layer: {best_layer}")

    fr_best = fr_embeddings[best_layer]
    values_arr = np.array(numbers)
    cats_arr = np.array(categories)

    # Get per-number probe errors
    from sklearn.model_selection import cross_val_predict
    probe = Ridge(alpha=1.0)
    y_pred_cv = cross_val_predict(probe, fr_best, values_arr, cv=5)
    abs_errors = np.abs(values_arr - y_pred_cv)

    vigesimal_mask = np.isin(cats_arr, ["vigesimal_70s", "vigesimal_80s", "vigesimal_90s"])
    decimal_mask = np.isin(cats_arr, ["decimal_tens"])

    vig_errors = abs_errors[vigesimal_mask]
    dec_errors = abs_errors[decimal_mask]

    print(f"  Vigesimal (70-99) MAE: {vig_errors.mean():.2f} ± {vig_errors.std():.2f} (n={len(vig_errors)})")
    print(f"  Decimal (20-69) MAE: {dec_errors.mean():.2f} ± {dec_errors.std():.2f} (n={len(dec_errors)})")

    # Mann-Whitney U test (non-parametric)
    u_stat, p_value = stats.mannwhitneyu(vig_errors, dec_errors, alternative="greater")
    effect_size = u_stat / (len(vig_errors) * len(dec_errors))  # rank-biserial
    print(f"  Mann-Whitney U: U={u_stat:.1f}, p={p_value:.6f}")
    print(f"  Rank-biserial effect size: {effect_size:.4f}")

    # Also compare French vs English on vigesimal range
    en_best = en_embeddings[best_layer]
    probe_en = Ridge(alpha=1.0)
    y_pred_en = cross_val_predict(probe_en, en_best, values_arr, cv=5)
    en_abs_errors = np.abs(values_arr - y_pred_en)

    fr_vig_errors = abs_errors[vigesimal_mask]
    en_vig_errors = en_abs_errors[vigesimal_mask]

    print(f"\n  French vigesimal MAE: {fr_vig_errors.mean():.2f}")
    print(f"  English vigesimal MAE: {en_vig_errors.mean():.2f}")
    u2, p2 = stats.mannwhitneyu(fr_vig_errors, en_vig_errors, alternative="greater")
    print(f"  Mann-Whitney U (Fr > En on vigesimal): U={u2:.1f}, p={p2:.6f}")

    # Save all results
    results_path = RESULTS_DIR / "probing_results.json"

    # Convert numpy types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = json.loads(json.dumps(all_results, default=convert_types))
    serializable["statistical_tests"] = {
        "vigesimal_vs_decimal": {
            "vigesimal_mae_mean": float(vig_errors.mean()),
            "vigesimal_mae_std": float(vig_errors.std()),
            "decimal_mae_mean": float(dec_errors.mean()),
            "decimal_mae_std": float(dec_errors.std()),
            "mann_whitney_u": float(u_stat),
            "p_value": float(p_value),
            "effect_size": float(effect_size),
        },
        "french_vs_english_vigesimal": {
            "french_mae": float(fr_vig_errors.mean()),
            "english_mae": float(en_vig_errors.mean()),
            "mann_whitney_u": float(u2),
            "p_value": float(p2),
        },
        "best_layer": best_layer,
    }

    # Save per-number errors for later analysis
    serializable["per_number_errors"] = {
        "french": abs_errors.tolist(),
        "english": en_abs_errors.tolist(),
        "numbers": numbers,
        "categories": categories,
    }

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()

    print("\nExperiment 1 complete!")


if __name__ == "__main__":
    main()
