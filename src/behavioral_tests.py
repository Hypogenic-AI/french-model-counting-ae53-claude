"""
Experiment 2: Behavioral testing of LLM French counting via API.

Tests GPT-4.1 on:
1. Number-to-digit conversion ("What number is quatre-vingt-dix-sept?" -> 97)
2. Next number prediction ("What comes after soixante-neuf?" -> soixante-dix)
3. Number comparison ("Which is larger: X or Y?")
4. Counting through vigesimal boundaries
"""

import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
from openai import OpenAI
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-4.1"


def load_number_data():
    data_path = PROJECT_ROOT / "datasets" / "french_numbers" / "french_numbers_0_999.jsonl"
    records = []
    with open(data_path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def categorize_number(n):
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


def call_api(prompt, system="You are a helpful assistant. Answer concisely.", temperature=0.0):
    """Call GPT-4.1 with retry logic."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=100,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return f"ERROR: {e}"


def extract_number(text):
    """Extract a number from model response text."""
    # Try to find a number in the response
    nums = re.findall(r'\b(\d+)\b', text)
    if nums:
        return int(nums[0])
    return None


def test_number_to_digit(records, sample_size=200):
    """Test: Given a French number word, what digit does it represent?"""
    print("\n--- Test 1: French Number → Digit Conversion ---")

    # Sample strategically: oversample vigesimal range
    vigesimal = [r for r in records if 70 <= r["number"] < 100]
    decimal = [r for r in records if 20 <= r["number"] < 70]
    other = [r for r in records if r["number"] < 20 or r["number"] >= 100]

    # Take all vigesimal (30), sample from others
    sampled = vigesimal.copy()
    remaining = sample_size - len(sampled)
    sampled += random.sample(decimal, min(remaining // 2, len(decimal)))
    sampled += random.sample(other, min(remaining - len(sampled) + len(vigesimal), len(other)))
    random.shuffle(sampled)

    results = []
    for r in tqdm(sampled, desc="Number→Digit"):
        prompt = f"What number (in digits) does the French word \"{r['french']}\" represent? Reply with just the number."
        response = call_api(prompt)
        predicted = extract_number(response)
        correct = (predicted == r["number"])

        results.append({
            "number": r["number"],
            "french": r["french"],
            "category": categorize_number(r["number"]),
            "response": response,
            "predicted": predicted,
            "correct": correct,
        })

    return results


def test_next_number(records, sample_size=100):
    """Test: What comes after X in French counting?"""
    print("\n--- Test 2: Next Number Prediction ---")

    # Focus on boundary numbers
    boundary_numbers = list(range(68, 101)) + list(range(8, 22)) + list(range(38, 42)) + list(range(58, 62))
    # Add some random numbers
    others = random.sample(range(1, 998), min(sample_size - len(boundary_numbers), 60))
    test_numbers = sorted(set(boundary_numbers + others))

    number_map = {r["number"]: r["french"] for r in records}

    results = []
    for n in tqdm(test_numbers[:sample_size], desc="Next number"):
        if n >= 999:
            continue
        fr_word = number_map.get(n, "")
        expected_next = number_map.get(n + 1, "")
        if not fr_word or not expected_next:
            continue

        prompt = f"In French counting, what number comes after \"{fr_word}\"? Reply with just the French number word."
        response = call_api(prompt)

        # Normalize for comparison
        response_clean = response.lower().strip().strip('"').strip("'").strip(".")
        expected_clean = expected_next.lower().strip()

        # Check if the response contains the digit
        predicted_digit = extract_number(response)
        correct_digit = (predicted_digit == n + 1) if predicted_digit is not None else False
        correct_word = (response_clean == expected_clean)

        results.append({
            "number": n,
            "french": fr_word,
            "expected_next": expected_next,
            "response": response,
            "category": categorize_number(n),
            "correct_word": correct_word,
            "correct_digit": correct_digit,
            "response_clean": response_clean,
        })

    return results


def test_comparison(records, sample_size=150):
    """Test: Which of two French numbers is larger?"""
    print("\n--- Test 3: Number Comparison ---")

    number_map = {r["number"]: r["french"] for r in records}

    # Generate pairs with strategic sampling
    pairs = []
    # Vigesimal vs decimal pairs
    for _ in range(50):
        a = random.randint(70, 99)
        b = random.randint(20, 69)
        pairs.append((a, b) if random.random() > 0.5 else (b, a))
    # Within vigesimal
    for _ in range(30):
        a, b = random.sample(range(70, 100), 2)
        pairs.append((a, b))
    # Random pairs
    for _ in range(sample_size - 80):
        a, b = random.sample(range(1, 999), 2)
        pairs.append((a, b))

    random.shuffle(pairs)

    results = []
    for a, b in tqdm(pairs[:sample_size], desc="Comparison"):
        fr_a = number_map.get(a, str(a))
        fr_b = number_map.get(b, str(b))

        prompt = f'Which is the larger number: "{fr_a}" or "{fr_b}"? Reply with just the larger French number word.'
        response = call_api(prompt)
        response_clean = response.lower().strip().strip('"').strip("'").strip(".")

        correct_answer = fr_a if a > b else fr_b
        # Check if response matches either number word
        predicted_larger = None
        if fr_a.lower() in response_clean:
            predicted_larger = a
        elif fr_b.lower() in response_clean:
            predicted_larger = b
        else:
            # Try digit extraction
            num = extract_number(response)
            if num == a:
                predicted_larger = a
            elif num == b:
                predicted_larger = b

        actual_larger = max(a, b)
        correct = (predicted_larger == actual_larger)

        # Categorize this pair
        pair_type = "both_decimal"
        if 70 <= a < 100 and 70 <= b < 100:
            pair_type = "both_vigesimal"
        elif 70 <= a < 100 or 70 <= b < 100:
            pair_type = "mixed"

        results.append({
            "num_a": a,
            "num_b": b,
            "french_a": fr_a,
            "french_b": fr_b,
            "response": response,
            "predicted_larger": predicted_larger,
            "actual_larger": actual_larger,
            "correct": correct,
            "pair_type": pair_type,
            "category_a": categorize_number(a),
            "category_b": categorize_number(b),
        })

    return results


def test_counting_sequences():
    """Test: Can the model count correctly through vigesimal boundaries?"""
    print("\n--- Test 4: Counting Sequences ---")

    sequences_path = PROJECT_ROOT / "datasets" / "french_numbers" / "counting_sequences.jsonl"
    sequences = []
    with open(sequences_path) as f:
        for line in f:
            sequences.append(json.loads(line))

    results = []
    for seq in tqdm(sequences, desc="Counting"):
        fr_words = seq["french_sequence"]
        start = seq["start"]

        # Give first 5, ask for next 5
        given = ", ".join(fr_words[:5])
        expected = fr_words[5:] if len(fr_words) > 5 else []
        if not expected:
            continue

        prompt = f"Continue this French counting sequence with the next {len(expected)} numbers: {given}\nReply with just the French number words separated by commas."
        response = call_api(prompt)

        # Parse response
        predicted_words = [w.strip().strip('"').strip("'") for w in response.split(",")]

        correct_count = 0
        for i, exp in enumerate(expected):
            if i < len(predicted_words):
                if predicted_words[i].lower().strip() == exp.lower().strip():
                    correct_count += 1

        results.append({
            "start": start,
            "given": given,
            "expected": expected,
            "response": response,
            "predicted_words": predicted_words,
            "correct_count": correct_count,
            "total": len(expected),
            "crosses_vigesimal": seq.get("crosses_vigesimal", False),
        })

    return results


def main():
    print("=" * 60)
    print("Experiment 2: Behavioral Testing via API")
    print(f"Model: {MODEL}")
    print("=" * 60)

    records = load_number_data()

    # Run all tests
    conversion_results = test_number_to_digit(records, sample_size=200)
    next_results = test_next_number(records, sample_size=100)
    comparison_results = test_comparison(records, sample_size=150)
    counting_results = test_counting_sequences()

    # Analyze results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Test 1: Conversion
    conv_correct = sum(1 for r in conversion_results if r["correct"])
    print(f"\n1. Number→Digit Conversion: {conv_correct}/{len(conversion_results)} ({100*conv_correct/len(conversion_results):.1f}%)")
    for cat in ["units", "teens", "decimal_tens", "vigesimal_70s", "vigesimal_80s", "vigesimal_90s", "hundreds"]:
        cat_results = [r for r in conversion_results if r["category"] == cat]
        if cat_results:
            cat_correct = sum(1 for r in cat_results if r["correct"])
            print(f"   {cat}: {cat_correct}/{len(cat_results)} ({100*cat_correct/len(cat_results):.1f}%)")

    # Test 2: Next number
    next_correct_word = sum(1 for r in next_results if r["correct_word"])
    next_correct_digit = sum(1 for r in next_results if r["correct_digit"])
    print(f"\n2. Next Number Prediction:")
    print(f"   Word-exact match: {next_correct_word}/{len(next_results)} ({100*next_correct_word/len(next_results):.1f}%)")
    print(f"   Digit match: {next_correct_digit}/{len(next_results)} ({100*next_correct_digit/len(next_results):.1f}%)")
    for cat in ["units", "teens", "decimal_tens", "vigesimal_70s", "vigesimal_80s", "vigesimal_90s", "hundreds"]:
        cat_results = [r for r in next_results if r["category"] == cat]
        if cat_results:
            cat_correct = sum(1 for r in cat_results if r["correct_word"])
            print(f"   {cat}: {cat_correct}/{len(cat_results)} ({100*cat_correct/len(cat_results):.1f}%)")

    # Test 3: Comparison
    comp_correct = sum(1 for r in comparison_results if r["correct"])
    print(f"\n3. Number Comparison: {comp_correct}/{len(comparison_results)} ({100*comp_correct/len(comparison_results):.1f}%)")
    for ptype in ["both_decimal", "both_vigesimal", "mixed"]:
        pt_results = [r for r in comparison_results if r["pair_type"] == ptype]
        if pt_results:
            pt_correct = sum(1 for r in pt_results if r["correct"])
            print(f"   {ptype}: {pt_correct}/{len(pt_results)} ({100*pt_correct/len(pt_results):.1f}%)")

    # Test 4: Counting
    count_mean = np.mean([r["correct_count"] / r["total"] for r in counting_results])
    print(f"\n4. Counting Sequences: {count_mean*100:.1f}% of numbers correct")
    vig_seqs = [r for r in counting_results if r["crosses_vigesimal"]]
    non_vig_seqs = [r for r in counting_results if not r["crosses_vigesimal"]]
    if vig_seqs:
        vig_mean = np.mean([r["correct_count"] / r["total"] for r in vig_seqs])
        print(f"   Crossing vigesimal boundary: {vig_mean*100:.1f}%")
    if non_vig_seqs:
        non_vig_mean = np.mean([r["correct_count"] / r["total"] for r in non_vig_seqs])
        print(f"   No vigesimal boundary: {non_vig_mean*100:.1f}%")

    # Save all results
    all_behavioral = {
        "model": MODEL,
        "seed": SEED,
        "conversion": conversion_results,
        "next_number": next_results,
        "comparison": comparison_results,
        "counting": counting_results,
        "summary": {
            "conversion_accuracy": conv_correct / len(conversion_results),
            "next_number_word_accuracy": next_correct_word / len(next_results),
            "next_number_digit_accuracy": next_correct_digit / len(next_results),
            "comparison_accuracy": comp_correct / len(comparison_results),
            "counting_accuracy": count_mean,
        }
    }

    results_path = RESULTS_DIR / "behavioral_results.json"
    with open(results_path, "w") as f:
        json.dump(all_behavioral, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_path}")

    print("\nExperiment 2 complete!")


if __name__ == "__main__":
    main()
