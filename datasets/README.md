# Downloaded Datasets

This directory contains datasets for the research project "How does the model count in French?"
Data files are NOT committed to git due to size. Follow the generation instructions below.

## Dataset 1: French Numbers 0-999

### Overview
- **Source**: Generated using `num2words` Python library
- **Size**: 1000 records (numbers 0-999)
- **Format**: JSONL
- **Task**: Number word mapping with vigesimal categorization
- **Location**: `french_numbers/french_numbers_0_999.jsonl`

### Fields
Each record contains:
- `number`: Integer value (0-999)
- `french`: French number word (e.g., "quatre-vingt-dix-sept")
- `english`: English number word (e.g., "ninety-seven")
- `category`: One of: units, teens, decimal_tens, soixante, vigesimal_70s, vigesimal_80s, vigesimal_90s, hundreds_1, hundreds_2_9
- `fr_word_count`: Number of words in French form
- `en_word_count`: Number of words in English form
- `is_vigesimal`: Boolean — true for numbers 70-99

### Generation Instructions

```python
from num2words import num2words
import json

records = []
for n in range(1000):
    fr = num2words(n, lang='fr')
    en = num2words(n, lang='en')
    records.append({"number": n, "french": fr, "english": en})

with open('datasets/french_numbers/french_numbers_0_999.jsonl', 'w') as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
```

**Dependencies**: `pip install num2words`

### Key Properties of French Counting System
- **1-69**: Regular decimal system (like English but with French words)
- **70-79** (vigesimal_70s): "soixante-dix" (60+10), "soixante et onze" (60+11), etc.
- **80-89** (vigesimal_80s): "quatre-vingts" (4×20), "quatre-vingt-un" (4×20+1), etc.
- **90-99** (vigesimal_90s): "quatre-vingt-dix" (4×20+10), "quatre-vingt-onze" (4×20+11), etc.

### Sample Data
See `french_numbers/samples.json` for examples.

## Dataset 2: Comparison Pairs

### Overview
- **Source**: Generated
- **Size**: 4994 pairs
- **Format**: JSONL
- **Task**: Binary classification (which number is larger?)
- **Location**: `french_numbers/comparison_pairs.jsonl`

### Fields
- `num1`, `num2`: Integer values
- `french1`, `french2`: French number words
- `english1`, `english2`: English number words
- `label`: 0 if num1 > num2, 1 if num1 < num2
- `num1_is_vigesimal`, `num2_is_vigesimal`: Whether numbers fall in 70-99 range

### Generation Instructions
```python
import random
from num2words import num2words

random.seed(42)
pairs = []
for _ in range(5000):
    a, b = random.randint(1, 999), random.randint(1, 999)
    if a != b:
        pairs.append({
            "num1": a, "num2": b,
            "french1": num2words(a, lang='fr'),
            "french2": num2words(b, lang='fr'),
            "label": 0 if a > b else 1
        })
```

## Dataset 3: Counting Sequences

### Overview
- **Source**: Generated
- **Size**: 20 sequences of 10 consecutive numbers each
- **Format**: JSONL
- **Task**: Next-number prediction, boundary analysis
- **Location**: `french_numbers/counting_sequences.jsonl`

### Notes
- Sequences span ranges 0-9, 5-14, 10-19, ..., 95-104
- `crosses_vigesimal` flag indicates if sequence crosses the 70-99 boundary
- Useful for testing whether models can count correctly through vigesimal transitions
