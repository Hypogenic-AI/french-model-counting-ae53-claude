# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "How does the model count in French?" — investigating how LLMs represent and process the French vigesimal counting system.

---

## Papers
Total papers downloaded: 24

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | Probing for Multilingual Numerical Understanding | Johnson et al. | 2020 | `papers/2010.06666_multilingual_numerical_probing.pdf` | **Most relevant** — French number probing with BERT/XLM |
| 2 | Language Models Encode Numbers Using Digit Representations in Base 10 | Levy & Geva | 2024 | `papers/2410.11781_digit_representations_base10.pdf` | Circular per-digit probes in base 10 |
| 3 | Unravelling the Mechanisms of Manipulating Numbers in LMs | Štefánik et al. | 2025 | `papers/2510.26285_mechanisms_manipulating_numbers.pdf` | Universal sinusoidal number representations |
| 4 | LLMs Know More About Numbers than They Can Say | Yuchi, Du, Eisner | 2025 | `papers/2602.07812_llms_know_more_numbers.pdf` | Internal knowledge vs verbalized output gap |
| 5 | What is a Number, That an LLM May Know It? | — | 2025 | `papers/2502.01540_what_is_a_number.pdf` | Cognitive science approach to LLM numeracy |
| 6 | Number Representations in LLMs: Human Parallel | — | 2025 | `papers/2502.16147_number_representations_human_parallel.pdf` | Logarithmic spacing like human cognition |
| 7 | Do NLP Models Know Numbers? | Wallace et al. | 2019 | `papers/1909.07940_probing_numeracy_embeddings.pdf` | Foundational numeracy probing work |
| 8 | Representing Numbers in NLP: Survey | Thawani et al. | 2021 | `papers/2103.13136_representing_numbers_nlp_survey.pdf` | Comprehensive numeracy taxonomy |
| 9 | Arithmetic in Transformers Explained | — | 2024 | `papers/2402.02619_arithmetic_in_transformers.pdf` | Mechanistic account of arithmetic circuits |
| 10 | Algorithmic Phase Transitions in LMs | — | 2024 | `papers/2412.07386_algorithmic_phase_transitions.pdf` | Phase transitions in arithmetic mechanisms |
| 11 | Non-literal Understanding of Number Words | — | 2025 | `papers/2502.06204_nonliteral_number_words.pdf` | Pragmatic number word interpretation |
| 12 | Exposing Numeracy Gaps (NumericBench) | — | 2025 | `papers/2502.11075_numeracy_gaps_benchmark.pdf` | Benchmark for fundamental numerical abilities |
| 13 | Pre-trained LMs Learn Accurate Number Representations | Kadlčík et al. | 2025 | `papers/2506.08966_accurate_number_representations.pdf` | Sinusoidal probes achieve near-perfect accuracy |
| 14 | Revealing the Numeracy Gap in Embeddings | — | 2025 | `papers/2509.05691_revealing_numeracy_gap.pdf` | Text embedding models struggle with numbers |
| 15 | A Fragile Number Sense | — | 2025 | `papers/2509.06332_fragile_number_sense.pdf` | Probing limits of LLM numerical reasoning |
| 16 | Efficient Numeracy (BitTokens) | — | 2025 | `papers/2510.06824_efficient_numeracy_bittokens.pdf` | Novel single-token number embeddings |
| 17 | Faith and Fate: Compositionality Limits | Dziri et al. | 2023 | `papers/2305.18654_faith_fate_compositionality.pdf` | Transformer limits on compositional tasks |
| 18 | Teaching Arithmetic to Small Transformers | — | 2023 | `papers/2307.03381_teaching_arithmetic_transformers.pdf` | Training transformers on arithmetic |
| 19 | Arithmetic: Memorization to Computation | Maltoni & Ferrara | 2023 | `papers/2305.14201_arithmetic_memorization_computation.pdf` | Memorization vs computation in arithmetic |
| 20 | Numeracy Enhances Literacy | Thawani et al. | 2021 | `papers/2103.13139_numeracy_enhances_literacy.pdf` | Number encoders improve word prediction |
| 21 | DICE: Numeracy-Preserving Embeddings | Sundararaman et al. | 2020 | `papers/2003.01058_dice_numeracy_embeddings.pdf` | Deterministic number embeddings |
| 22 | NumeroLogic: Number Encoding for LLMs | — | 2024 | `papers/2404.00459_numerologic.pdf` | Number encoding for numerical reasoning |
| 23 | CamemBERT-Bio Numbers in Clinical Narratives | Lompo & Le | 2024 | `papers/camembert_bio_numbers_clinical.pdf` | French BERT model for clinical numbers |
| 24 | FoNE: Fourier Number Embeddings | Zhou et al. | 2025 | `papers/fone_fourier_number_embeddings.pdf` | Fourier features for single-token numbers |

See `papers/README.md` for detailed descriptions.

---

## Datasets
Total datasets generated: 1 (with 3 files)

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| French Numbers 0-999 | Generated via `num2words` | 1000 records | Number word mapping + categorization | `datasets/french_numbers/` | Includes vigesimal category labels |
| Comparison Pairs | Generated | 4994 pairs | Value comparison (binary) | `datasets/french_numbers/comparison_pairs.jsonl` | French and English pairs |
| Counting Sequences | Generated | 20 sequences | Next-number prediction | `datasets/french_numbers/counting_sequences.jsonl` | Sequences crossing vigesimal boundaries |

See `datasets/README.md` for detailed descriptions including download/generation instructions.

---

## Code Repositories
Total repositories cloned: 3

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| Multilingual Num Probe | [github.com/dj1121/num_probe](https://github.com/dj1121/num_probe) | **Most relevant** — French number probing with BERT | `code/multilingual-num-probe/` | Has French templates, data gen scripts |
| Numeracy Probing (EACL 2026) | [github.com/VCY019/Numeracy-Probing](https://github.com/VCY019/Numeracy-Probing) | Internal number knowledge probing | `code/numeracy-probing/` | Linear probes, LoRA fine-tuning |
| Wallace Numeracy (EMNLP 2019) | [github.com/Eric-Wallace/numeracy](https://github.com/Eric-Wallace/numeracy) | Foundational numeracy probing | `code/wallace-numeracy/` | Original probing tasks |

See `code/README.md` for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder service with 4 diligent-mode searches covering: LLM numeracy, French number systems, mechanistic interpretability arithmetic, number representation probing
2. Web searches targeting: multilingual numeracy, French counting system, number words in LLMs
3. Semantic Scholar API results from paper-finder provided relevance-ranked papers
4. Targeted search for the specific multilingual probing paper (Johnson et al. 2020) which directly studies French

### Selection Criteria
- **Highest priority**: Papers studying number representation in multiple languages including French
- **High priority**: Papers on how LLMs represent numbers internally (probing, mechanistic interpretability)
- **Medium priority**: Papers on arithmetic circuits and number tokenization
- **Lower priority**: General math reasoning surveys (included for context)

### Challenges Encountered
- No paper-finder service initially available (httpx missing) — resolved by installing
- The specific intersection of "French counting" + "LLM representations" is underexplored — only one paper (Johnson et al. 2020) directly addresses this, and it uses older BERT-era models
- No existing dataset of French number words with vigesimal annotations — we generated one

### Gaps and Workarounds
- **No modern French number probing**: The only study (Johnson et al. 2020) uses BERT/DistilBERT/XLM. No study uses Llama, Mistral, or other modern LLMs on French numbers → This is the gap our experiment fills.
- **No Belgian/Swiss French comparison**: Would need `num2words` with `lang='fr_BE'` or custom mappings
- **No sinusoidal probing of number words**: Štefánik et al. probe digit tokens only, not number words → Our experiment should bridge this

---

## Recommendations for Experiment Design

Based on gathered resources, recommend:

1. **Primary dataset(s)**:
   - French numbers 0-999 generated via `num2words` (already in `datasets/french_numbers/`)
   - Comparison with English number words for the same range
   - Focus analysis on the vigesimal range (70-99) vs regular decimal range (1-69)

2. **Baseline methods**:
   - Replicate Johnson et al. (2020) probing approach on modern LLMs (Llama 3, Mistral)
   - Apply sinusoidal probing from Štefánik et al. (2025) to French number word embeddings
   - Apply circular digit-wise probing from Levy & Geva (2024) to French word forms

3. **Evaluation metrics**:
   - Probe accuracy (sinusoidal, circular, linear) on French vs English number words
   - Per-category accuracy (units, teens, decimal tens, vigesimal 70s/80s/90s, hundreds)
   - Representation similarity between French and English embeddings for same numbers
   - Layer-wise analysis of where French number information degrades

4. **Code to adapt/reuse**:
   - `code/multilingual-num-probe/` — Directly reusable for data generation and BERT-style probing
   - `code/numeracy-probing/` — Adapt embedding extraction and probe training for French words
   - Sinusoidal probing from `github.com/prompteus/numllama` (reference code)

5. **Key experimental questions**:
   - Do modern LLMs represent French vigesimal numbers (70-99) differently from decimal numbers (1-69)?
   - Can sinusoidal probes recover number values from French word embeddings as accurately as from digit tokens?
   - Is there a larger internal-knowledge-vs-output gap for French numbers than English?
   - Do models implicitly decompose "quatre-vingt-dix-sept" into arithmetic operations (4×20+10+7)?
