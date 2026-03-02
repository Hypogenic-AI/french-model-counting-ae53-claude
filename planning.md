# Research Plan: How Does the Model Count in French?

## Motivation & Novelty Assessment

### Why This Research Matters
French has an infamously counterintuitive counting system where numbers 70-99 use vigesimal (base-20) arithmetic: "soixante-dix" (60+10=70), "quatre-vingts" (4×20=80), "quatre-vingt-dix-neuf" (4×20+10+9=99). Understanding how LLMs internally represent these compositionally complex number words reveals how models handle non-transparent linguistic encoding of mathematical concepts — a fundamental question at the intersection of numerical cognition and language understanding.

### Gap in Existing Work
The only study directly examining French number representations in language models (Johnson et al. 2020) used BERT-era models and found French was consistently the worst-performing language for number probing. Since then, probing methodology has advanced dramatically — sinusoidal probes (Štefánik et al. 2025), circular digit-wise probes (Levy & Geva 2024) — but none have been applied to French number words. No study has examined how modern LLMs (Llama 3, Mistral) internally represent the vigesimal system, nor whether they implicitly decompose compositional French numbers into arithmetic operations.

### Our Novel Contribution
We conduct the first comprehensive mapping of how a modern LLM represents French counting, using three complementary approaches:
1. **Internal representation probing**: Extract hidden-state embeddings for French vs English number words and probe whether models encode correct numerical values, with special focus on the vigesimal range (70-99)
2. **API-level behavioral testing**: Systematically test LLMs on French counting tasks (next number, comparison, translation) to map where the vigesimal system causes errors
3. **Representation geometry analysis**: Visualize and quantify how the embedding space organizes French numbers — do vigesimal numbers cluster differently than decimal numbers?

### Experiment Justification
- **Experiment 1 (Embedding Probing)**: Tests whether model hidden states encode correct numerical magnitude for French number words. This is needed because prior work only probed BERT-era models and never compared vigesimal vs decimal subranges.
- **Experiment 2 (Behavioral Testing via API)**: Tests whether LLMs can correctly perform counting, comparison, and arithmetic with French number words. This reveals the output-level behavior that complements internal probing.
- **Experiment 3 (Representation Geometry)**: Visualizes how French numbers are organized in embedding space — PCA, cosine similarity matrices, cluster analysis. This maps the "shape" of how models represent the French counting system.

---

## Research Question
How do modern large language models internally represent and process the French vigesimal counting system (especially numbers 70-99), and where does this representation break down compared to regular decimal numbers and English equivalents?

## Background and Motivation
French counting uses a vigesimal system for 70-99:
- 70 = soixante-dix (60+10)
- 80 = quatre-vingts (4×20)
- 90 = quatre-vingt-dix (4×20+10)
- 97 = quatre-vingt-dix-sept (4×20+10+7)

This creates a natural experiment: the same mathematical quantities are encoded with radically different linguistic complexity in French vs English. By comparing how LLMs handle these two encodings of the same numbers, we can map exactly how language structure affects numerical representation.

## Hypothesis Decomposition

**H1**: Modern LLMs encode French number words in hidden states that correlate with numerical magnitude, but with lower accuracy for vigesimal-range numbers (70-99) than decimal-range numbers (1-69).

**H2**: LLMs make more errors on French counting/comparison tasks involving vigesimal numbers than decimal numbers, as measured via API behavioral testing.

**H3**: In the model's embedding space, vigesimal French numbers (70-99) show a different organizational structure than their English equivalents — specifically, numbers like 70-79 may cluster closer to 60-69 (due to "soixante-") rather than forming their own decade group.

**H4**: The model's internal number line for French is "bent" or distorted in the vigesimal region compared to a smooth English number line.

## Proposed Methodology

### Approach
We use a three-pronged approach, combining internal probing (via local model), behavioral testing (via OpenAI API), and geometric analysis:

1. **Local model probing** with a small multilingual model (e.g., Mistral 7B or multilingual-e5-large) on our 4x A6000 GPUs — extract embeddings for all French and English number words 0-999, train probes to recover numerical values
2. **API behavioral testing** with GPT-4.1 — systematic tests of counting, comparison, next-number prediction, and number-to-digit conversion in French
3. **Geometric analysis** of the extracted embeddings — PCA, t-SNE, cosine similarity heatmaps, cluster analysis

### Experimental Steps

#### Step 1: Data Preparation
- Load the pre-generated French numbers dataset (0-999 with vigesimal categories)
- Create prompt templates for behavioral testing
- Define number subranges for focused analysis: units (0-9), teens (10-19), decimal_tens (20-69), vigesimal_70s (70-79), vigesimal_80s (80-89), vigesimal_90s (90-99), hundreds (100-999)

#### Step 2: Embedding Extraction (Local Model)
- Load a multilingual model (Mistral 7B Instruct or similar)
- Extract hidden-state embeddings for each French and English number word at multiple layers
- Save embeddings for analysis

#### Step 3: Linear Probing
- Train linear probes to predict numerical value from embeddings
- Compare French vs English probe accuracy
- Compare vigesimal vs decimal subrange accuracy
- Layer-wise analysis: at which layers does French number information degrade?

#### Step 4: Behavioral Testing (API)
- Use GPT-4.1 API to test:
  - Next number prediction: "After soixante-neuf comes ___"
  - Number comparison: "Which is larger: quatre-vingt-sept or soixante-douze?"
  - Number-to-digit conversion: "What number is quatre-vingt-dix-sept?"
  - Counting sequences through vigesimal boundaries
- Compare accuracy across vigesimal vs decimal ranges

#### Step 5: Representation Geometry
- PCA/t-SNE visualization of French vs English number embeddings
- Cosine similarity heatmaps between consecutive numbers
- Decade clustering analysis
- Compare "number line" shape across languages

#### Step 6: Statistical Analysis
- Paired tests comparing vigesimal vs decimal accuracy
- Bootstrap confidence intervals
- Effect size calculations
- Correlation between linguistic complexity (word count) and probe error

### Baselines
- English number words (same range, same model) as cross-lingual baseline
- Digit tokens ("97") as representation ceiling
- Random probe performance as floor

### Evaluation Metrics
- **Probe R² and MAE**: How well can we predict numerical value from embeddings?
- **Behavioral accuracy**: % correct on counting, comparison, conversion tasks
- **Representational similarity**: Cosine similarity between French/English embeddings for same numbers
- **Clustering quality**: Silhouette score for decade groupings

### Statistical Analysis Plan
- Wilcoxon signed-rank tests for paired vigesimal vs decimal accuracy
- Bootstrap 95% CIs for all reported metrics
- Cohen's d for effect sizes
- Spearman correlation for complexity vs error analysis
- Significance level: α = 0.05 with Bonferroni correction for multiple comparisons

## Expected Outcomes
- **H1 supported if**: Probe accuracy on vigesimal range is significantly lower than decimal range
- **H2 supported if**: API error rate on vigesimal tasks is significantly higher
- **H3 supported if**: PCA/clustering shows 70-79 grouping with 60s rather than as own decade
- **H4 supported if**: French number line visualization shows distortion in 70-99 range

## Timeline and Milestones
1. Environment setup + data prep: 15 min
2. Embedding extraction: 30 min
3. Probing experiments: 30 min
4. API behavioral testing: 30 min
5. Geometric analysis + visualization: 30 min
6. Statistical analysis: 20 min
7. Documentation: 25 min

## Potential Challenges
- Model loading on GPU may require quantization for very large models → use 7B model
- API rate limits → batch requests, cache responses
- French tokenization may split number words inconsistently → use last-token or mean-pooling

## Success Criteria
- Complete embedding extraction and probing for 0-999 in both languages
- At least 100 API behavioral test instances
- Clear visualization showing how French numbers are organized in embedding space
- Statistical evidence (p < 0.05) for or against each hypothesis
- Comprehensive REPORT.md with actual numerical results
