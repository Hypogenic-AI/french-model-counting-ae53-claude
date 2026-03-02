# How Does the Model Count in French?

Investigating how large language models represent the French vigesimal (base-20) counting system, where 70-99 are expressed as arithmetic compositions (e.g., 97 = "quatre-vingt-dix-sept" = 4×20+10+7).

## Key Findings

- **French number words carry ~2× higher probing error than English** in Mistral-7B hidden states across all layers, reflecting the greater linguistic complexity of French number encoding.
- **Vigesimal numbers (80-99) are surprisingly well-encoded** in French — the shared "quatre-vingt-" prefix creates tight embedding clusters that actually outperform English equivalents in probing accuracy (French 80s MAE: 7.08 vs English: 11.08).
- **French 70-79 cluster with 60-69**, not with 80-89, in embedding space (cosine similarity 0.936 vs 0.825) — the "soixante-" prefix pulls these numbers toward their linguistic, not numeric, neighbors.
- **GPT-4.1 handles French counting near-perfectly** (100% conversion, 97.9% next-number, 100% comparison), with the only errors at vigesimal transition boundaries (e.g., 76→77, 379→380).
- **Cross-lingual similarity drops sharply for vigesimal numbers** (0.47 for 80s/90s vs 0.69 for units), showing where French and English representations diverge most.

## Project Structure

```
├── REPORT.md              # Full research report with all findings
├── planning.md            # Research plan and methodology
├── literature_review.md   # Synthesized literature review
├── resources.md           # Catalog of available resources
├── src/
│   ├── extract_embeddings.py    # Exp 1: Hidden-state probing
│   ├── behavioral_tests.py      # Exp 2: API behavioral testing
│   ├── representation_geometry.py # Exp 3: Embedding geometry
│   ├── run_analysis.py          # Statistical analysis
│   └── final_plots.py          # Publication-quality plots
├── results/
│   ├── probing_results.json     # Probing experiment data
│   ├── behavioral_results.json  # API test results
│   ├── statistical_analysis.json # Statistical tests
│   ├── geometry_results.json    # Geometry metrics
│   ├── embeddings/             # Saved embeddings (5 layers × 3 languages)
│   └── plots/                  # 15 visualizations
├── datasets/french_numbers/    # Number word dataset (0-999)
├── papers/                     # 24 related papers
└── code/                       # Baseline code repositories
```

## Reproduction

```bash
# Set up environment
uv venv && source .venv/bin/activate
uv pip install torch transformers numpy scipy scikit-learn matplotlib seaborn pandas num2words openai tqdm accelerate

# Run experiments (requires GPU for Exp 1, OpenAI API key for Exp 2)
python src/extract_embeddings.py    # ~15 min with GPU
python src/behavioral_tests.py      # ~8 min, needs OPENAI_API_KEY
python src/run_analysis.py          # ~2 min
python src/representation_geometry.py  # ~1 min
python src/final_plots.py           # ~10 sec
```

**Requirements**: Python 3.10+, CUDA-capable GPU (48GB for Mistral-7B), OpenAI API key.

## See Also

- [REPORT.md](REPORT.md) for the full research report
- [planning.md](planning.md) for the research plan
- [literature_review.md](literature_review.md) for background context
