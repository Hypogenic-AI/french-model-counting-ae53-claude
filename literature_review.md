# Literature Review: How Does the Model Count in French?

## Research Area Overview

This research investigates how large language models (LLMs) represent and process the French counting system. French is notable for its **vigesimal (base-20) number system** above 60: "soixante-dix" (60+10=70), "quatre-vingts" (4×20=80), "quatre-vingt-dix" (4×20+10=90). This makes French numbers an ideal testbed for studying how LLMs handle compositional, non-transparent number representations compared to more regular decimal systems.

The literature spans three intersecting areas: (1) how LLMs represent numbers internally, (2) mechanistic interpretability of arithmetic in transformers, and (3) multilingual numerical understanding.

---

## Key Papers

### 1. Probing for Multilingual Numerical Understanding in Transformer-Based Language Models
- **Authors**: Devin Johnson, Denise Mak, Drew Barker, Lexi Loessberg-Zahl
- **Year**: 2020
- **Source**: BlackboxNLP Workshop @ EMNLP 2020 (arXiv: 2010.06666)
- **Key Contribution**: The **most directly relevant paper** — probes BERT, DistilBERT, and XLM on French, English, Danish, and Japanese number understanding using two tasks: grammaticality judgment and value comparison.
- **Methodology**: MLP classifier on top of frozen pretrained embeddings. Numbers generated using `num2words` library (1-999), inserted into language-specific sentence templates. Task 1: Is a number word grammatical? Task 2: Which of two numbers is larger?
- **Key Results for French**:
  - Task 1 (grammaticality, bare): BERT 0.64, DistilBERT 0.91, XLM 0.87 — French is consistently the **worst-performing language** across all models
  - Task 1 (grammaticality, sentences): BERT 0.56, DistilBERT 0.77, XLM 0.81
  - Task 2 (value comparison, bare): BERT 0.54, DistilBERT 0.70, XLM 0.69 — near-chance for BERT
  - Task 2 (value comparison, sentences): all near chance (~0.50-0.56)
  - French's vigesimal system was expected to make it hardest, though the authors found model architecture mattered more than language transparency
  - When fine-tuned, all models reach >99% on both tasks for all languages, meaning the tasks are learnable
- **Code Available**: [github.com/dj1121/num_probe](https://github.com/dj1121/num_probe) — includes French sentence templates and data generation scripts
- **Relevance**: Directly establishes that pretrained models encode less numerical information for French than other languages, setting up our research question

### 2. Language Models Encode Numbers Using Digit Representations in Base 10
- **Authors**: Amit Arnold Levy, Mor Geva
- **Year**: 2024 (arXiv: 2410.11781, published NAACL 2025)
- **Key Contribution**: Shows LLMs use **orthogonal circular representations for each digit in base 10** rather than holistic magnitude representations
- **Methodology**: Trained circular probes on hidden representations of Llama 3 8B and Mistral 7B for numbers 0-999. Each digit position (ones, tens, hundreds) has an independent circular representation.
- **Key Results**:
  - Probes fail to recover exact number values directly, but can recover individual digits with high accuracy
  - Circular probes in base 10 achieve near-perfect accuracy (1.00 in Mistral 7B best layer); bases 2, 3, etc. perform much worse
  - Causal interventions: modifying the circular representation of a digit changes the model's output correspondingly
  - Numbers in word form (e.g., "twenty-two"): circular probes achieve 68.6% accuracy at peak layer — encouraging but lower than digit form
- **Code Available**: Yes (referenced in paper)
- **Relevance**: French numbers don't decompose neatly into digits — "quatre-vingt-dix-sept" = 4×20+10+7 = 97. If models use base-10 digit representations, French word forms may be harder to map to these representations.

### 3. Unravelling the Mechanisms of Manipulating Numbers in Language Models
- **Authors**: Michal Štefánik, Timothee Mickus, Marek Kadlčík, et al.
- **Year**: 2025 (arXiv: 2510.26285)
- **Key Contribution**: Demonstrates that LLMs converge to **universal sinusoidal representations** of numbers that are interchangeable across models and layers
- **Methodology**: Sinusoidal probes, Fourier decomposition, Representational Similarity Analysis (RSA), cross-layer generalization, error tracking across 8 models (OLMo 2, Llama 3, Phi 4)
- **Key Results**:
  - All models share the same top-63 Fourier base frequencies for number representations (perfect IoU agreement)
  - Sinusoidal probes outperform linear, logarithmic, and binary probes at all layers
  - The **residual stream** is the primary carrier of accurate numeric representation (98-100% accuracy); attention layers disrupt it, MLPs partially restore it
  - Models internally compute correct answers that fail to surface: 56.8% of subtraction errors, 94.4% of division errors have correct internal representations
  - Removing specific error-causing layers reduces errors by 27-64%
- **Code Available**: [github.com/prompteus/numllama](https://github.com/prompteus/numllama)
- **Relevance**: If number representations are universal sinusoidal patterns, how does the vigesimal French encoding interact with these patterns? The compositional nature of French numbers may require additional processing steps.

### 4. LLMs Know More About Numbers than They Can Say
- **Authors**: Fengting Yuchi, Li Du, Jason Eisner
- **Year**: 2025 (arXiv: 2602.07812, EACL 2026 Oral)
- **Key Contribution**: Demonstrates a sharp gap between LLMs' **internal numerical knowledge** and their **verbalized output**
- **Methodology**: Linear probes on hidden states for log-magnitude recovery and ranking classification. Tested on synthetic data and arXiv papers across multiple 7B-8B models.
- **Key Results**:
  - Linear probes recover numbers with ~2.3% relative error (synthetic) and ~19% (arXiv)
  - Hidden states encode ranking with >90% accuracy via linear classifier
  - But when explicitly asked, LLMs achieve only 50-70% verbalization accuracy
  - Probe-aware fine-tuning with LoRA improves verbalization by 3.22%
- **Code Available**: [github.com/VCY019/Numeracy-Probing](https://github.com/VCY019/Numeracy-Probing)
- **Relevance**: This internal-vs-output gap may be even larger for French due to the additional compositional complexity of the vigesimal system

### 5. What is a Number, That a Large Language Model May Know It?
- **Authors**: (arXiv: 2502.01540)
- **Year**: 2025
- **Key Contribution**: Uses **cognitive science tools** to study LLM number representations, finding log-linear representations paralleling human cognition
- **Methodology**: Probing with cognitive science frameworks. Tests whether LLMs exhibit psychologically plausible number representations.
- **Key Results**: In non-mathematical settings, LLMs exhibit log-linear representation blended with string distance, paralleling human approximate number system
- **Relevance**: French counting may create different cognitive-like representations due to its compositional complexity

### 6. Number Representations in LLMs: A Computational Parallel to Human Perception
- **Authors**: (arXiv: 2502.16147)
- **Year**: 2025
- **Key Contribution**: Shows LLM number representations exhibit **sublinear (logarithmic) spacing** like human number cognition
- **Methodology**: PCA and Partial Least Squares (PLS) with geometric regression across model layers
- **Key Results**: Number representations compress along a logarithmic scale, paralleling Weber-Fechner law in human perception
- **Relevance**: The vigesimal system groups numbers differently — 70-79 are "sixty-something" and 80-99 are "four-twenties-something" — which may create different spacing patterns

### 7. Do NLP Models Know Numbers? Probing Numeracy in Embeddings
- **Authors**: Eric Wallace, Yizhong Wang, Sujian Li, Sameer Singh, Matt Gardner
- **Year**: 2019 (EMNLP 2019, arXiv: 1909.07940)
- **Key Contribution**: Foundational work establishing that word embeddings encode numeracy to varying degrees
- **Methodology**: Probing tasks (list maximum, number decoding, addition) on GloVe, word2vec, ELMo, BERT
- **Key Results**: GloVe/word2vec encode magnitude up to 1000; character-level (ELMo) is most precise; BERT (subword) is less exact
- **Code Available**: [github.com/Eric-Wallace/numeracy](https://github.com/Eric-Wallace/numeracy)
- **Relevance**: Foundational probing methodology that later studies build upon

### 8. Representing Numbers in NLP: A Survey and a Vision
- **Authors**: Avijit Thawani, Jay Pujara, Pedro Szekely, Filip Ilievski
- **Year**: 2021 (arXiv: 2103.13136)
- **Key Contribution**: Comprehensive taxonomy of numeracy in NLP — 7 subtasks along granularity and units dimensions
- **Relevance**: Provides the theoretical framework for understanding different aspects of numeracy that our research touches on

### 9. Arithmetic in Transformers Explained
- **Authors**: (arXiv: 2402.02619)
- **Year**: 2024-2025
- **Key Contribution**: Mechanistic account of addition/subtraction circuits in small transformers, with cascading carry/borrow mechanisms
- **Key Results**: Small transformers trained from scratch achieve 99.999% accuracy on n-digit arithmetic; only 7% of 180 LLMs (1B-405B) can reliably add
- **Relevance**: Understanding arithmetic circuits helps understand how models might process the implicit arithmetic in French numbers (e.g., 4×20+10+7)

### 10. Pre-trained Language Models Learn Remarkably Accurate Representations of Numbers
- **Authors**: Marek Kadlčík, Michal Štefánik et al.
- **Year**: 2025 (arXiv: 2506.08966)
- **Key Contribution**: Shows sinusoidal probes with correct inductive bias achieve **near-perfect accuracy** in recovering numbers from embeddings
- **Methodology**: Sinusoidal probes across Llama 3, Phi 4, OLMo 2 series (1B-72B parameters)
- **Key Results**: Number embeddings follow sinusoidal wave-like patterns; precision explains large portion of arithmetic errors
- **Relevance**: Establishes the state-of-the-art probing methodology we should use for French number analysis

---

## Common Methodologies

1. **Linear/MLP Probing**: Train classifier on frozen embeddings (Johnson et al. 2020, Wallace et al. 2019)
2. **Sinusoidal Probing**: Probes with sinusoidal inductive bias — current state-of-the-art (Štefánik et al. 2025, Kadlčík et al. 2025)
3. **Circular Probing**: Per-digit circular representations in base 10 (Levy & Geva 2024)
4. **Causal Intervention**: Modify internal representations and observe output changes (Levy & Geva 2024)
5. **PCA/Fourier Analysis**: Dimensionality reduction and frequency analysis of embeddings (Štefánik et al. 2025)
6. **Representational Similarity Analysis (RSA)**: Cross-model comparison of number representations (Štefánik et al. 2025)

## Standard Baselines

- Random baseline (50% for binary classification tasks)
- BERT multilingual, DistilBERT, XLM (Johnson et al. 2020)
- Modern LLMs: Llama 3, Mistral, OLMo, Phi (2024-2025 papers)

## Evaluation Metrics

- **Classification accuracy** for grammaticality and comparison tasks
- **Relative error** for magnitude recovery probes
- **Probe accuracy per layer** for mechanistic analysis
- **IoU agreement** for cross-model representation comparison

## Datasets in the Literature

- Custom-generated number words via `num2words` (Johnson et al. 2020) — used for all 4 languages
- Synthetic integer/decimal comparison datasets (Yuchi et al. 2025)
- ArXiv text with numbers (Yuchi et al. 2025)
- Numbers 0-999 as digit tokens for probing (Levy & Geva 2024)

## Gaps and Opportunities

1. **No modern LLM probing of French numbers**: Johnson et al. (2020) used BERT-era models; no study has applied sinusoidal/circular probing to French number words in modern LLMs (Llama, Mistral, etc.)
2. **No analysis of vigesimal-specific effects**: No study has specifically examined how the 70-99 range (vigesimal) is represented differently from the 1-69 range (decimal) in French
3. **No cross-lingual comparison with modern probes**: The digit-level circular representation found by Levy & Geva (2024) has not been tested on French number words
4. **Belgian/Swiss French comparison**: No study compares standard French (soixante-dix) vs Belgian/Swiss French (septante) to isolate the effect of number system transparency
5. **Internal-vs-output gap for French**: The gap found by Yuchi et al. (2025) has not been measured for French specifically

## Recommendations for Our Experiment

Based on literature review:
- **Recommended datasets**: Generate French number words (1-999) using `num2words` following Johnson et al., with specific attention to vigesimal range (70-99); use comparison with English, and optionally Belgian French
- **Recommended baselines**: Modern multilingual LLMs (Llama 3, Mistral) with both French and English number words; compare digit tokens vs word tokens
- **Recommended metrics**: Sinusoidal probe accuracy, circular probe accuracy per digit, classification accuracy for comparison tasks, representation similarity between French and English number embeddings
- **Methodological considerations**:
  - Use sinusoidal probes (Štefánik et al.) rather than simple linear probes
  - Analyze per-layer representations to trace where French number information degrades
  - Compare vigesimal range (70-99) vs regular decimal range (1-69) specifically
  - Test whether models internally decompose "quatre-vingt-dix-sept" into 4×20+10+7
