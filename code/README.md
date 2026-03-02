# Cloned Repositories

## Repo 1: Multilingual Num Probe (Most Relevant)
- **URL**: https://github.com/dj1121/num_probe
- **Paper**: "Probing for Multilingual Numerical Understanding in Transformer-Based Language Models" (BlackboxNLP @ EMNLP 2020)
- **Purpose**: Probing BERT/DistilBERT/XLM for numerical understanding across English, French, Danish, Japanese
- **Location**: `code/multilingual-num-probe/`
- **Key files**:
  - `src/data_gen/gen_t1.py` — Generates grammaticality judgment data (Task 1)
  - `src/data_gen/gen_t2.py` — Generates value comparison data (Task 2)
  - `src/data_gen/sent_templates/fr_templates.txt` — 11 French sentence templates
  - `src/run_experiment.py` — Training/testing pipeline
  - `src/models.py` — Model loading (BERT, DistilBERT, XLM)
  - `environment.yml` — Conda environment specification
- **How to use**:
  - Data generation: `python src/data_gen/gen_t1.py -lang fr -range 999 -samples 100000`
  - Experiment: `python src/run_experiment.py -task t1 -model bert -lang fr`
- **Dependencies**: PyTorch, HuggingFace transformers, num2words, scikit-learn
- **Notes**: Uses frozen pretrained embeddings with MLP classifier on top. French sentence templates verified by native speakers. Can be adapted for modern LLMs by swapping model loading code.

## Repo 2: Numeracy Probing (EACL 2026)
- **URL**: https://github.com/VCY019/Numeracy-Probing
- **Paper**: "LLMs Know More About Numbers than They Can Say" (EACL 2026, Oral)
- **Purpose**: Probing internal numerical knowledge vs. verbalized output in modern LLMs
- **Location**: `code/numeracy-probing/`
- **Key files**:
  - `src/construct_data.py` — Generate synthetic comparison datasets
  - `src/get_embeds.py` — Extract embeddings from LLMs
  - `src/train_probe.py` — Train and evaluate probes (regression, classification, log-ratio)
  - `src/verbalization.py` — Test LLM verbalization accuracy
  - `src/finetune.py` — Probe-aware fine-tuning with LoRA
- **Models tested**: DeepSeek-R1, Llama 2/3.1, Mistral 7B, OLMo 2, Qwen3, GPT-4.1
- **Notes**: Requires 80GB GPU for fine-tuning. Highly relevant pipeline that can be adapted for French number word experiments. The embedding extraction and probe training scripts are directly reusable.

## Repo 3: Wallace Numeracy (EMNLP 2019)
- **URL**: https://github.com/Eric-Wallace/numeracy
- **Paper**: "Do NLP Models Know Numbers? Probing Numeracy in Embeddings"
- **Purpose**: Foundational numeracy probing (list max, number decoding, addition)
- **Location**: `code/wallace-numeracy/`
- **Notes**: Historic/foundational. Uses PyTorch + AllenNLP. Less directly applicable to modern LLMs but establishes the probing methodology.

## Additional Relevant Repositories (Not Cloned)
- **[prompteus/numllama](https://github.com/prompteus/numllama)** — Sinusoidal probing code from Štefánik et al. (2025). Key for applying state-of-the-art probing to French numbers.
- **[avi-otterai/numeracy-literacy](https://github.com/avi-otterai/numeracy-literacy)** — Wiki-Convert dataset and number encoder experiments
- **[ag1988/injecting_numeracy](https://github.com/ag1988/injecting_numeracy)** — Injecting numerical reasoning skills into LMs (ACL 2020)
