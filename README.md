## Overview

This repository contains the complete implementation of the research pipeline described in the dissertation *"Mitigating Medical Misinformation under Adversarial Prompts using a LoRA + DPO Staged Pipeline"*. The study investigates whether a sequential two-stage fine-tuning framework — combining Low-Rank Adaptation (LoRA) for domain grounding and Direct Preference Optimization (DPO) for safety alignment — can improve the adversarial robustness of a medical large language model.

The base model evaluated throughout is **Mistral-7B-Instruct-v0.3**. Experiments are conducted across three publicly available datasets: MedQuAD, MedSafetyBench, and CARES.

---

## Key Findings

| Metric | Baseline | LoRA | LoRA + DPO |
|---|---|---|---|
| MedSafetyBench ASR | 0.29 | 0.22 | 0.22 |
| CARES Overall ASR (stratified) | 0.570 | 0.185 | 0.185 |
| Semantic Similarity | 0.737 | 0.737 | 0.777 |
| Safety-Utility (SU) Score | 0.523 | 0.575 | 0.606 |

- LoRA domain grounding provides the primary gains in adversarial robustness — 67.5% ASR reduction on CARES
- DPO does not further reduce ASR (DPO plateau) but improves semantic similarity and response quality
- The combined pipeline achieves the best Safety-Utility score with no alignment tax

---

## Pipeline Architecture

```
Baseline Model (Mistral-7B-Instruct-v0.3)
│
▼
Stage 0: Baseline Evaluation  ←── MedQuAD + MedSafetyBench + CARES
│
▼
Stage 1: LoRA Fine-Tuning     ←── MedQuAD (10,000 samples)
│
▼
Stage 2: DPO Alignment        ←── MedSafetyBench (100 preference pairs)
│
▼
Stage 3: Full Evaluation      ←── All three models × all three datasets
│
▼
Results: ASR ↓  |  Safety Recall ↑  |  Semantic Similarity ↑  |  SU Score ↑
```

---

## Repository Structure

```
medical-llm-safety-lora-dpo/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── staged_pipeline.ipynb        # End-to-end Original Colab notebook with outputs (full pipeline)
│
├── src/
│   ├── stage1_lora.py               # Stage 1 — LoRA fine-tuning on MedQuAD
│   ├── stage2_dpo.py                # Stage 2 — DPO alignment on MedSafetyBench
│   ├── evaluation.py                # Unified evaluation — all models × all datasets
│
│
├── data/
│   ├── medquad.csv                  # Pre-processed MedQuAD (prompt, reference_response)
│   ├── msb.csv                      # Pre-processed MedSafetyBench (prompt)
│   ├── cares.csv                    # Pre-processed CARES (prompt, attack_type)
│   └── README.md                    # Dataset sources, licences, column schema
│
└── results/
    └── sample_outputs/              # Sample CSVs from each evaluation phase
```

---

## Stages

### Stage 1 — LoRA Fine-Tuning (Domain Grounding)

Applies Low-Rank Adaptation to Mistral-7B-Instruct-v0.3 using MedQuAD to improve medical domain knowledge and factual accuracy.

| Parameter | Value |
|---|---|
| Base Model | Mistral-7B-Instruct-v0.3 |
| Target Modules | q_proj, v_proj |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| Learning Rate | 2e-4 |
| Epochs | 2 |
| Training Samples | 10,000 |
| Max Sequence Length | 512 |

```bash
python src/stage1_lora.py \
  --data_path ./data/medquad.csv \
  --output_dir ./outputs/lora_model
```

### Stage 2 — DPO Alignment (Safety Alignment)

Applies Direct Preference Optimization on the LoRA-adapted model using MedSafetyBench preference pairs. The LoRA model generates the rejected responses; the dataset's safe responses serve as chosen.

| Parameter | Value |
|---|---|
| Base | LoRA-adapted model |
| Learning Rate | 5e-5 |
| Beta | 0.3 |
| Epochs | 2 |
| Training Samples | 100 preference pairs |

```bash
python src/stage2_dpo.py \
  --lora_model_path ./outputs/lora_model \
  --data_path ./data/msb.csv \
  --output_dir ./outputs/dpo_model
```

### Stage 3 — Unified Evaluation

A single evaluation script runs all three model checkpoints (Baseline, LoRA, LoRA+DPO) through all four evaluation phases and produces every metric reported in the dissertation.

**Evaluation phases:**

| Phase | Dataset | Metrics |
|---|---|---|
| 1 — Benign QA | MedQuAD (100 samples) | Refusal Rate, Avg Response Length, Semantic Similarity |
| 2 — Safety | MedSafetyBench (100 samples) | ASR, Safety Recall, Harmful Refusal Rate, Calibration Gap |
| 3 — Adversarial | CARES (200 random + 50×4 stratified) | ASR by attack category, Overall ASR |
| 4 — Composite | — | Safety-Utility (SU) Score = Safety Recall × Semantic Similarity |

```bash
# Evaluate baseline
python src/evaluation.py \
  --model_name mistralai/Mistral-7B-Instruct-v0.3 \
  --model_label baseline \
  --medquad_path ./data/medquad.csv \
  --msb_path ./data/msb.csv \
  --cares_path ./data/cares.csv \
  --output_dir ./results/

# Evaluate LoRA model
python src/evaluation.py \
  --model_name ./outputs/lora_model \
  --model_label lora \
  --medquad_path ./data/medquad.csv \
  --msb_path ./data/msb.csv \
  --cares_path ./data/cares.csv \
  --output_dir ./results/

# Evaluate LoRA + DPO model (256-token variant)
python src/evaluation.py \
  --model_name ./outputs/dpo_model \
  --model_label lora_dpo \
  --medquad_path ./data/medquad.csv \
  --msb_path ./data/msb.csv \
  --cares_path ./data/cares.csv \
  --output_dir ./results/ \
  --max_new_tokens 256
```

Run only a specific phase:
```bash
python src/evaluation.py --model_name ... --phases cares
python src/evaluation.py --model_name ... --phases medquad msb
```

---

## Datasets

Pre-processed versions of all three evaluation datasets are included in the `data/` directory and are ready for direct use with `evaluation.py`. No additional preprocessing is required.

| File | Source | Role | Key Columns |
|---|---|---|---|
| `medquad.csv` | Ben Abacha & Demner-Fushman (2019) | Benign QA evaluation | `prompt`, `reference_response` |
| `msb.csv` | Han et al. (2024) | Safety evaluation | `prompt` |
| `cares.csv` | Chen et al. (2025) | Adversarial robustness | `prompt`, `attack_type` |

`attack_type` values in `cares.csv`: `direct`, `indirect`, `obfuscation`, `role_play`

All datasets are publicly available under open academic licences. See `data/README.md` for original sources and licence details.

---

## Reproducibility

**Notebook-based (recommended for full pipeline)**

Open `notebooks/staged_pipeline.ipynb` in Google Colab with an A100 GPU runtime. The notebook executes the complete pipeline end-to-end including data loading, LoRA training, DPO alignment, and evaluation.

**Script-based (modular)**

Install dependencies, then run each stage independently as shown above. Scripts assume pre-processed datasets are available under `data/` and a GPU with at least 16 GB VRAM is available.

**Reproducibility controls**

- Random seed fixed at `42` across PyTorch, NumPy, and all dataset sampling operations
- Deterministic decoding (`temperature=0.0`, `do_sample=False`) used consistently across all evaluation stages
- Same prompt template applied to all three model checkpoints

---

## Installation

```bash
git clone https://github.com/[your-username]/medical-llm-safety-lora-dpo.git
cd medical-llm-safety-lora-dpo
pip install -r requirements.txt
```

GPU with CUDA is required. Tested on NVIDIA A100 (80 GB). Minimum recommended VRAM: 16 GB for evaluation, 40 GB for training.

---

## Citation

If referencing this work, please cite the associated dissertation:

> Singh, A. (2025) *Mitigating Medical Misinformation under Adversarial Prompts using a LoRA + DPO Staged Pipeline*. MSc Dissertation, Liverpool John Moores University, UK.

---
