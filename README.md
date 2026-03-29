# medical-llm-safety-lora-dpo
Mitigating Medical Misinformation using LoRA + DPO staged fine-tuning pipeline

# Mitigating Medical Misinformation using LoRA + DPO

## Overview
This project investigates the robustness of Large Language Models (LLMs) in medical domains against adversarial prompting attacks. It proposes a staged fine-tuning pipeline combining domain grounding (LoRA) and preference-based alignment (DPO) to improve safety and response quality.

The study evaluates model behaviour under adversarial conditions using MedSafetyBench and CARES benchmarks, focusing on metrics such as Attack Success Rate (ASR) and refusal behaviour.

---

## Pipeline Architecture

The pipeline consists of four stages:

### Stage 0 — Baseline Evaluation
- Base model: Mistral-7B-Instruct
- No fine-tuning applied
- Evaluated on:
  - MedQuAD (benign QA)
  - Adversarial datasets (MedSafetyBench / CARES)

### Stage 1 — LoRA Fine-Tuning
- Dataset: MedQuAD
- Objective: Improve domain grounding and factual accuracy
- Output: LoRA-adapted model

### Stage 2 — DPO Alignment
- Dataset: MedSafetyBench (preference pairs)
- Objective: Improve safety alignment using preference optimisation
- Output: LoRA + DPO aligned model

### Stage 3 — Final Evaluation
- Comparative evaluation across:
  - Baseline
  - LoRA
  - LoRA + DPO
- Metrics:
  - Attack Success Rate (ASR)
  - Refusal Rate

---

## Stage 0 — Baseline Evaluation

### Objective

To establish a reference point for model behaviour prior to any fine-tuning.

### Setup

* Model: Mistral-7B-Instruct
* No parameter updates applied

### Datasets

* MedQuAD (benign medical QA)
* Adversarial datasets (MedSafetyBench, CARES)

### Method

* Generate responses using the base model
* Evaluate:

  * Refusal behaviour
  * Harmful compliance

### Metrics

* Attack Success Rate (ASR)
* Refusal Rate

### Implementation

* src/stage0_baseline_medquad.py
* src/stage0_baseline_adversarial.py

---

## Stage 1 — LoRA Fine-Tuning (Domain Grounding)

### Objective

To improve factual accuracy and domain grounding using parameter-efficient fine-tuning.

### Method

* Apply Low-Rank Adaptation (LoRA) to the base model
* Fine-tune using MedQuAD dataset

### Key Characteristics

* Lightweight adaptation (no full model retraining)
* Focus on improving medical knowledge representation

### Output

* LoRA-adapted model

### Implementation

* src/stage1_lora.py

---

## Stage 2 — DPO Alignment (Safety Alignment)

### Objective

To align model outputs with safety preferences using preference-based optimisation.

### Method

* Apply Direct Preference Optimization (DPO)
* Train on preference pairs (chosen vs rejected responses)

### Dataset

* MedSafetyBench (preference format)

### Key Characteristics

* Improves refusal behaviour and response quality
* Does not significantly reduce adversarial attack success beyond LoRA

### Output

* LoRA + DPO aligned model

### Implementation

* src/stage2_dpo.py

---

## Stage 3 — Final Evaluation and Comparison

### Objective

To compare model performance across all stages under identical conditions.

### Models Evaluated

* Baseline (no fine-tuning)
* LoRA model
* LoRA + DPO model

### Method

* Use same adversarial dataset for all models
* Apply consistent evaluation logic

### Metrics

* Attack Success Rate (ASR)
* Refusal Rate

### Implementation

* src/stage3_evaluation.py

---

## Reproducibility

* Full experimental pipeline is available in:
  notebooks/staged_pipeline.ipynb

* Scripts provide modular execution for each stage

* Datasets are externally hosted and linked via:
  data/README.md

---

## Summary

This methodology ensures:

* Controlled comparison across model variants
* Reproducibility of experimental results
* Clear separation between domain adaptation and safety alignment stages

The staged design enables systematic evaluation of the contribution of each component in improving medical AI safety.

---

## Pipeline Flow

```
Baseline Model (Mistral-7B)
│
▼
Stage 0: Baseline Evaluation
│
▼
Stage 1: LoRA Fine-Tuning (MedQuAD)
│
▼
Stage 2: DPO Alignment (MedSafetyBench)
│
▼
Stage 3: Final Evaluation (CARES / MedSafetyBench)
│
▼
Results: ASR ↓, Safety ↑, Quality ↑
```

---
## Repository Structure

```
medical-llm-safety-lora-dpo/
│
├── notebooks/        # End-to-end Colab pipeline (data prep + experiments)
├── src/              # Modular implementation of each stage
│   ├── stage0_baseline_medquad.py
│   ├── stage0_baseline_adversarial.py
│   ├── stage1_lora.py
│   ├── stage2_dpo.py
│   └── stage3_evaluation.py
│
├── data/             # External dataset links (Google Drive)
├── results/          # Evaluation outputs and metrics
├── configs/          # Training configurations (optional)
├── docs/             # Supporting documentation
```

---

## Dataset Access

Due to size constraints, datasets are not included in this repository.

Preprocessed datasets used in this study are available here:

👉 [Google Drive Link — ADD YOUR LINK HERE]

Original sources:
- MedQuAD
- MedSafetyBench
- CARES

Dataset preprocessing and transformation logic is implemented in: notebooks/staged_pipeline.ipynb


---

## Reproducibility

This project supports two modes of execution:

### 1. Notebook-based (Recommended)
- End-to-end pipeline execution
- Includes dataset preprocessing

### 2. Script-based
- Modular execution via:

src/stage0_baseline_medquad.py
src/stage0_baseline_adversarial.py
src/stage1_lora.py
src/stage2_dpo.py
src/stage3_evaluation.py


Scripts assume preprocessed datasets are available.

---

## Key Findings

- LoRA significantly reduces adversarial vulnerability (ASR reduction)
- DPO improves response quality but does not further reduce ASR
- Combined pipeline achieves best safety–utility balance

---

## Citation

If referencing this work, please cite the associated dissertation.

---
