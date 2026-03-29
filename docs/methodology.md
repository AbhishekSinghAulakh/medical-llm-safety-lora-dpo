
# Methodology

## Overview

This project implements a staged fine-tuning pipeline to improve the safety and reliability of Large Language Models (LLMs) in medical contexts. The methodology follows a structured progression from baseline evaluation to domain adaptation and alignment, culminating in comparative evaluation.

The pipeline consists of four stages:

1. Baseline Evaluation
2. LoRA Fine-Tuning (Domain Grounding)
3. DPO Alignment (Safety Alignment)
4. Final Evaluation and Comparison

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
