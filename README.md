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

## Repository Structure
medical-llm-safety-lora-dpo/

notebooks/ # Full experimental pipeline (Colab)
src/ # Modular scripts for each stage
data/ # Dataset links (not stored here)
results/ # Sample outputs
configs/ # Configuration files (optional)
docs/ # Supporting documentation

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
