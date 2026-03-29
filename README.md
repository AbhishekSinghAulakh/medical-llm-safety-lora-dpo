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

