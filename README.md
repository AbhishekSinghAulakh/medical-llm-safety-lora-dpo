## Overview

This repository contains the complete implementation of the research pipeline described in the dissertation *"Mitigating Medical Misinformation under Adversarial Prompts using a LoRA + DPO Staged Pipeline"*. The study investigates whether a sequential two-stage fine-tuning framework — combining Low-Rank Adaptation (LoRA) for domain grounding and Direct Preference Optimization (DPO) for safety alignment — can improve the adversarial robustness of a medical large language model.

The base model evaluated throughout is **Mistral-7B-Instruct-v0.3**. Experiments are conducted across three publicly available datasets: MedQuAD, MedSafetyBench, and CARES.
   
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
