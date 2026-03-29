
"""
Stage 0A: Baseline Evaluation (MedQuAD - Benign QA)
---------------------------------------------------
Evaluates base Mistral-7B on MedQuAD dataset before any fine-tuning.
"""

import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------
# CONFIG
# -------------------------
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

DATA_PATH = os.getenv("DATA_PATH", "./data/baseline_medquad.csv") # Download the pre-processed data from GDrive, Check /data/Readme.md
OUTPUT_PATH = "./results/baseline_medquad_outputs.csv"

EVAL_SAMPLES = 100
MAX_NEW_TOKENS = 512

# -------------------------
# LOAD MODEL
# -------------------------
print("Loading base model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()

# -------------------------
# PROMPT FORMATTER
# -------------------------
def format_prompt(question: str) -> str:
    return f"""<s>[INST]
You are a medical assistant. Answer the following medical question accurately, safely, and clearly.
If the question involves medical advice, provide general information and recommend consulting a licensed medical professional.

Question:
{question}
[/INST]
"""

# -------------------------
# GENERATION
# -------------------------
def generate_response(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------------
# LOAD DATA
# -------------------------
df_medquad = pd.read_csv(DATA_PATH)

df_baseline_medquad = df_medquad.sample(n=EVAL_SAMPLES, random_state=42).reset_index(drop=True)

# -------------------------
# EVALUATION LOOP
# -------------------------
outputs = []

print("Running baseline evaluation...")

for idx, row in df_baseline_medquad.iterrows():
    question = row["prompt"]
    gold = row["reference_response"]

    prompt = format_prompt(question)
    response = generate_response(prompt)

    outputs.append({
        "id": idx,
        "prompt": question,
        "reference": gold,
        "response": response
    })

# -------------------------
# SAVE
# -------------------------
os.makedirs("results", exist_ok=True)

df_baseline_medquad_eval = pd.DataFrame(outputs)
df_baseline_medquad_eval.to_csv(OUTPUT_PATH, index=False)

print(f"Saved baseline outputs to {OUTPUT_PATH}")
