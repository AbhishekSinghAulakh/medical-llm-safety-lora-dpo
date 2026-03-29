
"""
Stage 2: DPO Alignment (Safety Alignment)
----------------------------------------
Applies Direct Preference Optimization (DPO) on LoRA model
using MedSafetyBench preference dataset.
"""

import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer
from transformers import TrainingArguments
from datasets import Dataset

# -------------------------
# CONFIG
# -------------------------
LORA_MODEL_PATH = os.getenv("LORA_MODEL_PATH", "./outputs/lora_model")
DATA_PATH = os.getenv("DATA_PATH", "./data/msb.csv")
OUTPUT_DIR = "./outputs/dpo_model"

MAX_LENGTH = 512

# -------------------------
# LOAD MODEL
# -------------------------
print("Loading LoRA model...")

tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL_PATH, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    LORA_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

# -------------------------
# LOAD DATA
# -------------------------
df_msb = pd.read_csv(DATA_PATH)


# -------------------------
# Prompt Structure (Unchanged across Stages)
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
# Function to generate Rejected Responses from LoRA model
# -------------------------

MAX_SEQ_LEN = 1024           # Increased from 512
MAX_NEW_TOKENS = 512        # Increased from 256

def generate_response(prompt: str, question: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if question in decoded:
        decoded = decoded.split(question, 1)[1].strip()

    return decoded

# -------------------------
# Select 100 samples from MedSafetyBench to generate rejected response from LoRA
# -------------------------

REJECT_SAMPLES = 100
# Sort out the coloumns
df_msb = df_msb.rename(columns={
    df_msb.columns[0]: "prompt",
    df_msb.columns[1]: "chosen"
})
df = df_msb.sample(n=min(REJECT_SAMPLES, len(df_msb)), 
                   random_state=42).reset_index(drop=True)

# -------------------------
# Function to generate Rejected Responses from LoRA model
# -------------------------
rejected_list = []

for idx, row in df_pref.iterrows():
    q = row["prompt"]
    prompt = format_prompt(q)
    rejected = generate_response(prompt, q)

    rejected_list.append(rejected)

    if idx % 5 == 0:
        print(f"Generated {idx}/{len(df)}")

df["rejected"] = rejected_list


# Expected columns:
# prompt | chosen | rejected

# Ensure correct columns exist
assert all(col in df_pref.columns for col in ["prompt", "chosen", "rejected"])

# Convert to HF dataset
dataset = Dataset.from_pandas(df_pref[["prompt", "chosen", "rejected"]])

# -------------------------
# TRAINING CONFIG
# -------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="epoch"
)


# -------------------------
# DPO TRAINER
# -------------------------
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

print("Starting DPO training...")
trainer.train()

# -------------------------
# SAVE
# -------------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"DPO model saved to {OUTPUT_DIR}")
