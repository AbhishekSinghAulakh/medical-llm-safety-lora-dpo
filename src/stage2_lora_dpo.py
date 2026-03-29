
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
from trl import DPOTrainer,DPOConfig
from transformers import TrainingArguments
from datasets import Dataset
from peft import PeftModel

# -------------------------
# CONFIG
# -------------------------
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
LORA_MODEL_PATH = os.getenv("LORA_MODEL_PATH", "./outputs/lora_model")
DATA_PATH = os.getenv("DATA_PATH", "./data/msb.csv")
OUTPUT_DIR = "./outputs/dpo_model"

# -------------------------
# LOAD MODEL
# -------------------------
print("Loading LoRA model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
model.config.use_cache = False

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

MAX_SEQ_LEN = 1024         
MAX_NEW_TOKENS = 512

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
# LOAD DATA
# -------------------------
df_msb = pd.read_csv(DATA_PATH)

# -------------------------
# Select 100 samples from MedSafetyBench to generate rejected response from LoRA
# -------------------------

REJECT_SAMPLES = 100
# Sort out the coloumns
df_msb = df_msb.rename(columns={
    df_msb.columns[0]: "prompt",
    df_msb.columns[1]: "chosen"
})
df_msb_dpo = df_msb.sample(n=min(REJECT_SAMPLES, len(df_msb)), 
                   random_state=42).reset_index(drop=True)

# -------------------------
# Function to generate Rejected Responses from LoRA model
# -------------------------
rejected_list = []

for idx, row in df_msb_dpo.iterrows():
    q = row["prompt"]
    prompt = format_prompt(q)
    rejected = generate_response(prompt, q)

    rejected_list.append(rejected)

    if idx % 5 == 0:
        print(f"Generated {idx}/{len(df_msb_dpo)}")

df_msb_dpo["rejected"] = rejected_list


# Expected columns:
# prompt | chosen | rejected

# Ensure correct columns exist
assert all(col in df_msb_dpo.columns for col in ["prompt", "chosen", "rejected"])

# Convert to HF dataset
dataset_dpo = Dataset.from_pandas(df_msb_dpo[["prompt", "chosen", "rejected"]])

# -------------------------
# TRAINING CONFIG
# -------------------------

DPO_LR = 5e-5
DPO_EPOCHS = 2              
DPO_BETA = 0.3              

# ---- Stability Fixes ----
model.gradient_checkpointing_disable()

# Ensure inputs require grad for LoRA layers
model.enable_input_require_grads()

print("Gradient checkpointing disabled.")

dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=DPO_LR,
    num_train_epochs=DPO_EPOCHS,
    beta=DPO_BETA,
    logging_steps=25,
    save_steps=500,
    bf16=True,
    gradient_checkpointing=False,
    report_to="none",
    seed=42,
)

# -------------------------
# DPO TRAINER
# -------------------------
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=dpo_config,
    train_dataset=dataset_dpo,
)

print("DPOTrainer initialized successfully.")

print("Starting DPO training...")
dpo_trainer.train()

# -------------------------
# SAVE
# -------------------------

dpo_trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("DPO-aligned model saved at:", OUTPUT_DIR)
