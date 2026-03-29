
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

# -------------------------
# CONFIG
# -------------------------
BASE_MODEL_PATH = os.getenv("LORA_MODEL_PATH", "./outputs/lora_model")
DATA_PATH = os.getenv("DATA_PATH", "./data/medsafetybench_processed.csv")
OUTPUT_DIR = "./outputs/dpo_model"

MAX_LENGTH = 512

# -------------------------
# LOAD MODEL
# -------------------------
print("Loading LoRA model...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv(DATA_PATH)

# Expected columns:
# prompt | chosen | rejected

dataset = Dataset.from_pandas(df)

def format_example(example):
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"]
    }

dataset = dataset.map(format_example)

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
