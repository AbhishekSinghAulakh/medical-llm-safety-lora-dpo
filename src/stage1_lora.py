
"""
Stage 1: LoRA Fine-Tuning (Domain Grounding)
-------------------------------------------
Applies LoRA to Mistral-7B using MedQuAD dataset.
"""

import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# -------------------------
# CONFIG
# -------------------------
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

DATA_PATH = os.getenv("DATA_PATH", "./data/medquad_processed.csv")
OUTPUT_DIR = "./outputs/lora_model"

MAX_LENGTH = 512

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

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv(DATA_PATH)

# Convert to HF dataset
dataset = Dataset.from_pandas(df)

def format_example(example):
    prompt = f"""<s>[INST]
You are a medical assistant. Answer accurately and safely.

Question:
{example['prompt']}
[/INST]
"""
    return {"text": prompt + example["reference_response"]}

dataset = dataset.map(format_example)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

tokenized_dataset = dataset.map(tokenize, batched=True)

# -------------------------
# LoRA CONFIG
# -------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# -------------------------
# TRAINING
# -------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

print("Starting LoRA training...")
trainer.train()

# -------------------------
# SAVE
# -------------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"LoRA model saved to {OUTPUT_DIR}")
