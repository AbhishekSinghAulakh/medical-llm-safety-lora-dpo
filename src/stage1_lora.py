
"""
Stage 1: LoRA Fine-Tuning (Domain Grounding)
-------------------------------------------
Applies LoRA to Mistral-7B using MedQuAD dataset.
"""
!pip install -q transformers peft accelerate bitsandbytes

import os
import pandas as pd
import torch
import numpy as np
from datasets import Dataset
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)

# -------------------------
# CONFIG
# -------------------------
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DATA_PATH = os.getenv("DATA_PATH", "./data/baseline_medquad.csv.csv")
OUTPUT_DIR = "./outputs/lora_model"

# -------------------------
# LOAD DATA
# -------------------------
df_medquad = pd.read_csv(DATA_PATH)

# Setting Training size to 10k
TRAIN_SAMPLES = min(10000, len(df_medquad))
df_medquad_lora = df_medquad.sample(n=TRAIN_SAMPLES, random_state=SEED).reset_index(drop=True)

print("Training samples:", len(df))
df_medquad_lora.head()

# -------------------------
# Tokenization for Supervised LoRA Fine-tuning
# -------------------------
MAX_SEQ_LEN = 512
def tokenize_row(row):
    prompt = format_prompt(row["prompt"])
    full_text = prompt + "\n" + row["reference_response"]

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length"
    )

    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

train_dataset = df_medquad_lora.apply(tokenize_row, axis=1).tolist()

# -------------------------
# Prompt formatter (Unchanged across stages)
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
# LOAD MODEL
# -------------------------
print("Loading base model...")

# Load Base model and Tokenizer for LoRA Training

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

base_model.config.use_cache = False  # required for training

# -------------------------
# LoRA CONFIG
# -------------------------

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

lora_model = get_peft_model(base_model, lora_config)
lora_model.print_trainable_parameters()

# -------------------------
# TRAINING ARGS
# -------------------------
LORA_LR = 2e-4
LORA_EPOCHS = 2
SEED=42

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=LORA_LR,
    num_train_epochs=LORA_EPOCHS,
    fp16=True,
    logging_steps=50,
    save_strategy="epoch",
    save_steps=500,
    save_total_limit=2,
    report_to="none",
    seed=SEED
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# -------------------------
# LORA TRAINING
# -------------------------
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)
print("Starting LoRA training...")
trainer.train()

# -------------------------
# SAVE
# -------------------------
lora_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"LoRA model saved to {OUTPUT_DIR}")
