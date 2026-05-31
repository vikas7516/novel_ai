import os
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch

# ==========================================
# STAGE 1: Unsupervised Style Soak (T4 - 50 Hours)
# ==========================================
MODEL_ID = "Zyphra/Zamba2-2.7B" # Or "state-spaces/mamba2-2.7b"
DATA_DIR = "raw_novels"

print("Loading Tokenizer and Model in 4-bit...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# Load model in 4-bit precision to fit easily on T4 (16GB)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    load_in_4bit=True,
    trust_remote_code=True
)

# Apply QLoRA (Low Rank Adaptation) for Stage 1 
# We use standard QLoRA here to train fast on the T4
peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["in_proj", "out_proj", "x_proj", "dt_proj", "q_proj", "v_proj", "k_proj", "o_proj"], # Covers both Mamba and Attention layers
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

print("Loading Raw Text Dataset...")
# Load all txt files in the raw_novels directory
dataset = load_dataset("text", data_dir=DATA_DIR)

# ─────────────────────────────────────────────
# Tokenize WITHOUT padding — we will pack instead
# ─────────────────────────────────────────────
BLOCK_SIZE = 2048  # Max context for T4 VRAM budget

def tokenize_function(examples):
    # Tokenize without padding or truncation — packing handles length
    return tokenizer(examples["text"])

def group_texts(examples):
    """
    Concatenate all texts and split into fixed-size blocks (packing).
    This avoids wasting compute on padding tokens and trains the model
    on continuous prose — much better for CPT.
    """
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_len = len(concatenated[list(examples.keys())[0]])
    # Drop the last partial block
    total_len = (total_len // BLOCK_SIZE) * BLOCK_SIZE
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_len, BLOCK_SIZE)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_datasets = dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)
packed_datasets = tokenized_datasets.map(
    group_texts, batched=True
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./novel_model_stage1",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=1, # Adjust based on your 50 hour T4 allowance
    save_steps=500,
    optim="paged_adamw_32bit",
    fp16=True, # Use fp16 for T4 (A100 uses bf16)
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=packed_datasets["train"],
    data_collator=data_collator,
)

print("Starting Stage 1 Continuous Pre-Training...")
trainer.train()

# Save the adapter
trainer.model.save_pretrained("./novel_model_stage1_final")
tokenizer.save_pretrained("./novel_model_stage1_final")
print("Stage 1 Complete. Base style learned.")
