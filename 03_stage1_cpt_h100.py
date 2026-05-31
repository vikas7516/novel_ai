import os
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch

# ==========================================
# STAGE 1: Unsupervised Style Soak (H100 Optimized)
# ==========================================
MODEL_ID = "Zyphra/Zamba2-2.7B" # Or "state-spaces/mamba2-2.7b"
DATA_DIR = "raw_novels"

print("Loading Tokenizer and Model in Native Bfloat16...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# Load model in full bfloat16 (H100 has plenty of VRAM, no quantization overhead needed)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# Apply LoRA
peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["in_proj", "out_proj", "x_proj", "dt_proj", "q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

print("Loading Raw Text Dataset...")
dataset = load_dataset("text", data_dir=DATA_DIR)

# ─────────────────────────────────────────────
# Tokenize WITHOUT padding — packing
# ─────────────────────────────────────────────
BLOCK_SIZE = 2048

def tokenize_function(examples):
    return tokenizer(examples["text"])

def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_len = len(concatenated[list(examples.keys())[0]])
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
    per_device_train_batch_size=8,        # Increased batch size for H100
    gradient_accumulation_steps=4,       # Total batch size = 32 (65,536 tokens/step)
    learning_rate=2e-4,
    logging_steps=5,
    num_train_epochs=3,                  # Let's train for 3 epochs on the test data
    save_steps=100,
    optim="adamw_torch",                 # Native PyTorch AdamW is faster on H100 than paged versions
    bf16=True,                           # native bfloat16 on H100
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
