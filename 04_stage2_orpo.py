import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import ORPOTrainer, ORPOConfig

# ==========================================
# STAGE 2: ORPO Alignment (H100 - 3 Hours)
# ==========================================
# IMPORTANT: Before running this, run the merge step:
#   python 00_merge_stage1.py
# This loads the Stage 1 merged weights (not the raw base model).
# If you load BASE_MODEL_ID directly, Stage 1 training has NO effect here.
BASE_MODEL_ID = "./novel_model_stage1_merged"  # Stage 1 merged output
ORIGIN_MODEL_ID = "Zyphra/Zamba2-2.7B"  # Fallback if merge not done
DATASET_PATH = "orpo_dataset.jsonl"

import os
model_path = BASE_MODEL_ID if os.path.isdir(BASE_MODEL_ID) else ORIGIN_MODEL_ID
if model_path == ORIGIN_MODEL_ID:
    print("[WARN] Stage 1 merged model not found. Loading base model directly.")
    print("       Run 00_merge_stage1.py first for best results.")
print(f"[→] Loading model from: {model_path}")

tokenizer = AutoTokenizer.from_pretrained(ORIGIN_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# On an H100 (80GB), you have enough VRAM to load the model in bf16
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# Apply a very wide LoRA adapter (mimics full fine tuning without the optimizer overhead)
peft_config = LoraConfig(
    r=256,             # Massive rank for deep stylistic learning
    lora_alpha=512,
    target_modules=["in_proj", "out_proj", "x_proj", "dt_proj", "q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

print("Loading ORPO Dataset...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# ORPO requires the data in a specific format
def format_orpo(example):
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"]
    }

orpo_dataset = dataset.map(format_orpo)

orpo_config = ORPOConfig(
    output_dir="./novel_model_final_orpo",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    warmup_ratio=0.1,        # Warmup prevents odds-ratio loss destabilizing early steps
    beta=0.1,
    max_length=4096,
    max_prompt_length=1024,
    logging_steps=10,
    num_train_epochs=3,
    optim="paged_adamw_8bit",
    bf16=True,
    save_steps=200,
    report_to="none"
)

trainer = ORPOTrainer(
    model=model,
    args=orpo_config,
    train_dataset=orpo_dataset,
    tokenizer=tokenizer,
)

print("Starting Stage 2 ORPO Alignment on H100...")
trainer.train()

trainer.model.save_pretrained("./novel_model_final_weights")
tokenizer.save_pretrained("./novel_model_final_weights")
print("Training Complete! You now have a custom Writer's Soul model.")
