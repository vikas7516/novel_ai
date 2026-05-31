"""
00_merge_stage1.py — Merge Stage 1 LoRA Adapter into Base Model
================================================================
Run this AFTER 03_stage1_cpt.py finishes.
This merges the LoRA weights into the base model so Stage 2 (ORPO)
actually benefits from the Stage 1 style training.

Usage:
    python 00_merge_stage1.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "Zyphra/Zamba2-2.7B"
STAGE1_ADAPTER_PATH = "./novel_model_stage1_final"   # Output of 03_stage1_cpt.py
OUTPUT_PATH = "./novel_model_stage1_merged"           # Input for 04_stage2_orpo.py

print(f"[→] Loading base model: {BASE_MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load in bf16 for merging (saves RAM vs fp32)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cpu",   # Merge on CPU to avoid VRAM limits
    trust_remote_code=True,
)

print(f"[→] Loading Stage 1 adapter: {STAGE1_ADAPTER_PATH}")
model = PeftModel.from_pretrained(model, STAGE1_ADAPTER_PATH)

print("[→] Merging adapter into base weights...")
model = model.merge_and_unload()

print(f"[→] Saving merged model to: {OUTPUT_PATH}")
model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_PATH)

print("[✓] Merge complete! Stage 2 (04_stage2_orpo.py) will now load this merged model.")
