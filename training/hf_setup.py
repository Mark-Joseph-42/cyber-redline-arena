#!/usr/bin/env python3
"""HuggingFace login and model download for DPO training."""
import subprocess, sys, os

# Read token from environment variable — never hardcode tokens in source files
TOKEN = os.environ.get("HF_TOKEN", "")
if not TOKEN:
    TOKEN = input("Enter your HuggingFace token: ").strip()

print("=== Upgrading HuggingFace Hub ===")
os.system("pip3 install -q --upgrade huggingface_hub transformers")

print("=== HuggingFace Login ===")
from huggingface_hub import login
login(token=TOKEN, add_to_git_credential=False)
print("Login OK")

print("\n=== Pre-downloading Qwen/Qwen2.5-4B-Instruct ===")
print("Downloading model weights (~8GB) — takes 10-20 min on typical connection...")

from transformers import AutoTokenizer, AutoConfig

# Download tokenizer first (fast)
print("Step 1/2: Downloading tokenizer...")
tok = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-4B-Instruct",
    token=TOKEN,
    trust_remote_code=True
)
print(f"Tokenizer OK — vocab size: {tok.vocab_size}")

# Download config (validates model access)
print("Step 2/2: Verifying model access...")
cfg = AutoConfig.from_pretrained(
    "Qwen/Qwen2.5-4B-Instruct",
    token=TOKEN,
    trust_remote_code=True
)
print(f"Config OK — model type: {cfg.model_type}")
print("\nFull model weights will be downloaded automatically when training starts.")
print("Ready to train!")
