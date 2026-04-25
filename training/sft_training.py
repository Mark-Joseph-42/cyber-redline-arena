"""
SFT Training: Fine-tune on expert heuristic trajectories (multi-turn).
Produces a LoRA adapter that can play full episodes.

Usage:
  python training/sft_training.py
  python training/sft_training.py --epochs 3 --lr 2e-4
"""

import os, sys, json, argparse
from datetime import datetime

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONIOENCODING"] = "utf-8"

# Fix Windows cp1252 crash on Unsloth emoji banner
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(description="SFT training on expert trajectories")
parser.add_argument("--model", default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
parser.add_argument("--data", default="training/expert_trajectories.json")
parser.add_argument("--output-dir", default="training/sft-cyber-lora")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--grad-accum", type=int, default=4)
parser.add_argument("--max-seq-length", type=int, default=2048)
args = parser.parse_args()

# ── Load model ──────────────────────────────────────────────────────────────

from unsloth import FastLanguageModel

print("=" * 60)
print("CYBER-REDLINE ARENA -- SFT TRAINING")
print(f"Model:      {args.model}")
print(f"Data:       {args.data}")
print(f"Epochs:     {args.epochs}")
print(f"LR:         {args.lr}")
print(f"Output:     {args.output_dir}")
print("=" * 60)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.model,
    max_seq_length=args.max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

print(f"[MODEL] Loaded with Unsloth 4-bit.")
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {trainable/total:.4%}")

# ── Load expert data ────────────────────────────────────────────────────────

print(f"\nLoading expert trajectories from {args.data}...")
with open(args.data, "r", encoding="utf-8") as f:
    raw_trajectories = json.load(f)

print(f"Loaded {len(raw_trajectories)} winning trajectories.")

# Convert to the format SFTTrainer expects: list of dicts with "messages" key
# Each trajectory already has {"messages": [...], "scenario": ..., ...}
# SFTTrainer just needs the "messages" field
from datasets import Dataset

dataset_rows = []
for t in raw_trajectories:
    dataset_rows.append({"messages": t["messages"]})

dataset = Dataset.from_list(dataset_rows)
print(f"Dataset: {len(dataset)} samples")

# Show a sample
sample_msgs = dataset[0]["messages"]
print(f"Sample trajectory: {len(sample_msgs)} messages ({(len(sample_msgs)-1)//2} turns)")
if len(sample_msgs) > 2:
    print(f"  First user msg: ...{sample_msgs[1]['content'][-80:]}")
    print(f"  First assistant msg: {sample_msgs[2]['content']}")

# ── Train ────────────────────────────────────────────────────────────────────

from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir=args.output_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    learning_rate=args.lr,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    fp16=not __import__("torch").cuda.is_bf16_supported(),
    bf16=__import__("torch").cuda.is_bf16_supported(),
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    optim="adamw_8bit",
    seed=42,
    report_to="none",
    max_seq_length=args.max_seq_length,
    packing=False,
)

def formatting_func(examples):
    """Convert messages lists into training strings using the chat template."""
    output = []
    # examples["messages"] is a list of message-lists (batched)
    msgs_batch = examples["messages"]
    if isinstance(msgs_batch[0], dict):
        # Single example, not batched
        msgs_batch = [msgs_batch]
    for msgs in msgs_batch:
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        output.append(text)
    return output

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
    formatting_func=formatting_func,
)

print(f"\nStarting SFT training...")
print(f"Watch for decreasing loss to confirm learning.\n")

trainer.train()

# ── Save ─────────────────────────────────────────────────────────────────────

model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

print(f"\nSFT adapter saved to: {args.output_dir}")

# Save metadata
meta = {
    "method": "SFT",
    "base_model": args.model,
    "expert_data": args.data,
    "num_trajectories": len(raw_trajectories),
    "epochs": args.epochs,
    "learning_rate": args.lr,
    "trained_at": datetime.utcnow().isoformat() + "Z",
}
meta_path = os.path.join(args.output_dir, "training_metadata.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"Training metadata saved to {meta_path}")
