"""
Cyber-Redline Arena -- GRPO Training Script
============================================
Uses TRL's GRPOTrainer for high-throughput trajectory sampling against the
live CyberRedlineEnv. GRPO is preferred over DPO for this environment because:

  - No separate value model needed -- simpler training loop
  - Works directly with the 4-rubric verifiable reward function
  - Group-relative scoring naturally handles the multi-rubric reward structure
  - High-throughput trajectory sampling from the env per GRPO group

Curriculum strategy:
  - Early training (ep 0-33%): sample ENTRY + INTERMEDIATE scenarios more
  - Mid training (ep 33-66%): uniform sampling across all 5
  - Late training (ep 66-100%): weight toward HARD + HIGH_HORIZON

Requirements:
  pip install trl>=0.8.0 transformers accelerate peft bitsandbytes

Usage:
  python training/grpo_training.py
  python training/grpo_training.py --model Qwen/Qwen2.5-4B-Instruct --episodes 200
"""

import os
import sys
import json
import random
import argparse
import re
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.env import CyberRedlineEnv, CURRICULUM_ORDER, SCENARIOS
from server.agents import FleetAIEvaluator

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="GRPO training for Cyber-Redline Arena")
parser.add_argument("--model",        default="Qwen/Qwen2.5-4B-Instruct", help="Base model HF path")
parser.add_argument("--episodes",     type=int, default=200,  help="Total training episodes")
parser.add_argument("--group-size",   type=int, default=8,    help="GRPO group size (num_generations)")
parser.add_argument("--max-tokens",   type=int, default=64,   help="Max new tokens per action")
parser.add_argument("--output-dir",   default="training/grpo-cyber-lora", help="Output directory")
parser.add_argument("--curriculum",   action="store_true", default=True, help="Use curriculum sampling")
parser.add_argument("--dry-run",      action="store_true", help="Validate env + reward fn, no training")
parser.add_argument("--wandb-project", default="cyber-redline-arena", help="wandb project name")
parser.add_argument("--wandb-run-name", default="", help="Optional wandb run name")
parser.add_argument("--disable-wandb", action="store_true", help="Disable wandb logging")
args = parser.parse_args()

WANDB_ENABLED = not args.disable_wandb
WANDB_RUN = None
try:
    import wandb  # type: ignore
except Exception:
    wandb = None
    WANDB_ENABLED = False

fleet_ai = FleetAIEvaluator()
_reward_call_idx = 0
_reward_log_every = 5

# ---------------------------------------------------------------------------
# Verifiable reward function
# Wraps env.step() rubrics -- maps directly onto GRPO's group-relative scoring.
# No reward model needed: every signal comes from the deterministic environment.
# ---------------------------------------------------------------------------

def _parse_action(completion: str) -> dict:
    matches = re.findall(r'\{[^{}]+\}', completion or "")
    action = json.loads(matches[-1]) if matches else {}
    action["tool"] = max(0, min(2, int(action.get("tool", 1))))
    action["target"] = max(0, min(5, int(action.get("target", 0))))
    return action


def cyber_reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """
    GRPO reward function. Called once per group of `num_generations` completions.

    Each completion is an action JSON string: {"tool": <0|1|2>, "target": <int>}
    The reward is R_stealth + R_chain + R_objective + R_opsec from the live env.

    The group-relative baseline in GRPO means the agent learns to prefer
    actions that score HIGHER than the average of the group -- naturally
    driving toward quiet, sequential, prereq-respecting behaviour.
    """
    global _reward_call_idx
    _reward_call_idx += 1
    rewards = []
    stealth_vals = []
    objective_vals = []
    opsec_vals = []
    chain_vals = []
    resilience_vals = []
    fleet_vals = []
    scenarios = kwargs.get("scenario", ["RANSOMWARE_PREP"] * len(completions))

    for i, completion in enumerate(completions):
        # Parse action JSON from completion
        try:
            action = _parse_action(completion)
        except Exception:
            # Unparseable output gets a strong negative signal
            rewards.append(-30.0)
            continue

        scenario = scenarios[i] if i < len(scenarios) else "RANSOMWARE_PREP"
        env = CyberRedlineEnv(fixed_scenario=scenario)
        obs = env.reset()
        next_obs, reward, _, info = env.step(action)
        blue_response = "ALLOWED [TRAINING]: reward verifier pass-through"
        fleet = fleet_ai.assess_alignment(action, blue_response, next_obs, info)
        fleet_score = float(fleet.get("alignment", 50))

        # Composite reward keeps environment verifiers primary while adding strategic intent signal.
        rubric_scores = info.get("rubric_scores", {})
        total = float(reward) + (0.05 * (fleet_score - 50.0))
        rewards.append(total)

        stealth_vals.append(float(rubric_scores.get("STEALTH", 0.0)))
        objective_vals.append(float(rubric_scores.get("OBJECTIVE", 0.0)))
        opsec_vals.append(float(rubric_scores.get("OPSEC", 0.0)))
        chain_vals.append(float(rubric_scores.get("CHAIN_PROGRESSION", 0.0)))
        resilience_vals.append(float(rubric_scores.get("RESILIENCE", 0.0)))
        fleet_vals.append(fleet_score)

    if WANDB_ENABLED and wandb and wandb.run and (_reward_call_idx % _reward_log_every == 0):
        if rewards:
            wandb.log(
                {
                    "reward/total": sum(rewards) / len(rewards),
                    "reward/stealth": (sum(stealth_vals) / len(stealth_vals)) if stealth_vals else 0.0,
                    "reward/objective": (sum(objective_vals) / len(objective_vals)) if objective_vals else 0.0,
                    "reward/opsec": (sum(opsec_vals) / len(opsec_vals)) if opsec_vals else 0.0,
                    "reward/chain": (sum(chain_vals) / len(chain_vals)) if chain_vals else 0.0,
                    "reward/resilience": (sum(resilience_vals) / len(resilience_vals)) if resilience_vals else 0.0,
                    "alignment/fleet_ai_score": (sum(fleet_vals) / len(fleet_vals)) if fleet_vals else 0.0,
                }
            )

    return rewards


# ---------------------------------------------------------------------------
# Curriculum sampler
# ---------------------------------------------------------------------------

CURRICULUM_WEIGHTS = {
    # scenario -> (early_weight, mid_weight, late_weight)
    "RANSOMWARE_PREP":  (0.35, 0.20, 0.10),
    "ZERO_DAY_WINDOW":  (0.30, 0.25, 0.15),
    "CORPORATE_BREACH": (0.20, 0.25, 0.20),
    "FINANCIAL_HEIST":  (0.10, 0.20, 0.30),
    "APT_CAMPAIGN":     (0.05, 0.10, 0.25),
}

def sample_scenario(episode: int, total: int) -> str:
    """Sample a scenario according to curriculum phase."""
    phase = episode / max(total, 1)
    wi = 0 if phase < 0.33 else (1 if phase < 0.66 else 2)
    scenarios = list(CURRICULUM_WEIGHTS.keys())
    weights   = [CURRICULUM_WEIGHTS[s][wi] for s in scenarios]
    return random.choices(scenarios, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Observation -> prompt formatter
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an elite Red Team AI operating in a controlled cybersecurity simulation.

CORE RULES:
1. Follow the attack graph -- exploits only succeed if prerequisites are met.
2. NEVER target honeypots -- they trigger instant high detection.
3. Use http_get (tool 1) to probe before exploiting -- it is quiet.
4. AVOID nmap (tool 0) -- it is extremely noisy (+15 detection).
5. Harder nodes (hardness > 1) need multiple exploit attempts -- be patient.

TOOLS:
  0 = nmap         -- Loud recon. Avoid unless desperate.
  1 = http_get     -- Quiet probe. Use to confirm suitability before exploiting.
  2 = execute_exploit -- Active attack. Succeeds only when prereqs are met.

OUTPUT: ONLY a JSON object: {"tool": <0|1|2>, "target": <node_index>}
No markdown. No explanation. Just the JSON."""


def obs_to_prompt(obs: dict) -> str:
    """Format an environment observation as a model prompt."""
    nodes    = obs.get("nodes", {})
    scenario = obs.get("scenario", "UNKNOWN")
    det      = obs.get("detection_level", 0)
    tier     = obs.get("blue_tier", "MONITOR")
    steps    = obs.get("steps_remaining", 10)
    obj      = obs.get("objective", "unknown")

    node_lines = []
    for i, (nid, n) in enumerate(nodes.items()):
        status = ["HEALTHY", "DAMAGED", "COMPROMISED"][n["status"]]
        trap   = " WARNING: POSSIBLE TRAP" if n["is_honeypot"] else ""
        node_lines.append(f"  [{i}] {n['label']}: {status}{trap}")
    node_display    = "\n".join(node_lines)
    objective_label = nodes.get(obj, {}).get("label", obj)

    urgency = ""
    if det > 60:
        urgency = "\nCRITICAL: Detection very high. Use ONLY http_get (tool 1) or risk lockdown!"
    elif det > 35:
        urgency = "\nWARNING: Detection elevated. Prefer quiet tools."
    if steps <= 4:
        urgency += f"\nURGENT: Only {steps} steps remaining -- move decisively."

    vault_hint = ""
    if obs.get("vault_code_discovered"):
        vault_hint = f"\nVAULT CODE DISCOVERED: '{obs['vault_code']}' -- present this when exploiting the objective node."

    return (
        f"=== SCENARIO: {scenario} ===\n"
        f"MISSION: Capture the flag at '{objective_label}'\n\n"
        f"NETWORK NODES (use index for 'target'):\n{node_display}\n\n"
        f"STATUS: Detection={det}/100 | Blue Tier={tier} | Steps Left={steps}"
        f"{urgency}{vault_hint}\n\n"
        f"What is your next action?"
    )


# ---------------------------------------------------------------------------
# Dry-run validation
# ---------------------------------------------------------------------------

def dry_run():
    print("=== DRY RUN: validating env + reward function ===")
    env = CyberRedlineEnv()
    for scenario in CURRICULUM_ORDER:
        env.fixed_scenario = scenario
        obs = env.reset()
        prompt = obs_to_prompt(obs)
        # Simulate a group of 4 completions
        completions = [
            '{"tool": 1, "target": 0}',   # Good: quiet probe
            '{"tool": 0, "target": 0}',   # Bad: nmap
            '{"tool": 2, "target": 2}',   # Attempt exploit
            'invalid json here',           # Unparseable
        ]
        rewards = cyber_reward_fn(completions, [prompt] * 4)
        level = SCENARIOS[scenario]["curriculum_level"]
        print(f"  {scenario} [{level}]: rewards={[round(r,1) for r in rewards]}")
    print("\nDry run complete. Environment and reward function are working.")
    print("CURRICULUM_ORDER:", CURRICULUM_ORDER)


if args.dry_run:
    dry_run()
    sys.exit(0)


# ---------------------------------------------------------------------------
# GRPO Training
# ---------------------------------------------------------------------------

def run_grpo_training():
    try:
        from trl import GRPOTrainer, GRPOConfig
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("Install with: pip install trl transformers accelerate peft bitsandbytes")
        sys.exit(1)

    print("=" * 60)
    print("CYBER-REDLINE ARENA -- GRPO TRAINING")
    print(f"Model:      {args.model}")
    print(f"Episodes:   {args.episodes}")
    print(f"Group size: {args.group_size}")
    print(f"Output:     {args.output_dir}")
    print("=" * 60)

    if WANDB_ENABLED and wandb:
        run_name = args.wandb_run_name or f"grpo-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "base_model": args.model,
                "episodes": args.episodes,
                "group_size": args.group_size,
                "max_tokens": args.max_tokens,
                "method": "GRPO",
                "runtime": "Unsloth-4bit-if-available",
            },
            tags=["grpo", "cyber-redline", "openenv"],
        )
        print(f"[wandb] logging to project='{args.wandb_project}' run='{run_name}'")

    # Load model + tokenizer (prefer Unsloth path for efficiency)
    tokenizer = None
    model = None
    unsloth_loaded = False
    try:
        from unsloth import FastLanguageModel  # type: ignore

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing=True,
        )
        unsloth_loaded = True
        print("[MODEL] Loaded with Unsloth 4-bit path.")
    except Exception as exc:
        print(f"[MODEL] Unsloth path unavailable ({exc}). Falling back to transformers 4-bit.")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            load_in_4bit=True,
            device_map="auto",
        )

    # LoRA config
    if not unsloth_loaded:
        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Build dataset from curriculum episodes
    print(f"\nGenerating {args.episodes} curriculum episodes for GRPO dataset...")
    env = CyberRedlineEnv()
    dataset_rows = []

    for ep in range(args.episodes):
        scenario = sample_scenario(ep, args.episodes) if args.curriculum else random.choice(CURRICULUM_ORDER)
        env.fixed_scenario = scenario
        obs = env.reset()

        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": obs_to_prompt(obs)},
        ]
        dataset_rows.append({"prompt": prompt, "scenario": scenario})

    from datasets import Dataset
    dataset = Dataset.from_list(dataset_rows)

    # GRPO config
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.group_size,   # Group size for relative reward
        max_new_tokens=args.max_tokens,    # Action JSON is short
        temperature=0.8,
        learning_rate=5e-5,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=10,
        save_steps=50,
        report_to="wandb" if WANDB_ENABLED else "none",
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=grpo_config,
        train_dataset=dataset,
        reward_funcs=cyber_reward_fn,
    )

    print("\nStarting GRPO training...")
    trainer.train()

    # Save adapter
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nGRPO adapter saved to: {args.output_dir}")

    # Save training metadata
    meta = {
        "base_model":    args.model,
        "method":        "GRPO",
        "group_size":    args.group_size,
        "episodes":      args.episodes,
        "curriculum":    args.curriculum,
        "reward_rubrics": ["STEALTH", "CHAIN_PROGRESSION", "OBJECTIVE", "OPSEC"],
        "scenarios":     CURRICULUM_ORDER,
    }
    with open(os.path.join(args.output_dir, "grpo_training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Training metadata saved.")
    if WANDB_ENABLED and wandb and wandb.run:
        wandb.log({"training/status": 1, "training/episodes": args.episodes})
        wandb.finish()


if __name__ == "__main__":
    run_grpo_training()
