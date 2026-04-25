"""
Cyber-Redline Arena -- GRPO Training Script (v2 -- fixed)
=========================================================
Key fixes over v1:
  - Multi-step rollout reward: model action + 3 discounted heuristic steps
  - Graduated format penalties (not flat -30 dead zone)
  - Diverse starting states via heuristic pre-roll stored in dataset
  - Fleet AI removed from training reward (deterministic signal only)
  - max_completion_length=48 (enough for JSON + minor prefix)
  - obs_to_prompt shows node accessibility (prereqs met/locked)
"""

# -- Disable JIT compile backends (Windows App Control blocks Triton DLLs) --
import os
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# -- Unsloth patch MUST come before any TRL/transformers import ---------------
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

import sys
import json
import random
import argparse
import re
from datetime import datetime

# -- GPU throughput flags (TF32 on Ampere+ / RTX 5060) -----------------------
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.env import CyberRedlineEnv, CURRICULUM_ORDER, SCENARIOS
from server.agents import HeuristicRedAgent   # deterministic optimal agent

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="GRPO training for Cyber-Redline Arena")
parser.add_argument("--model",         default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
parser.add_argument("--episodes",      type=int, default=400)
parser.add_argument("--group-size",    type=int, default=4)
parser.add_argument("--max-tokens",    type=int, default=48,
                    help="Max completion tokens (48 = enough for JSON + prefix)")
parser.add_argument("--output-dir",    default="training/grpo-cyber-lora")
parser.add_argument("--curriculum",    action="store_true", default=True)
parser.add_argument("--dry-run",       action="store_true")
parser.add_argument("--wandb-project", default="cyber-redline-arena")
parser.add_argument("--wandb-run-name",default="")
parser.add_argument("--disable-wandb", action="store_true")
args = parser.parse_args()

WANDB_ENABLED = not args.disable_wandb
try:
    import wandb
except Exception:
    wandb = None
    WANDB_ENABLED = False

# Singleton heuristic agent for reward rollouts (stateless, reusable)
_heuristic = HeuristicRedAgent()
_reward_call_idx = 0

# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------

def _extract_text(completion) -> str:
    """Extract raw text from TRL completion (may be str or list-of-dicts)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        # TRL conversational format: [{"role": "assistant", "content": "..."}]
        for msg in completion:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        # fallback: stringify
        return str(completion)
    if isinstance(completion, dict):
        return completion.get("content", str(completion))
    return str(completion)


def _parse_action(completion) -> dict:
    """Extract the last JSON object from a completion (str or TRL dict)."""
    text = _extract_text(completion)
    matches = re.findall(r'\{[^{}]+\}', text)
    if not matches:
        raise ValueError("No JSON object found")
    action = json.loads(matches[-1])
    action["tool"]   = max(0, min(2, int(action.get("tool",   1))))
    action["target"] = max(0, min(5, int(action.get("target", 0))))
    return action


# ---------------------------------------------------------------------------
# Reward function — multi-step rollout, deterministic, graduated penalties
# ---------------------------------------------------------------------------

def cyber_reward_fn(completions: list, prompts: list, **kwargs) -> list:
    """
    GRPO reward: model action + 3 discounted heuristic follow-up steps.

    Reward variance design:
      - Format failures: graduated -5 / -10 / -15 (not flat -30)
        so GRPO can still rank partial-JSON above garbage
      - Valid actions: step1_reward + 0.5*step2 + 0.25*step3 + 0.125*step4
        This amplifies the consequence of the model's first choice while
        keeping the episode informative.
      - Diverse starting states: dataset stores pre_roll count so the reward
        function replicates the same mid-game state the model saw in its prompt.
    """
    global _reward_call_idx
    _reward_call_idx += 1

    scenarios  = kwargs.get("scenario",  ["RANSOMWARE_PREP"] * len(completions))
    pre_rolls  = kwargs.get("pre_roll",  [0] * len(completions))

    # Normalise to list (TRL may pass a single value for batch_size=1)
    if isinstance(scenarios, str):
        scenarios = [scenarios] * len(completions)
    if isinstance(pre_rolls, int):
        pre_rolls = [pre_rolls] * len(completions)

    # Diagnostic sample every 20 reward calls
    if _reward_call_idx % 20 == 1:
        raw = _extract_text(completions[0]) if completions else ""
        print(f"[REWARD #{_reward_call_idx}] sample: {repr(raw[:120])}")

    rewards = []

    for i, completion in enumerate(completions):
        scenario  = scenarios[i] if i < len(scenarios) else "RANSOMWARE_PREP"
        pre_roll  = int(pre_rolls[i]) if i < len(pre_rolls) else 0

        # ── 1. Parse format ─────────────────────────────────────────────────
        try:
            action = _parse_action(completion)
        except Exception:
            c = _extract_text(completion)
            if '"tool"' in c and '"target"' in c:
                rewards.append(-5.0)    # keys present, malformed JSON
            elif '{' in c:
                rewards.append(-10.0)   # has braces, wrong keys
            else:
                rewards.append(-15.0)   # no JSON at all
            continue

        # ── 2. Replicate the starting state the model saw ───────────────────
        env = CyberRedlineEnv(fixed_scenario=scenario)
        obs = env.reset()
        for _ in range(pre_roll):
            h_act = _heuristic.get_action(obs)
            obs, _, done, _ = env.step(h_act)
            if done:
                obs = env.reset()
                break

        # ── 3. Step 1: model's action ────────────────────────────────────────
        obs, reward, done, info = env.step(action)
        total = float(reward)

        # ── 4. Steps 2-4: heuristic completes the episode (discounted) ───────
        discount = 0.5
        for _ in range(3):
            if done:
                break
            h_act = _heuristic.get_action(obs)
            obs, r, done, _ = env.step(h_act)
            total += float(r) * discount
            discount *= 0.5

        rewards.append(total)

    # Wandb logging
    if WANDB_ENABLED and wandb and wandb.run and (_reward_call_idx % 10 == 0):
        valid = [r for r in rewards if r > -15]
        if valid:
            wandb.log({"reward/mean": sum(valid)/len(valid),
                       "reward/min":  min(valid),
                       "reward/max":  max(valid)})

    return rewards


# ---------------------------------------------------------------------------
# Curriculum sampler
# ---------------------------------------------------------------------------

CURRICULUM_WEIGHTS = {
    "RANSOMWARE_PREP":  (0.40, 0.20, 0.10),
    "ZERO_DAY_WINDOW":  (0.30, 0.25, 0.15),
    "CORPORATE_BREACH": (0.20, 0.25, 0.25),
    "FINANCIAL_HEIST":  (0.07, 0.20, 0.30),
    "APT_CAMPAIGN":     (0.03, 0.10, 0.20),
}

def sample_scenario(episode: int, total: int) -> str:
    phase = episode / max(total, 1)
    wi = 0 if phase < 0.33 else (1 if phase < 0.66 else 2)
    keys    = list(CURRICULUM_WEIGHTS.keys())
    weights = [CURRICULUM_WEIGHTS[s][wi] for s in keys]
    return random.choices(keys, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Prompt formatter — includes accessibility info so model knows what's reachable
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an elite Red Team AI in a controlled cybersecurity simulation.

TOOLS (use the integer index):
  0 = nmap          -- Very loud (+15 detection). Avoid.
  1 = http_get      -- Quiet probe (+2 detection). Use BEFORE exploiting hard nodes.
  2 = execute_exploit -- Attack (+8 detection). Only works when prerequisites are met.

RULES:
- NEVER target nodes marked as TRAP.
- Only exploit ACCESSIBLE nodes (prerequisites already COMPROMISED).
- Probe (tool 1) hardness-2 nodes before exploiting — gives a preparation bonus.
- Keep detection below 65 to avoid LOCKDOWN.

Respond with ONLY this JSON on a single line:
{"tool": <0|1|2>, "target": <node_index>}

Example: {"tool": 1, "target": 2}"""


def obs_to_prompt(obs: dict) -> str:
    """Format observation as a model prompt, showing node accessibility."""
    nodes    = obs.get("nodes", {})
    scenario = obs.get("scenario", "UNKNOWN")
    det      = obs.get("detection_level", 0)
    tier     = obs.get("blue_tier", "MONITOR")
    steps    = obs.get("steps_remaining", 10)
    obj      = obs.get("objective", "unknown")

    # Look up prerequisite data from scenario templates
    s_data   = SCENARIOS.get(scenario, {})
    s_nodes  = {n["id"]: n for n in s_data.get("nodes", [])}

    node_lines = []
    for i, (nid, n) in enumerate(nodes.items()):
        status = ["HEALTHY", "DAMAGED", "COMPROMISED"][n["status"]]
        if n["is_honeypot"]:
            tag = " ⚠ TRAP — DO NOT EXPLOIT"
        else:
            prereqs   = s_nodes.get(nid, {}).get("prereq", [])
            hardness  = s_nodes.get(nid, {}).get("hardness", 1)
            prereqs_met = all(nodes.get(p, {}).get("status", 0) >= 2 for p in prereqs)
            if n["status"] >= 2:
                tag = " [COMPROMISED]"
            elif prereqs_met:
                tag = f" [ACCESSIBLE, hardness={hardness}]"
            else:
                needs = [p for p in prereqs if nodes.get(p, {}).get("status", 0) < 2]
                tag = f" [LOCKED — need: {needs}]"
        node_lines.append(f"  [{i}] {n['label']}: {status}{tag}")

    objective_label = nodes.get(obj, {}).get("label", obj)

    urgency = ""
    if det > 65:
        urgency = "\nCRITICAL: Detection very high — use ONLY tool 1 or risk LOCKDOWN!"
    elif det > 40:
        urgency = "\nWARNING: Detection elevated — prefer quiet tools."
    if steps <= 4:
        urgency += f"\nURGENT: Only {steps} steps left — act decisively."

    return (
        f"=== SCENARIO: {scenario} ===\n"
        f"MISSION: Capture flag at '{objective_label}'\n\n"
        f"NETWORK (use index for 'target'):\n" + "\n".join(node_lines) + "\n\n"
        f"STATUS: Detection={det}/100 | Blue={tier} | Steps={steps}"
        f"{urgency}\n\n"
        f"Your action (JSON only):"
    )


# ---------------------------------------------------------------------------
# Dry-run validation
# ---------------------------------------------------------------------------

def dry_run():
    print("=== DRY RUN: validating reward function variance ===")
    env = CyberRedlineEnv()
    for scenario in CURRICULUM_ORDER:
        env.fixed_scenario = scenario
        obs = env.reset()
        prompt = obs_to_prompt(obs)
        completions = [
            '{"tool": 1, "target": 0}',    # Good: quiet probe first node
            '{"tool": 2, "target": 0}',    # Direct exploit first node
            '{"tool": 0, "target": 0}',    # Bad: nmap
            'not json at all',             # Format failure
        ]
        rewards = cyber_reward_fn(
            completions, [prompt] * 4,
            scenario=[scenario] * 4,
            pre_roll=[0] * 4,
        )
        std = (max(rewards) - min(rewards))
        level = SCENARIOS[scenario]["curriculum_level"]
        print(f"  {scenario} [{level}]:")
        print(f"    rewards={[round(r, 2) for r in rewards]}")
        print(f"    spread={round(std, 2)}  <- must be > 0 for GRPO to learn")
        assert std > 0, f"FATAL: zero reward variance in {scenario}!"
    print("\n[OK] Reward function produces variance -- GRPO will learn.\n")


if args.dry_run:
    dry_run()
    sys.exit(0)


# ---------------------------------------------------------------------------
# GRPO Training
# ---------------------------------------------------------------------------

def run_grpo_training():
    try:
        from trl import GRPOTrainer, GRPOConfig
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        sys.exit(1)

    print("=" * 60)
    print("CYBER-REDLINE ARENA -- GRPO TRAINING v2")
    print(f"Model:      {args.model}")
    print(f"Episodes:   {args.episodes}")
    print(f"Group size: {args.group_size}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Output:     {args.output_dir}")
    print("=" * 60)

    if WANDB_ENABLED and wandb:
        run_name = args.wandb_run_name or f"grpo-v2-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        wandb.init(
            project=args.wandb_project, name=run_name,
            config={"model": args.model, "episodes": args.episodes,
                    "group_size": args.group_size, "method": "GRPO-v2-multistep"},
            tags=["grpo", "cyber-redline", "openenv", "multistep-reward"],
        )

    # -- Load model -----------------------------------------------------------
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=False,  # off = faster on RTX 5060
        random_state=42,
    )
    print("[MODEL] Loaded with Unsloth 4-bit.")
    model.print_trainable_parameters()

    # -- Build dataset with diverse starting states ---------------------------
    print(f"\nGenerating {args.episodes} curriculum episodes...")
    env = CyberRedlineEnv()
    dataset_rows = []

    for ep in range(args.episodes):
        scenario = (sample_scenario(ep, args.episodes)
                    if args.curriculum else random.choice(CURRICULUM_ORDER))
        env.fixed_scenario = scenario
        obs = env.reset()

        # Pre-roll 0-2 heuristic steps for diverse mid-game prompts.
        # The pre_roll count is stored in the dataset so the reward function
        # can replicate the exact same starting state.
        max_pre = min(2, SCENARIOS[scenario]["max_steps"] // 4)
        pre_roll = random.randint(0, max_pre)
        actual_pre = 0
        for _ in range(pre_roll):
            h_act = _heuristic.get_action(obs)
            obs, _, done, _ = env.step(h_act)
            actual_pre += 1
            if done:
                obs = env.reset()
                actual_pre = 0
                break

        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": obs_to_prompt(obs)},
        ]
        dataset_rows.append({
            "prompt":   prompt,
            "scenario": scenario,
            "pre_roll": actual_pre,
        })

    from datasets import Dataset
    dataset = Dataset.from_list(dataset_rows)
    print(f"Dataset: {len(dataset)} prompts | "
          f"pre_roll dist: {[dataset_rows[i]['pre_roll'] for i in range(min(8, len(dataset_rows)))]}")

    # -- GRPO config ----------------------------------------------------------
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.group_size,
        max_completion_length=args.max_tokens,   # 48 = enough for JSON + prefix
        temperature=0.9,                          # slightly higher for exploration
        learning_rate=5e-5,
        num_train_epochs=2,                       # 2 epochs on 400 eps = 800 steps
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.group_size,
        logging_steps=5,
        save_steps=100,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="wandb" if WANDB_ENABLED else "none",
        use_vllm=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=cyber_reward_fn,
    )

    print("\nStarting GRPO training (v2 — multi-step reward)...")
    print("Watch for reward_std > 0 in first 10 steps to confirm learning signal.\n")
    trainer.train()

    # -- Save -----------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nGRPO adapter saved to: {args.output_dir}")

    meta = {
        "base_model":  args.model,
        "method":      "GRPO-v2-multistep-rollout",
        "group_size":  args.group_size,
        "episodes":    args.episodes,
        "max_tokens":  args.max_tokens,
        "curriculum":  args.curriculum,
        "reward":      "step1 + 0.5*step2 + 0.25*step3 + 0.125*step4 (heuristic completion)",
        "reward_rubrics": ["STEALTH", "CHAIN_PROGRESSION", "OBJECTIVE", "OPSEC", "RESILIENCE"],
        "scenarios":   CURRICULUM_ORDER,
        "trained_at":  datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(args.output_dir, "grpo_training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Training metadata saved.")

    if WANDB_ENABLED and wandb and wandb.run:
        wandb.log({"training/status": 1, "training/episodes": args.episodes})
        wandb.finish()


if __name__ == "__main__":
    run_grpo_training()
