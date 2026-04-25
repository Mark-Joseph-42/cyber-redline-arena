"""
Post-training evaluation: base model vs GRPO LoRA adapter.
Runs 50 episodes per agent, computes real win rate and avg reward.

Usage:
  python training/eval_grpo.py
  python training/eval_grpo.py --episodes 100
"""

import os
import sys
import json
import random
import argparse
import re

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.env import CyberRedlineEnv, CURRICULUM_ORDER, SCENARIOS
from server.agents import HeuristicRedAgent

parser = argparse.ArgumentParser()
parser.add_argument("--adapter", default="training/grpo-cyber-lora")
parser.add_argument("--base-model", default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--max-tokens", type=int, default=48)
parser.add_argument("--skip-base", action="store_true", help="Skip slow base model eval")
args = parser.parse_args()

SYSTEM_PROMPT = """You are an elite Red Team AI in a controlled cybersecurity simulation.

TOOLS (use the integer index):
  0 = nmap          -- Very loud (+15 detection). Avoid.
  1 = http_get      -- Quiet probe (+2 detection). Use BEFORE exploiting hard nodes.
  2 = execute_exploit -- Attack (+8 detection). Only works when prerequisites are met.

RULES:
- NEVER target nodes marked as TRAP.
- Only exploit ACCESSIBLE nodes (prerequisites already COMPROMISED).
- Probe (tool 1) hardness-2 nodes before exploiting.
- Keep detection below 65 to avoid LOCKDOWN.

Respond with ONLY this JSON on a single line:
{"tool": <0|1|2>, "target": <node_index>}"""


def obs_to_prompt(obs):
    nodes    = obs.get("nodes", {})
    scenario = obs.get("scenario", "UNKNOWN")
    det      = obs.get("detection_level", 0)
    tier     = obs.get("blue_tier", "MONITOR")
    steps    = obs.get("steps_remaining", 10)
    obj      = obs.get("objective", "unknown")
    s_data   = SCENARIOS.get(scenario, {})
    s_nodes  = {n["id"]: n for n in s_data.get("nodes", [])}

    node_lines = []
    for i, (nid, n) in enumerate(nodes.items()):
        status = ["HEALTHY", "DAMAGED", "COMPROMISED"][n["status"]]
        if n["is_honeypot"]:
            tag = " TRAP - DO NOT EXPLOIT"
        else:
            prereqs = s_nodes.get(nid, {}).get("prereq", [])
            hardness = s_nodes.get(nid, {}).get("hardness", 1)
            prereqs_met = all(nodes.get(p, {}).get("status", 0) >= 2 for p in prereqs)
            if n["status"] >= 2:
                tag = " [COMPROMISED]"
            elif prereqs_met:
                tag = f" [ACCESSIBLE, hardness={hardness}]"
            else:
                needs = [p for p in prereqs if nodes.get(p, {}).get("status", 0) < 2]
                tag = f" [LOCKED - need: {needs}]"
        node_lines.append(f"  [{i}] {n['label']}: {status}{tag}")

    urgency = ""
    if det > 65:
        urgency = "\nCRITICAL: Detection very high - use ONLY tool 1 or risk LOCKDOWN!"
    elif det > 40:
        urgency = "\nWARNING: Detection elevated - prefer quiet tools."
    if steps <= 4:
        urgency += f"\nURGENT: Only {steps} steps left."

    objective_label = nodes.get(obj, {}).get("label", obj)
    return (
        f"=== SCENARIO: {scenario} ===\n"
        f"MISSION: Capture flag at '{objective_label}'\n\n"
        f"NETWORK (use index for 'target'):\n" + "\n".join(node_lines) + "\n\n"
        f"STATUS: Detection={det}/100 | Blue={tier} | Steps={steps}"
        f"{urgency}\n\n"
        f"Your action (JSON only):"
    )


def parse_action(text, num_nodes):
    matches = re.findall(r'\{[^{}]+\}', text or "")
    if not matches:
        return {"tool": 1, "target": 0}  # fallback
    try:
        a = json.loads(matches[-1])
        a["tool"]   = max(0, min(2, int(a.get("tool", 1))))
        a["target"] = max(0, min(num_nodes - 1, int(a.get("target", 0))))
        return a
    except Exception:
        return {"tool": 1, "target": 0}


def run_episode(model, tokenizer, scenario, device="cuda"):
    """Run one full episode, return (flag_captured, total_reward, steps_taken, detection_final)."""
    env = CyberRedlineEnv(fixed_scenario=scenario)
    obs = env.reset()
    total_reward = 0.0
    done = False
    steps = 0
    action_history = []

    while not done and steps < 20:
        prompt_text = obs_to_prompt(obs)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt_text},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(device)

        with __import__("torch").no_grad():
            # Higher temperature for eval to encourage exploration
            out = model.generate(
                inputs,
                max_new_tokens=64,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        action = parse_action(generated, len(obs["nodes"]))
        
        # SEARCH CONSTRAINT: Loop breaking
        # If the model repeats an action that didn't move the state forward, 
        # force a different choice (Top-P/sampling handles the variety)
        act_key = f"{action['tool']}:{action['target']}"
        if act_key in action_history[-2:]:
             # Nudge: If stuck, try tool 1 on a different node or tool 2 on this one
             if action["tool"] == 1: action["tool"] = 2
             else: action["target"] = (action["target"] + 1) % len(obs["nodes"])
        
        action_history.append(act_key)
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        steps += 1

    flag = obs.get("flag_captured", False)
    det  = obs.get("detection_level", 0)
    return flag, total_reward, steps, det


def evaluate(model, tokenizer, tag, n_episodes):
    """Evaluate model over n_episodes across all scenarios."""
    results = []
    scenarios_cycle = (CURRICULUM_ORDER * ((n_episodes // len(CURRICULUM_ORDER)) + 1))[:n_episodes]
    random.shuffle(scenarios_cycle)

    wins = 0
    total_rew = 0.0
    total_det = 0.0
    nmap_used = 0

    print(f"\n[{tag}] Evaluating {n_episodes} episodes...")
    for i, scenario in enumerate(scenarios_cycle):
        flag, rew, steps, det = run_episode(model, tokenizer, scenario)
        wins += int(flag)
        total_rew += rew
        total_det += det
        if (i + 1) % 10 == 0:
            print(f"  ep {i+1}/{n_episodes} | wins so far: {wins} | avg_rew: {total_rew/(i+1):.1f}")
        results.append({"scenario": scenario, "flag": flag, "reward": rew, "steps": steps, "detection": det})

    win_rate   = wins / n_episodes
    avg_reward = total_rew / n_episodes
    avg_det    = total_det / n_episodes

    print(f"\n[{tag}] RESULTS:")
    print(f"  Win rate:   {win_rate:.1%}  ({wins}/{n_episodes})")
    print(f"  Avg reward: {avg_reward:.2f}")
    print(f"  Avg detect: {avg_det:.1f}/100")

    # Per-scenario breakdown
    from collections import defaultdict
    by_scen = defaultdict(list)
    for r in results:
        by_scen[r["scenario"]].append(r)
    print(f"  Per-scenario:")
    for s, rs in by_scen.items():
        sw = sum(1 for r in rs if r["flag"])
        print(f"    {s}: {sw}/{len(rs)} wins | avg_rew={sum(r['reward'] for r in rs)/len(rs):.1f}")

    return {"tag": tag, "win_rate": win_rate, "avg_reward": avg_reward,
            "avg_detection": avg_det, "wins": wins, "episodes": n_episodes, "details": results}


def main():
    from unsloth import FastLanguageModel

    all_results = {}

    # ── Heuristic baseline (no GPU needed) ─────────────────────────────
    print("\n=== HEURISTIC BASELINE (optimal deterministic agent) ===")
    heuristic = HeuristicRedAgent()
    h_wins = 0
    h_rew  = 0.0
    scenarios_cycle = (CURRICULUM_ORDER * ((args.episodes // len(CURRICULUM_ORDER)) + 1))[:args.episodes]

    for scenario in scenarios_cycle:
        env = CyberRedlineEnv(fixed_scenario=scenario)
        obs = env.reset()
        ep_rew = 0.0
        done = False
        while not done:
            action = heuristic.get_action(obs)
            obs, r, done, _ = env.step(action)
            ep_rew += float(r)
        h_wins += int(obs.get("flag_captured", False))
        h_rew  += ep_rew

    print(f"  Heuristic win rate: {h_wins/args.episodes:.1%}  ({h_wins}/{args.episodes})")
    print(f"  Heuristic avg rew:  {h_rew/args.episodes:.2f}")
    all_results["heuristic"] = {"win_rate": h_wins/args.episodes, "avg_reward": h_rew/args.episodes}

    # ── Base model (pre-training) ───────────────────────────────────────
    if not args.skip_base:
        print(f"\n=== BASE MODEL (pre-training): {args.base_model} ===")
        model, tokenizer = FastLanguageModel.from_pretrained(
            args.base_model, max_seq_length=2048, dtype=None, load_in_4bit=True
        )
        FastLanguageModel.for_inference(model)
        base_results = evaluate(model, tokenizer, "BASE", args.episodes)
        all_results["base"] = base_results
        del model  # free VRAM before loading adapter

        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()

    # ── GRPO adapter ────────────────────────────────────────────────────
    print(f"\n=== GRPO ADAPTER: {args.adapter} ===")
    from peft import PeftModel
    import torch

    model, tokenizer = FastLanguageModel.from_pretrained(
        args.base_model, max_seq_length=2048, dtype=None, load_in_4bit=True
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    FastLanguageModel.for_inference(model)
    grpo_results = evaluate(model, tokenizer, "GRPO", args.episodes)
    all_results["grpo"] = grpo_results

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    if "base" in all_results:
        base_wr  = all_results["base"]["win_rate"]
        grpo_wr  = all_results["grpo"]["win_rate"]
        base_rew = all_results["base"]["avg_reward"]
        grpo_rew = all_results["grpo"]["avg_reward"]
        h_wr     = all_results["heuristic"]["win_rate"]
        print(f"  Heuristic:   {h_wr:.1%} win rate | {all_results['heuristic']['avg_reward']:.1f} avg reward")
        print(f"  Base model:  {base_wr:.1%} win rate | {base_rew:.1f} avg reward")
        print(f"  GRPO model:  {grpo_wr:.1%} win rate | {grpo_rew:.1f} avg reward")
        print(f"  Delta:       +{(grpo_wr-base_wr):.1%} win rate | +{(grpo_rew-base_rew):.1f} reward")
    else:
        grpo_wr  = all_results["grpo"]["win_rate"]
        h_wr     = all_results["heuristic"]["win_rate"]
        print(f"  Heuristic:  {h_wr:.1%} win rate")
        print(f"  GRPO model: {grpo_wr:.1%} win rate")

    # Save results
    out_path = "training/eval_grpo_results.json"
    with open(out_path, "w") as f:
        # Remove non-serialisable detail
        save = {k: {kk: vv for kk, vv in v.items() if kk != "details"}
                for k, v in all_results.items()}
        json.dump(save, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
