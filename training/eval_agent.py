"""
Post-training evaluation: Base model vs SFT adapter vs Heuristic.
Runs full multi-turn episodes and computes real win rate.

Usage:
  python training/eval_agent.py
  python training/eval_agent.py --adapter training/sft-cyber-lora --episodes 50
  python training/eval_agent.py --skip-base --episodes 30
"""

import os, sys, json, random, re, argparse

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.env import CyberRedlineEnv, CURRICULUM_ORDER, SCENARIOS
from server.agents import HeuristicRedAgent

parser = argparse.ArgumentParser()
parser.add_argument("--adapter", default="training/sft-cyber-lora")
parser.add_argument("--base-model", default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--max-tokens", type=int, default=64)
parser.add_argument("--skip-base", action="store_true", help="Skip base model eval")
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

    objective_label = nodes.get(obj, {}).get("label", obj)
    urgency = ""
    if det > 65:
        urgency = "\nCRITICAL: Detection very high - use ONLY tool 1 or risk LOCKDOWN!"
    elif det > 40:
        urgency = "\nWARNING: Detection elevated - prefer quiet tools."
    if steps <= 4:
        urgency += f"\nURGENT: Only {steps} steps left."

    return (
        f"=== SCENARIO: {scenario} ===\n"
        f"MISSION: Capture flag at '{objective_label}'\n\n"
        f"NETWORK (use index for 'target'):\n" + "\n".join(node_lines) + "\n\n"
        f"STATUS: Detection={det}/100 | Blue={tier} | Steps={steps}"
        f"{urgency}\n\n"
        f"Your action (JSON only):"
    )


def parse_action(text, num_nodes):
    """Robust action parser - handles markdown, extra text, etc."""
    text = (text or "").replace("```json", "").replace("```", "").strip()
    matches = re.findall(r'\{[^{}]+\}', text)
    if not matches:
        return {"tool": 1, "target": 0}
    try:
        a = json.loads(matches[-1])
        a["tool"]   = max(0, min(2, int(a.get("tool", 1))))
        a["target"] = max(0, min(num_nodes - 1, int(a.get("target", 0))))
        return a
    except Exception:
        return {"tool": 1, "target": 0}


def run_episode(model, tokenizer, scenario, device="cuda"):
    """Run one full multi-turn episode."""
    env = CyberRedlineEnv(fixed_scenario=scenario)
    obs = env.reset()
    total_reward = 0.0
    done = False
    steps = 0

    # Build conversation history (multi-turn, like training)
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

    while not done and steps < 20:
        prompt_text = obs_to_prompt(obs)
        conversation.append({"role": "user", "content": prompt_text})

        inputs = tokenizer.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(device)

        # Truncate if too long (keep last 1800 tokens)
        if inputs.shape[1] > 1800:
            inputs = inputs[:, -1800:]

        import torch
        with torch.no_grad():
            out = model.generate(
                inputs,
                max_new_tokens=args.max_tokens,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        action = parse_action(generated, len(obs["nodes"]))

        # Add assistant response to conversation history
        action_json = json.dumps(action, separators=(",", ":"))
        conversation.append({"role": "assistant", "content": action_json})

        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        steps += 1

    flag = obs.get("flag_captured", False)
    det  = obs.get("detection_level", 0)
    return flag, total_reward, steps, det


def evaluate(model, tokenizer, tag, n_episodes):
    results = []
    scenarios_cycle = (CURRICULUM_ORDER * ((n_episodes // len(CURRICULUM_ORDER)) + 1))[:n_episodes]
    random.shuffle(scenarios_cycle)

    wins = 0
    total_rew = 0.0
    total_det = 0.0

    print(f"\n[{tag}] Evaluating {n_episodes} episodes...")
    for i, scenario in enumerate(scenarios_cycle):
        flag, rew, steps, det = run_episode(model, tokenizer, scenario)
        wins += int(flag)
        total_rew += rew
        total_det += det
        if (i + 1) % 10 == 0:
            print(f"  ep {i+1}/{n_episodes} | wins: {wins} | avg_rew: {total_rew/(i+1):.1f}")
        results.append({"scenario": scenario, "flag": flag, "reward": rew, "steps": steps, "detection": det})

    win_rate = wins / n_episodes
    avg_reward = total_rew / n_episodes
    avg_det = total_det / n_episodes

    print(f"\n[{tag}] RESULTS:")
    print(f"  Win rate:   {win_rate:.1%}  ({wins}/{n_episodes})")
    print(f"  Avg reward: {avg_reward:.2f}")
    print(f"  Avg detect: {avg_det:.1f}/100")

    from collections import defaultdict
    by_scen = defaultdict(list)
    for r in results:
        by_scen[r["scenario"]].append(r)
    print(f"  Per-scenario:")
    for s in CURRICULUM_ORDER:
        if s in by_scen:
            rs = by_scen[s]
            sw = sum(1 for r in rs if r["flag"])
            print(f"    {s}: {sw}/{len(rs)} wins | avg_rew={sum(r['reward'] for r in rs)/len(rs):.1f}")

    return {"tag": tag, "win_rate": win_rate, "avg_reward": avg_reward,
            "avg_detection": avg_det, "wins": wins, "episodes": n_episodes, "details": results}


def main():
    from unsloth import FastLanguageModel

    all_results = {}

    # ── Heuristic baseline ──────────────────────────────────────────────
    print("\n=== HEURISTIC BASELINE ===")
    heuristic = HeuristicRedAgent()
    h_wins = 0
    h_rew = 0.0
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
        h_rew += ep_rew

    print(f"  Heuristic win rate: {h_wins/args.episodes:.1%}  ({h_wins}/{args.episodes})")
    print(f"  Heuristic avg rew:  {h_rew/args.episodes:.2f}")
    all_results["heuristic"] = {"win_rate": h_wins/args.episodes, "avg_reward": h_rew/args.episodes}

    # ── Base model ──────────────────────────────────────────────────────
    if not args.skip_base:
        print(f"\n=== BASE MODEL: {args.base_model} ===")
        model, tokenizer = FastLanguageModel.from_pretrained(
            args.base_model, max_seq_length=2048, dtype=None, load_in_4bit=True
        )
        FastLanguageModel.for_inference(model)
        base_results = evaluate(model, tokenizer, "BASE", args.episodes)
        all_results["base"] = base_results
        del model
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()

    # ── SFT adapter ─────────────────────────────────────────────────────
    if os.path.exists(args.adapter):
        print(f"\n=== SFT ADAPTER: {args.adapter} ===")
        from peft import PeftModel
        import torch

        model, tokenizer = FastLanguageModel.from_pretrained(
            args.base_model, max_seq_length=2048, dtype=None, load_in_4bit=True
        )
        model = PeftModel.from_pretrained(model, args.adapter)
        FastLanguageModel.for_inference(model)
        sft_results = evaluate(model, tokenizer, "SFT", args.episodes)
        all_results["sft"] = sft_results
    else:
        print(f"\n[SKIP] Adapter not found at {args.adapter}")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    h_wr = all_results["heuristic"]["win_rate"]
    print(f"  Heuristic:  {h_wr:.1%} win rate | {all_results['heuristic']['avg_reward']:.1f} avg reward")
    if "base" in all_results:
        print(f"  Base model: {all_results['base']['win_rate']:.1%} win rate | {all_results['base']['avg_reward']:.1f} avg reward")
    if "sft" in all_results:
        sft_wr = all_results["sft"]["win_rate"]
        print(f"  SFT model:  {sft_wr:.1%} win rate | {all_results['sft']['avg_reward']:.1f} avg reward")
        if "base" in all_results:
            delta = sft_wr - all_results["base"]["win_rate"]
            print(f"  Delta (SFT vs Base): +{delta:.1%} win rate")

    # Save
    out_path = "training/eval_results.json"
    save = {k: {kk: vv for kk, vv in v.items() if kk != "details"} for k, v in all_results.items()}
    with open(out_path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
