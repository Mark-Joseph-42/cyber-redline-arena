"""
Generate expert trajectories from the HeuristicRedAgent.
Each trajectory is a multi-turn conversation: system + alternating user/assistant.
Only winning episodes are kept.

Usage: python training/gen_expert_data.py
"""

import os, sys, json, random

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.env import CyberRedlineEnv, CURRICULUM_ORDER, SCENARIOS
from server.agents import HeuristicRedAgent

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


def generate_trajectories(n_episodes=500, seed=42):
    random.seed(seed)
    agent = HeuristicRedAgent()
    trajectories = []
    wins = 0
    losses = 0

    # Cycle through all scenarios
    scenarios = (CURRICULUM_ORDER * ((n_episodes // len(CURRICULUM_ORDER)) + 1))[:n_episodes]
    random.shuffle(scenarios)

    for ep_idx, scenario in enumerate(scenarios):
        env = CyberRedlineEnv(fixed_scenario=scenario)
        obs = env.reset()
        done = False

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        ep_reward = 0.0
        steps = 0

        while not done:
            prompt_text = obs_to_prompt(obs)
            action = agent.get_action(obs)
            action_json = json.dumps(action, separators=(",", ":"))

            messages.append({"role": "user", "content": prompt_text})
            messages.append({"role": "assistant", "content": action_json})

            obs, reward, done, info = env.step(action)
            ep_reward += float(reward)
            steps += 1

        flag = obs.get("flag_captured", False)
        if flag:
            wins += 1
            trajectories.append({
                "messages": messages,
                "scenario": scenario,
                "reward": ep_reward,
                "steps": steps,
            })
        else:
            losses += 1

        if (ep_idx + 1) % 100 == 0:
            print(f"  Generated {ep_idx+1}/{n_episodes} | wins={wins} losses={losses}")

    print(f"\nDone: {wins} winning trajectories from {n_episodes} episodes ({wins/n_episodes:.0%} win rate)")
    return trajectories


if __name__ == "__main__":
    print("=== Generating Expert Trajectories ===")
    trajectories = generate_trajectories(n_episodes=500)

    # Save as JSON for SFT
    out_path = "training/expert_trajectories.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(trajectories, f, indent=2)
    print(f"Saved {len(trajectories)} trajectories to {out_path}")

    # Show stats
    avg_steps = sum(t["steps"] for t in trajectories) / len(trajectories)
    avg_reward = sum(t["reward"] for t in trajectories) / len(trajectories)
    avg_turns = sum(len(t["messages"]) for t in trajectories) / len(trajectories)
    print(f"Avg steps/episode: {avg_steps:.1f}")
    print(f"Avg reward: {avg_reward:.1f}")
    print(f"Avg messages/trajectory: {avg_turns:.1f}")
