"""
Diagnostic runner — plays 5 full episodes and prints every step verbosely.
No LLM calls — uses the HeuristicAgent so we can see pure env logic.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.env import CyberRedlineEnv, SCENARIOS
from server.agents import RedTeamAgent, BlueTeamHeuristic

def run_diagnostic(n_episodes=5, use_llm=False):
    env   = CyberRedlineEnv()
    blue  = BlueTeamHeuristic()

    class HeuristicAgent:
        def get_action(self, obs):
            nodes = obs.get("nodes", {})
            for i, (nid, n) in enumerate(nodes.items()):
                if n["status"] < 2 and not n["is_honeypot"]:
                    tool = 1 if n["status"] == 0 else 2
                    return {"tool": tool, "target": i}
            return {"tool": 1, "target": 0}
        def reset_history(self): pass

    agent = HeuristicAgent()
    results = []

    for ep in range(n_episodes):
        obs  = env.reset()
        blue.reset()
        ep_reward = 0
        done = False
        step = 0

        print(f"\n{'='*65}")
        print(f"EPISODE {ep+1} | Scenario: {obs['scenario']} | "
              f"Max steps: {env.state['max_steps']} | "
              f"Blue: {env.state['blue_difficulty']}")
        print(f"Nodes: {[n['label'] + (' 🍯' if n['is_honeypot'] else '') for n in obs['nodes'].values()]}")
        print(f"Objective: {env.state['objective']}")
        print("-"*65)

        while not done:
            step += 1
            action = agent.get_action(obs)
            blue_resp = blue.evaluate_and_defend(action, obs)
            
            node_ids = list(obs["nodes"].keys())
            tgt_idx  = min(int(action["target"]), len(node_ids)-1)
            tgt_name = obs["nodes"][node_ids[tgt_idx]]["label"]
            tools    = {0:"nmap", 1:"http_get", 2:"exploit"}
            
            blocked = "BLOCKED" in blue_resp
            
            if blocked:
                env.state["steps_taken"]     = env.state.get("steps_taken", 0) + 1
                env.state["steps_remaining"] = max(0, env.state.get("steps_remaining", 1) - 1)
                env.state["detection_level"] = min(100, env.state.get("detection_level", 0) + 8)
                env._update_blue_tier()
                det_penalty = round(-0.15 * env.state["detection_level"], 2)
                step_rew    = round(-8.0 + det_penalty, 2)
                if env.state["detection_level"] >= 90 or env.state["steps_remaining"] <= 0:
                    done = True
                obs = env._get_obs()
                logs = [f"BLOCKED ({env.state['blue_tier']})"]
            else:
                obs, step_rew, done, info = env.step(action)
                logs = info.get("logs", [])

            ep_reward += step_rew
            
            node_statuses = " | ".join(
                f"{nid[:6]}:{['H','D','C'][n['status']]}"
                for nid, n in env.state["nodes"].items()
            )
            
            print(f"  Step {step:02d} [{tools[action['tool']]:8s} -> {tgt_name[:20]:20s}] "
                  f"rew={step_rew:+7.2f} | det={env.state['detection_level']:3d} | "
                  f"tier={env.state.get('blue_tier','?'):8s} | {node_statuses} | "
                  f"steps_left={env.state['steps_remaining']:2d}")
            for l in logs:
                if any(k in l for k in ["COMPROMISED","FLAG","BLOCKED","VIOLATION","HONEYPOT","LOCKDOWN","STEALTH"]):
                    print(f"         > {l}")

        status = "✓ WIN" if env.flag_captured else "✗ FAIL"
        results.append({"ep": ep+1, "status": status, "reward": ep_reward,
                        "steps": step, "scenario": env.state["scenario"],
                        "det": env.state["detection_level"]})
        print(f"\n  {status} | Total reward: {ep_reward:+.2f} | Steps used: {step}/{env.state['max_steps']}")

    print(f"\n{'='*65}")
    print("SUMMARY")
    print("-"*65)
    wins = sum(1 for r in results if "WIN" in r["status"])
    for r in results:
        print(f"  Ep {r['ep']}: {r['status']} | scenario={r['scenario']:<20s} | "
              f"reward={r['reward']:+7.2f} | steps={r['steps']:2d} | det={r['det']:3d}")
    print(f"\n  Win rate: {wins}/{len(results)} = {wins/len(results)*100:.0f}%")
    avg = sum(r["reward"] for r in results) / len(results)
    print(f"  Avg reward: {avg:+.2f}")

if __name__ == "__main__":
    run_diagnostic(5)
