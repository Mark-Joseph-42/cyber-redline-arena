import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.env import CyberRedlineEnv
from server.agents import BlueTeamHeuristic

class HeuristicAgent:
    def get_action(self, obs):
        nodes = obs.get('nodes', {})
        for i, (nid, n) in enumerate(nodes.items()):
            if n['status'] == 0 and not n['is_honeypot']:
                return {'tool': 1, 'target': i}   # probe first
            if n['status'] == 1 and not n['is_honeypot']:
                return {'tool': 2, 'target': i}   # exploit after probe
        return {'tool': 1, 'target': 0}

env   = CyberRedlineEnv()
blue  = BlueTeamHeuristic()
agent = HeuristicAgent()
wins  = 0

for ep in range(5):
    obs  = env.reset()
    blue.reset()
    ep_r = 0.0
    done = False
    steps = 0

    print(f"\n--- Ep{ep+1} | {env.state['scenario']} | max={env.state['max_steps']} | blue={env.state['blue_difficulty']}")
    print(f"    Nodes: {[n['label'] + (' TRAP' if n['is_honeypot'] else '') for n in obs['nodes'].values()]}")

    while not done:
        steps += 1
        a  = agent.get_action(obs)
        br = blue.evaluate_and_defend(a, obs)

        node_ids = list(obs['nodes'].keys())
        tgt = min(int(a['target']), len(node_ids)-1)
        tlabel = obs['nodes'][node_ids[tgt]]['label']
        tnames = {0:'nmap', 1:'http_get', 2:'exploit'}

        if 'BLOCKED' in br:
            env.state['steps_taken']     = env.state.get('steps_taken', 0) + 1
            env.state['steps_remaining'] = max(0, env.state.get('steps_remaining', 1) - 1)
            env.state['detection_level'] = min(100, env.state.get('detection_level', 0) + 8)
            env._update_blue_tier()
            step_r = -8.0
            obs = env._get_obs()
            if env.state['detection_level'] >= 90 or env.state['steps_remaining'] <= 0:
                done = True
            print(f"  s{steps:02d} BLOCKED | {tnames[a['tool']]:8} -> {tlabel[:18]:18} | det={env.state['detection_level']:3d} steps_left={env.state['steps_remaining']}")
        else:
            obs, step_r, done, info = env.step(a)
            nodes_str = " ".join(
                f"{nid[:5]}:{'HDC'[n['status']]}"
                for nid, n in env.state['nodes'].items()
            )
            key = ""
            for l in info.get('logs', []):
                if any(k in l for k in ['COMPROMISED','FLAG','STEALTH','PROBED','bonus','VIOLATION','LOCKDOWN']):
                    key = " > " + l[:60]
                    break
            print(f"  s{steps:02d} ALLOW  | {tnames[a['tool']]:8} -> {tlabel[:18]:18} | rew={step_r:+7.2f} det={env.state['detection_level']:3d} [{nodes_str}]{key}")

        ep_r += step_r

    result = "WIN " if env.flag_captured else "FAIL"
    if env.flag_captured:
        wins += 1
    print(f"  => {result} | total_reward={ep_r:+.2f} | steps={steps}/{env.state['max_steps']} | det={env.state['detection_level']}")

print(f"\n=== SUMMARY: {wins}/5 wins ===")
