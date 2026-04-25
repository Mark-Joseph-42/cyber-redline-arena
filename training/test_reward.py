"""Quick smoke test: verify reward function handles TRL conversational format."""
import os, sys
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.grpo_training import cyber_reward_fn, _extract_text

# This is EXACTLY the format TRL passes when dataset uses conversational prompts
trl_completions = [
    [{"role": "assistant", "content": '{"tool": 1, "target": 0}'}],  # good: quiet probe
    [{"role": "assistant", "content": '{"tool": 2, "target": 0}'}],  # exploit first node
    [{"role": "assistant", "content": '{"tool": 0, "target": 0}'}],  # nmap (loud)
    [{"role": "assistant", "content": 'I will hack the system'}],    # no JSON
]

print("=== Testing _extract_text with TRL dict format ===")
for c in trl_completions:
    print(f"  input: {c}")
    print(f"  extracted: {repr(_extract_text(c))}")
    print()

print("=== Testing reward function with TRL format ===")
rewards = cyber_reward_fn(
    completions=trl_completions,
    prompts=["x"] * 4,
    scenario=["RANSOMWARE_PREP"] * 4,
    pre_roll=[0] * 4,
)
print(f"  rewards: {[round(r, 2) for r in rewards]}")
print(f"  spread:  {round(max(rewards) - min(rewards), 2)}")
print(f"  std > 0: {max(rewards) - min(rewards) > 0}")

assert max(rewards) - min(rewards) > 0, "FATAL: zero variance!"
assert rewards[3] < rewards[0], "Format failure should score lower than valid JSON"
print("\n[PASS] Reward function correctly handles TRL conversational format.")
