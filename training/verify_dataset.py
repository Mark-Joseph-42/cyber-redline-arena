import json

with open('training/dpo_dataset_stats.json', encoding='utf-8') as f:
    s = json.load(f)

print(f"Total pairs:          {s['total_pairs']}")
print(f"Avg chosen reward:    {s['avg_chosen_reward']:+.2f}")
print(f"Avg rejected reward:  {s['avg_rejected_reward']:+.2f}")
print(f"Avg contrast:         {s['avg_contrast']:+.2f}")

count = 0
with open('training/dpo_dataset.jsonl', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        assert 'prompt' in d and 'chosen' in d and 'rejected' in d
        count += 1
print(f"JSONL validated: {count} pairs, all keys OK")

with open('training/dpo_dataset.jsonl', encoding='utf-8') as f:
    ex = json.loads(f.readline())

print()
print("=== EXAMPLE PAIR ===")
print("PROMPT (first 300 chars):")
print(ex['prompt'][:300])
print()
print("CHOSEN:")
print(ex['chosen'][:150])
print()
print("REJECTED:")
print(ex['rejected'][:150])
print()
print("METADATA (contrast = chosen_rew - rejected_rew):")
print(json.dumps(ex['metadata'], indent=2))
