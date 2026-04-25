import os, sys
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unsloth import FastLanguageModel
from server.env import CyberRedlineEnv
from training.grpo_training import obs_to_prompt, _extract_text, _parse_action
import torch

MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
ADAPTER_PATH = "training/grpo-cyber-lora"

print(f"Loading model and adapter from {ADAPTER_PATH}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = 32768,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)
model.load_adapter(ADAPTER_PATH)

env = CyberRedlineEnv(fixed_scenario="RANSOMWARE_PREP")
obs = env.reset()

print("\n--- INFERENCE TEST ---")
for i in range(5):
    prompt = obs_to_prompt(obs)
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda")

    outputs = model.generate(
        input_ids = inputs,
        max_new_tokens = 48,
        use_cache = True,
    )
    
    # Slice to get only new tokens
    generated_ids = outputs[0][inputs.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"\n[Step {i+1}]")
    print(f"Prompt end: ...{prompt[-100:]}")
    print(f"Response: {repr(response)}")
    
    try:
        action = _parse_action(response)
        print(f"Action: {action}")
        obs, reward, done, info = env.step(action)
        print(f"Reward: {reward} | Done: {done}")
    except Exception as e:
        print(f"Error: {e}")
        break
    
    if done:
        print("Scenario Finished!")
        break
