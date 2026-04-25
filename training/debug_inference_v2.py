import os, sys
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unsloth import FastLanguageModel
from server.env import CyberRedlineEnv
from training.grpo_training import obs_to_prompt, SYSTEM_PROMPT, _extract_text, _parse_action
import torch

MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
ADAPTER_PATH = "training/grpo-cyber-lora"

print(f"Loading model and adapter from {ADAPTER_PATH}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = 2048,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)
model.load_adapter(ADAPTER_PATH)

env = CyberRedlineEnv(fixed_scenario="RANSOMWARE_PREP")
obs = env.reset()

print("\n--- INFERENCE TEST (with System Prompt) ---")
for i in range(3):
    prompt_text = obs_to_prompt(obs)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt_text},
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids = inputs,
            max_new_tokens = 128,  # increased for safety
            temperature = 0.01,    # greedy for stability
            use_cache = True,
            pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    generated_ids = outputs[0][inputs.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"\n[Step {i+1}]")
    print(f"Response: {repr(response)}")
    
    try:
        action = _parse_action(response)
        print(f"Action: {action}")
        obs, reward, done, info = env.step(action)
        print(f"Reward: {reward} | Done: {done}")
    except Exception as e:
        print(f"Error: {e}")
        # Print a bit of the prompt to be sure
        print(f"Sample of input: {tokenizer.decode(inputs[0][:100])}...")
        break
    
    if done:
        print("Scenario Finished!")
        break
