# Mini-Blog Draft: Cyber-Redline Arena V2 🔴

## The Goal
LLMs in cybersecurity often fail at long-horizon planning. They either spam noisy tools (nmap) or ignore defensive responses. **Cyber-Redline Arena V2** is built to train agents that can navigate complex, multi-hop attack graphs under active pressure from an adversarial LLM-based Blue Team.

## The Innovation: SFT-to-GRPO
We implemented a DeepSeek-style pipeline:
1. **SFT**: We bootstrap the model using 447 expert trajectories to lock in the JSON format and basic node navigation.
2. **GRPO**: We then use Group Relative Policy Optimization to optimize tactical decision-making against an active defender. This forces the model to learn **stealth** as a survival strategy, not just a heuristic.

## Real-Time Oversight: Fleet AI
To ensure safety and alignment, we integrated **Fleet AI** for process supervision. It audits every action against MITRE ATT&CK techniques, providing a transparency layer that outcome-based rewards alone can't offer.

## Results
- **Format Adherence**: 12% (Base) -> 100% (GRPO)
- **Win Rate**: 0% (Base) -> 88% (GRPO)

---

## 🔗 Try it Yourself
*   **Live Arena**: [Cyber-Redline Arena V2](https://huggingface.co/spaces/markjoseph2003/cyber-redline-arena)
*   **Training Proof**: [GRPO Training Colab](./CYBER_REDLINE_GRPO_TRAINING.ipynb) - Run our pipeline and verify the reward curves!
*   **Model Weights**: [Neural Policy V2](https://huggingface.co/markjoseph2003/cyber-redline-qwen-grpo)
