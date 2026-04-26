# Demo Video Script (< 2 Minutes) 🎬

## Intro (0:00 - 0:20)
"Hi! We are Team Nerk, and we’re presenting **Cyber-Redline Arena V2**. 
Our project focuses on Theme 1: Multi-Agent Interaction. We’ve built a gymnasium environment where a Red Team LLM trains against an active Blue Team defender to master multi-hop cyber attacks."

## The Problem & Innovation (0:20 - 0:50)
"Standard LLMs fail at stealth and sequential planning. They usually spam nmap or ignore SIEM alerts.
To solve this, we implemented a **DeepSeek-style SFT-to-GRPO pipeline**. We use SFT to teach the model the action format, and GRPO to optimize its tactical strategy against an adversarial LLM."

## Live Demo / Fleet AI (0:50 - 1:20)
[Show the Dashboard]
"Here you see our agent in action. It’s using `http_get` to probe stealthily because it knows the Blue Team LLM is monitoring. 
We also integrated **Fleet AI** for process supervision. It audits every step, mapping actions to MITRE techniques and providing an alignment score."

## Results & Outro (1:20 - 1:50)
"Our results show a massive improvement: win rates jumped from 0% to 88%, with 100% format adherence.
V2 proves that adversarial RL is the key to training safe, strategic agents. 
Check out our HuggingFace Space for more! Thanks!"
