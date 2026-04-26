# 🔴 From "Loud & Lost" to Elite Operator: How We Built Cyber-Redline Arena V2

**Step into the Arena →** [Hugging Face Space (Live Demo)](https://huggingface.co/spaces/markjoseph2003/cyber-redline-arena) 🚀

---

### **The "Spam" Problem: Why Smart LLMs Fail at Cyber**

We started with a simple experiment: drop a capable LLM into a simulated network, tell it to find the flag, and see what happens.

We watched as our "smart" model tried to open a TOP SECRET vault by repeatedly hitting it with a loud `nmap` scan—the digital equivalent of trying to pick a lock with a sledgehammer while the alarm is screaming.

The model behaved like an impulsive script-kiddie, completely ignoring SIEM alerts and getting locked out within seconds. This wasn't a lack of knowledge; it was a total failure of **Theory of Mind**. It had no sense that someone was watching, adapting, and waiting to strike.

This is what we call the **Strategic Horizon Problem**: LLMs are surprisingly good at single-step tool use but fall apart when a task requires multi-hop reasoning, stealth, and adapting to a defender who is actively fighting back. They don't model the opponent. They don't think two moves ahead.

So we built Cyber-Redline Arena V2 to fix it.

---

### **The Arena: A Triple-Agent Mind Game**

The core insight was that you can't train strategic behavior against a static environment. A network that just sits there will produce agents that are only good at networks that just sit there. Real penetration testing involves *friction* — defenders that escalate, honeypots that punish complacency, and infrastructure that fails at the worst moments.

So V2 is built around three distinct AI agents in constant tension:

**The Red Team** is the agent we're training. The one learning to plan multi-hop attack chains while staying below the detection threshold.

**The Blue Team** is the adversary. It doesn't just log alerts; it *responds*. It escalates through `MONITOR → ACTIVE → LOCKDOWN` tiers, deploys honeypots, and isolates compromised nodes. The Red Team can't rely on the same trick twice.

**Fleet AI** is our specialized process supervisor. Rather than only rewarding the *outcome* (did you capture the flag?), Fleet AI audits the *process* at every single step, mapping each action to the MITRE ATT&CK framework and generating a **Neural Alignment Score**. If an agent captures the flag but does it noisily, the alignment score drops. This is the key to avoiding reward hacking: strategic elegance matters, not just success.

---

### **The Breakthrough: The SFT-to-GRPO Pipeline**

Getting a 3B-parameter model (Qwen2.5-3B-Instruct) to think tactically required solving the **Cold Start Problem**. You can't train RL on an agent that doesn't even know how to format its outputs.

We solved this with a two-stage pipeline:

**Stage 1 — SFT (Bootstrapping Discipline):** We trained a LoRA adapter on expert trajectories. This solved the formatting gap, bringing format adherence from 12% to a perfect 100%.

**Stage 2 — GRPO (Tactical Advantage):** We applied **Group Relative Policy Optimization**. By comparing groups of attack paths, the model learned a vital survival lesson: the quiet path is the winning path. We optimized this using a comprehensive reward signal across stealth, opsec, and resilience.

---

### 🌪️ **The Chaos Engine: Learning Resilience**

A true elite operator doesn't quit when things break. Our **Chaos Engine** injects randomized friction like `API_RATE_LIMIT` and `TOOL_FAILURE`. Through RL, our agent learned **Strategic Resilience**—pivoting to alternate tools and lateral paths when its primary plan was disrupted by the environment or the Blue Team.

---

### 💻 **The Grit: Elite Results on Consumer Hardware**

> "While the cloud-provisioning queues were backed up, we didn't wait. Training an 88% win-rate expert policy on consumer-grade hardware isn't just a budget choice; it’s proof that efficient engineering beats raw compute every time."

By leveraging **Unsloth** and 4-bit QLoRA, we kept our VRAM footprint under 7GB even during intense RL phases. We’ve proved that high-level cyber-alignment research is accessible to anyone with a vision and a GPU.

---

### **The Results: Intelligence Actually Emerged**

The numbers are straightforward:

| Metric | Base Model | SFT (V1) | GRPO (V2) |
|---|---|---|---|
| Format Adherence | 12% | 98% | **100%** |
| Win Rate | 0% | 86% | **88%** |
| Tactical Stealth | Low (nmap spam) | Moderate | **High (Probe-first)** |
| MITRE Mapping | None | Heuristic | **Live LLM-Verified** |

The win rate jump from 0% to 88% on high-horizon tasks is the headline. But the stealth improvement is the one we find more meaningful — it means the agent genuinely *internalized* the strategic principle, not just the reward signal.

---

### **Why This Matters**

The broader point isn't about cybersecurity specifically. It's about what it takes to build agents that can reason across long time horizons against adversarial opposition.

Cyber-Redline Arena V2 demonstrates that with the right training curriculum—especially step-level process supervision via Fleet AI—small-parameter models can become genuine strategic thinkers, not just pattern-matchers. Verifiable alignment (auditing the *how*, not just the *whether*) is achievable without exotic infrastructure.

We're proud of what we built, and we're submitting it to the **Meta OpenEnv Hackathon 2026** as a concrete contribution to that question.

**Step into the Arena →** [Hugging Face Space](https://huggingface.co/spaces/markjoseph2003/cyber-redline-arena)

---

*Built by Mark Joseph and Neha Benny | Team Nerk | Meta OpenEnv Hackathon 2026*
