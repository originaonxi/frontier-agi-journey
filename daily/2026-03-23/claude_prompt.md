# Day 3 — Claude Code Prompt
# Binary vs Dense Reward Ablation on 200 Live Leads
# Copy this entire file and paste into Claude Code

I am Anmol Chaudhary (Sam Anmol). CTO @ Aonxi, $650K ARR.
Ex-Meta Ads ML. Ex-Apple Face ID. NeurIPS 2026 submission.
Day 3 of 365. Goal: $1M/year frontier lab role.

TODAY'S TASK: Binary vs Dense Reward Ablation on 200 Live Leads

{
  "paper_title": "RLVR is Not RL: Understanding the Role of Verifiable Rewards in LLM Post-Training",
  "task_title": "Binary vs Dense Reward Ablation on 200 Live Leads",
  "task_description": "**Hour 1 — Setup & Framing (60 min)**\n\nCreate `experiments/rlvr_vs_dense_reward/` in your GitHub repo. Write `README.md` first — this forces clarity. The README must state the hypothesis in one sentence: 'Binary conversion outcome (RLVR-style verifiable reward) matches or exceeds dense RewardFlow shaping on outreach conversion.' Define your three reward conditions explicitly in code:\n\n

EXPECTED OUTPUT: 

WHY FRONTIER LABS CARE: 

TODAY'S PAPERS:
Paper 1: RLVR is Not RL: Understanding the Role of Verifiable Rewards in LLM Post-Training
Why: Definitively shows that RLVR (the training paradigm behind DeepSeek-R1, GRPO) works primarily through outcome filtering rather than true RL policy improvement — this reframes how reward signals should be designed for reasoning agents and directly challenges assumptions baked into most current RLHF pipelines.
For me: Anmol's RewardFlow and PRM work is built on the assumption that dense reward signals compound over steps. This paper's finding that sparse verifiable rewards dominate means his ASM multi-session memory system could use outcome verification (did the outreach convert?) as the primary training signal rather than per-step reward shaping — simpler to implement and likely more stable at his 83% beat rate baseline.
Paper 2: Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training
Why: Introduces a self-training loop where agents learn to detect when they are mid-error in a trajectory and backtrack — achieving up to 20% improvement on WebArena and SWE-Bench style tasks — which is the most concrete published mechanism for agent self-correction without human labels released this week.
For me: Anmol's autonomous revenue agent at Aonxi makes sequential decisions across multi-session outreach. Agent-R's iterative self-training framework maps directly onto his ASM architecture: failed outreach sessions (no reply, wrong persona) become backtrack signals. He can adapt their MCTS-guided trajectory relabeling to relabel failed lead sequences using his existing 2,452-lead labeled dataset as ground truth.
Paper 3: Memory-Efficient LLM Training with Online Subspace Descent
Why: Extends GaLore-style subspace gradient training with an online descent schedule that closes 80% of the perplexity gap to full-rank Adam while using 40% less memory — published by DeepMind and directly relevant to anyone fine-tuning large models on constrained infrastructure.
For me: Anmol runs LoRA fine-tuning for his ASM persona models. This method is positioned as a LoRA alternative that doesn't constrain the weight update to fixed low-rank adapters — meaning his persona specialization could be higher fidelity without additional GPU spend. Given he's shipping at $0 raised, the memory efficiency directly reduces inference and training costs at Aonxi.
Paper 4: VideoAgent2: Memory-Augmented Multi-Agent Framework for Long-horizon Video Understanding
Why: Demonstrates that hierarchical episodic memory — where a coordinator agent indexes and retrieves from sub-agent memory traces — dramatically outperforms single-context approaches on tasks requiring reasoning across hours of content; provides the most concrete working multi-agent memory architecture released this week.
For me: Anmol's NeurIPS 2026 submission is explicitly on multi-session memory (ASM-Outreach). VideoAgent2's hierarchical episodic store (session-level summaries → retrievable trace chunks → working context injection) is architecturally isomorphic to what ASM needs across prospect interaction sessions. He can cite this as concurrent work and differentiate on the text/agent domain, or directly adopt their retrieval indexing design for ASM.
Paper 5: SWEET-RL: Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks with Stepwise Rewards
Why: Introduces a principled credit assignment method for multi-turn agent RL that decomposes sparse end-of-episode rewards into per-turn advantage estimates without requiring a process reward model — directly solving the hardest open problem in training conversational agents: who gets credit for a sale made 6 turns later.
For me: This is the most directly relevant paper to Anmol's exact production problem. His revenue agent closes deals across multi-turn email/LinkedIn sequences, but training signal only arrives at conversion (sparse, delayed). SWEET-RL's stepwise advantage decomposition lets him assign credit back to individual outreach turns that causally contributed to the close — without building a separate PRM. This could replace his current RewardFlow heuristic with a theoretically grounded alternative. Levine's lab has code.

DO ALL 8 STEPS WITHOUT STOPPING:

STEP 1: Read ~/asm-replication/asm/ — understand current code.

STEP 2: Create ~/frontier-agi-journey/daily/2026-03-23/implementation/
Write complete working Python implementing today's task.
Use ~/aonxi-outreach-agent/aonxi.db as test data if needed.

STEP 3: Run it. Show real numbers. Save results.json.

STEP 4: Write ~/frontier-agi-journey/daily/2026-03-23/README.md
What I built · paper extended (arxiv ID) · my contribution
· results table · why it matters for AGI · production connection.

STEP 5: Push to GitHub
cd ~/frontier-agi-journey && git add -A
git commit -m "Day 3: Binary vs Dense Reward Ablation on 200 Live Leads"
git push origin main
Show commit hash and URL.

STEP 6: Update ~/frontier-agent/cv_tracker/progress.json

STEP 7: Write 50-word LinkedIn post. One number. github.com/originaonxi

STEP 8: Write lab email under 100 words: 