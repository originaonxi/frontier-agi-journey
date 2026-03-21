# Day 1 — Claude Code Prompt
# Execute today's research task
# Copy this entire file and paste into Claude Code

I am Anmol Chaudhary (Sam Anmol). CTO @ Aonxi, $650K ARR.
Ex-Meta Ads ML. Ex-Apple Face ID. NeurIPS 2026 submission.
Day 1 of 365. Goal: $1M/year frontier lab role.

TODAY'S TASK: Execute today's research task

{
  "paper_title": "Process Reward Models for Multi-Step Reasoning: Scaling Laws and Failure Modes",
  "task_title": "Train PRM on Live Production Data, Plot Scaling Curve",
  "task_description": "**Hour 1 — Data Extraction (60 min)**\n\nCreate `aonxi/prm_data_extractor.py`. Your 4-step ASM-Outreach pipeline (Prospect → Enrich → Personalize → Send) maps cleanly onto a multi-step reasoning chain — each step is a 'reasoning token' with a verifiable outcome. Extract from your 2,452 processed leads:\n\n

EXPECTED OUTPUT: 

WHY FRONTIER LABS CARE: 

TODAY'S PAPERS:
Paper 1: RLVR Is Not RL: On the Importance of Reward Design in Reinforcement Learning for Reasoning
Why: Rigorously separates RLVR (verifier-reward RL) from classical RL, showing that most 'reasoning gains' in LLMs come from reward hacking rather than genuine policy improvement — directly attacks the foundations of post-training pipelines at every major lab.
For me: His ASM-Outreach paper uses a reward signal (83% beat rate over baseline) as the empirical backbone. This paper gives him the formal language to characterize whether his RewardFlow signal is genuine RL or RLVR, and how to make the reward design more robust — directly citable in his NeurIPS 2026 submission to defend his reward architecture.
Paper 2: Memorization vs. Generalization: The Role of Context Length in Transformer Memory for Long-Horizon Tasks
Why: First controlled empirical study showing exactly where transformer memory breaks down across context lengths for multi-step agentic tasks — provides a clean phase diagram of when you need external memory vs. in-context is sufficient.
For me: ASM (Adaptive Session Memory) is his NeurIPS submission's core contribution. This paper gives him the ablation baseline he needs: a principled chart showing where in-context fails and where ASM's external memory kicks in. He can directly overlay his 2,452 lead sessions onto their phase diagram to demonstrate real-world validation.
Paper 3: Process Reward Models for Multi-Step Reasoning: Scaling Laws and Failure Modes
Why: First systematic scaling law paper for PRMs (Process Reward Models) — shows exactly how PRM quality scales with verifier size, data quantity, and step granularity, with empirical failure mode taxonomy. Critical for anyone building agentic pipelines that use intermediate reward.
For me: He lists PRM as a core skill and has replications on GitHub. This paper gives him the scaling playbook to take his PRM replications from toy-scale to production-grade. His Aonxi pipeline processes 2,452 leads with intermediate decision steps — each step is a natural PRM training signal. He can instrument Aonxi to auto-generate PRM training data from live production.
Paper 4: Agent-FLAN: Generalizable Agent Instruction Tuning via Data Mixing and Reward Shaping
Why: Shows that instruction-tuned agents trained with carefully mixed synthetic + real task data plus lightweight reward shaping dramatically outperform RLHF-only agents on multi-step tool-use benchmarks — with full training code released.
For me: His agent orchestration work at Aonxi (autonomous revenue agent) is functionally identical to the tool-use agents in this paper. The data-mixing recipe they publish can be applied directly to his lead-processing pipeline. Code is open, he can fine-tune a 7B model on his 2,452-lead dataset using their recipe in a weekend.
Paper 5: Long-Term Memory Architectures for LLM Agents: A Systematic Comparison of Retrieval, Compression, and Consolidation Strategies
Why: The first head-to-head empirical benchmark comparing RAG, memory consolidation, hierarchical compression, and episodic replay for persistent LLM agents across 6 task domains — directly from the lab Anmol wants to join, with reproducible eval harness.
For me: This is the Anthropic-adjacent paper most directly aligned with his NeurIPS submission on multi-session memory (ASM-Outreach). Their benchmark taxonomy maps 1:1 onto his ASM architecture. He should: (1) run his system through their eval harness to get comparable numbers, (2) email the authors with his production results (2,452 real leads, 83% beat rate), and (3) cite this as the primary related work framing his contribution.

DO ALL 8 STEPS WITHOUT STOPPING:

STEP 1: Read ~/asm-replication/asm/ — understand current code.

STEP 2: Create ~/frontier-agi-journey/daily/2026-03-21/implementation/
Write complete working Python implementing today's task.
Use ~/aonxi-outreach-agent/aonxi.db as test data if needed.

STEP 3: Run it. Show real numbers. Save results.json.

STEP 4: Write ~/frontier-agi-journey/daily/2026-03-21/README.md
What I built · paper extended (arxiv ID) · my contribution
· results table · why it matters for AGI · production connection.

STEP 5: Push to GitHub
cd ~/frontier-agi-journey && git add -A
git commit -m "Day 1: Execute today's research task"
git push origin main
Show commit hash and URL.

STEP 6: Update ~/frontier-agent/cv_tracker/progress.json

STEP 7: Write 50-word LinkedIn post. One number. github.com/originaonxi

STEP 8: Write lab email under 100 words: 