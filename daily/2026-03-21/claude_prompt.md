# Day 1 — Claude Code Prompt
# MEM1 vs ASM: Memory Compression Benchmark on Live Agent Traces
# Copy this entire file and paste into Claude Code

I am Anmol Chaudhary (Sam Anmol). CTO @ Aonxi, $650K ARR.
Ex-Meta Ads ML. Ex-Apple Face ID. NeurIPS 2026 submission.
Day 1 of 365. Goal: $1M/year frontier lab role.

TODAY'S TASK: MEM1 vs ASM: Memory Compression Benchmark on Live Agent Traces

{
  "paper_title": "MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agent Tasks",
  "task_title": "MEM1 vs ASM: Memory Compression Benchmark on Live Agent Traces",
  "task_description": "**Goal:** Empirically compare MEM1-style memory compression against your production ASM (Autonomous Session Memory) system using real Aonxi lead-processing traces. This directly connects your NeurIPS submission to a concurrent frontier paper, producing a benchmark artifact that no one else on earth can produce — because only you have 2,452 real production agent traces.\n\n**Hour 1 — Paper Read + Hypothesis Formation (60 min)**\n\nRead MEM1 focusing on three things only:\n1. Their memory compression objective: what is the formal definition of what gets kept vs. discarded after each turn?\n2. Their evaluation metric: how do they measure long-horizon task performance vs. memory footprint?\n3. Their baseline: what does their ablation look like without compression?\n\nWrite a 5-bullet `HYPOTHESIS.md` in `experiments/mem1_vs_asm/` answering: 'Where should ASM outperform MEM1, and why?' Your bet: ASM's multi-session persistence across leads (not just within a session) is a dimension MEM1 doesn't address. State this falsifiably.\n\n**Hour 2 — Data Pipeline (60 min)**\n\nCreate `experiments/mem1_vs_asm/build_trace_dataset.py`:\n\n

EXPECTED OUTPUT: 

WHY FRONTIER LABS CARE: 

TODAY'S PAPERS:
Paper 1: RLVR is Not RL: Understanding the Difference Between Reinforcement Learning from Verifiable Rewards and Standard RL
Why: Clarifies the mechanistic distinction between GRPO/RLVR (used in DeepSeek-R1, Qwen) and true RL, showing RLVR is closer to filtered supervised learning — directly challenges assumptions powering the current reasoning model wave and has empirical ablations proving it.
For me: Anmol's ASM-Outreach system uses RLHF-style reward signals for multi-session memory optimization and his RewardFlow work. This paper reframes what his reward model is actually doing — if RLVR ≠ RL, his 83% ASM beat rate may be attributable to filtered supervision, not policy gradient learning. He can run ablations on his production data to publish a follow-up empirical note, directly strengthening his NeurIPS submission narrative.
Paper 2: Retrieval-Augmented Generation with Conflicting Information
Why: Directly measures what happens when an agent's retrieved memory conflicts with parametric knowledge — shows standard RAG degrades catastrophically under conflict, and proposes a conflict-aware fusion layer that recovers ~18 F1 points on multi-hop QA benchmarks. This is the most empirically grounded memory-vs-knowledge paper of the week.
For me: Anmol's ASM (Autonomous Session Memory) system retrieves multi-session context and injects it into LLM prompts for outreach agents. When a lead's CRM data conflicts with what the model learned in pretraining (e.g., company size, role changes), the agent likely hallucinates — this paper gives him an explicit architectural fix (conflict-aware fusion) he can drop into his Aonxi production stack and measure lift on his 2,452-lead dataset. Direct publishable experiment.
Paper 3: Agent-FLAN: Designing Data and Methods of Effective Agent Tuning for Large Language Models
Why: Shows that naive instruction-following fine-tuning degrades agent tool-use capability, and identifies the exact data mixture ratio and formatting that recovers it — with open weights and code, making it the most actionable agent fine-tuning recipe released this week.
For me: Anmol does LoRA fine-tuning for his autonomous revenue agents at Aonxi. He's almost certainly hitting the exact degradation this paper describes — models that follow instructions well but lose structured tool-call formatting. The paper's data mixture recipe (ratio of agent traces to general SFT) is directly applicable to his fine-tuning runs. He can replicate in <12 hours using his existing PyTorch + LoRA stack and measure impact on his lead-processing pipeline.
Paper 4: SWEET-RL: Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks
Why: Introduces a credit assignment method for multi-turn agent RL that correctly attributes reward to individual turns rather than the full episode — solving the core problem that makes RLHF brittle for long-horizon agents, with strong empirical results on collaborative reasoning benchmarks.
For me: This is the most direct technical connection to Anmol's NeurIPS submission on multi-session memory. His ASM system operates across multiple turns and sessions, and reward attribution across turns is exactly the unsolved problem in his current RLHF setup. SWEET-RL's per-turn credit assignment could replace his current episode-level reward and is the kind of methodological upgrade that strengthens a NeurIPS paper from 'interesting system' to 'novel contribution.' He should reach out to Aviral Kumar (Berkeley) — overlapping research territory is a hiring signal.
Paper 5: MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agent Tasks
Why: Proposes an RL-trained memory consolidation mechanism that teaches agents WHEN to write, update, and delete memory — achieving state-of-the-art on long-horizon agent benchmarks while reducing context length by 60%, directly addressing the memory bottleneck in autonomous agents.
For me: This paper is essentially a formalization of what Anmol's ASM system is doing intuitively — but with a learned memory gating mechanism instead of heuristic rules. His current ASM decides what to persist across sessions via hand-crafted logic; MEM1 makes this RL-trained. Given his NeurIPS submission is on multi-session memory, MEM1 is both a competitive reference he must cite AND an architectural upgrade he can implement. His production dataset of 2,452 leads with 83% beat rates is a stronger real-world benchmark than any of the paper's evals — he should replicate MEM1 on his data and publish the comparison.

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
git commit -m "Day 1: MEM1 vs ASM: Memory Compression Benchmark on Live Agent Traces"
git push origin main
Show commit hash and URL.

STEP 6: Update ~/frontier-agent/cv_tracker/progress.json

STEP 7: Write 50-word LinkedIn post. One number. github.com/originaonxi

STEP 8: Write lab email under 100 words: 