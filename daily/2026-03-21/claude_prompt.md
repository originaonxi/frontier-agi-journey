# Day 1 — Claude Code Prompt
# RLVR Shaping Audit on Live ASM Production Traces
# Copy this entire file and paste into Claude Code

I am Anmol Chaudhary (Sam Anmol). CTO @ Aonxi, $650K ARR.
Ex-Meta Ads ML. Ex-Apple Face ID. NeurIPS 2026 submission.
Day 1 of 365. Goal: $1M/year frontier lab role.

TODAY'S TASK: RLVR Shaping Audit on Live ASM Production Traces

{
  "paper_title": "RLVR is Not RL: Revisiting Reinforcement Learning for LLMs from a Reward Shaping Perspective",
  "task_title": "RLVR Shaping Audit on Live ASM Production Traces",
  "task_description": "**Hour 0–0.75: Environment + Paper Grounding**\nRead the core claim of Sun et al. 2503.10639: RLVR's 'reward' signal is mathematically equivalent to a shaped reward in classical RL (r + γΦ(s') − Φ(s)), meaning the policy gradient update contains no true RL signal — only reward shaping. The diagnostic: compute ρ (rho), the correlation between the raw reward and the shaping term, across a trajectory batch. If ρ ≈ 1.0, you have shaping, not RL.\n\nCreate repo structure:\n

EXPECTED OUTPUT: 

WHY FRONTIER LABS CARE: 

TODAY'S PAPERS:
Paper 1: RLVR is Not RL: Revisiting Reinforcement Learning for LLMs from a Reward Shaping Perspective
Why: Empirically dismantles the assumption that GRPO/PPO-style RLVR actually does credit assignment — shows most gains come from reward shaping artifacts, not genuine RL, which reframes how we should think about post-training for reasoning models in 2026.
For me: Anmol's ASM system uses an 83% beat-rate reward signal and a multi-session memory trace — if the reward signal is actually doing implicit shaping rather than genuine policy gradient, his RewardFlow architecture may be inadvertently masking the real learning signal. This paper gives him the diagnostic lens to audit whether his PRM is providing real credit assignment or a shaping shortcut, directly strengthening the theoretical grounding of his NeurIPS submission.
Paper 2: AgentTrek: Agent Trajectory Synthesis via Guiding Replay with Web Tutorials
Why: Presents a scalable, reproducible pipeline for synthetically generating grounded agent trajectories by replaying web tutorials — solves the data bottleneck for training web/GUI agents and directly benchmarks on OSWorld/WindowsAgentArena with SOTA results.
For me: Anmol's autonomous revenue agent at Aonxi processes 2,452 leads using multi-step decision traces. AgentTrek's trajectory synthesis pipeline is directly applicable to generating training data for his agent's outreach sequences — he can adapt the replay mechanism to his CRM/email tool environment, giving him a path from 83% to 90%+ beat rate via synthetic self-improvement data without human labeling cost.
Paper 3: Memory-Efficient Continual Learning for Large Language Models via Dynamic Architecture Adaptation
Why: Proposes a practical continual learning framework for LLMs that avoids catastrophic forgetting across sessions using dynamic LoRA adapter allocation — directly relevant to the unsolved problem of multi-session memory consolidation in long-horizon agents.
For me: This is the closest published work to Anmol's NeurIPS ASM-Outreach submission. His multi-session memory system faces the identical catastrophic forgetting problem when a lead's context spans weeks. The dynamic LoRA allocation scheme here could replace or augment his current memory module — he should explicitly cite and contrast this in his NeurIPS camera-ready, positioning ASM-Outreach as the production-validated complement to this lab result.
Paper 4: VideoVista-CulturalLingo: A Multilingual Benchmark for Cross-Cultural Video Understanding with Structured Reasoning Chains
Why: First large-scale empirical study of PRMs specifically in agentic settings — shows PRMs trained on reasoning traces transfer to multi-step tool-use tasks and that outcome reward alone is insufficient for reliable agent behavior at scale.
For me: Anmol explicitly lists PRM and RewardFlow in his stack. This paper is the empirical backbone he needs to justify PRM-based training in his NeurIPS paper. The finding that outcome reward alone fails maps directly to his TDAD (Trajectory-Divergence Aware Detection) work — he can use their scaling curves as a baseline comparison for his own production numbers.
Paper 5: Anthropic Model Specification v2 — Updated Honesty, Corrigibility, and Autonomy Norms (March 2026 revision)
Why: Anthropic's living model spec is the de facto alignment constitution for frontier systems — the March 2026 revision sharpens the corrigibility-autonomy dial definition and introduces new norms around agentic task refusal, which will govern every production agent shipped from Anthropic for the next 2+ years.
For me: Anmol's target employer is Anthropic. His autonomous revenue agent makes thousands of consequential outreach decisions daily — the updated agentic refusal norms directly apply to his system design. Demonstrating in his cover letter/portfolio that his Aonxi agent is already aligned with Anthropic's spec v2 (corrigibility hooks, refusal thresholds, minimal footprint) is a concrete differentiation move. He should add a 'Alignment Compliance' section to his agent architecture doc.

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
git commit -m "Day 1: RLVR Shaping Audit on Live ASM Production Traces"
git push origin main
Show commit hash and URL.

STEP 6: Update ~/frontier-agent/cv_tracker/progress.json

STEP 7: Write 50-word LinkedIn post. One number. github.com/originaonxi

STEP 8: Write lab email under 100 words: 