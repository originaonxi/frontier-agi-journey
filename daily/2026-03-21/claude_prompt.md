# Day 1 — Claude Code Prompt
# 
# Copy this entire file and paste into Claude Code

I am Anmol Chaudhary (Sam Anmol). CTO @ Aonxi, $650K ARR.
Ex-Meta Ads ML. Ex-Apple Face ID. NeurIPS 2026 submission.
Day 1 of 365. Goal: $1M/year frontier lab role.

TODAY'S TASK: 

{
  "paper_title": "RLVR is Not RL: Understanding the Role of Verifiable Rewards in Language Model Alignment",
  "task_title": "Test RLVR Distribution Sharpening on Live Production Agent Data",
  "task_description": "## Step-by-Step Execution Plan\n\n### Hour 1 (0:00–1:00): Set up experiment scaffold\n\nCreate `experiments/rlvr_distribution_analysis/` with:\n\n

EXPECTED OUTPUT: 

WHY FRONTIER LABS CARE: 

TODAY'S PAPERS:
Paper 1: RLVR is Not RL: Understanding the Role of Verifiable Rewards in Language Model Alignment
Why: Directly challenges whether GRPO/RLVR is doing 'real' RL at all — argues verifiable reward signals mostly amplify pre-trained distributions rather than explore novel reasoning paths, which reshapes how we should think about reward shaping in production agent systems.
For me: Anmol's ASM-Outreach system uses outcome-based reward signals (reply rate, meeting booked) as verifiable rewards — this paper's analysis maps precisely onto his RewardFlow design. The 83% beat rate may be explained not by true policy improvement but by distribution amplification, which is a testable hypothesis he can run on his 2,452-lead dataset this week.
Paper 2: Agents Are Not Enough: Toward a Unified Theory of Cognitive Architectures for AGI
Why: DeepMind proposes a formal taxonomy distinguishing reactive agents, memory-augmented agents, and full cognitive architectures — provides the theoretical scaffolding that frontier labs are using to evaluate agent research submissions right now.
For me: Anmol's NeurIPS submission (ASM-Outreach, multi-session memory) fits squarely in the 'memory-augmented agent' tier of this taxonomy. Using this framework to reframe his paper's contribution — and explicitly positioning ASM as a step toward the cognitive architecture tier — would significantly strengthen the submission's framing for NeurIPS reviewers and Anthropic/DeepMind hiring readers.
Paper 3: ReasonFlux: Hierarchical LLM Reasoning via Scaling Thought Templates
Why: Achieves SOTA on MATH500 and competitive AIME results by learning a hierarchical library of reusable reasoning templates — empirically demonstrates that structured thought decomposition > raw chain-of-thought scaling, with code released.
For me: Anmol's outreach agent already implicitly uses 'templates' (sequence of reasoning steps: prospect research → pain identification → message construction). ReasonFlux's hierarchical template library is a direct architectural upgrade he can apply to ASM's planning module. The released code means he can benchmark his agent's reasoning steps against this within a weekend.
Paper 4: Long-Term Memory for AI Agents: Taxonomy, Challenges, and Future Directions
Why: First comprehensive empirical taxonomy of long-term memory mechanisms for agents — tests retrieval, compression, and consolidation strategies at scale with head-to-head benchmarks, giving practitioners a clear decision matrix for memory architecture choices.
For me: This paper is the reference document for Anmol's NeurIPS submission. His ASM multi-session memory system implements a specific point in their taxonomy (episodic + semantic hybrid with session-level compression). Citing this taxonomy and benchmarking ASM against their evaluation framework would immediately elevate the paper's empirical rigor. The 2,452 leads processed is a real-world dataset that exceeds any benchmark in this survey.
Paper 5: Process Reward Models for LLM Agents: Empirical Analysis and Scaling Behavior
Why: Largest empirical study to date on PRM (Process Reward Models) applied to multi-step agentic tasks — finds PRMs outperform ORMs at decision points but degrade with horizon length, with specific architectural fixes and scaling curves published.
For me: Anmol explicitly lists PRM and RewardFlow in his skills and existing work. This paper provides the empirical grounding and failure mode analysis for exactly what he's building. The finding that PRMs degrade with horizon length directly maps to a known weakness in long outreach sequences (5+ touchpoints) — Anmol likely has production data showing this exact degradation he can now explain and fix with the paper's proposed architectural patches.

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
git commit -m "Day 1: "
git push origin main
Show commit hash and URL.

STEP 6: Update ~/frontier-agent/cv_tracker/progress.json

STEP 7: Write 50-word LinkedIn post. One number. github.com/originaonxi

STEP 8: Write lab email under 100 words: 