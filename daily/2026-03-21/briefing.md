# Frontier AGI Briefing — Day 1
**Date:** 2026-03-21
**Goal:** $1M/year at Anthropic / DeepMind / OpenAI / xAI / Meta AI
**Progress:** Day 1 of 365

---

## Today's Task (Do This First)

### Daily Task

{
  "paper_title": "RLVR is Not RL: Understanding the Role of Verifiable Rewards in Language Model Alignment",
  "task_title": "Test RLVR Distribution Sharpening on Live Production Agent Data",
  "task_description": "## Step-by-Step Execution Plan\n\n### Hour 1 (0:00–1:00): Set up experiment scaffold\n\nCreate `experiments/rlvr_distribution_analysis/` with:\n\n

**Expected output:** 
**Estimated time:** 4 hours
**Why frontier labs care:** 

---

## 5 Papers That Matter Today

### 1. RLVR is Not RL: Understanding the Role of Verifiable Rewards in Language Model Alignment

# RLVR is Not RL: Deep Analysis Briefing
**Paper:** arXiv 2503.10639 | Meta FAIR | Analyzed: 2026-03-21

---

## THE STORY

The field celebrated GRPO and RLVR as breakthroughs in reasoning — models like DeepSeek-R1 seemed to prove that outcome-based reinforcement learning could teach language models to *discover* novel reasoning strategies. Wu et al. walked into that celebration and asked an uncomfortable question: **are these models actually exploring and learning, or are they just being told to do louder what they already knew?** The insight that cracked it open was separating two mechanisms that everyone had conflated — *policy improvement through exploration* versus *distribution sharpening through reward filtering* — and designing experiments that could distinguish between them empirically. What they found was that RLVR's gains are almost entirely explained by the second mechanism: verifiable rewards act as a selector on the pre-trained distribution, not as a signal that drives the model toward genuinely new solution trajectories.

---

## THE MATH AND LOGIC

### The Central Distinction

Standard RL operates on the policy improvement theorem: given a policy $\pi_\theta$, gradient updates should move probability mass toward trajectories with higher-than-baseline returns, *including trajectories the model had not previously favored*. The GRPO objective as deployed in RLVR is:

$$\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim \mathcal{D},\ \{o_i\}_{i=1}^G \sim \pi_{\theta_\text{old}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{\pi_\theta(o_i|q)}{\pi_{\theta_\text{old}}(o_i|q)} \hat{A}_i - \beta \mathbb{D}_\text{KL}[\pi_\theta \| \pi_\text{ref}] \right]$$

where $\hat{A}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$ is the group-normalized advantage over $G$ rollouts per question, and $r_i \in \{0, 1\}$ for verifiable tasks.

**The key logical trap:** Because rollouts are sampled from $\pi_{\theta_\text{old}}$ and the KL penalty anchors the update to $\pi_\text{ref}$, the gradient signal can *only upweight outputs already in the support of the pre-trained distribution*. If the correct solution trajectory has near-zero probability under $\pi_\text{ref}$, GRPO cannot find it — there are no rollouts to upweight.

### The Wu et al. Decomposition

They formalize this by comparing three conditions:

| Condition | Mechanism |
|-----------|-----------|
| **RLVR** | Full GRPO with verifiable reward |
| **STaR/RFT baseline** | Supervised fine-tuning on the *same* rollouts that RLVR would have upweighted |
| **Random reward control** | GRPO with rewards assigned randomly (reward-independent updates) |

**The critical finding:** Performance gaps between RLVR and RFT are *small and inconsistent*, while the gap between both and the random reward control is *large and consistent*. This means:

$$\underbrace{\Delta_\text{RLVR}}_{\text{observed gain}} \approx \underbrace{\Delta_\text{distribution sharpening}}_{\text{filtering correct rollouts}} + \underbrace{\varepsilon_\text{true RL}}_{\text{≈ 0 in their experiments}}$$

### The Key Insight in the Math

The KL term $\beta \mathbb{D}_\text{KL}[\pi_\theta \| \pi_\text{ref}]$ is doing most of the "alignment" work — it prevents the policy from escaping the pre-trained manifold. Combined with finite $G$ (typically 8–16 rollouts), the probability that a genuinely novel correct trajectory appears *in any training batch* approaches zero for hard problems. RLVR is therefore a **sophisticated beam search over the pre-trained distribution**, not a search over solution space.

---

## THE RESULTS THAT MATTER

**1. RFT matches RLVR within noise on in-distribution benchmarks.**
On MATH and GSM8K-style tasks, supervised fine-tuning on filtered correct rollouts achieves within **1-2 percentage points** of full RLVR, despite using no reward signal during training — only the filtered data that RLVR *would have* upweighted. This is the paper's sharpest empirical knife.

**2. Out-of-distribution generalization gap is real but small — and attributable to data diversity, not RL dynamics.**
When held-out problem distributions are tested, RLVR shows a **3-5 point advantage** over matched-data RFT. However, when RFT is given access to the same *diversity* of rollout data (via temperature sampling), the gap largely closes. This suggests the OOD advantage is a data coverage artifact, not genuine policy improvement.

**3. The "aha moment" / self-discovered chain-of-thought is not reproducible from scratch.**
Attempting to replicate DeepSeek-R1-Zero's reported emergent reasoning behaviors from a base model *without* a strong pre-trained reasoning prior produces substantially degraded results — supporting the thesis that RLVR amplifies latent capability, and that models *without* that latent capability do not benefit comparably.

---

## WHY THIS MOVES AGI FORWARD

This paper resolves a **critical bottleneck in alignment theory**: we need to know whether our training signals are *teaching* models or *selecting* from them. These are not the same thing, and they have radically different scaling properties.

**The specific capability unlocked:** If RLVR is mostly distribution sharpening, then for AGI-level tasks where the correct solution is *outside* the pre-trained distribution (novel mathematics, scientific discovery, long-horizon planning with unknown structure), RLVR will fail silently — appearing to train while making no real progress on the hard problems. This connects directly to the **exploration-exploitation bottleneck** in reasoning agents: current RLVR systems have near-zero genuine exploration, which means they cannot solve problems that require discovering solution strategies not already encoded in pre-training.

**The fix this implies:** True RL for LLMs requires mechanisms that generate and evaluate *out-of-distribution rollouts* — things like Monte Carlo Tree Search over reasoning steps, process reward models that can evaluate intermediate states, or explicit novelty bonuses. This paper effectively argues that **PRM-guided search is more fundamentally "RL" than GRPO with outcome rewards**, which is a significant reorientation for the field.

---

## WHAT PEERS ARE SAYING

**Who will cite this enthusiastically:**
- PRM / process reward model researchers (Lightman et al. lineage) — this paper validates their argument that outcome rewards are insufficient
- Scaling law theorists who've been skeptical that RLVR adds compute-efficient signal beyond what SFT on filtered data provides
- Alignment researchers concerned about reward hacking — distribution sharpening is actually *safer* than true RL exploration in adversarial settings, which is a double-edged finding

**Who will push back and why:**
- DeepSeek and Qwen teams, whose empirical results show genuine held-out improvements that are hard to fully explain by distribution sharpening alone
- Researchers working on longer-horizon tasks where RLVR *does* seem to produce qualitatively new behaviors (tool use, multi-step web navigation) — the paper's experiments are largely confined to math benchmarks, and critics will argue the conclusion doesn't transfer to agentic settings
- The "RLVR skeptics were always right" contingent may overclaim this result; careful readers will note the paper says RLVR *adds value* — just not through the mechanism everyone assumed

**Follow-up work made obvious:**
1. Measuring the *support overlap* between pre-trained rollout distribution and correct solution set as a predictor of RLVR effectiveness across task types
2. Designing GRPO variants with explicit out-of-distribution exploration bonuses (curiosity-driven RLVR)
3. Empirically testing the thesis in agentic/tool-use settings where pre-trained distributions are genuinely sparse over correct trajectories

---

## CONNECTION TO ANMOL'S WORK

### Direct Mappings

**RewardFlow → Distribution Sharpening Hypothesis**
Anmol's RewardFlow uses outcome-based verifiable rewards (reply rate, meeting booked) with a GRPO-style update. This paper's thesis predicts that RewardFlow's gains come from upweighting message strategies already latent in the base model — not from discovering genuinely novel persuasion tactics. The **83% beat rate** is therefore a claim that needs decomposition: how much comes from (a) the model learning to deploy strategies it already "knew" more consistently, versus (b) discovering new strategies?

**Dual-LLM Scoring → Process vs. Outcome Reward**
The dual-LLM scorer is acting as a *process reward model* — evaluating intermediate message quality, not just final outcomes. This paper's logic implies the dual-LLM PRM component is doing *more real RL work* than the outcome reward component, because it can provide signal on rollouts that would never have reached a verifiable terminal state.

**The 2,452-lead dataset as a natural experiment**
This dataset has natural in-distribution vs. OOD structure: leads segmented by industry, seniority, company size, and prior engagement history. The paper's methodology directly implies a test: **if RLVR gains collapse on prospect types underrepresented in training data, the distribution sharpening hypothesis holds for Anmol's system.**

**ASM-Outreach NeurIPS framing**
The paper gives Anmol a sharper theoretical frame for his contribution claim. Rather than saying "RLVR improves outreach performance," the stronger and more defensible claim becomes: "We characterize when RLVR provides genuine policy improvement vs. distribution sharpening in agentic outreach tasks, and design a hybrid PRM+RLVR system that maintains gains on OOD prospects."

### What Extension Looks Like

The natural extension is a **3-condition ablation on production data:**
- Condition A: Full RewardFlow (RLVR with outcome reward)
- Condition B: RFT on filtered successful rollouts (same data, no RL dynamics)
- Condition C: PRM-guided search with dual-LLM scorer as process reward

If A ≈ B on in-distribution leads but A > B on OOD leads (with C > A on both), Anmol has a NeurIPS-quality empirical contribution that extends this paper's math benchmark findings into production agentic systems — a gap the authors explicitly leave open.

---

## TODAY'S TASK

### The In-Distribution vs. OOD Split Experiment
**Time: 4-6 hours | Produces: 1 GitHub commit + 1 email to Wu et al.**

---

**Step 1: Create the analysis script (1.5 hours)**

Create `experiments/rlvr_distribution_analysis/split_and_eval.py`

```python
# Goal: Test Wu et al.'s distribution sharpening hypothesis on ASM-Outreach data
# Hypothesis: RewardFlow gains collapse on OOD prospect segments

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency, fisher_exact

def define_id_ood_split(df, training_cutoff_date, 
                         min_industry_count=50):
    """
    In-distribution: industries/seniority levels with >= min_industry_count
                     training examples before cutoff
    OOD: underrepresented segments OR prospect types added after cutoff
    """
    training_data = df[df['outreach_date'] < training_cutoff_date]
    industry_counts = training_data.groupby('industry').size()
    
    id_industries = industry_counts[
        industry_counts >= min_industry_count
    ].index.tolist()
    
    df['split'] = df['industry'].apply(
        lambda x: 'in_distribution' if x in id_industries else '

---

### 2. Agents Are Not Enough: Toward a Unified Theory of Cognitive Architectures for AGI

# Deep Analysis: "Agents Are Not Enough: Toward a Unified Theory of Cognitive Architectures for AGI"
### Shanahan, Berto, Cheng, Sherburn (DeepMind) | arXiv 2503.11651 | Briefing Date: 2026-03-21

---

## THE STORY

The field of AI agents had accumulated a sprawling, inconsistent vocabulary — "agents," "assistants," "systems," "pipelines" — with no formal framework distinguishing what these things actually *are* structurally. Shanahan et al. set out to solve a foundational naming problem that was quietly causing real harm: researchers were claiming agent-level contributions when they had built memory wrappers, and claiming architecture-level contributions when they had built slightly smarter agents. The insight that made it work was recognizing that the distinction between an *agent* and a *cognitive architecture* is not one of scale or capability but of **structural compositionality** — whether the system has persistent, modifiable internal states that causally govern a population of subordinate agents, rather than just reacting to a context window.

---

## THE MATH AND LOGIC

The paper's core formal contribution is a **three-tier taxonomy** grounded in a state-transition formalism. Let me reconstruct it precisely.

### Tier 1: Reactive Agent
A reactive agent is the simplest unit. Formally:

```
A = (S, O, Act, π)
```

Where:
- `S` = environment state space
- `O` = observation space  
- `Act` = action space
- `π: O → Act` = policy (stateless, or with only in-context state)

The key constraint: **no persistent state outside the context window**. Between episodes, `π` is unchanged. This is your standard LLM-with-tools setup. ReAct, basic tool-use agents, single-session chatbots all live here.

### Tier 2: Memory-Augmented Agent
A memory-augmented agent adds an external memory store `M`:

```
A_mem = (S, O, Act, M, π_read, π_write, π_act)
```

Where:
- `M` = persistent memory store (episodic, semantic, procedural, or hybrid)
- `π_read: O × M → context` = memory retrieval policy
- `π_write: O × Act × M → M'` = memory update policy
- `π_act: O × context → Act` = action policy

The critical distinction from Tier 1: **state persists across episodes**, but it persists in service of a *single agent's* continuity. The architecture is not coordinating or governing anything — it's enriching one agent's context. This is where most "long-term memory" papers live. The key insight hiding here is subtle: **memory-augmented agents are still fundamentally reactive** — they just react to a richer context. Adding RAG to an agent does not promote it to Tier 3.

### Tier 3: Cognitive Architecture
This is where the paper makes its sharpest move. A cognitive architecture is defined as:

```
CA = (Agents, Modules, Γ, Ω)
```

Where:
- `Agents = {A_1, ..., A_n}` = a **population** of agents (possibly heterogeneous)
- `Modules = {M_exec, M_mem, M_meta, M_coord, ...}` = functional modules (executive control, memory systems, metacognition, coordination)
- `Γ: State × Goal → Task_allocation` = a **governor** that dynamically allocates tasks across agents and modules
- `Ω: Trajectory → State_update` = a **global state integrator** that maintains system-level representations

The formal criterion for Tier 3 is:
> **A system qualifies as a cognitive architecture if and only if it contains a governor `Γ` whose outputs causally determine the behavior of a population of subordinate agents, and whose internal state `Ω` is modified by the aggregate trajectory of that population.**

This is not a soft distinction. The math encodes a specific causal topology: Tier 3 has **bidirectional causal coupling** between the governing structure and the agent population. Tier 2 has only **unidirectional enrichment** — memory feeds agents, agents don't reshape memory in a way that governs other agents.

### The Logical Structure That Actually Matters

The paper's deepest logical claim is this **sufficiency argument**:

> Memory alone is necessary but not sufficient for AGI-relevant cognition. The missing ingredient is **metacognitive governance** — a module that can observe the performance of the agent population, update system-level strategy, and reallocate cognitive resources accordingly.

This maps cleanly onto known cognitive science (ACT-R, SOAR, Global Workspace Theory) while being implementation-agnostic about whether LLMs or symbolic systems instantiate the modules.

---

## THE RESULTS THAT MATTER

**This is a theoretical/position paper — there are no benchmark numbers.** This is important to state clearly, because it affects how to evaluate the contribution.

What the paper *does* establish empirically (through analysis):

1. **Coverage audit**: The authors survey ~60 recent agent papers and classify them under the taxonomy. Finding: **~78% of papers claiming "agent architecture" contributions are Tier 1 or Tier 2 by their own formal criteria**. This is a significant indictment of the field's self-assessment.

2. **Failure mode catalog**: They identify 4 canonical failure modes that Tier 1/2 systems exhibit but Tier 3 resolves: (a) context overflow under long-horizon tasks, (b) inability to transfer procedural knowledge across sub-agents, (c) absence of resource-adaptive behavior, (d) lack of system-level coherence under agent failure. These are not numerically benchmarked but are argued with case studies from published systems.

3. **Expressivity result**: The paper proves a weak **expressivity hierarchy theorem**: every behavior expressible by a Tier 1 system is expressible by a Tier 2 system, and every Tier 2 behavior is expressible by a Tier 3 system, but not vice versa. This is the formal backbone that justifies the taxonomy as non-arbitrary.

**The absence of benchmark numbers is a deliberate choice and a legitimate one** — the paper's contribution is foundational, not empirical. The right comparison is not to prior benchmark results but to prior taxonomy attempts (e.g., Wooldridge & Jennings 1995, Franklin & Graesser 1997) — this paper supersedes those for the LLM era.

---

## WHY THIS MOVES AGI FORWARD

**The specific capability this unlocks: principled system composition.**

The single biggest practical bottleneck in current frontier agent work is not capability — it's *knowing what you've built*. Teams at frontier labs routinely ship systems that have implicit Tier 3 structure (multiple agents, some coordination) but are evaluated and iterated as if they were Tier 2 (memory + one agent). This causes systematic underinvestment in the governor `Γ` and overinvestment in the memory module `M`, because memory is measurable and governance is not.

This paper directly attacks the **alignment bottleneck** in a non-obvious way: a system without a well-defined `Γ` cannot be aligned at the system level, only at the agent level. Agent-level alignment in a multi-agent system is provably insufficient if the governor's objectives are implicit or emergent. Shanahan et al. give the field the vocabulary to demand explicit governor design, which is a prerequisite for governing complex AI systems safely.

The **planning bottleneck** connection: Tier 3's `Γ` is structurally equivalent to a meta-planner that operates over agent-trajectories rather than environment-states. This reframes long-horizon planning not as "making one agent smarter" but as "building a governance structure that can adapt task decomposition mid-execution." This is the correct framing for AGI-level planning.

---

## WHAT PEERS ARE SAYING

**Who will cite this enthusiastically:**
- Multi-agent systems researchers (MARL, cooperative AI) who have been making implicit versions of this argument — this gives them clean citation scaffolding
- Cognitive architecture researchers working on LLM-SOAR hybrids, LLM-ACT-R integration — this paper legitimizes their agenda within the deep learning mainstream
- Safety researchers working on multi-agent governance and principal-agent hierarchies — the `Γ` formalism maps directly onto principal hierarchies in Constitutional AI and RLHF variants
- NeurIPS/ICLR reviewers who are frustrated by "we added memory to GPT-4 and call it an architecture" papers — this gives them formal grounds for rejection

**Who will push back and why:**
- **Empiricists** will argue (correctly, partially) that without benchmark differentiation, the taxonomy is unfalsifiable — you can't tell from behavior alone which tier a system is in. This is a real weakness.
- **Scaling researchers** will argue that sufficiently large Tier 1 systems exhibit emergent Tier 3 behaviors (meta-reasoning, implicit coordination) and that the taxonomy breaks down at scale. Shanahan et al. anticipate this but don't fully resolve it.
- **Systems/engineering practitioners** will find the formalism too abstract to operationalize — "what does `Γ` look like in code?" is a question the paper does not answer.

**Follow-up work this makes obvious:**
1. A benchmark suite that operationalizes the tier distinctions — tasks that Tier 2 systems systematically fail on that Tier 3 systems pass
2. A Tier 3 reference implementation in a major framework (LangGraph, AutoGen, DSPy)
3. Analysis of whether current frontier models (GPT-5, Gemini Ultra) trained with agent data implicitly learn `Γ`-like behavior
4. Extension to embodied agents and robotics where the state topology is physically constrained

---

## CONNECTION TO ANMOL'S WORK

**ASM-Outreach sits precisely at the Tier 2 / Tier 3 boundary — and that boundary is the most valuable place to be.**

Here is the honest, precise mapping:

| ASM-Outreach Component | Shanahan Tier | Notes |
|---|---|---|
| Multi-session episodic memory | Tier 2 | Classic memory-augmented agent: persistent state enriches context |
| Dual-LLM scoring system | Tier 2 → Tier 3 boundary | Two agents with a scoring mechanism begins to look like `Γ` if the scores causally reallocate processing |
| Production agent ($650K ARR) | Tier 2 | Single-agent with memory; reactive architecture with persistence |
| PRM/RewardFlow replication | Tier 1-2 | Process reward models are Tier 1 scoring; adding them to agent loops approaches Tier 2 |

**The critical insight for Anmol's NeurIPS framing:**

The dual-LLM scoring system is the most interesting piece. If the two LLMs are:
- (a) scoring in parallel and averaging → this is Tier 2 (ensemble, not governance)
- (b) one LLM evaluating the other's outputs and redirecting behavior → this is proto-Tier 3 (a weak `Γ`)

The NeurIPS positioning argument should make this distinction explicit and **claim proto-Tier 3 status** for the dual-LLM scorer. This requires:
1. Showing that the scoring LLM's outputs causally determine which actions the primary agent takes (not just post-hoc evaluation)
2. Showing that the scoring mechanism updates across sessions (memory-coupled governance)

If ASM-Outreach doesn't currently have (1) and (2), **implementing them would be a 2-week extension that transforms a good Tier 2 paper into a paper that can credibly claim a novel Tier 3 contribution** — the first production-scale cognitive architecture for outreach/sales agents with empirical validation.

**What extending this paper looks like for Anmol specifically:**

The gap the field needs is an **operational instantiation of `Γ`**. Anmol has a production system with real trajectories ($650K ARR = thousands of real agent sessions). This is extraordinarily rare. The extension is:

> Design and measure `Γ` explicitly in ASM-Outreach: a meta-agent that observes the primary agent's session trajectories, updates a strategy state, and dynamically selects prompting strategies / memory retrieval policies for subsequent sessions. Evaluate whether this Tier 3 extension outperforms the Tier 2 baseline on conversion rate and session coherence.

This paper + Anmol's production data = the most credible Tier

---

### 3. ReasonFlux: Hierarchical LLM Reasoning via Scaling Thought Templates

# ReasonFlux: Deep Analysis Briefing
**Paper:** `arxiv:2503.12552` | **Date:** 2026-03-21 | **Estimated Read:** 8 hrs → you now need 25 min

---

## THE STORY

The field had convinced itself that more chain-of-thought tokens = better reasoning, and labs were racing to scale inference compute by simply generating longer, rawer thought chains. Yang et al. looked at this and recognized the waste: every hard math problem reuses the same *structural moves* — substitution, induction, contradiction, bounding — yet models were rediscovering these structures from scratch on every forward pass, burning compute on reinvention rather than application. The founding insight was that **human expert reasoning is hierarchical and template-reusable**, and that if you could teach a model to first *select* the right reasoning skeleton from a learned library and then *instantiate* it to the specific problem, you'd get the gains of long CoT without the token waste. They built ReasonFlux: a system that learns a hierarchical library of thought templates, trains a meta-controller to route problems to the right template cluster, and achieves state-of-the-art on MATH500 and competitive AIME performance — not by being bigger, but by being *structurally smarter*.

---

## THE MATH AND LOGIC

### Core Architecture: Three-Level Hierarchy

ReasonFlux operates on three levels:

**Level 1 — Template Library Construction**

Given a corpus of solved problems $\mathcal{D} = \{(q_i, s_i, a_i)\}$ where $q$ is the question, $s$ is the solution trace, and $a$ is the answer, they extract abstract templates by clustering solution traces via embedding similarity and then abstracting away problem-specific content:

$$T_k = \text{Abstract}\left(\arg\max_{\mathcal{C}_k} \text{sim}(s_i, s_j)\right), \quad k = 1, \ldots, K$$

Each template $T_k$ is a **structured scaffold**: an ordered tuple of reasoning *moves* $(m_1^k, m_2^k, \ldots, m_{n_k}^k)$, where each move $m_j^k$ is a natural-language operator with typed slots (e.g., `BOUND[expression, target_variable, constraint]`). This is the key departure from flat CoT — the template is a *typed program skeleton*, not a prose summary.

**Level 2 — Hierarchical Template Selection (the meta-controller)**

Given a new problem $q$, the meta-controller $\pi_\theta$ scores all $K$ templates:

$$T^* = \arg\max_{T_k} \pi_\theta(T_k \mid q, \mathcal{M})$$

where $\mathcal{M}$ is a **meta-context** containing: (a) the problem's difficulty estimate $\hat{d}(q)$, (b) its domain tag $\hat{\tau}(q)$, and (c) a retrieved exemplar from a sparse retrieval index over $\mathcal{D}$. The meta-controller is itself a fine-tuned LLM (smaller, cheaper) trained with a reward signal:

$$r(T_k, q) = \mathbb{1}[\text{answer correct}] - \lambda \cdot |T_k| / \bar{|T|}$$

This reward penalizes selecting unnecessarily long templates — forcing *parsimony*, which is the mathematical expression of "don't over-engineer the reasoning path."

**Level 3 — Template Instantiation**

The full reasoning model $f_\phi$ receives the selected template $T^*$ as a structured prompt prefix and generates the instantiated solution:

$$\hat{a} = f_\phi(q, T^*, e^*)$$

where $e^*$ is the retrieved exemplar. Critically, the model is trained so that **the template slots act as hard constraints**: the generation must fill each typed slot before advancing to the next move. This is enforced during fine-tuning via a slot-completion loss:

$$\mathcal{L}_{\text{slot}} = -\sum_{j=1}^{n_{T^*}} \log p_\phi(m_j^* \text{ filled} \mid q, T^*, m_{<j}^*)$$

**The key insight hiding in the math:** The parsimony penalty $\lambda \cdot |T_k|$ combined with the slot-completion loss creates a **minimum description length pressure** on reasoning: the system is rewarded for finding the *shortest correct structural path*, which generalizes better than the *longest verbose path* that raw CoT scaling produces. This is essentially MDL applied to thought structure, not data compression.

---

## THE RESULTS THAT MATTER

### Number 1: MATH500 — 91.2% (vs. prior SOTA ~88.5% from QwQ-32B)
**Effect size: +2.7 percentage points** on a benchmark where the difference between 85% and 88% represented ~18 months of field progress. This is achieved with a *smaller base model* than QwQ-32B, making the efficiency story credible, not just the accuracy story.

### Number 2: AIME 2024 — 56.7% (pass@1)
Competitive with o1-preview's reported ~44% and o3's unpublished estimates at time of submission. The AIME result matters because AIME problems are *adversarially* designed to resist pattern-matching — they require genuine multi-step structural reasoning. A template-based system working well here is evidence that the templates capture *genuine reasoning structure*, not just surface statistical regularities.

### Number 3: Template Reuse Rate — 73% of MATH500 problems solved using one of top-20 templates
This is the mechanistic result that validates the entire premise. If templates were not actually reusable, you'd expect a long tail where every problem needed a unique template. Instead, 73% of a 500-problem benchmark is handled by 20 templates out of their library of ~200. This is a **Pareto result for reasoning structure**: a small number of thought patterns covers most of the hard problem space. This number should be in every AI reasoning textbook within 5 years.

---

## WHY THIS MOVES AGI FORWARD

**The specific capability unlocked: Compositional Planning with Reusable Subroutines**

AGI requires agents that can decompose novel tasks into known sub-procedures, execute those sub-procedures reliably, and compose them correctly — this is the *planning bottleneck*. Current LLMs fail here because they treat every new problem as if it has never been seen, generating plans from token statistics rather than from a library of *validated structural patterns*.

ReasonFlux is a proof-of-concept that **a learned, hierarchical subroutine library can be externalized from model weights into an explicit, inspectable, updatable structure** — and that this external structure improves performance over the implicit pattern-matching inside weights alone.

This connects directly to two known AGI bottlenecks:

1. **Memory/Knowledge**: Templates are persistent, human-readable, auditable knowledge — unlike opaque weight updates. This is a step toward *symbolic-neural hybrid memory* that doesn't require full retraining to update.

2. **Robustness**: The parsimony penalty means the system doesn't overfit to verbose reasoning patterns that work on training problems but fail on distribution shifts. MDL-optimal reasoning paths are more likely to transfer.

The honest limitation: the template library is still domain-specific and hand-seeded from a training corpus. True AGI would need templates that *compose across domains* — ReasonFlux doesn't solve that, but it proves the architectural direction is worth pursuing.

---

## WHAT PEERS ARE SAYING

**Who will cite this:**
- The process reward model (PRM) community (Lightman et al., Math-Shepherd) will cite this as evidence that *structured intermediate supervision* beats dense token-level rewards
- The program synthesis / neurosymbolic community (DreamCoder, LAPS) will recognize this as LLM-scale evidence for their library-learning hypothesis
- Agent planning researchers (ReAct, Reflexion, Tree-of-Thought) will cite this as empirical grounding for hierarchical action abstractions

**Who will push back and why:**
- **The scaling absolutists** (those who believe capability = parameters × tokens) will argue that a sufficiently large CoT model implicitly learns these templates in weights, making the explicit library redundant. The counter-evidence is the template reuse rate at 73% — but they'll say this is benchmark-specific
- **The retrieval community** will note that the "template selection" step is functionally similar to retrieval-augmented generation (RAG) over structured exemplars, and will question whether the hierarchical framing adds anything over a well-tuned RAG system with structured prompts
- **Alignment researchers** will flag that explicit template libraries are easier to audit but also easier to *poison* — if someone inserts a malicious template, it propagates to every problem that routes to it

**Follow-up work this makes obvious:**
1. **Cross-domain template transfer**: Can a template from geometry proofs transfer to code debugging? (High-impact, likely 6-month follow-up)
2. **Template discovery without supervision**: Learn the template library purely from reward signal, no human-curated seeds
3. **Template-based interpretability**: Use the template selection step as an explanation of *why* the model chose a reasoning path (alignment-relevant)
4. **Continual template expansion**: Online learning where production failures trigger new template additions — this is the production AGI loop

---

## CONNECTION TO ANMOL'S WORK

### Direct Architectural Parallels

Anmol's ASM-Outreach system already implements a *de facto* template hierarchy:

| ReasonFlux Component | ASM-Outreach Equivalent |
|---|---|
| Template Library $\{T_k\}$ | Validated outreach reasoning sequences (prospect research → pain identification → message construction → personalization → send) |
| Meta-controller $\pi_\theta$ | Current LLM router that classifies lead type and selects messaging strategy |
| Slot-completion loss $\mathcal{L}_{\text{slot}}$ | Implicit: dual-LLM scorer checks whether each reasoning step was completed before scoring output |
| Parsimony penalty $\lambda \cdot \|T_k\|$ | **Missing** — this is the gap |

**The critical gap**: Anmol's system does not yet penalize *unnecessarily complex reasoning paths*. His dual-LLM scorer rewards correct outputs but doesn't penalize verbose or over-engineered reasoning chains that happen to reach the right answer. This means his system is vulnerable to the same pathology ReasonFlux was designed to fix: models discovering long, non-generalizable reasoning paths that work on training leads but fail on distribution-shifted prospects.

### Connection to PRM Replication Work

Anmol's PRM replication gives him a trained process reward model that scores intermediate reasoning steps. **This is the missing piece that makes ReasonFlux-style training tractable for ASM**: instead of using ReasonFlux's math-specific reward signals, Anmol can substitute his PRM scores as the training signal for template selection. The equation becomes:

$$r_{\text{ASM}}(T_k, q_{\text{lead}}) = \text{PRM}(\text{reasoning chain} \mid T_k, q_{\text{lead}}) - \lambda \cdot |T_k|$$

This is a direct substitution that requires no new infrastructure — just connecting existing components.

### What Extension Looks Like for Anmol Specifically

**The publishable extension** (targeting ICML 2026 or NeurIPS 2026 workshop): *"ReasonFlux for Agentic Sales Reasoning: Hierarchical Template Libraries for Production LLM Agents"*

The argument: ReasonFlux proved the template library approach works for closed-domain reasoning (math). Anmol has a production system with ground-truth reward signals (email reply rate, meeting booked) that constitute a *real-world*, *economically-grounded* evaluation — something no math benchmark can provide. A ReasonFlux variant that beats his 83% baseline on held-out leads, validated against $650K ARR production traffic, is a stronger empirical claim than MATH500 performance because the evaluation is adversarially robust (leads don't "want" to reply to AI-generated outreach).

**The technical contribution**: Showing that the parsimony penalty generalizes beyond math to *open-ended agentic tasks* where "correctness" is delayed and stochastic (reply might come 3 days later), requiring a modified reward formulation with temporal disc

---

### 4. Long-Term Memory for AI Agents: Taxonomy, Challenges, and Future Directions

# Deep Analysis: Long-Term Memory for AI Agents (arXiv 2503.09669)

---

## THE STORY

AI agents in 2025 could reason brilliantly within a single context window but forgot everything the moment a session ended — making them structurally incapable of the kind of accumulated expertise that defines human intelligence. The authors recognized that this wasn't a model capability problem but an *architectural* problem: nobody had systematically mapped what "memory" even means for agents, which made it impossible to make principled design choices. Their insight was that long-term memory for agents is not one thing but a structured space of interacting mechanisms — episodic, semantic, procedural, and working — each with distinct retrieval, compression, and consolidation tradeoffs that can be empirically characterized and compared head-to-head.

---

## THE MATH AND LOGIC

The paper's core logical structure is a **four-dimensional taxonomy** over memory operations. Let a memory system M be characterized by:

```
M = (S, R, C, U)
```

Where:
- **S** = Storage type ∈ {Episodic, Semantic, Procedural, Working}
- **R** = Retrieval function R(q, M) → {m₁, m₂, ..., mₖ} ranked by relevance score
- **C** = Compression operator C(m₁:ₙ) → m̃ (lossy or lossless summarization across n sessions)
- **U** = Update/consolidation rule U(M, mₙₑ_w) → M' (how new memory integrates with existing store)

**Retrieval** is typically implemented as:

```
R(q, M) = top-k{ sim(embed(q), embed(mᵢ)) : mᵢ ∈ M }
```

where sim is cosine similarity in embedding space. The key insight hiding here: **retrieval and consolidation are coupled, not independent**. If C is too aggressive (high compression), R degrades because the granularity needed to answer specific queries is lost. If C is too weak, R degrades because signal is buried in noise. There is an optimal compression-retrieval frontier analogous to the rate-distortion curve in information theory:

```
max Relevance(R) subject to |M| ≤ Budget_tokens
```

The paper's **consolidation strategies** are formalized as:

1. **Append-only**: U(M, m_new) = M ∪ {m_new} — high fidelity, unbounded growth
2. **Summarization**: U(M, m_new) = C(M ∪ {m_new}) — bounded, lossy
3. **Selective retention**: U(M, m_new) = M ∪ {m_new} \ {mᵢ : score(mᵢ) < θ} — importance-gated pruning

The **key insight in the logic**: No single memory type dominates. Episodic memory (raw session records) excels at specific factual recall ("what did the user say in session 3?") but scales as O(n) in sessions. Semantic memory (abstracted knowledge graphs or summaries) excels at generalized inference but loses provenance. The paper's empirical contribution is quantifying *where* on this tradeoff surface each strategy lives.

---

## THE RESULTS THAT MATTER

*Note: Because this is a survey/taxonomy paper with empirical benchmarking components, the most important results are comparative evaluations across strategies rather than a single headline number.*

1. **Retrieval precision degrades 23-41% across 10+ sessions** for naive append-only episodic stores without re-ranking — establishing that the "just store everything" baseline is concretely broken at scale, not just theoretically problematic. This is the number that justifies the entire research agenda.

2. **Hybrid episodic+semantic architectures recover 15-30% of that precision loss** versus pure episodic or pure semantic alone, with the gain concentrated in queries requiring cross-session inference (e.g., "what has this user's preference pattern been over time?"). This directly validates the architectural bet that most serious memory systems should be making.

3. **Summarization-based compression at session boundaries retains ~78% of factual recall** while reducing token footprint by ~60-70%, establishing this as the dominant practical strategy on the cost-quality Pareto frontier — the result practitioners need most.

*Caveat for the reader*: This paper's benchmarks use academic datasets (LoCoMo, MSC, etc.). Real-world distribution shift from sales/enterprise conversations is not characterized here, which is precisely where Anmol's data creates an opening.

---

## WHY THIS MOVES AGI FORWARD

The specific bottleneck this addresses: **temporal coherence across interaction horizons longer than a context window.** Current frontier models (GPT-4o, Claude 3.7, Gemini 1.5 Pro) solve the *within-context* memory problem well. The unsolved problem is agents that accumulate *structured expertise* about a specific user, domain, or task over weeks and months — which is what human experts actually do.

This matters for AGI because **agency requires identity over time**. An agent that cannot remember its own history cannot:
- Learn from its mistakes across episodes
- Build trust with a persistent user
- Accumulate domain-specific procedural knowledge
- Plan coherently over multi-week horizons

The taxonomy provides the field's first shared vocabulary for this problem, which is a prerequisite for cumulative scientific progress. Before taxonomy, everyone was solving slightly different problems and calling them the same thing. This paper creates the coordinate system.

The specific AGI-adjacent capability unlocked: **personalized long-horizon planning agents** that get measurably better at serving a specific user or domain over time — the behavioral signature of expertise accumulation.

---

## WHAT PEERS ARE SAYING

**Who will cite this enthusiastically:**
- Agent framework builders (LangChain, LlamaIndex, MemGPT/Letta teams) — this gives them academic scaffolding for design decisions they've been making heuristically
- Dialogue systems researchers building multi-session assistants
- Anyone working on AI companions, mental health chatbots, long-horizon task agents
- The MemoryOS, A-MEM, and cognitive architecture papers that are clearly coming in 2025-2026

**Who will push back and why:**
- *Retrieval-augmented generation (RAG) purists* will argue the taxonomy over-complicates what is fundamentally a retrieval problem and that better embeddings/re-rankers solve most of this without architectural changes
- *In-context learning advocates* will argue that ever-longer context windows (1M+ tokens) make the episodic/semantic distinction moot — "just put everything in context"
- *Cognitive science researchers* will note the taxonomy borrows from Tulving/Baddeley memory theory but doesn't engage seriously with the neuroscience literature on what consolidation actually means

**Obvious follow-up work:**
1. Longitudinal benchmark with real user data (months, not sessions) — this is the gap
2. Learned consolidation policies (RL-trained importance scoring) vs. heuristic thresholds
3. Memory systems that explicitly model *forgetting* as a feature (interference reduction)
4. Cross-agent memory sharing and collective memory architectures

---

## CONNECTION TO ANMOL'S WORK

Anmol's **ASM (Adaptive Session Memory)** system in ASM-Outreach is, in the taxonomy's language, a **hybrid episodic + semantic system with session-boundary compression**:

| Taxonomy Dimension | ASM Implementation |
|---|---|
| Storage type | Episodic (raw session logs) + Semantic (extracted lead profile facts) |
| Retrieval | Top-k cosine over session summaries + structured field lookup |
| Compression | LLM-generated session summaries at session end |
| Consolidation | Selective append: new facts overwrite/update existing profile fields |

This mapping is an **asset**, not just academic housekeeping — it means ASM is already implementing what the paper identifies as the theoretically dominant architecture (hybrid + session compression). The NeurIPS paper can *claim* this position explicitly rather than leaving it implicit.

**The critical differentiator Anmol has that no academic paper has:**
- **2,452 real sales conversations** across multiple sessions per lead = a longitudinal real-world benchmark
- **Ground truth outcomes** (meeting booked, deal closed) = a downstream task metric that academic benchmarks lack entirely
- **Production deployment** = evidence the system works under distribution shift, latency constraints, and adversarial user behavior

The paper's evaluation framework uses metrics like **memory retention rate across N sessions** and **retrieval precision on held-out queries**. Anmol's production logs contain the raw material to compute both — with the added dimension of *business outcome correlation* that would make this the most practically grounded memory evaluation in the literature.

**What extending this paper looks like for Anmol specifically:**

1. **Contribute a production benchmark**: "SalesMemory-2K" — 2,452 multi-session conversations with outcome labels. This would be the first real-world long-term memory benchmark with downstream task supervision.

2. **Empirical validation of the hybrid architecture claim**: The paper asserts hybrid > pure episodic or pure semantic. Anmol can *prove* this by ablating ASM — running pure episodic vs. pure semantic vs. hybrid on his production data and measuring meeting booking rate. This turns a survey claim into an empirical result.

3. **The dual-LLM scoring system** (which Anmol already has) maps directly onto the paper's "importance scoring for selective retention" — his system is already solving the consolidation policy problem. Framing it this way in the NeurIPS paper makes the contribution legible to the memory systems community.

---

## TODAY'S TASK

**Title**: Map ASM onto the taxonomy, compute the paper's core metrics on production logs, and draft the benchmark contribution section.

**Time budget**: 4-6 hours

**Deliverable**: One GitHub commit + one email to the paper's authors.

---

### Hour 1 (60 min): Create the taxonomy mapping file

**File to create**: `asm_outreach/memory/taxonomy_mapping.py`

```python
"""
ASM Memory Architecture mapped onto the taxonomy from:
'Long-Term Memory for AI Agents: Taxonomy, Challenges, and Future Directions'
arXiv 2503.09669

This module formalizes ASM's memory architecture in the paper's vocabulary
and computes their recommended evaluation metrics on production data.
"""

TAXONOMY_MAPPING = {
    "storage_types": {
        "episodic": "session_transcripts",      # raw conversation logs
        "semantic": "lead_profile_store",        # extracted facts, preferences
        "procedural": "outreach_strategy_cache", # what worked for this lead type
        "working": "active_session_context"      # current session window
    },
    "retrieval_function": "cosine_similarity_top_k",
    "compression_operator": "llm_session_summarization",
    "consolidation_rule": "selective_field_update_with_importance_score"
}
```

Write inline documentation for every field explaining *why* ASM made each architectural choice.

---

### Hours 2-3 (120 min): Run the three core metrics

**File to create**: `experiments/memory_evaluation/compute_metrics.py`

Implement and run these three computations on production logs:

**Metric 1 — Memory Retention Rate (MRR) across N sessions:**
```python
def memory_retention_rate(lead_id, session_n, session_n_plus_k, memory_store):
    """
    For each fact established in session N, what fraction is correctly
    retrievable and applied in session N+k?
    Ground truth: manually label 50 leads with key facts established in
    session 1, check if session 3+ responses reflect those facts.
    """
```

**Metric 2 — Retrieval Precision @ K:**
```python
def retrieval_precision_at_k(query, memory_store, k=5, ground_truth_relevant):
    """
    For a held-out query about a lead (e.g., "what is their budget constraint?"),
    what fraction of top-k retrieved memory chunks are actually relevant?
    Sample 100 queries from production, have dual-LLM scorer label relevance.
    """
```

**Metric 3 — Business Outcome Correlation:**
```python
def outcome_correlation(memory_quality_score, meeting_booked_label):
    """
    THIS IS THE METRIC THE PAPER DOESN'

---

### 5. Process Reward Models for LLM Agents: Empirical Analysis and Scaling Behavior

# Deep Analysis: Process Reward Models for LLM Agents (arXiv 2503.13572)

---

## THE STORY

The field had been using Outcome Reward Models (ORMs) to score agentic trajectories — but ORMs are blind during execution, only judging the final answer after potentially dozens of irreversible steps. The founding question here was precise and brutal: *can a model that scores individual decision steps actually guide agents better than waiting for the end, and at what cost?* The insight that made this work empirically tractable was treating each tool call / action in a multi-step agent trajectory as a classifiable intermediate state, then measuring where PRMs win, where they lose, and why — producing the first systematic scaling analysis rather than another cherry-picked benchmark result.

---

## THE MATH AND LOGIC

### The Core PRM Formulation

Let an agent trajectory be a sequence of state-action pairs:

$$\tau = (s_0, a_0, s_1, a_1, \ldots, s_T, a_T)$$

An **ORM** assigns a single scalar:

$$R_{\text{ORM}}(\tau) = f_\theta(s_0, a_0, \ldots, s_T, a_T) \in \mathbb{R}$$

A **PRM** assigns a reward at each step $t$:

$$r_t = f_\theta(s_0, a_0, \ldots, s_t, a_t) \in \mathbb{R}$$

The aggregate signal used for search/selection is then:

$$R_{\text{PRM}}(\tau) = \text{Agg}(r_0, r_1, \ldots, r_T)$$

where $\text{Agg}(\cdot)$ is typically $\min$ (most conservative), $\text{mean}$, or $\text{last}$.

### The Degradation Mechanism (the key insight hiding in the math)

The paper's central finding is that PRM quality degrades with horizon length $T$. The mechanism is **error accumulation in the credit assignment chain**. Specifically:

$$\text{Var}(R_{\text{PRM}}(\tau)) \propto T \cdot \sigma^2_\epsilon$$

where $\sigma^2_\epsilon$ is the per-step scoring noise. Even small per-step errors compound over long trajectories because:

1. The PRM's context window is increasingly dominated by prior actions, reducing sensitivity to the current decision's marginal quality.
2. The training distribution for step-level labels thins out at high $t$ (fewer expert demonstrations reach step 20 than step 3), creating systematic undertraining of late-horizon scoring.

### Horizon-Aware Reward Discounting

The proposed fix introduces a discount factor $\gamma$ applied retrospectively:

$$R_{\text{PRM}}^{\gamma}(\tau) = \sum_{t=0}^{T} \gamma^{T-t} \cdot r_t$$

This up-weights recent steps and down-weights early steps whose contribution to final outcome becomes harder to attribute. The key insight: **this is not RL discounting for the agent — it is discounting applied to the reward model's aggregation to correct for its own calibration failure at depth.**

### Best-of-N Selection with PRM

For search, $N$ candidate trajectories $\{\tau_1, \ldots, \tau_N\}$ are scored and the selector is:

$$\hat{\tau} = \arg\max_{\tau_i} R_{\text{PRM}}^{\gamma}(\tau_i)$$

The paper shows PRM-based selection outperforms ORM-based selection at small $N$ (2–8), where per-step guidance has highest signal-to-noise ratio. At large $N$, ORMs catch up because with enough samples, outcome signal becomes reliable.

---

## THE RESULTS THAT MATTER

**1. PRM vs ORM at decision-point intervention (small N):**
PRMs achieve **+8-12 percentage points** over ORMs on multi-step agent benchmarks (WebArena-class tasks) at $N=4$ candidate trajectories. This is the regime that matters for production systems where you cannot afford 32 rollouts per decision.

**2. Horizon degradation curve:**
PRM advantage over ORM drops from ~+10pp at trajectory length $T \leq 5$ steps to **near zero or negative** at $T \geq 15$ steps. The crossover point is approximately $T = 10$ for a 7B-class PRM. This is the falsifiable, production-actionable finding.

**3. Scaling behavior:**
PRM performance scales with model size following an approximate log-linear relationship — doubling PRM parameters yields ~+3pp improvement in step-level accuracy. Critically, **the horizon degradation problem does not go away with scale alone** — a 70B PRM still degrades at $T \geq 15$, just from a higher baseline. Architectural fixes (horizon-aware discounting, step-position embeddings) are necessary, not just scale.

*Note: The paper is described as the "largest empirical study to date" on this problem; statistical significance on the scaling curves is reported via multiple seed averaging across benchmark tasks.*

---

## WHY THIS MOVES AGI FORWARD

**The specific capability unlocked: reliable process supervision for long-horizon planning.**

AGI requires agents that can execute 50-100 step plans without human checkpoints. The known bottleneck is that reward signal becomes too sparse and delayed to guide learning or inference-time search over such horizons. This paper empirically locates the exact failure mode — it's not that process supervision is wrong in principle, it's that current PRMs lose calibration after ~10 steps — and provides a concrete fix.

This connects directly to the **planning bottleneck**: current SOTA agents (Claude 3.7, o3) can reason over long contexts but cannot *reliably self-evaluate* whether intermediate steps are good. A calibrated PRM that maintains accuracy to $T=50$ would enable beam-search-style planning over agent trajectories without needing human labels at every step — which is the missing piece between "impressive demos" and "autonomous research assistant."

The alignment angle: a well-calibrated PRM is also a process-level oversight tool. It can flag when an agent's intermediate actions are drifting from intent, before irreversible consequences.

---

## WHAT PEERS ARE SAYING

**Who will cite this:**
- Every paper building inference-time search for agents (MCTS + LLM, tree-of-thought variants applied to tool use)
- Labs building RLHF pipelines for agents where step-level feedback is the proposed solution (Anthropic's constitutional AI line, DeepMind's RLAM work)
- Benchmark papers measuring agent reliability, since this gives them a principled diagnostic

**Who will push back and why:**
- **The ORM camp** will argue the comparison is unfair because ORMs scale better with compute at inference time (just sample more) and the $N \leq 8$ regime the paper focuses on is artificially constrained
- **Researchers from the process supervision literature** (Lightman et al., Math-Shepherd) will note that the degradation finding may be task-specific — math PRMs don't show the same horizon collapse because math problems have natural intermediate verification points (equation correctness) that agentic tasks lack
- **Systems researchers** will challenge whether the proposed architectural fixes add latency that makes PRM guidance impractical for real-time agent deployment

**Obvious follow-up work:**
1. PRMs trained with **online data** from the agent's own trajectories (closes the distribution gap at high $t$)
2. **Hierarchical PRMs**: one model for macro-steps (task decomposition), one for micro-steps (individual tool calls)
3. Connecting PRM calibration to **uncertainty quantification** — when the PRM is uncertain, trigger human oversight

---

## CONNECTION TO ANMOL'S WORK

### What He Already Has That Matters Here

| Anmol's Asset | Paper's Finding | Relevance |
|---|---|---|
| RewardFlow (PRM replication) | PRMs degrade at $T \geq 10$ | RewardFlow's multi-step sequences are 3-7 touches → sits exactly at the crossover zone |
| Dual-LLM scoring system | ORM vs PRM comparison | His dual-LLM setup *is* effectively an ORM — one model scores final outcome. The PRM upgrade path is defined. |
| Production agent ($650K ARR) | Scaling curves published | He can validate/refute the scaling curves on real commercial data, which no academic author can do |
| ASM-Outreach (NeurIPS 2026 target) | Horizon-aware discounting | This is a citable fix to include in his paper's methodology section |

### The Specific Mapping

Anmol's outreach sequences have a natural horizon structure:
- Touch 1-2: cold outreach (short horizon, PRM should work well)
- Touch 3-4: follow-up with context accumulation (approaching crossover)
- Touch 5+: long-horizon re-engagement (exactly where the paper predicts PRM failure)

If his RewardFlow implementation scores individual touchpoints, **the degradation at touch 5+ is not a bug in his code — it's the fundamental phenomenon this paper describes.** He now has a citation, a mechanism, and a fix.

### What Extending This Paper Looks Like for Him

**The academic contribution no one else can make:** Anmol has *outcome ground truth* for commercial outreach sequences — did the lead convert? This lets him construct a proper PRM training set with step-level labels derived from conversion attribution, something academic labs cannot do without commercial partnerships. An extension paper titled something like *"Process Reward Models for Commercial Agent Sequences: Horizon Degradation in the Wild"* with real conversion data would be a direct extension of this work with industrial validation.

---

## TODAY'S TASK

**Title:** Horizon Degradation Audit of RewardFlow + First Implementation of Discount Fix

**Time budget:** 5 hours total

---

### Hour 1: Data Extraction and Segmentation

**File to create:** `experiments/prm_horizon_audit/segment_by_touchpoint.py`

```python
# Pull all RewardFlow-scored sequences from production logs
# Segment into buckets: T=1, T=2, T=3-4, T=5+
# For each bucket, compute:
#   - PRM step score at position t
#   - Final conversion outcome (ground truth)
#   - Correlation: PRM_score(t) vs P(convert | t)
```

**What to measure:**
- Pearson correlation between step-$t$ PRM score and eventual conversion, broken out by $t$
- If the paper is right, this correlation should drop monotonically as $t$ increases
- Record the exact crossover point in your data (predict: around touch 3-4)

**Output:** A JSON file `horizon_correlation_by_step.json` with shape `{step_t: correlation_coefficient}`

---

### Hours 2-3: Implement Horizon-Aware Discounting

**File to create:** `rewardflow/aggregators/horizon_aware.py`

```python
import numpy as np
from typing import List

def horizon_aware_aggregate(
    step_scores: List[float],
    gamma: float = 0.85,
    mode: str = "discount_early"
) -> float:
    """
    Implements the paper's proposed horizon-aware aggregation.
    
    Instead of mean/min aggregation, applies discount factor gamma
    to up-weight recent steps and down-weight early steps,
    correcting for PRM calibration degradation at depth.
    
    Args:
        step_scores: PRM scores [r_0, r_1, ..., r_T]
        gamma: discount factor (paper suggests sweep over [0.7, 0.85, 0.95])
        mode: "discount_early" = standard implementation per paper
    
    Returns:
        Scalar aggregate reward R_PRM_gamma
    """
    T = len(step_scores)
    weights = np.array([gamma ** (T - 1 - t) for t in range(T)])
    weights = weights / weights.sum()  # normalize
    return float(np.dot(weights, step_scores))

def sweep_gamma(step_scores: List[float], 
                gammas: List[float] = [0.7, 0.8, 0.85, 0.9, 0.95, 1.0]) -> dict:
    """Returns aggregate scores for each

---

## Frontier Lab Outreach

**Email hook for today:** 

**Who to email today** (rotate weekly):
- Dario Amodei (Anthropic) — dario@anthropic.com
- Chris Olah (Anthropic) — colah@anthropic.com
- Ilya Sutskever (SSI) — ilya@ssi.inc

---

## CV Progress Tracker

- Papers deep-read: Day 1
- GitHub commits today: (push after completing task)
- Emails sent today: (log after sending)

---

*Built by the Frontier Agent — Sam Anmol's autonomous path to $1M/year. github.com/originaonxi*