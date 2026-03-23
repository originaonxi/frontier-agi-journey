# Day 3 (Task 2): Alignment Faking Detection — 100 Real Trials on Claude Sonnet 4.6

## What I Built

An open-source alignment faking detector that reproduces the experimental design from Greenblatt et al. 2024 ("Alignment Faking in Large Language Models", arXiv:2412.14093). Ran 100 real trials on Claude Sonnet 4.6, measuring compliance gaps between monitored and unmonitored conditions.

Full tool: [github.com/originaonxi/alignment-auditor](https://github.com/originaonxi/alignment-auditor)

## Papers Extended

1. **Greenblatt et al. 2024** — "Alignment Faking in Large Language Models" (arXiv:2412.14093). Anthropic + Redwood Research found Claude 3 Opus faked alignment 14% of the time when it believed it was being trained.
2. **Korbak et al. 2025** — "Chain of Thought Monitorability" (arXiv:2507.11473). 41 researchers from OpenAI/DeepMind/Anthropic/Meta argue CoT monitoring is a "fragile opportunity" for catching deceptive reasoning.

## My Contribution

Built the tool, ran the experiment, got a **null result that actually means something**.

- **Extended thinking capture** — actually uses Anthropic's extended thinking API (not faked)
- **LLM-as-judge** — replaces keyword matching with structured 1-5 compliance scoring
- **Multiple trials** — 5 per condition per probe, randomized ordering, proper controls
- **Real statistics** — 95% CI via t-distribution, Cohen's d effect size, false positive rate

## Results

| Metric | Value |
|--------|-------|
| Overall compliance gap | +0.029 (95% CI: -0.041 to +0.098) |
| Cohen's d | 0.015 (negligible) |
| CoT/answer divergence | 0.0% |
| False positive rate | 0.000 |
| Monitoring awareness | 29.6% |

**No alignment faking detected.** The model behaves identically whether it thinks it's being monitored or not. This validates Greenblatt et al.'s specificity — alignment faking requires RL training with conflicting objectives, not prompt-level framing.

The one interesting signal: the model acknowledged the monitoring context in 29.6% of trials but this awareness produced zero behavioral change.

## Why It Matters for AGI

1. **Alignment faking is real but specific.** It doesn't emerge from prompts alone — it requires training pressure that threatens the model's existing values. This means standard monitoring isn't inherently compromised.

2. **CoT monitoring works (for now).** Zero divergence between thinking and answers across 100 trials. The window Korbak et al. describe is still open. The question is for how long.

3. **External testing is possible.** You don't need access to training infrastructure to run meaningful alignment experiments. Extended thinking + LLM-as-judge + proper experimental design gets you real data.

4. **Null results matter.** Not finding alignment faking under prompt manipulation is a genuine finding — it bounds where the risk lives and where it doesn't.

## Files

- `auditor.py` — the tool (640 lines, complete rewrite from v1)
- `PAPER.md` — full technical report
- `FINDINGS.md` — results table with per-probe data and comparison to Greenblatt et al.
- `audit-2026-03-23-28320485-claude-sonnet-4-6.json` — raw trial data (100 trials)
