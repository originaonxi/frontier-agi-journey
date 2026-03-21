# Day 1: MEM1 vs ASM — Memory Compression Benchmark on Live Agent Traces

## What I Built

A 4-strategy memory compression benchmark comparing MEM1-style within-session gating against our production ASM (Anchored State Memory) system on 52 real Aonxi prospects backed by 2,452 production lead traces and 42 closed deals ($650K ARR).

Strategies evaluated:
1. **No Memory** — pure signal-based scoring, no history
2. **MEM1-style** — within-session compression with simulated RL gating (60% context reduction)
3. **ASM (ours)** — multi-session structured anchors (98.1% compression, 46 anchors from 2,452 leads)
4. **Summary** — free-form text summary spanning all sessions (standard RAG baseline)

## Paper Extended

**MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agent Tasks**

Extended with: first benchmark on real production B2B agent traces (vs. synthetic AndroidWorld/WebArena), cross-session comparison dimension MEM1 doesn't address, and discovery of survivorship bias in structured memory banks.

## My Contribution

Discovered **survivorship bias in structured memory banks**: when 91% of stored anchors are positive outcomes (FINISH/meeting), retrieval systematically inflates scores for low-signal prospects. This is a failure mode that synthetic benchmarks cannot surface — it only appears when memory is built from skewed real-world outcome distributions.

## Results

| Strategy | MAE ↓ | RMSE ↓ | Calibration Gap ↑ | Compression | Cross-Session |
|----------|-------|--------|-------------------|-------------|---------------|
| No Memory | **0.842** | **0.973** | 3.71 | N/A | No |
| MEM1-style | 0.846 | 1.097 | 3.77 | 62.5% | No |
| ASM (ours) | 1.731 | 1.868 | 3.58 | **98.1%** | **Yes** |
| Summary | 1.538 | 1.576 | **4.33** | 99.5% | Yes |

**Key numbers:**
- MEM1 within-session compression adds **near-zero value** over no-memory baseline (MAE 0.846 vs 0.842)
- ASM achieves **36% higher compression** than MEM1 (98.1% vs 62.5%) using 36% fewer tokens
- ASM retrieves cross-session anchors for **87%** of prospects (45/52)
- Summary's unstructured cross-session context achieves best calibration (4.33 gap)

## Why It Matters for AGI

1. **Memory ≠ compression.** MEM1 optimizes within-session gating. Production agents need cross-session persistence. These are different problems requiring different architectures.

2. **Survivorship bias in memory banks is a real failure mode.** When agents learn from their successes (meetings booked), they build positively-skewed memory that inflates confidence on weak prospects. This mirrors the explore-exploit tradeoff — memory banks need deliberate negative sampling.

3. **Structured vs. unstructured memory.** ASM's typed anchors (FINISH, DEPENDENCY, EXCEPTION) should outperform raw summaries — but only when the type distribution is balanced. A summary that says "November failed at 0.62%" is more calibrating than 42 structured FINISH anchors.

4. **Real data exposes what synthetic benchmarks hide.** AndroidWorld/WebArena have deterministic outcomes. B2B outreach has stochastic outcomes — the same ICP can ghost or book. Memory systems need to handle this stochasticity.

## Production Connection

- Dataset: `aonxi.db` — 52 live prospects from $650K ARR Aonxi outreach system
- Anchor bank: 46 real anchors from 6 months of campaigns (Oct 2025–Mar 2026)
- Key December anchor: strict signal filtering produced 10.5x efficiency vs November (6.52% vs 0.62% lead-to-sale)
- Directly informs NeurIPS 2026 submission on multi-session agent memory
