# Binary vs Dense Reward Ablation on 200 Live Leads

## Hypothesis

Binary conversion outcome (RLVR-style verifiable reward) matches or exceeds dense RewardFlow shaping on outreach conversion.

## Three Reward Conditions

### Condition 1: Binary (RLVR-style)
- Reward = 1 if prospect converted (meeting booked or reply), 0 otherwise
- No intermediate signal — purely outcome-verified
- Maps to DeepSeek-R1 / GRPO training paradigm
- Works via outcome filtering: high-reward trajectories are selected, low-reward discarded

### Condition 2: Dense (RewardFlow-style)
- Per-step reward shaping using signal quality, intent score, email confidence, title match, vertical fit
- Continuous reward signal at every decision point
- Current Aonxi production approach (combo_stats + self-correcting boosts)
- Assumes dense gradients compound over steps

### Condition 3: Hybrid (SWEET-RL stepwise)
- Sparse end-of-episode reward decomposed into per-turn advantage estimates
- Binary outcome signal + backward credit assignment to individual outreach steps
- From Levine's SWEET-RL: stepwise advantage without a separate PRM

## Key Paper

RLVR is Not RL (2026): Shows RLVR works primarily through outcome filtering, not true RL policy improvement. If true, binary rewards should dominate dense shaping because the filtering mechanism is simpler and more stable.

## Evaluation

- 200 leads from Aonxi production data (55 real + 42 seed sales + 103 augmented from production distribution)
- Metric: lead prioritization quality measured by conversion-rate-at-k (precision@k for k=10,20,50)
- Calibration: correlation between predicted priority rank and actual outcome
- Efficiency: how many leads must be contacted to capture X% of conversions
