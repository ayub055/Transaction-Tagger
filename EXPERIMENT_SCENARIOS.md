# Experiment Scenarios — Observations & Next Actions

A running log of training scenarios. For each scenario, record observations and enumerate the concrete next actions worth trying.

---

## How to Use This Document

Each scenario entry follows this structure:
- **Config** — dataset, freeze strategy, loss, key hyperparameters
- **Observations** — raw numbers per epoch (full table + epoch-over-epoch deltas)
- **Collapse trend** — embedding health trajectory, not just single-epoch snapshots
- **Patience tracker** — where early stopping stands and what the options are
- **Diagnosis** — what the numbers are collectively telling you
- **Possible Next Actions** — ranked from highest to lowest expected impact

### Dimensions to always check in diagnosis

| Dimension | What to look for | Red flag |
|-----------|-----------------|----------|
| **Loss trend** | Monotonically decreasing | Flattening before epoch 5, or increasing val loss while train loss drops |
| **MAP vs Recall@1 gap** | MAP < Recall@1 means embeddings separate top-1 well but ranking beyond top-1 is noisy | MAP stalling while R@1 improves = representation space not tight enough |
| **Recall@5 vs Recall@10** | Small gap = most positives cluster near the query | Large gap = positives are scattered, negative mining too easy |
| **Collapse: avg_cos_sim** | Should be < 0.3 and decreasing | Rising toward 0.9 = collapse; already below 0.1 = may be under-constrained |
| **Collapse: dead_dims** | Should be low and decreasing | Rising dead dims = model not using embedding capacity |
| **Collapse: effective_rank** | Should be high and stable | Dropping rank = representational collapse beginning |
| **Class volatility** | Which classes rotate in/out of worst-5 | Persistent worst class across ≥3 epochs = structural data issue, not a training issue |
| **Primary metric plateau + secondary still moving** | R@5 plateaued but MAP improving → more to gain; switch the primary metric or extend patience | Both plateaued = genuine convergence |
| **LR schedule phase** | Cosine: warming up → peak → decay. Where are you? | Validating improvement at the peak — may drop once LR decays below useful threshold |
| **Best epoch gap** | How many epochs since best? | If best was early (epoch 1-2) and counter is high → LR too large or data issue |

### Class behaviour taxonomy

| Pattern | Meaning | Action |
|---------|---------|--------|
| Class in worst-5 every epoch | Structural: label noise, class ambiguity, or too few samples | Audit raw texts; check confusion target; use PKSampler |
| Class in worst-5 then recovers | Boundary class, model refining decision surface | Normal — watch for re-emergence |
| New class enters worst-5 at later epochs | Model tightened other classes; this one is now relatively the weakest | Check if absolute recall is still acceptable (e.g. 0.94 is fine) |
| Class flips between 0.93–1.00 across epochs | High within-class variance (few samples) | Increase K in PKSampler for this class |

---

## Scenario 1 — cleaned_merchant · Full Unfreeze · Triplet Loss

**Date:** 2026-04-23  
**Status:** Running — early stopping counter reset at epoch 6 (new best). Best model at epoch 6.

### Config
| Setting | Value |
|---------|-------|
| Dataset | cleaned_merchant |
| Freeze strategy | Full unfreeze (all BERT layers trainable from epoch 1) |
| Loss | Triplet (semi-hard mining) |
| Pooling | mean |
| Scheduler | Cosine + warmup |
| batch_size | 256, accumulation=4 → effective=1024 |
| base_lr | 0.00002 |
| Patience | 4 (monitored on Recall@5) |

---

### Observations — Full Epoch Table

| Epoch | Val Loss | Accuracy | MRR | MAP | nDCG@10 | R@1 | R@5 | R@10 | avg_cos_sim | dead_dims | eff_rank |
|-------|----------|----------|-----|-----|---------|-----|-----|------|-------------|-----------|----------|
| 1 | 0.0916 | 0.9836 | 0.9849 | 0.7929 | 0.9596 | 0.9836 | 0.9862 | 0.9868 | 0.164 | 12/256 | 117 |
| 2 | 0.0699 | 0.9888 | 0.9895 | 0.8537 | 0.9724 | 0.9888 | 0.9896 | 0.9910 | 0.116 | 11/256 | 144 |
| 3 | 0.0581 | 0.9908 | 0.9915 | 0.8676 | 0.9782 | 0.9908 | **0.9922** | 0.9926 | — | — | — |
| 4 | 0.0516 | 0.9906 | 0.9914 | 0.8850 | 0.9770 | 0.9906 | 0.9922 | 0.9928 | — | — | — |
| 5 | 0.0466 | 0.9914 | 0.9918 | 0.8970 | 0.9785 | 0.9914 | 0.9922 | 0.9926 | 0.102 | 5/256 | 130 |
| 6 | 0.0393 | 0.9932 | 0.9937 | 0.9118 | 0.9850 | 0.9932 | **0.9940** | 0.9942 | 0.103 | 4/256 | 122 |

**Bold** = best Recall@5 (primary early-stopping metric). Best model saved at epoch 6.

#### Epoch-over-epoch deltas (Δ)

| Epoch | ΔVal Loss | ΔAccuracy | ΔMAP | ΔR@5 | Δdead_dims |
|-------|-----------|-----------|------|------|------------|
| 1→2 | −0.0217 | +0.0052 | +0.0608 | +0.0034 | −1 |
| 2→3 | −0.0118 | +0.0020 | +0.0139 | +0.0026 | — |
| 3→4 | −0.0065 | −0.0002 | +0.0174 | 0.0000 | — |
| 4→5 | −0.0050 | +0.0008 | +0.0120 | 0.0000 | — |
| 5→6 | −0.0073 | +0.0018 | +0.0148 | +0.0018 | −1 |

---

### Per-Class Recall@1 Worst-5 History

| Epoch | #1 Worst | #2 | #3 | #4 | #5 |
|-------|----------|----|-----|-----|-----|
| 1 | 28 (0.9429) | 3 (0.9438) | 14 (0.9444) | 33 (0.9524) | 17 (0.9580) |
| 2 | 32 (0.9524) | 11 (0.9588) | 7 (0.9604) | 28 (0.9619) | 5 (0.9647) |
| 3 | 32 (0.9434) | 14 (0.9592) | 15 (0.9655) | 22 (0.9659) | 43 (0.9770) |
| 4 | 32 (0.9231) | 22 (0.9439) | 0 (0.9720) | 28 (0.9730) | 17 (0.9789) |
| 5 | 22 (0.9474) | 32 (0.9515) | 17 (0.9727) | 11 (0.9787) | 0 (0.9787) |
| 6 | 32 (0.9459) | 17 (0.9651) | 22 (0.9762) | 35 (0.9773) | 28 (0.9802) |

**Class volatility analysis:**
- **Class 32** — worst or second-worst in epochs 2, 3, 4, 5, 6. **Persistent structural issue.** Dipped to 0.9231 at epoch 4, partial recovery to 0.9459 at epoch 6. Likely label ambiguity or insufficient samples.
- **Class 22** — entered at epoch 3, peaked as #1 worst at epoch 5 (0.9474), recovered to 0.9762 at epoch 6. Boundary refinement — not structural. Monitor to confirm.
- **Class 17** — returned at epoch 6 (0.9651) after recovery at epochs 3–4. Worth watching — second appearance suggests persistent boundary ambiguity.
- **Class 28** — was #1 worst at epoch 1, recovered. Brief reappearance at epoch 4, back in worst-5 at epoch 6 (0.9802) at a much higher absolute recall. Not a concern at this level.
- **Class 35** — new entry at epoch 6 (0.9773). Likely exposed as other classes tightened. Absolute recall is acceptable; monitor for persistence.
- **Classes 0, 3, 14, 33** — no longer in worst-5. Normal boundary refinement resolved.

---

### Collapse Trend

| Epoch | avg_cos_sim | dead_dims | eff_rank | Interpretation |
|-------|------------|-----------|----------|----------------|
| 1 | 0.164 | 12 | 117 | Starting point — moderate diversity |
| 2 | 0.116 | 11 | 144 | Improving — more diverse, higher rank |
| 5 | 0.102 | **5** | 130 | Dead dims dropped sharply (12→5). Rank dipped slightly (144→130) |
| 6 | 0.103 | **4** | 122 | avg_cos_sim flat (+0.001). Dead dims still falling (5→4). Rank dipped 130→122 |

Dead dims now at 4 — near floor, model is using nearly all embedding capacity. avg_cos_sim held essentially flat (0.102→0.103), no collapse concern. Effective rank dipping gradually (144→122 over epochs 2–6); the model is consolidating onto a tighter manifold consistent with convergence, not collapse. The rank decline is modest and parallel to MAP gains — representations are becoming more structured, not more degenerate.

---

### Patience Tracker

| Epoch | R@5 | Counter | Status |
|-------|-----|---------|--------|
| 1 | 0.9862 | — | Baseline |
| 2 | 0.9896 | reset | Improved |
| 3 | **0.9922** | reset | **Best — checkpoint saved** |
| 4 | 0.9922 | 1/4 | No improvement |
| 5 | 0.9922 | 2/4 | No improvement |
| 6 | **0.9940** | reset | **New best — checkpoint saved** |

Counter reset at epoch 6. 0 epochs of patience used. Training continues; if R@5 does not improve for 4 more epochs (by epoch 10), training stops.

**Critical observation:** The R@5 ceiling-effect concern from epoch 5 resolved — R@5 jumped +0.0018 to 0.9940. Headroom is now only 0.0060. MAP broke through 0.90 for the first time (0.9118), gaining +0.0148 in a single epoch. Val loss continues monotonically downward (0.0916 → 0.0393, −57% from epoch 1). The model is still learning.

---

### Diagnosis

**Breakthrough at epoch 6:** R@5 finally broke its plateau (+0.0018 to 0.9940), MAP crossed 0.91 for the first time, and a new best checkpoint was saved. The concern about premature early stopping is resolved for now — the model is still on an upward trajectory.

**MAP is the real story:** 0.793 → 0.912 across 6 epochs, +0.0148 in epoch 6 alone. This is the metric that matters most for retrieval quality at inference. It is not saturated — further gains are expected.

**Class 32 is the structural problem:** Worst class in 5 of 6 epochs, absolute recall stuck between 0.92–0.95. This is not a training issue. Audit raw texts for class 32 — likely label ambiguity or insufficient samples rather than a solvable training configuration problem.

**Class 22 is recovering:** Was #1 worst at epoch 5 (0.9474), recovered to 0.9762 at epoch 6 — consistent with boundary refinement, not structural noise. Monitor; if it stays out of worst-5 at epoch 7, consider it resolved.

**Class 17 re-emerged:** Was in worst-5 at epoch 1, recovered, now back at #2 worst (0.9651). Two appearances suggest persistent boundary overlap with another class. Worth investigating alongside class 32.

**Collapse is healthy:** avg_cos_sim flat at ~0.103 (no collapse), dead dims at 4 (near-floor — model using nearly all capacity). Effective rank declining gradually (144→122) in line with manifold consolidation; not a red flag given parallel MAP gains.

**R@5 ceiling is tightening:** At 0.9940, only 0.60pp of headroom remains. Another plateau is likely. Watch MAP and val loss as the signal once R@5 saturates again.

---

### Possible Next Actions

#### A. Immediate — watch for R@5 plateau recurrence (highest priority)
1. **If R@5 plateaus again at 0.9940, switch early stopping metric to MAP.** R@5 has only 0.60pp of headroom; MAP is still rising +0.015/epoch and has clear capacity for further gains. Alternatively, increase patience to 6–8 so a 2–3 epoch plateau doesn't prematurely end a run where MAP and val loss are still improving.
2. **Or: switch primary metric to Val Loss.** Monotonically decreasing, gives the cleanest signal once all retrieval metrics near their ceiling.

#### B. Class 32 investigation (act now — 5 of 6 epochs, structural)
3. **Audit raw texts for class 32.** Pull all misclassified class-32 samples and find what class they are confused with. If they consistently retrieve class X, inspect whether 32 and X share narration patterns.
4. **Check class 32 sample count.** If underrepresented, use PKSampler with high K for that class. If overrepresented but still failing, the issue is label ambiguity, not data volume.

#### C. Class 17 watch (second appearance — borderline)
5. **Monitor class 17 for 1–2 more epochs.** Was worst-5 at epoch 1 (recovered), now #2 worst at epoch 6 (0.9651). If it persists at epoch 7, treat it alongside class 32 as a structural issue and inspect confusion targets.

#### D. Class 22 — tentatively resolved
6. **No action needed on class 22 yet.** It recovered from 0.9474 to 0.9762 in one epoch. Consistent with boundary refinement. Watch passively — act only if it re-enters worst-5 at epoch 8+.

#### E. Continue training with current setup
7. **Let training continue to epoch 10.** Patience counter reset at epoch 6. MAP at 0.9118 is still rising; val loss at 0.0393 shows no overfitting. Trajectory suggests MAP could reach 0.93+ by epoch 8–10.
8. **Watch for train/val loss divergence.** When val loss starts rising while train loss falls, stop regardless of patience.

#### F. Extraction of current best
9. **The epoch-6 checkpoint is the current best** by R@5 (0.9940), MAP (0.9118), and val loss (0.0393). The epoch-5 model is now superseded. If deploying before training ends, use epoch 6.

#### G. Next experiment configuration (when this run ends)
10. **Re-run with `val_every_n_steps` active** so you can catch the MAP peak mid-epoch rather than waiting for epoch end.
11. **Try SupCon loss** as the next ablation. MAP stalling at 0.93+ is the point where SupCon's full-batch negatives have the most advantage over triplet.

---

*Add new scenarios below this line as experiments progress.*

---
