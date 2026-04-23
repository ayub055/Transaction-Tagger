# Training Health Checklist

Use this document after every training run to decide whether the experiment is worth keeping, needs tuning, or must be discarded. Work through the sections in order — loss first, then retrieval quality, then embedding health, then per-class behaviour.

All values come from `experiments/<exp_name>/logs/training_logs.json` and the plots in `experiments/<exp_name>/plots/`.

---

## How to Read the Log Quickly

```
experiments/<exp_name>/
├── logs/
│   ├── training.log          ← human-readable timestamped run log
│   └── training_logs.json    ← all numeric history (load with json.load)
└── plots/
    ├── loss_curve.png
    ├── step_loss_curve.png
    ├── grad_norms.png
    ├── validation_metrics.png
    ├── collapse_metrics.png
    ├── retrieval_metrics.png
    ├── per_class_recall1.png
    ├── per_class_recall_history.png
    ├── tsne_epoch1.png
    ├── tsne_best_epoch{N}.png
    └── tsne_final.png
```

---

## Section 1 — Training Loss

**Source:** `epoch_losses`, `step_losses` in JSON → `loss_curve.png`, `step_loss_curve.png`

### What it measures
The contrastive/triplet loss on the training batches. It tells you whether the model is actually learning to push same-class embeddings together and different-class embeddings apart.

### Checklist

| # | Check | Healthy | Yellow Flag | 🔴 Red Alert |
|---|-------|---------|-------------|--------------|
| 1.1 | **Epoch loss direction** | Decreasing across epochs | Flat after epoch 2 | Increasing or oscillating wildly |
| 1.2 | **Epoch loss magnitude (triplet)** | Starts ~0.3–0.5, drops toward 0.05–0.15 by final epoch | Starts or stays above 0.5 | Never drops below starting value |
| 1.3 | **Epoch loss magnitude (SupCon)** | Starts ~4–6, drops toward 1–2 | Starts high and barely moves | Increases after epoch 1 |
| 1.4 | **Step loss within an epoch** | Noisy but downward trend within epoch | Spikes of >5× the running average | Persistent NaN or Inf values |
| 1.5 | **Train vs Val loss gap** | Val loss within 20–30% of train loss | Val loss 2× train loss from epoch 3+ | Val loss increases while train loss decreases (overfitting) |
| 1.6 | **Loss at epoch 1** | Already below starting value | Same as before training | Higher than random initialisation would suggest |

### Interpretation notes
- **Triplet loss hitting 0.0 too fast** (within 1–2 epochs) usually means all triplets are trivially satisfied — the model has collapsed or the mining strategy is not finding hard enough negatives. Cross-check against collapse metrics (Section 4).
- **SupCon loss plateau around 4–5** means the model is treating all pairs as equally negative — check batch composition (too few classes per batch, PKSampler P too low).
- **Sudden loss spike mid-training** is usually a learning rate issue or a corrupt batch. One spike is tolerable; two consecutive spikes need investigation.

---

## Section 2 — Validation Loss

**Source:** `val_losses` in JSON → `validation_metrics.png` (top-left panel)

### What it measures
Triplet loss computed on the held-out validation set using random triplet sampling. Independently confirms the training loss signal is not overfitting.

### Checklist

| # | Check | Healthy | Yellow Flag | 🔴 Red Alert |
|---|-------|---------|-------------|--------------|
| 2.1 | **Direction** | Decreasing, tracking train loss | Flat while train loss still drops | Increasing after initial drop |
| 2.2 | **Gap from train loss** | Within 2× of train loss | 3–5× train loss | >5× train loss (severe overfit) |
| 2.3 | **Consistency with retrieval metrics** | Val loss and Recall@5 move together | Val loss improves but Recall@5 stagnates | Val loss improves but Recall@5 degrades |

### Interpretation notes
- Val loss is a **weak proxy** for retrieval quality because it uses random triplets, not exhaustive nearest-neighbour search. Always use Recall@K as the primary decision metric (Section 3).
- A low val loss with poor Recall@5 usually means the model has learned to separate a few easy clusters perfectly but handles the hard confusable classes poorly.

---

## Section 3 — Retrieval Quality Metrics

**Source:** `val_recall5`, `val_accuracies`, `val_map`, `val_ndcg10` → `validation_metrics.png`, `retrieval_metrics.png`

These are the **primary metrics** for deciding whether a model is production-ready. They measure real nearest-neighbour retrieval quality over 5,000 validation samples.

### Metric Definitions

| Metric | What it asks |
|--------|-------------|
| **Recall@1** (= Accuracy) | Is the single nearest neighbour the correct class? |
| **Recall@5** | Is the correct class anywhere in the top-5 neighbours? (primary training objective) |
| **Recall@10** | Is the correct class anywhere in the top-10 neighbours? |
| **MRR** | On average, at what rank does the first correct neighbour appear? (1.0 = always rank 1) |
| **MAP** | Precision averaged over all correct retrievals — penalises correct hits buried deep in results |
| **nDCG@10** | Like MAP but with position-based discount — retrievals at rank 1 count more than rank 10 |

### Checklist

| # | Check | Healthy | Yellow Flag | 🔴 Red Alert |
|---|-------|---------|-------------|--------------|
| 3.1 | **Recall@5 final epoch** | > 0.80 | 0.60–0.80 | < 0.60 |
| 3.2 | **Recall@1 final epoch** | > 0.60 | 0.40–0.60 | < 0.40 |
| 3.3 | **Recall@1 vs Recall@5 gap** | < 0.25 (model is confident) | 0.25–0.40 | > 0.40 (model often gets it in top-5 but not top-1 — confusable classes) |
| 3.4 | **MRR final epoch** | > 0.65 | 0.45–0.65 | < 0.45 |
| 3.5 | **MAP final epoch** | > 0.55 | 0.35–0.55 | < 0.35 |
| 3.6 | **nDCG@10 final epoch** | > 0.70 | 0.50–0.70 | < 0.50 |
| 3.7 | **Recall@5 trajectory** | Monotonically increasing or near-monotonic | Plateau after epoch 3 | Up-down oscillation with no clear trend |
| 3.8 | **Best epoch vs final epoch** | Final ≤ 3% below best | Final 5–10% below best | Final >10% below best (severe late-epoch regression) |
| 3.9 | **MAP vs nDCG@10 ratio** | nDCG ≈ MAP or slightly above | nDCG << MAP (correct hits are buried deep) | — |

### Interpretation notes
- **High Recall@5 but low MRR** means the model reliably finds the right class within 5 guesses but rarely puts it first. Acceptable for top-K inference; needs work for top-1 deployment.
- **MRR > Recall@1** is impossible — if you see this, there's a bug in metric logging.
- **Recall@5 plateau at 0.70–0.75** with no improvement across experiments usually points to class confusion in the data (overlapping categories), not a model issue. Check per-class recall (Section 5).
- **Recall@5 regresses in later epochs** while train loss continues to fall → classic overfitting. Lower patience or add dropout.

---

## Section 4 — Embedding Collapse

**Source:** `collapse_history` in JSON → `collapse_metrics.png`

These metrics are computed every epoch on a subsample of validation embeddings. They are the **early warning system** — collapse can happen well before retrieval metrics visibly degrade.

### The Four Collapse Signals

#### 4a. Average Pairwise Cosine Similarity (`avg_cosine_similarity`)

Measures how similar all embeddings are to each other. If everything collapses to a single point or a thin cone, this shoots toward 1.0.

| Value | Interpretation |
|-------|---------------|
| 0.0 – 0.2 | ✅ Healthy: embeddings are well spread |
| 0.2 – 0.5 | ✅ Acceptable: some clustering, which is desired |
| 0.5 – 0.7 | ⚠️ Watch: embeddings are becoming dense — check if classes are genuinely confusable |
| 0.7 – 0.9 | ⚠️ Warning: significant collapse — model may be shortcutting |
| > 0.9 | 🔴 **COLLAPSE**: all embeddings are pointing in roughly the same direction |

**Red alert trigger:** `avg_cosine_similarity > 0.9` for two consecutive epochs.

#### 4b. Dead Dimensions (`dead_dimensions` / `total_dimensions`)

Counts embedding dimensions with variance < 1e-4 — dimensions the model has learned to ignore. A 256-dim embedding space with 200 dead dims is effectively 56-dimensional.

| Dead dims (out of 256 total) | Interpretation |
|------------------------------|---------------|
| 0–10 | ✅ Healthy |
| 10–30 | ✅ Acceptable |
| 30–60 | ⚠️ Model is underusing capacity — try higher dropout or lower final_dim |
| 60–128 | ⚠️ Warning: significant waste — consider architecture changes |
| > 128 (> 50%) | 🔴 **DIMENSIONAL COLLAPSE**: embedding space is wasted |

**What dead dimensions mean in practice:** The FAISS index and nearest-neighbour search still work, but you are paying the full inference cost of a 256-dim model while only getting ~(256−dead_dims)-dim discriminative power. This directly hurts retrieval quality.

**Red alert trigger:** `dead_dimensions > 50%` of `total_dimensions`.

#### 4c. Effective Rank (`effective_rank`)

Number of singular values above 1% of the largest singular value. Measures how many truly independent directions the embedding space uses.

| Value (for 256-dim embeddings) | Interpretation |
|-------------------------------|---------------|
| > 100 | ✅ Rich, diverse representation |
| 50–100 | ✅ Acceptable |
| 20–50 | ⚠️ Model is over-specialising |
| 10–20 | ⚠️ Warning: very low rank |
| < 10 | 🔴 **RANK COLLAPSE**: effectively a 10-dim model in a 256-dim wrapper |

**Red alert trigger:** `effective_rank < 10` at any point after epoch 2.

#### 4d. Stable Rank (`stable_rank`)

Scale-invariant version of effective rank. Computed as `sum(s²) / s0²` where `s` are singular values. Robust to L2 normalisation effects.

| Value | Interpretation |
|-------|---------------|
| > 50 | ✅ Healthy |
| 20–50 | ✅ Acceptable |
| 10–20 | ⚠️ Watch |
| < 10 | 🔴 **LOW STABLE RANK**: same concern as effective rank, confirmed by a different method |

### Collapse Checklist

| # | Check | Healthy | Yellow Flag | 🔴 Red Alert |
|---|-------|---------|-------------|--------------|
| 4.1 | `avg_cosine_similarity` at epoch 1 | < 0.3 | 0.3–0.5 | > 0.5 (starts collapsed) |
| 4.2 | `avg_cosine_similarity` trend | Stable or gently rising | Steadily rising | Rapid rise to > 0.9 |
| 4.3 | `dead_dimensions` at final epoch | < 30 / 256 | 30–60 / 256 | > 128 / 256 |
| 4.4 | `dead_dimensions` trend | Stable or decreasing | Slowly rising | Rapidly rising epoch over epoch |
| 4.5 | `effective_rank` at epoch 1 | > 50 | 20–50 | < 20 |
| 4.6 | `effective_rank` trend | Rising or stable | Slowly declining | Sharp drop across 2 consecutive epochs |
| 4.7 | `stable_rank` vs `effective_rank` | Both agree (< 20% difference) | One much higher — investigate SVD distribution | Both < 10 simultaneously |
| 4.8 | Collapse + good Recall@5 | Fine — model has learned tight clusters | — | Recall@5 also low → the model has truly failed |

### Collapse diagnosis and fixes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `avg_cos_sim` → 1.0 fast (epoch 1–2) | Learning rate too high, or L2 norm pushing everything together | Reduce LR by 5–10×; check L2 normalisation is applied after projection head, not before |
| Dead dims rising but rank OK | Specific dimensions are saturating | Increase dropout; try smaller `final_dim` |
| Rank collapse with low dead dims | Whole space is shrinking proportionally | Likely mode collapse in SupCon — try lower temperature (0.04 instead of 0.07) |
| Collapse only at specific epoch | Learning rate schedule step | Check if StepLR drop coincides with collapse epoch |
| Good rank, high cos similarity | Tight clusters that are all near each other | This is actually OK if per-class recall is good — classes may be genuinely similar in the data |

---

## Section 5 — Per-Class Recall@1

**Source:** `per_class_recall_history`, `per_class_recall_last_epoch` in JSON → `per_class_recall1.png`, `per_class_recall_history.png`

### What it measures
Recall@1 broken down by individual transaction category. Shows which classes the model handles well and which it confuses.

### Checklist

| # | Check | Healthy | Yellow Flag | 🔴 Red Alert |
|---|-------|---------|-------------|--------------|
| 5.1 | **Minimum per-class Recall@1** | > 0.50 for all classes | Any class < 0.30 | Any class at 0.0 (model never correctly predicts it) |
| 5.2 | **Class recall variance** | Most classes > 0.70, a few stragglers | Many classes in 0.30–0.60 range | Bimodal distribution: some at > 0.9, many at < 0.2 |
| 5.3 | **Recall vs class size** | Low-support classes can be lower, but not zero | Large classes (> 1000 samples) with recall < 0.5 | Large-class recall < 0.3 |
| 5.4 | **Per-class recall history (heatmap)** | Worst classes improving epoch over epoch | Worst classes stuck, not improving | Worst classes getting worse over time |
| 5.5 | **Number of classes with recall < 0.50** | < 10% of total classes | 10–25% | > 25% |

### Interpretation notes
- **A class stuck at 0.0 recall** almost always means that class's text descriptions are identical or near-identical to another class, or the class has too few training samples to form a meaningful cluster.
- **Per-class heatmap showing a block of permanently red rows** means those classes are structurally confusable in the data — investigate with the confusion suite (`confusion_pairs_path`).
- **High-support class with low recall** is more concerning than a low-support class with low recall.

---

## Section 6 — Gradient Norms

**Source:** `grad_norms` in JSON → `grad_norms.png`

### What it measures
Global L2 norm of all gradients before the clipping step, tracked per optimiser step. Reveals training stability — gradient explosion/vanishing before clip.

### Checklist

| # | Check | Healthy | Yellow Flag | 🔴 Red Alert |
|---|-------|---------|-------------|--------------|
| 6.1 | **Typical norm magnitude** | 0.1 – 2.0 | 2.0 – 5.0 | Consistently > 10 |
| 6.2 | **Direction over training** | Gradually decreasing or stable | High variance throughout | Increasing trend across epochs |
| 6.3 | **Spike frequency** | Occasional spikes (< 1% of steps) | Regular spikes (> 5% of steps) | Every 10–20 steps |
| 6.4 | **Spike magnitude** | 3–5× running average | 10–20× running average | > 50× running average |
| 6.5 | **Norm at epoch 1 vs final epoch** | Final < epoch-1 norm | Same | Final > epoch-1 norm |

### Interpretation notes
- **Gradient clip is set at max_norm=1.0**. If you see `grad_norms` consistently near or above 1.0, the model is hitting the clip ceiling — gradients are being significantly truncated every step. This slows learning and can cause instability.
- **Very small grad norms (< 0.01)** after epoch 1 indicate the model has converged or gradients are vanishing (frozen BERT layers passing no gradient). Check the freeze strategy is working as intended.
- **Spikes coinciding with new BERT layers being unfrozen** (gradual freeze strategy) are expected — the newly unfrozen layers have large gradients initially.

---

## Section 7 — TSNE Visualisations

**Source:** `plots/tsne_epoch1.png`, `plots/tsne_best_epoch{N}.png`, `plots/tsne_final.png`

### What to look for

| Plot | What it should show | 🔴 Red Alert |
|------|-------------------|--------------|
| `tsne_epoch1.png` | Loose, somewhat overlapping clusters — model just started | Uniform blob with no structure at all |
| `tsne_best_epochN.png` | Tight, well-separated clusters; each colour should form a distinct island | Clusters still heavily overlapping; no visible separation |
| `tsne_final.png` | Same as best, ideally identical if no late-epoch regression | Noticeably worse than best epoch — confirms late-epoch overfit |

### Checklist

| # | Check | Healthy | Yellow Flag | 🔴 Red Alert |
|---|-------|---------|-------------|--------------|
| 7.1 | **Cluster separation (best epoch)** | Clear gaps between colour groups | Overlapping but distinct centres | Single blob, no grouping by colour |
| 7.2 | **Intra-cluster compactness** | Tight colour patches | Elongated or scattered patches per class | Each class is scattered across the entire plot |
| 7.3 | **Epoch 1 → best epoch progression** | Clear improvement in cluster definition | Marginal improvement | No change — model never learned |
| 7.4 | **Best → final regression** | No visible degradation | Minor fraying at cluster boundaries | Final plot looks like epoch 1 |
| 7.5 | **Number of colours visible** | Should match number of classes in validation (capped at 10 by tab10 cmap) | Some colours missing — class not in retrieval eval subsample | — |

---

## Section 8 — Early Stopping Behaviour

**Source:** `best_epoch`, `training.log` early-stopping messages

### Checklist

| # | Check | Healthy | Yellow Flag | 🔴 Red Alert |
|---|-------|---------|-------------|--------------|
| 8.1 | **Best epoch** | Epoch 5 – (max_epochs − patience) | Epoch 1 or 2 (model peaked too early) | Epoch = max_epochs (model never converged, ran out of budget) |
| 8.2 | **Patience counter at stop** | Exactly `patience` epochs of no improvement | — | Triggered at epoch 2–3 (trained less than 5 epochs total) |
| 8.3 | **Gap: best epoch → final epoch** | < 5 epochs of wasted training | 5–10 epochs | > 10 epochs (patience too high, wasted compute) |
| 8.4 | **Best Recall@5 vs final Recall@5** | < 2% difference | 2–5% regression | > 5% regression at final epoch |

---

## Section 9 — Freeze Strategy Check

**Source:** `training.log` lines `Total Layers : N | # Unfrozen Layers...`, config key `freeze_strategy`

### Checklist

| # | Check | Healthy | Yellow Flag | 🔴 Red Alert |
|---|-------|---------|-------------|--------------|
| 9.1 | **`full` strategy: recall improvement** | Improves from epoch 1 | Slow start (epoch 1–3 near-zero recall) — BERT weight initialisation dominating | No improvement at any epoch |
| 9.2 | **`gradual` strategy: unfreezing log** | Layer count increases 1 per epoch as expected | — | Layers not reported in log (freeze function not called) |
| 9.3 | **`gradual` vs `full` Recall@5** | Gradual should match or exceed full by final epoch | Gradual lags full by > 5% — consider more epochs | Gradual significantly worse (BERT not unfreezing enough) |
| 9.4 | **Grad norms at unfreeze point** | Spike when new layers unlock, then settle | Spike never settles | Spike causes loss explosion |

---

## Section 10 — Artifacts and Files Check

A run is only usable for inference if all these files exist.

### Checklist

```
experiments/<exp_name>/
├── fusion_encoder_best.pth          ← must exist
├── training_artifacts.pkl           ← must exist (contains vocab, scaler, label_mapping, config)
├── logs/
│   ├── training.log                 ← must exist
│   └── training_logs.json           ← must exist and be valid JSON
└── plots/
    ├── loss_curve.png               ← sanity visual
    ├── collapse_metrics.png         ← sanity visual
    └── tsne_best_epoch*.png         ← if missing, TSNE was skipped — check log for the reason
```

| # | Check | Action if missing |
|---|-------|------------------|
| 10.1 | `fusion_encoder_best.pth` exists | Training crashed before first validation — check log for OOM/exception |
| 10.2 | `training_artifacts.pkl` exists | Training crashed before saving artifacts — model is unusable for inference |
| 10.3 | `training_logs.json` is valid JSON | `json.load` it; if corrupt, check if disk ran out of space during write |
| 10.4 | TSNE plots exist | If absent: check log for `TSNE snapshot skipped` — likely a sklearn version issue (fixed in `plotting.py`) or OOM |
| 10.5 | `training_artifacts.pkl` contains correct `config.text_col` | Load and print `pkl['config']['text_col']` — must match what the experiment YAML specified |

---

## Quick Decision Summary

After checking all sections above, use this table to decide what to do with the run.

| Overall Status | Criteria | Action |
|---------------|----------|--------|
| ✅ **Ship** | Recall@5 > 0.80, no collapse, all artifacts present | Use for inference |
| ✅ **Keep (watch)** | Recall@5 0.70–0.80, minor collapse signals, no hard red alerts | Acceptable; try one more run with tweaked LR |
| ⚠️ **Investigate** | Recall@5 0.55–0.70, OR any single red alert in sections 4 or 6 | Debug before using — check which section triggered |
| 🔴 **Discard** | Recall@5 < 0.55, OR `avg_cos_sim` > 0.9, OR `effective_rank` < 10, OR loss never decreased | Discard; do not use for inference or comparison |

---

## Comparison Across Experiments

When comparing multiple experiments on the same test data, rank them by this priority:

1. **Recall@5** (primary — used for best-model checkpointing)
2. **Recall@1** (production relevance — is top-1 prediction correct?)
3. **MRR** (ranking quality — how high up is the correct answer?)
4. **nDCG@10** (penalises burying correct answers)
5. **`avg_cosine_similarity`** as a tiebreaker (lower is better — richer embedding space)
6. **`dead_dimensions`** as a tiebreaker (lower is better)

Do **not** compare runs on `val_loss` alone — two models can have identical val loss with very different retrieval quality.

---

## Reading `training_logs.json` Programmatically

```python
import json
import os

exp = "tagger_proj256_final256_fd1_gradual_bs768_lr3.46e-05_ext_cleaned_merchant_triplet"
with open(f"experiments/{exp}/logs/training_logs.json") as f:
    log = json.load(f)

print(f"Best Recall@5        : {log['best_val_recall5']:.4f}  (epoch {log['best_epoch']})")
print(f"Final epoch Recall@5 : {log['val_recall5'][-1]:.4f}")
print(f"Final avg_cos_sim    : {log['collapse_history'][-1]['avg_cosine_similarity']:.3f}")
print(f"Final dead_dims      : {log['collapse_history'][-1]['dead_dimensions']}/{log['collapse_history'][-1]['total_dimensions']}")
print(f"Final effective_rank : {log['collapse_history'][-1]['effective_rank']}")
print(f"Final stable_rank    : {log['collapse_history'][-1]['stable_rank']:.1f}")
print(f"Epochs trained       : {len(log['epoch_losses'])}")
```
