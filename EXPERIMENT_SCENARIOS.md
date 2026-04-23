# Experiment Scenarios — Observations & Next Actions

A running log of training scenarios. For each scenario, record observations and enumerate the concrete next actions worth trying.

---

## How to Use This Document

Each scenario entry follows this structure:
- **Config** — dataset, freeze strategy, loss
- **Observations** — raw numbers from training logs
- **Diagnosis** — what the numbers are telling you
- **Possible Next Actions** — ranked from highest to lowest expected impact

---

## Scenario 1 — cleaned_merchant · Full Unfreeze · Triplet Loss

**Date:** 2026-04-23

### Config
| Setting | Value |
|---------|-------|
| Dataset | cleaned_merchant |
| Freeze strategy | Full unfreeze (all BERT layers trainable from epoch 1) |
| Loss | Triplet |

### Observations

#### Epoch 1
| Metric | Value |
|--------|-------|
| Val Loss | 0.0916 |
| Accuracy | 0.9836 |
| MRR | 0.9849 |
| MAP | 0.7929 |
| nDCG@10 | 0.9596 |
| Recall@1 | 0.9836 |
| Recall@5 | 0.9862 |
| Recall@10 | 0.9868 |
| avg_cos_sim | 0.164 |
| dead_dims | 12 / 256 |
| effective_rank | 117 |

Worst classes: 28 (0.9429), 3 (0.9438), 14 (0.9444), 33 (0.9524), 17 (0.9580)
Best classes: 44, 47, 4, 1, 24 (all 1.0000)

#### Epoch 2
| Metric | Value |
|--------|-------|
| Val Loss | 0.0699 |
| Accuracy | 0.9888 |
| MRR | 0.9895 |
| MAP | 0.8537 |
| nDCG@10 | 0.9724 |
| Recall@1 | 0.9888 |
| Recall@5 | 0.9896 |
| Recall@10 | 0.9910 |
| avg_cos_sim | 0.116 |
| dead_dims | 11 / 256 |
| effective_rank | 144 |

Worst classes: 32 (0.9524), 11 (0.9588), 7 (0.9604), 28 (0.9619), 5 (0.9647)
Best classes: 47, 10, 15, 41, 2 (all 1.0000)

### Diagnosis

**Embedding health:** Healthy. avg_cos_sim dropped from 0.164 → 0.116 (embeddings are becoming more diverse, not collapsing). dead_dims slightly improved (12 → 11). effective_rank rose 117 → 144, meaning the model is using more of its 256-d space — good sign that BERT is actually contributing.

**Loss trend:** Dropping cleanly (0.0916 → 0.0699). No signs of overfitting yet. Model is still learning.

**Retrieval quality:** Strong. MAP jumped +7.6 pp (0.793 → 0.854) in one epoch — the model is getting better at ranking, not just top-1. nDCG@10 and Recall@10 are also improving, suggesting the representation space is tightening.

**Worst classes:** Class 28 is consistently the worst across both epochs. Classes 3, 14, 33, 17 (epoch 1) mostly recovered. New weak classes (32, 11, 7) appeared in epoch 2 — likely because the model is refining decision boundaries and the easy gains have been made.

**Risk flags:**
- Full unfreeze from epoch 1 can cause catastrophic forgetting in early epochs. The metrics suggest this hasn't happened yet, but watch the loss curve carefully at epochs 3–4.
- effective_rank=144/256 is fine but not maximal — some capacity may still be underutilised.

### Possible Next Actions

#### A. Continue and observe (lowest effort, highest value right now)
1. **Run to epoch 4–5 and watch for signs of overfitting.** Val loss is still dropping sharply. Don't stop early. Check whether MAP continues improving — it jumped a lot epoch 1→2 and may plateau.
2. **Check the loss curve plot** (`loss_curve.png`). If step loss within epoch 2 is still noisy-but-decreasing, the model has headroom. If it flattened mid-epoch, the LR may be too low by epoch 3.

#### B. Target the weak classes
3. **Inspect the raw texts for class 28.** It is the only class that was weak in epoch 1 AND epoch 2 — implying it has genuine ambiguity or noisy labels, not just an easy gap to close.
4. **Pull all misclassified class 28 samples and check what class they are retrieved as.** If they are consistently confused with one other class, that is a labelling or feature issue, not a model issue.
5. **Try upweighting class 28 in the sampler (PKSampler P/K).** If class 28 has fewer samples than others, the balanced sampler may not be sampling it enough per batch.

#### C. Loss / learning signal
6. **Switch to SupCon loss** (Supervised Contrastive). Triplet only sees one negative per anchor per step; SupCon sees all negatives in the batch. With good class balance, SupCon typically gives a bigger MAP jump than Recall@1, which matches your bottleneck (MAP=0.85 vs Recall@1=0.99).
7. **Add online hard negative mining to the triplet sampler.** The current triplet sampler may be selecting semi-hard negatives. Mining harder negatives would give the model a stronger signal on the already-well-separated classes and potentially help the weak ones.

#### D. Freeze strategy comparison
8. **Run the same config with gradual unfreeze** (freeze BERT epoch 1, unfreeze top 2 layers epoch 2, full unfreeze epoch 3). Compare epoch-2 MAP — if gradual unfreeze gives similar MAP with lower variance across classes, it is safer. If MAP is lower, full unfreeze is justified.
9. **Add a warmup LR schedule for the BERT layers specifically.** Full unfreeze with a flat LR can apply too large a gradient to the lower BERT layers early on. A layerwise LR decay (lower LR for lower layers) is the most principled fix.

#### E. Architecture / embedding
10. **Lower embedding dim from 256 → 128.** effective_rank=144 with only 11 dead dims at epoch 2 is healthy, but if you are leaving capacity on the table (rank=144 out of 256), a smaller head might regularise better and train faster.
11. **Try mean-pooled last_hidden_state instead of pooler_output** if not already done. The pooler_output CLS head is fine-tuned for NSP, not retrieval. Mean pooling over all tokens is generally better for sentence-level similarity tasks.

#### F. Data
12. **Audit the cleaned_merchant cleaning pipeline.** If cleaning was aggressive (e.g., removing merchant-specific tokens), it may have erased discriminative signals for classes 28, 32, 11. Compare a raw vs cleaned sample from class 28.
13. **Run with uncleaned data as a baseline** for 2 epochs and compare class-28 Recall@1. If cleaning hurt class 28, the cleaning step needs revision.

---

*Add new scenarios below this line as experiments progress.*

---
