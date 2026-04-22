# TAGGER — Training & Inference Commands

All commands are run from the project root.

---

## Command Flow (quick reference)

```
[0] Smoke test          → verify pipeline runs without errors
[1] Full training       → produces best checkpoint + artifacts per experiment
[2] Artifacts (opt.)    → only if checkpoint exists but artifacts file is missing
[3] Build FAISS index   → one-time per experiment, after picking best run
[4] Run inference       → tag new transactions using the built index
[5] Confusion suite     → evaluate hard-pair discrimination (requires curated JSON)
```

---

## Step 0 — Smoke test (always run first)

Runs 2 epochs on a 10k stratified sample with frozen BERT.  
Verifies the full pipeline executes without errors before committing to a long run.

```bash
python -m src.train_multi_expt_ --config experiments_smoke.yaml --format yaml
```

Expected output folder:
```
experiments/tagger_proj256_final256_fd1_freeze_bs128_lr1.00e-04_val0.2_triplet/
```

> The `fd1` segment encodes `fusion_depth` (1 = default single-layer MLP; 2 = two residual blocks).

---

## Step 1 — Full training

Runs all 4 experiments sequentially (2×2 ablation: loss × freeze strategy).

```bash
python -m src.train_multi_expt_ --config experiments.yaml --format yaml
```

**Data split modes** (set in `experiments.yaml`):

| Mode | Config keys | Behaviour |
|------|-------------|-----------|
| Separate files (preferred) | `csv_path` + `val_csv_path` | Loads train and test CSVs independently; no splitting |
| Single file (fallback) | `csv_path` + `val_split` | Splits the single CSV by `val_split` fraction internally |

When `val_csv_path` is present it takes priority and `val_split` is ignored.  
The experiment directory name uses `_ext_` instead of `_val0.15_` when a separate val file is used.

To run a single experiment, comment out the others in `experiments.yaml` before running.

Each experiment auto-saves to its own directory under `experiments/`:

Experiment directory names now include the `fusion_depth` segment (`fd1` = default, `fd2` = deeper fusion):

| Config | Directory |
|--------|-----------|
| Triplet + full BERT | `experiments/tagger_proj256_final256_fd1_full_bs1024_lr4.00e-05_val0.15_triplet/` |
| Triplet + gradual unfreeze | `experiments/tagger_proj256_final256_fd1_gradual_bs1024_lr4.00e-05_val0.15_triplet/` |
| SupCon + full BERT | `experiments/tagger_proj256_final256_fd1_full_bs1024_lr4.00e-05_val0.15_supcon/` |
| SupCon + gradual unfreeze | `experiments/tagger_proj256_final256_fd1_gradual_bs1024_lr4.00e-05_val0.15_supcon/` |

Each experiment directory contains:
```
fusion_encoder_best.pth           ← best checkpoint (by val Recall@5)
fusion_encoder_epoch_N.pth        ← checkpoint every 5 epochs
training_artifacts.pkl            ← vocab, scaler, label mapping, fusion_depth (used by inference)
logs/training_logs.json           ← all metrics history (see Logging section)
logs/confusion_results.json       ← confusion suite results (if run)
plots/loss_curve.png
plots/grad_norms.png
plots/validation_metrics.png
plots/collapse_metrics.png        ← cosine sim, dead dims, effective rank + stable rank
plots/retrieval_metrics.png       ← MAP and nDCG@10 per epoch
plots/per_class_recall1.png       ← bar chart, final epoch
plots/per_class_recall_history.png ← heatmap of worst-30 classes × epochs
plots/tsne_epoch1.png             ← TSNE of validation embeddings at epoch 1
plots/tsne_best_epochN.png        ← TSNE at best recall epoch
plots/tsne_final.png              ← TSNE at final epoch
```

---

## Step 2 — Save artifacts (standalone, only if needed outside training)

The training loop already saves `training_artifacts.pkl` per experiment.  
Only run this separately if you need to rebuild artifacts from scratch
(e.g. you have a pre-trained checkpoint but lost the artifacts file).

```bash
python -m src.saving_artifacts
```

Output: `training_artifacts/training_artifacts.pkl`

---

## Step 3 — Build FAISS retrieval index

Run this after picking the best experiment from Step 1.
Set `EXP` to the experiment name shown in Step 1.

```bash
EXP=tagger_proj256_final256_full_bs1024_lr4.00e-05_val0.15_triplet

python run_inference.py \
  --artifacts experiments/$EXP/training_artifacts.pkl \
  --model     experiments/$EXP/fusion_encoder_best.pth \
  --csv       data/sample_txn.csv \
  --index     experiments/$EXP/golden_records.faiss \
  --index-type HNSW \
  --batch-size 512
```

**Index type options:**

| Type | Use case |
|------|----------|
| `HNSW` | Default. Fast approximate search, no training needed |
| `IVF` | Better recall for very large corpora (>1M records), slower to build |
| `L2` | Exact search, use only for small corpora (<100K) |
| `IP` | Exact inner-product search (use only if embeddings are not L2-normalised) |

Output files:
```
experiments/$EXP/golden_records.faiss
experiments/$EXP/golden_records_metadata.pkl
```

---

## Step 4 — Run inference (tag new transactions)

Tag transactions from a CSV file.

```bash
EXP=tagger_proj256_final256_full_bs1024_lr4.00e-05_val0.15_triplet

python run_inference.py \
  --artifacts  experiments/$EXP/training_artifacts.pkl \
  --model      experiments/$EXP/fusion_encoder_best.pth \
  --index      experiments/$EXP/golden_records.faiss \
  --skip-build \
  --input-csv  data/sample_txn.csv \
  --output-csv inference_results.csv \
  --top-k      5 \
  --batch-size 512
```

**Input CSV required columns:**
`cust_id`, `tran_date`, `tran_partclr`, `dr_cr_indctor`, `tran_amt_in_ac`, `tran_mode`

Optional: `sal_flag` (defaults to `N` if missing)

**Output CSV columns:**
`cust_id`, `tran_date`, `tran_partclr`, `tran_amt_in_ac`, `tran_mode`,
`dr_cr_indctor`, `category`, `confidence`, `top_k_matches`

Rows with `confidence < 0.7` are left with an empty `category` (majority vote too split).

---

## Step 5 — Confusion Suite (C3)

Evaluates hard-pair discrimination on a curated set of (query, positive, confuser) triples.  
Requires `data/confusion_pairs.json` to be hand-curated first (see format below).

```bash
EXP=tagger_proj256_final256_fd1_full_bs1024_lr4.00e-05_val0.15_triplet

python -m src.confusion_suite \
  --exp_dir experiments/$EXP \
  --pairs   data/confusion_pairs.json \
  --out     experiments/$EXP/logs/confusion_results.json
```

**Pair file format** (`data/confusion_pairs.json`):
```json
[
  {
    "name": "salary_vs_p2p",
    "query":    {"tran_partclr": "SALARY FROM EMPLOYER", "tran_mode": "NEFT", "dr_cr_indctor": "C", "sal_flag": "Y", "tran_amt_in_ac": 50000.0},
    "positive": {"tran_partclr": "MONTHLY SALARY NEFT", "tran_mode": "NEFT", "dr_cr_indctor": "C", "sal_flag": "Y", "tran_amt_in_ac": 48000.0},
    "confuser": {"tran_partclr": "NEFT TRANSFER FROM FRIEND", "tran_mode": "NEFT", "dr_cr_indctor": "C", "sal_flag": "N", "tran_amt_in_ac": 5000.0}
  }
]
```

A pair **passes** iff `cos(query, positive) > cos(query, confuser)`.  
Results are printed per-pair with margin and saved to `logs/confusion_results.json`.

**Run automatically at end of training** by adding to your config:
```yaml
confusion_pairs_path: "data/confusion_pairs.json"
```

---

## Deeper Fusion (E2)

To run an experiment with the 2-layer residual fusion MLP, add to a config:
```yaml
fusion_depth: 2
```

`fusion_depth: 1` (default) preserves the original single-layer behaviour.  
Old checkpoints are incompatible with `fusion_depth: 2` — always rebuild index after changing depth.

---

## Logging & Experiment Tracking

### What is logged automatically (no setup needed)

Every training run writes to its experiment directory:

`logs/training_logs.json` keys:

| Key | Description |
|-----|-------------|
| `epoch_losses` | Train loss per epoch |
| `val_losses` | Val triplet loss per epoch |
| `val_recall5` | Recall@5 per epoch |
| `val_accuracies` | Recall@1 (accuracy) per epoch |
| `val_map` | MAP per epoch |
| `val_ndcg10` | nDCG@10 per epoch |
| `collapse_history` | Per-epoch: `avg_cosine_similarity`, `dead_dimensions`, `effective_rank`, `stable_rank` |
| `per_class_recall_history` | Per-epoch list of `{class_str: recall}` dicts |
| `per_class_recall_last_epoch` | Final epoch per-class recall (convenience key) |
| `confusion_results` | Pass rate + per-pair detail (null if no pairs file configured) |
| `best_val_recall5` / `best_epoch` | Best recall and the epoch it occurred |

To inspect logs from a finished run:

```bash
EXP=tagger_proj256_final256_fd1_full_bs1024_lr4.00e-05_val0.15_triplet
python -c "
import json
d = json.load(open('experiments/$EXP/logs/training_logs.json'))
print(f'Best Recall@5: {d[\"best_val_recall5\"]:.4f} at epoch {d[\"best_epoch\"]}')
print(f'Final MAP:     {d[\"val_map\"][-1]:.4f}')
print(f'Final nDCG@10: {d[\"val_ndcg10\"][-1]:.4f}')
"
```

---

### Weights & Biases (wandb)

wandb is **not wired into the training code yet**. The section below documents what to add and the commands to run once it is integrated.

#### One-time setup

```bash
pip install wandb
wandb login          # paste your API key from wandb.ai/authorize
```

#### Add to each config in experiments.yaml

```yaml
wandb_project: "tagger-financial-encoder"
wandb_entity:  "your-wandb-username"   # or team name
wandb_group:   "phase1"                # groups related runs together
```

#### Integration points in src/train_multi_expt_.py

Add at the top of `run_experiment()`, after `exp_dir` is created:

```python
import wandb
run = wandb.init(
    project=config.get("wandb_project", "tagger-financial-encoder"),
    entity=config.get("wandb_entity", None),
    name=exp_name,
    group=config.get("wandb_group", "default"),
    config=config,
    dir=exp_dir,
    reinit=True,
)
```

Log step-level metrics inside the optimizer step block:

```python
wandb.log({
    "train/step_loss": loss.item() * accumulation_steps,
    "train/grad_norm": grad_norm,
    "train/lr": scheduler.get_last_lr()[0],
}, step=global_step)
```

Log epoch-level metrics after `print_validation_report()`:

```python
wandb.log({
    "train/epoch_loss":              avg_loss,
    "val/loss":                      val_metrics['val_loss'],
    "val/recall@1":                  val_metrics.get('recall@1', 0),
    "val/recall@5":                  val_metrics['recall@5'],
    "val/recall@10":                 val_metrics.get('recall@10', 0),
    "val/accuracy":                  val_metrics['accuracy'],
    "val/mrr":                       val_metrics.get('mrr', 0),
    "val/map":                       val_metrics.get('map', 0),
    "val/ndcg@10":                   val_metrics.get('ndcg@10', 0),
    "collapse/avg_cosine_similarity": collapse['avg_cosine_similarity'],
    "collapse/dead_dimensions":       collapse['dead_dimensions'],
    "collapse/effective_rank":        collapse['effective_rank'],
    "collapse/stable_rank":           collapse.get('stable_rank', 0),
    "epoch": epoch + 1,
}, step=global_step)
```

Log per-class recall as an interactive table:

```python
if val_metrics.get('per_class_recall1'):
    inv_label_map = {v: k for k, v in train_dataset.label_mapping.items()}
    table = wandb.Table(columns=["class_name", "recall@1"])
    for code, recall in val_metrics['per_class_recall1'].items():
        table.add_data(inv_label_map.get(code, str(code)), recall)
    wandb.log({"val/per_class_recall": table}, step=global_step)
```

Log best model checkpoint as a versioned artifact:

```python
# Inside the best-model save block:
artifact = wandb.Artifact(
    name=f"model-{run.id}", type="model",
    metadata={"epoch": epoch + 1, "val_recall5": best_val_recall5}
)
artifact.add_file(best_model_path)
run.log_artifact(artifact)
```

Log plots at the end of the run:

```python
wandb.log({
    "plots/loss_curve":                wandb.Image(os.path.join(exp_dir, "plots", "loss_curve.png")),
    "plots/validation_metrics":        wandb.Image(os.path.join(exp_dir, "plots", "validation_metrics.png")),
    "plots/collapse_metrics":          wandb.Image(os.path.join(exp_dir, "plots", "collapse_metrics.png")),
    "plots/retrieval_metrics":         wandb.Image(os.path.join(exp_dir, "plots", "retrieval_metrics.png")),
    "plots/per_class_recall1":         wandb.Image(os.path.join(exp_dir, "plots", "per_class_recall1.png")),
    "plots/per_class_recall_history":  wandb.Image(os.path.join(exp_dir, "plots", "per_class_recall_history.png")),
    "plots/tsne_final":                wandb.Image(os.path.join(exp_dir, "plots", "tsne_final.png")),
})
wandb.finish()
```

#### Viewing runs

```bash
wandb sync experiments/<name>/wandb/   # push an offline run
wandb status                           # check sync status
```

Or open the web UI: `wandb.ai/<entity>/tagger-financial-encoder`

#### Hyperparameter sweeps (after baseline runs are done)

Create `sweep.yaml`:

```yaml
program: src/train_multi_expt_.py
method: bayes
metric:
  name: val/recall@5
  goal: maximize
parameters:
  base_lr:
    distribution: log_uniform_values
    min: 1.0e-5
    max: 1.0e-3
  dropout:
    values: [0.1, 0.2, 0.3]
  supcon_temperature:
    distribution: uniform
    min: 0.05
    max: 0.3
  freeze_strategy:
    values: ["full", "gradual"]
  loss_type:
    values: ["triplet", "supcon"]
```

```bash
wandb sweep sweep.yaml                     # prints SWEEP_ID
wandb agent <entity>/tagger-financial-encoder/<SWEEP_ID>
```

---

## GPU vs CPU notes

- Training: AMP (`use_amp: true`) requires CUDA. On CPU, set `use_amp: false`.
- Index building: FP16 (`--no-fp16` to disable) requires CUDA.
- Inference: automatically uses CUDA if available, falls back to CPU.
- FAISS GPU index: only available if `faiss-gpu` is installed (vs `faiss-cpu`).

---

## Troubleshooting

**`PKSampler: p=32 but only N classes have >= 2 samples`**  
Reduce `pk_p` in `experiments.yaml` to a value ≤ N.

**`Dimension mismatch: FAISS index has dim=X but model outputs dim=Y`**  
The index was built with a different model. Rebuild: rerun Step 3.

**`RuntimeError: Error(s) in loading state_dict` after changing `fusion_depth`**  
The checkpoint was saved with a different `fusion_depth`. Retrain or use the matching `fusion_depth` value when loading. Old checkpoints (no `fd` in the name) used `fusion_depth=1`.

**`--model is required`**  
You must always pass `--model` to `run_inference.py`. It has no default.

**Validation is very slow**  
Retrieval evaluation is capped at 5000 samples by default (set in `validation.py:MAX_RETRIEVAL_EVAL`).  
If validation still hangs, reduce `val_split` or `batch_size` in your config.
