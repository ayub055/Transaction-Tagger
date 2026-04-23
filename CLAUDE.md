# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Commands

```bash
# Smoke test — always run first to verify the pipeline before a long run
python -m src.train_multi_expt_ --config experiments_smoke.yaml --format yaml

# Full multi-experiment training run
python -m src.train_multi_expt_ --config experiments.yaml --format yaml

# Save artifacts standalone (only if checkpoint exists but .pkl is missing)
python -m src.saving_artifacts

# Run inference + build FAISS index
python run_inference.py \
  --artifacts experiments/$EXP/training_artifacts.pkl \
  --model     experiments/$EXP/fusion_encoder_best.pth \
  --csv       data/sample_txn.csv \
  --index     experiments/$EXP/golden_records.faiss \
  --index-type HNSW --batch-size 512

# Run inference only (index already built)
python run_inference.py --skip-build \
  --artifacts experiments/$EXP/training_artifacts.pkl \
  --model     experiments/$EXP/fusion_encoder_best.pth \
  --index     experiments/$EXP/golden_records.faiss \
  --input-csv data/sample_txn.csv --output-csv results.csv --top-k 5

# Confusion suite (requires data/confusion_pairs.json)
python -m src.confusion_suite \
  --exp_dir experiments/$EXP \
  --pairs   data/confusion_pairs.json \
  --out     experiments/$EXP/logs/confusion_results.json

# Audit the dataset
python dataset_audit.py

# Inspect a finished run's key metrics
python -c "
import json; d = json.load(open('experiments/$EXP/logs/training_logs.json'))
print(f'Best Recall@5: {d[\"best_val_recall5\"]:.4f} at epoch {d[\"best_epoch\"]}')
print(f'Final MAP: {d[\"val_map\"][-1]:.4f}')
"
```

---

## Architecture

### Pipeline overview

```
CSV → TransactionDataset → DataLoader → FusionEncoder → triplet/SupCon loss → validation → FAISS index → inference
```

### `FusionEncoder` (`src/fusion_encoder.py`)

Three modalities fused into a single L2-normalised embedding:
1. **BERT** (`bert-base-uncased`) → pooled via `pooling_strategy` → Linear → `text_proj_dim` (256-d)
2. **Categorical** (`tran_mode`, `dr_cr_indctor`, `sal_flag`) → `nn.Embedding(vocab, 4)` each → concat 12-d
3. **Numeric** (`tran_amt_in_ac`) → StandardScaler → 1-d

The three are concatenated and passed through a fusion MLP (`fusion_depth=1` = single linear; `fusion_depth>=2` adds residual blocks). Optionally, a projection head is appended for SupCon training (the pre-projection representation is returned at inference time).

`pooling_strategy` options: `'mean'` (recommended), `'cls'`, `'pooler'` (legacy, not recommended).

### Training loop (`src/train_multi_expt_.py`)

`run_experiment(config)` is the single entry point. Key flow:

1. Build experiment directory name from config fields; create `experiments/<name>/logs/` and `plots/`.
2. Load train/val data — prefers separate `val_csv_path`; falls back to `val_split` on single CSV.
3. Val labels are re-encoded in train's label space; unknown val labels are dropped.
4. Instantiate `FusionEncoder`, `AdamW`, and either `StepLR` or cosine+warmup scheduler.
5. Loss: `nn.TripletMarginLoss` with `sample_triplets()` (random or semi-hard mining) or `SupConLoss`.
6. After each epoch: run `evaluate_validation_metrics()`, compute collapse metrics, call `print_validation_report()`, save best checkpoint, optionally save TSNE snapshot.
7. After training ends: generate all plots, save `training_logs.json` and `training_artifacts.pkl`.

**Step-based validation** is not yet implemented — validation only fires once per epoch. With 5500+ batches per epoch this is a known gap.

**Gradient accumulation** is supported via `accumulation_steps`. Effective batch size = `batch_size × accumulation_steps`. LR is auto-scaled via square-root rule when `base_lr` is set.

**AMP** (`use_amp: true`) requires CUDA.

### Freeze strategies (`apply_freeze_strategy`)

| `freeze_strategy` | Behaviour |
|---|---|
| `freeze` | All BERT layers frozen |
| `gradual` | Unfreeze top-N layers each epoch (epoch 1 → top 1, epoch 2 → top 2, …) — top-down, most task-specific layers first |
| `full` | All BERT layers trainable from epoch 1 |

### Validation (`src/validation.py`)

`evaluate_validation_metrics()` does a single forward pass over the full val set, then:
- Builds a pairwise distance matrix on a **random** subsample capped at `MAX_RETRIEVAL_EVAL=5000`
- Computes Recall@{1,5,10}, MRR, MAP (full ranking), nDCG@10, per-class Recall@1
- Returns `all_embeddings` (full set, CPU) for collapse detection

`EarlyStopping` monitors Recall@5 (mode='max', patience configurable).

### Collapse detection (`compute_collapse_metrics`)

Called after each validation. Subsamples up to 512 embeddings for cosine similarity, up to 1000 for SVD. Returns `avg_cosine_similarity`, `dead_dimensions` (var < 1e-4), `effective_rank` (singular values > 1% of s₀), `stable_rank` (sum(sᵢ²)/s₀²).

### Plots (`src/plotting.py`)

All plots are generated **only at the end of training** (not live). Functions take `log_data` dict (same structure as `training_logs.json`) and a save path. Each function is standalone and can be called with partial history for mid-run plots.

### Experiment output layout

```
experiments/<name>/
├── fusion_encoder_best.pth          # best checkpoint (by val Recall@5)
├── fusion_encoder_epoch_N.pth       # rolling periodic checkpoint (kept latest only)
├── training_artifacts.pkl           # vocab, scaler, label mapping, config (used by inference)
├── logs/
│   ├── training.log                 # loguru timestamped log
│   └── training_logs.json           # all numeric history
└── plots/                           # all PNGs generated at end of run
```

`training_logs.json` keys: `epoch_losses`, `step_losses`, `grad_norms`, `val_losses`, `val_recall5`, `val_accuracies`, `val_map`, `val_ndcg10`, `collapse_history`, `per_class_recall_history`, `best_val_recall5`, `best_epoch`, `confusion_results`, `config`.

### Key config fields

| Field | Effect |
|---|---|
| `loss_type` | `"triplet"` or `"supcon"` |
| `freeze_strategy` | `"freeze"`, `"gradual"`, `"full"` |
| `mining_strategy` | `"random"` or `"semi_hard"` (triplet only) |
| `pooling_strategy` | `"mean"`, `"cls"`, `"pooler"` |
| `scheduler_type` | `"cosine"` (warmup + cosine) or `"step"` (StepLR γ=0.9) |
| `use_pk_sampler` | PKSampler: P classes × K samples per batch |
| `fusion_depth` | 1 = single MLP, ≥2 = residual blocks |
| `text_col` | column name or list; supports `"cleaned_merchant"`, `["merchant","tran_partclr"]`, etc. |
| `val_csv_path` | if set, overrides `val_split`; experiment dir uses `_ext_` tag |
| `text_cleaning` | applies `clean_narration()` (strips dates, long alphanums) |
| `filter_null_label` | drops rows with missing or `"NULL"` labels |
| `confusion_pairs_path` | if set, runs confusion suite at end of training |

### Inference (`src/inference_pipeline.py`)

`GoldenRecordIndexer` encodes a labeled CSV and builds a FAISS index (HNSW/IVF/L2/IP).  
`TransactionInferencePipeline` loads model + index, runs batch encoding, retrieves top-K neighbours, and applies majority voting (confidence threshold 0.7).

### PKSampler (`src/triplet_sampler.py`)

`PKSampler` yields batches of exactly P×K items (P classes, K samples each). Used with `use_pk_sampler: true`. Requires all P classes to have ≥2 samples; reduce `pk_p` if the dataset has fewer classes.

---

## Known gaps / active work

- **wandb not wired in** — integration code is documented in `COMMANDS.md` but not yet implemented.

## Observability features

- **Step-based validation** fires every `val_every_n_steps` optimizer steps (default = `n_batches // 4`, ≈4 per epoch) plus mandatory epoch-end. Override via `val_every_n_steps` in config.
- **Live plots** — all PNGs under `plots/` are overwritten after every validation (step or epoch). Open in any auto-refreshing viewer.
- **Validation report in log file** — `format_validation_report()` in `validation.py` returns the report as a string; the training loop logs it via loguru so it appears in `logs/training.log`.
- **TSNE legends** — `plot_embedding_projection()` accepts `label_mapping` and renders a per-class legend (≤30 classes) or colorbar (>30 classes). TSNE files are tagged `tsne_step{N}.png` (mid-epoch) or `tsne_epoch{N}.png` (epoch-end).
