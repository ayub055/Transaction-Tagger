import os
import math
import pickle
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split  # used only when val_csv_path is absent
from torch.cuda.amp import autocast, GradScaler
import json
import yaml
import argparse
from loguru import logger

# Remove default stderr sink; we add per-experiment file sinks in run_experiment()
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}", level="INFO")

from src.data_loader import TransactionDataset
from src.fusion_encoder import FusionEncoder
from src.plotting import plot_loss_curves, plot_grad_norms, plot_embedding_projection, plot_validation_curves, plot_collapse_metrics, plot_per_class_recall, plot_per_class_recall_history, plot_retrieval_metrics
from src.validation import EarlyStopping, print_validation_report, format_validation_report, evaluate_validation_metrics
from src.confusion_suite import run_confusion_suite


def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    categorical = torch.stack([item['categorical'] for item in batch])
    numeric = torch.stack([item['numeric'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    metadata = [item['metadata'] for item in batch]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'categorical': categorical,
        'numeric': numeric,
        'labels': labels,
        'metadata': metadata
    }


# ---------------------- Freeze Strategy ----------------------
def apply_freeze_strategy(encoder, strategy, epoch=None):
    if strategy == "freeze":
        for param in encoder.bert.parameters(): param.requires_grad = False
    elif strategy == "gradual":
        total_layers = len(encoder.bert.encoder.layer)
        layers_to_unfreeze = min(epoch, total_layers)
        # Unfreeze top-down: start from the highest (most task-specific) layers
        logger.info(f'Total Layers : {total_layers} | # Unfrozen Layers (top-down): {layers_to_unfreeze}')
        for param in encoder.bert.parameters(): param.requires_grad = False
        for i in range(total_layers - layers_to_unfreeze, total_layers):
            for param in encoder.bert.encoder.layer[i].parameters(): param.requires_grad = True
    elif strategy == "full":
        for param in encoder.bert.parameters(): param.requires_grad = True


def sample_triplets(embeddings, labels, margin=0.5, max_triplets=64,
                    mining_strategy='random'):
    # fp32 cast — cdist under AMP (fp16) can produce NaNs / wrong semi-hard picks
    dist_matrix = torch.cdist(embeddings.float(), embeddings.float(), p=2)
    triplets = []
    batch_size = len(labels)

    for anchor_idx in range(batch_size):
        anchor_label = labels[anchor_idx]
        pos_candidates = (labels == anchor_label).nonzero(as_tuple=True)[0]
        pos_candidates = pos_candidates[pos_candidates != anchor_idx]
        if len(pos_candidates) == 0: continue

        pos_idx = pos_candidates[torch.randint(0, len(pos_candidates), (1,)).item()]
        neg_candidates = (labels != anchor_label).nonzero(as_tuple=True)[0]
        if len(neg_candidates) == 0: continue

        if mining_strategy == 'semi_hard':
            d_ap = dist_matrix[anchor_idx, pos_idx]
            neg_dists = dist_matrix[anchor_idx, neg_candidates]
            semi_hard_mask = (neg_dists > d_ap) & (neg_dists < d_ap + margin)
            if semi_hard_mask.any():
                pool = neg_candidates[semi_hard_mask]
            else:
                pool = neg_candidates  # fallback to random
            neg_idx = pool[torch.randint(0, len(pool), (1,)).item()]
        else:
            neg_idx = neg_candidates[torch.randint(0, len(neg_candidates), (1,)).item()]

        triplets.append((anchor_idx, pos_idx.item(), neg_idx.item()))
        if len(triplets) >= max_triplets: break

    return triplets


# ---------------------- SupCon Loss ----------------------
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        device = embeddings.device
        batch_size = embeddings.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Cosine similarity matrix
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Mask out self-contrast
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        # For numerical stability
        logits_max, _ = similarity.max(dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        # Log-sum-exp over negatives
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Mean of log-likelihood over positives
        positives_per_row = mask.sum(dim=1)
        # Skip anchors with no positives in the batch
        valid = positives_per_row > 0
        if not valid.any():
            # keep in graph so AMP GradScaler can unscale without error
            return embeddings.sum() * 0.0

        mean_log_prob_pos = (mask * log_prob).sum(dim=1)[valid] / positives_per_row[valid]
        loss = -mean_log_prob_pos.mean()
        return loss


# ---------------------- Collapse Detection ----------------------
def compute_collapse_metrics(embeddings):
    with torch.no_grad():
        # Average pairwise cosine similarity (on a subsample for speed)
        n = min(512, embeddings.shape[0])
        sample = embeddings[:n]
        normed = nn.functional.normalize(sample, p=2, dim=1)
        cos_sim = torch.matmul(normed, normed.T)
        # Exclude diagonal
        mask = ~torch.eye(n, dtype=torch.bool, device=cos_sim.device)
        avg_cos_sim = cos_sim[mask].mean().item()

        # Per-dimension variance
        dim_variance = torch.var(embeddings, dim=0)
        dead_dims = (dim_variance < 1e-4).sum().item()
        total_dims = embeddings.shape[1]

        # Effective rank via SVD + stable rank (scale-invariant, robust to L2-norm)
        sample_for_svd = embeddings[:min(1000, embeddings.shape[0])].float()
        _, s, _ = torch.linalg.svd(sample_for_svd, full_matrices=False)
        threshold = 0.01 * s[0]
        effective_rank = (s > threshold).sum().item()
        stable_rank = (s.pow(2).sum() / s[0].pow(2).clamp(min=1e-12)).item()

    return {
        'avg_cosine_similarity': avg_cos_sim,
        'dead_dimensions': dead_dims,
        'total_dimensions': total_dims,
        'effective_rank': effective_rank,
        'stable_rank': stable_rank,
    }


# ---------------------- TSNE Snapshot Helper ----------------------
def _save_tsne_snapshot(val_metrics, exp_dir, tag, max_points=5000, label_mapping=None):
    """Stratified-sample embeddings and save a TSNE png with class-name legends."""
    if 'all_embeddings' not in val_metrics or 'eval_labels' not in val_metrics:
        return
    emb = val_metrics['all_embeddings']
    labels = val_metrics['eval_labels']
    n = min(max_points, len(labels))
    idx = torch.randperm(len(labels))[:n]
    emb_sample = emb[idx].numpy() if hasattr(emb, 'numpy') else np.asarray(emb)[idx.numpy()]
    lab_sample = labels[idx].numpy()
    out = os.path.join(exp_dir, 'plots', f'tsne_{tag}.png')
    plot_embedding_projection(emb_sample, lab_sample, out, label_mapping=label_mapping)


# ---------------------- Validation Helper ----------------------
def _do_validation(
    encoder, val_dataloader, config, device,
    epoch_losses, step_losses, grad_norms,
    val_losses, val_recall5, val_accuracies,
    val_map_history, val_ndcg_history,
    per_class_recall_history, collapse_history,
    best_val_recall5_ref,   # [float] wrapper so value is mutable across calls
    label_mapping, exp_dir, tag,
    val_train_losses=None,  # avg train loss per validation window — aligns with val_losses
    # dataset state — bundled into best checkpoint so recovery needs no CSV re-read
    cat_vocab=None, scaler=None, categorical_dims=None, config_snapshot=None,
):
    """
    Run one validation pass, update all history lists in-place, regenerate all
    plots, save TSNE snapshot, and save best checkpoint if recall@5 improved.
    Returns val_metrics dict.
    """
    encoder.eval()
    logger.info(f"Evaluating on validation set (tag={tag})...")

    val_metrics = evaluate_validation_metrics(
        encoder, val_dataloader,
        nn.TripletMarginLoss(margin=config["margin"]),
        device, k_values=[1, 5, 10],
        margin=config["margin"], max_triplets=64)

    val_losses.append(val_metrics['val_loss'])
    val_recall5.append(val_metrics['recall@5'])
    val_accuracies.append(val_metrics['accuracy'])
    val_map_history.append(val_metrics.get('map', 0.0))
    val_ndcg_history.append(val_metrics.get('ndcg@10', 0.0))
    per_class_recall_history.append(
        {str(k): float(v) for k, v in val_metrics.get('per_class_recall1', {}).items()})

    # Collapse detection
    if 'all_embeddings' in val_metrics:
        collapse = compute_collapse_metrics(val_metrics['all_embeddings'])
    else:
        sample_batch = next(iter(val_dataloader))
        with torch.no_grad():
            sample_emb = encoder(
                sample_batch['input_ids'].to(device),
                sample_batch['attention_mask'].to(device),
                sample_batch['categorical'].to(device),
                sample_batch['numeric'].to(device))
        collapse = compute_collapse_metrics(sample_emb)
    collapse_history.append(collapse)

    logger.info(
        f"  Collapse check: avg_cos_sim={collapse['avg_cosine_similarity']:.3f}, "
        f"dead_dims={collapse['dead_dimensions']}/{collapse['total_dimensions']}, "
        f"effective_rank={collapse['effective_rank']}")

    # Log validation report to stdout AND log file
    report = format_validation_report(val_metrics, tag)
    print(report)
    logger.info(report)

    # Save best checkpoint
    if val_metrics['recall@5'] > best_val_recall5_ref[0]:
        best_val_recall5_ref[0] = val_metrics['recall@5']
        best_model_path = os.path.join(exp_dir, "fusion_encoder_best.pth")
        logger.info(f'New best model! Saving to {best_model_path}')
        slim_val_metrics = {k: v for k, v in val_metrics.items()
                            if k not in ('all_embeddings', 'eval_labels')}
        torch.save({
            'tag': tag,
            'model_state_dict': encoder.state_dict(),
            'val_recall5': best_val_recall5_ref[0],
            'val_metrics': slim_val_metrics,
            'cat_vocab': cat_vocab,
            'scaler': scaler,
            'label_mapping': label_mapping,
            'categorical_dims': categorical_dims,
            'config': config_snapshot,
        }, best_model_path)

    # TSNE snapshot
    try:
        _save_tsne_snapshot(val_metrics, exp_dir, tag=tag, label_mapping=label_mapping)
    except Exception as e:
        logger.warning(f"TSNE snapshot skipped ({tag}): {e}")

    # Regenerate all plots with current data (live update)
    log_data = {
        "epoch_losses": epoch_losses,
        "step_losses": step_losses,
        "grad_norms": grad_norms,
        "val_losses": val_losses,
        "val_train_losses": val_train_losses or [],
        "val_recall5": val_recall5,
        "val_accuracies": val_accuracies,
        "val_map": val_map_history,
        "val_ndcg10": val_ndcg_history,
        "collapse_history": collapse_history,
        "per_class_recall_history": per_class_recall_history,
    }
    plots_dir = os.path.join(exp_dir, "plots")
    try:
        plot_loss_curves(log_data, os.path.join(plots_dir, "loss_curve.png"))
        plot_grad_norms(log_data, os.path.join(plots_dir, "grad_norms.png"))
        plot_validation_curves(log_data, os.path.join(plots_dir, "validation_metrics.png"))
        plot_collapse_metrics(log_data, os.path.join(plots_dir, "collapse_metrics.png"))
        plot_retrieval_metrics(log_data, os.path.join(plots_dir, "retrieval_metrics.png"))
        if val_metrics.get('per_class_recall1'):
            plot_per_class_recall(
                val_metrics['per_class_recall1'],
                os.path.join(plots_dir, "per_class_recall1.png"),
                label_mapping=label_mapping)
            plot_per_class_recall_history(
                per_class_recall_history,
                os.path.join(plots_dir, "per_class_recall_history.png"),
                label_mapping=label_mapping)
    except Exception as e:
        logger.warning(f"Live plot update failed ({tag}): {e}")

    return val_metrics


# ---------------------- Training  ----------------------
def run_experiment(config):
    # Gradient accumulation setup
    accumulation_steps = config.get("accumulation_steps", 1)
    base_batch_size = config.get("base_batch_size", 256)

    # Config flags with backward-compatible defaults
    loss_type = config.get("loss_type", "triplet")
    mining_strategy = config.get("mining_strategy", "random")
    pooling_strategy = config.get("pooling_strategy", "mean")
    scheduler_type = config.get("scheduler_type", "step")
    warmup_ratio = config.get("warmup_ratio", 0.05)
    fusion_depth = config.get("fusion_depth", 1)
    use_projection_head = (loss_type == "supcon")
    use_pk_sampler = config.get("use_pk_sampler", False)
    pk_p = config.get("pk_p", 32)
    pk_k = config.get("pk_k", 8)

    # Use the actual per-step batch size (PKSampler overrides config["batch_size"])
    actual_batch = (pk_p * pk_k) if use_pk_sampler else config["batch_size"]
    effective_batch_size = actual_batch * accumulation_steps

    # Learning rate scaling
    if "base_lr" in config:
        batch_ratio = effective_batch_size / base_batch_size
        scaled_lr = config["base_lr"] * (batch_ratio ** 0.5)
        config["lr"] = scaled_lr
        logger.info(f"LR Scaling: base_lr={config['base_lr']:.2e}, batch_ratio={batch_ratio:.2f}, scaled_lr={scaled_lr:.2e}")

    _val_tag = "ext" if config.get("val_csv_path") else f"val{config.get('val_split', 0.15)}"
    _text_col_raw = config.get("text_col", "tran_partclr")
    _text_tag = "+".join(_text_col_raw) if isinstance(_text_col_raw, list) else _text_col_raw
    exp_name = f"tagger_proj{config['text_proj_dim']}_final{config['final_dim']}_fd{fusion_depth}_{config['freeze_strategy']}_bs{effective_batch_size}_lr{config['lr']:.2e}_{_val_tag}_{_text_tag}_{loss_type}"
    exp_dir = os.path.join("experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)

    _log_sink_id = logger.add(
        os.path.join(exp_dir, "logs", "training.log"),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
        rotation="50 MB",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running experiment: {exp_name} on {device}")
    logger.info(f"Batch config: per_device={config['batch_size']}, accumulation={accumulation_steps}, effective={effective_batch_size}")
    logger.info(f"Loss: {loss_type} | Mining: {mining_strategy} | Pooling: {pooling_strategy} | Scheduler: {scheduler_type}")

    # Load data — two modes:
    #   Preferred: separate csv_path (train) + val_csv_path (test/val)
    #   Fallback:  single csv_path split internally by val_split
    val_csv_path = config.get("val_csv_path")

    train_df = pd.read_csv(config["csv_path"])

    # Drop NULL/unclassified labels before any subsampling so they never enter training
    if config.get("filter_null_label", False):
        _before = len(train_df)
        _null_mask = (
            train_df[config["label_col"]].isna() |
            train_df[config["label_col"]].astype(str).str.strip().str.upper().eq("NULL")
        )
        train_df = train_df[~_null_mask].reset_index(drop=True)
        logger.info(f"NULL filter (train): dropped {_before - len(train_df):,} rows → {len(train_df):,} remaining")

    # Optional stratified subsample on train (used for smoke tests and fast iteration)
    sample_size = config.get("sample_size", None)
    if sample_size and len(train_df) > sample_size:
        train_df = (
            train_df.groupby(config["label_col"], group_keys=False)
            .apply(lambda x: x.sample(
                n=max(1, round(sample_size * len(x) / len(train_df))),
                random_state=42
            ))
            .reset_index(drop=True)
        )
        logger.info(f"Stratified sample: using {len(train_df)} rows (target={sample_size})")

    if val_csv_path:
        val_df = pd.read_csv(val_csv_path)
        if config.get("filter_null_label", False):
            _bv = len(val_df)
            _null_mask_v = (
                val_df[config["label_col"]].isna() |
                val_df[config["label_col"]].astype(str).str.strip().str.upper().eq("NULL")
            )
            val_df = val_df[~_null_mask_v].reset_index(drop=True)
            logger.info(f"NULL filter (val): dropped {_bv - len(val_df):,} rows → {len(val_df):,} remaining")
        val_df = val_df.sample(n=min(50000, len(val_df)), random_state=42).reset_index(drop=True)
        logger.info(f"Separate val file: {val_csv_path}")
    else:
        val_split = config.get("val_split", 0.15)
        train_df, val_df = train_test_split(
            train_df, test_size=val_split, random_state=42,
            stratify=train_df[config["label_col"]])
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.sample(n=min(50000, len(val_df)), random_state=42).reset_index(drop=True)

    logger.info(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")

    tokenizer = BertTokenizer.from_pretrained(config["bert_model"])
    text_cleaning = config.get("text_cleaning", False)
    text_col = config.get("text_col", "tran_partclr")
    logger.info(f"Text column(s): {text_col}")
    train_dataset = TransactionDataset(train_df, tokenizer, config["categorical_cols"], config["numeric_cols"], config["label_col"], text_cleaning=text_cleaning, text_col=text_col)
    val_dataset = TransactionDataset(val_df, tokenizer, config["categorical_cols"], config["numeric_cols"], config["label_col"], text_cleaning=text_cleaning, text_col=text_col)

    val_dataset.cat_vocab = train_dataset.cat_vocab
    val_dataset.scaler = train_dataset.scaler
    val_dataset.numeric_data = val_dataset.scaler.transform(val_df[config["numeric_cols"]])
    val_dataset.label_mapping = train_dataset.label_mapping

    # Re-encode val labels in train's label space — val_df.cat.codes can silently
    # drift if any class is missing from one split. Unknown labels -> -1; drop those.
    inv_label_mapping = {v: k for k, v in train_dataset.label_mapping.items()}
    val_codes = val_df[config["label_col"]].map(inv_label_mapping)
    known_mask = val_codes.notna().to_numpy()
    n_dropped = int((~known_mask).sum())
    if n_dropped > 0:
        logger.warning(f"Dropping {n_dropped} val rows with labels absent from train.")
        val_df_aligned = val_df.loc[known_mask].reset_index(drop=True)
        val_dataset.df = val_df_aligned
        val_dataset.numeric_data = val_dataset.scaler.transform(val_df_aligned[config["numeric_cols"]])
        val_codes = val_codes.loc[known_mask].reset_index(drop=True)
    val_dataset.labels = pd.Series(val_codes.astype(int).values)

    num_workers = config.get("num_workers", 4)

    if use_pk_sampler:
        from src.triplet_sampler import PKSampler
        pk_sampler = PKSampler(train_dataset.labels.tolist(), p=pk_p, k=pk_k)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_sampler=pk_sampler,
            collate_fn=collate_fn, num_workers=num_workers,
            pin_memory=True, persistent_workers=True)
        logger.info(f"PKSampler: P={pk_p} classes, K={pk_k} samples/class, batch_size={pk_p*pk_k}")
    else:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    categorical_dims = [len(train_dataset.cat_vocab[col]) for col in config["categorical_cols"]]

    # ENCODER
    encoder = FusionEncoder(
        bert_model_name=config["bert_model"],
        categorical_dims=categorical_dims,
        numeric_dim=len(config["numeric_cols"]),
        text_proj_dim=config["text_proj_dim"],
        final_dim=config["final_dim"],
        p=config.get("dropout", 0.1),
        pooling_strategy=pooling_strategy,
        use_projection_head=use_projection_head,
        fusion_depth=fusion_depth).to(device)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, encoder.parameters()), lr=config["lr"])

    # Scheduler
    total_steps = len(train_dataloader) * config["epochs"] // accumulation_steps
    warmup_steps = int(total_steps * warmup_ratio)

    if scheduler_type == "cosine":
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_steps)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_steps - warmup_steps, 1))
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, [warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps])
        step_scheduler_per_batch = True
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        step_scheduler_per_batch = False

    # Loss
    if loss_type == "supcon":
        temperature = config.get("supcon_temperature", 0.07)
        loss_fn = SupConLoss(temperature=temperature)
        logger.info(f"SupCon Loss with temperature={temperature}")
    else:
        triplet_loss_fn = nn.TripletMarginLoss(margin=config["margin"])

    # Mixed precision training
    use_amp = config.get("use_amp", False) and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    if use_amp: logger.info("Mixed precision training enabled (AMP)")

    # Early stopping setup - Using recall@5 as primary metric
    early_stopping = EarlyStopping(patience=config.get("patience", 5),min_delta=config.get("min_delta", 0.001), mode='max',  verbose=True)

    epoch_losses, step_losses, grad_norms = [], [], []
    val_losses, val_recall5, val_accuracies = [], [], []
    val_map_history, val_ndcg_history = [], []
    val_train_losses = []          # avg train loss in the window leading up to each validation
    _train_loss_buf = []           # step losses since the last validation — flushed at each val
    per_class_recall_history = []
    collapse_history = []
    best_val_recall5_ref = [0.0]   # mutable wrapper so _do_validation can update it
    global_step = 0
    val_num = 0                    # global validation counter (used by early stopping)

    config_snapshot = {
        'categorical_cols': config['categorical_cols'],
        'numeric_cols': config['numeric_cols'],
        'label_col': config['label_col'],
        'bert_model': config['bert_model'],
        'text_proj_dim': config['text_proj_dim'],
        'final_dim': config['final_dim'],
        'dropout': config.get('dropout', 0.1),
        'num_categories': len(train_dataset.label_mapping),
        'text_cleaning': text_cleaning,
        'text_col': text_col,
        'pooling_strategy': pooling_strategy,
        'fusion_depth': fusion_depth,
    }

    def _save_snapshot(tag: str):
        """Persist artifacts + logs to disk. Safe to call mid-run or on interrupt."""
        _artifacts = {
            'cat_vocab': train_dataset.cat_vocab,
            'scaler': train_dataset.scaler,
            'label_mapping': train_dataset.label_mapping,
            'categorical_dims': categorical_dims,
            'config': config_snapshot,
        }
        with open(os.path.join(exp_dir, "training_artifacts.pkl"), 'wb') as f:
            pickle.dump(_artifacts, f)

        _log_data = {
            "epoch_losses": epoch_losses,
            "step_losses": step_losses,
            "grad_norms": grad_norms,
            "val_losses": val_losses,
            "val_train_losses": val_train_losses,
            "val_recall5": val_recall5,
            "val_accuracies": val_accuracies,
            "val_map": val_map_history,
            "val_ndcg10": val_ndcg_history,
            "best_val_recall5": best_val_recall5_ref[0],
            "best_epoch": early_stopping.best_epoch,
            "collapse_history": collapse_history,
            "per_class_recall_history": per_class_recall_history,
            "per_class_recall_last_epoch": per_class_recall_history[-1] if per_class_recall_history else {},
            "confusion_results": None,
            "config": {k: v for k, v in config.items() if isinstance(v, (str, int, float, bool, list))},
            "snapshot_tag": tag,
        }
        with open(os.path.join(exp_dir, "logs", "training_logs.json"), "w") as f:
            json.dump(_log_data, f, indent=2)

        logger.info(f"Snapshot saved ({tag}): artifacts + logs written to {exp_dir}")

    # Step-based validation: default = ~4 mid-epoch validations.
    # global_step counts optimizer steps (increments every accumulation_steps batches),
    # so the default must be in optimizer-step units, not batch units.
    n_batches = len(train_dataloader)
    optimizer_steps_per_epoch = max(1, n_batches // accumulation_steps)
    # 3 validations/epoch = 2 mid-epoch step fires + 1 mandatory epoch-end.
    # Interval = steps/3 so fires at ~33% and ~66%; epoch-end covers the final third.
    val_every_n_steps = config.get("val_every_n_steps") or max(1, optimizer_steps_per_epoch // 3)
    logger.info(f"Step-based validation every {val_every_n_steps} optimizer steps "
                f"(2 mid-epoch + 1 epoch-end = 3 validations/epoch, "
                f"{optimizer_steps_per_epoch} opt-steps/epoch). "
                f"Epoch-end validation always fires.")

    try:
     for epoch in tqdm(range(config["epochs"]), desc="Training Epochs"):
        encoder.train()
        apply_freeze_strategy(encoder, config["freeze_strategy"], epoch)
        total_loss, batch_count = 0.0, 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_dataloader, start=1):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            categorical = batch['categorical'].to(device)
            numeric = batch['numeric'].to(device)
            labels = batch['labels'].to(device)

            # Mixed precision forward pass
            with autocast(enabled=use_amp):
                embeddings = encoder(input_ids, attention_mask, categorical, numeric)

                if loss_type == "supcon":
                    loss = loss_fn(embeddings, labels)
                else:
                    triplets = sample_triplets(embeddings, labels, config["margin"],
                                               max_triplets=64,
                                               mining_strategy=mining_strategy)
                    if not triplets: continue

                    anchor_emb = torch.stack([embeddings[a] for a, _, _ in triplets])
                    pos_emb = torch.stack([embeddings[p] for _, p, _ in triplets])
                    neg_emb = torch.stack([embeddings[n] for _, _, n in triplets])

                    loss = triplet_loss_fn(anchor_emb, pos_emb, neg_emb)

                loss = loss / accumulation_steps

            # Backward pass with gradient scaling
            if use_amp: scaler.scale(loss).backward()
            else:loss.backward()

            # Accumulate gradients and step optimizer every accumulation_steps
            if batch_idx % accumulation_steps == 0:
                if use_amp: scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)

                # Track global L2 grad norm: sqrt(sum p.grad.norm^2)
                with torch.no_grad():
                    grad_sq = 0.0
                    for p in encoder.parameters():
                        if p.grad is not None:
                            gn = p.grad.detach().norm(2).item()
                            grad_sq += gn * gn
                    grad_norm = grad_sq ** 0.5
                grad_norms.append(grad_norm)

                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
                global_step += 1

                if step_scheduler_per_batch:
                    scheduler.step()

                # Mid-epoch step-based validation
                if global_step % val_every_n_steps == 0:
                    val_num += 1
                    val_train_losses.append(
                        float(np.mean(_train_loss_buf)) if _train_loss_buf else float('nan'))
                    _train_loss_buf.clear()
                    val_metrics = _do_validation(
                        encoder, val_dataloader, config, device,
                        epoch_losses, step_losses, grad_norms,
                        val_losses, val_recall5, val_accuracies,
                        val_map_history, val_ndcg_history,
                        per_class_recall_history, collapse_history,
                        best_val_recall5_ref,
                        train_dataset.label_mapping, exp_dir,
                        tag=f"step{global_step}",
                        val_train_losses=val_train_losses,
                        cat_vocab=train_dataset.cat_vocab,
                        scaler=train_dataset.scaler,
                        categorical_dims=categorical_dims,
                        config_snapshot=config_snapshot,
                    )
                    if early_stopping(val_metrics['recall@5'], val_num):
                        logger.info(f"Early stopping triggered at step {global_step} (val #{val_num})")
                        break
                    encoder.train()
                    apply_freeze_strategy(encoder, config["freeze_strategy"], epoch)

            _loss_val = loss.item() * accumulation_steps
            step_losses.append(_loss_val)
            _train_loss_buf.append(_loss_val)
            total_loss += _loss_val
            batch_count += 1

            if batch_idx % (8 * accumulation_steps) == 0:
                logger.info(f"Epoch [{epoch+1}/{config['epochs']}], Batch [{batch_idx}], Loss: {loss.item() * accumulation_steps:.4f}")

        if not step_scheduler_per_batch:
            scheduler.step()

        avg_loss = total_loss / max(batch_count, 1)
        epoch_losses.append(avg_loss)
        logger.info(f"Epoch [{epoch+1}/{config['epochs']}] Train Loss: {avg_loss:.4f}")

        # If mid-epoch early stopping fired, propagate the break
        if early_stopping.early_stop:
            break

        # EPOCH-END VALIDATION (always runs)
        val_num += 1
        val_train_losses.append(
            float(np.mean(_train_loss_buf)) if _train_loss_buf else float('nan'))
        _train_loss_buf.clear()
        val_metrics = _do_validation(
            encoder, val_dataloader, config, device,
            epoch_losses, step_losses, grad_norms,
            val_losses, val_recall5, val_accuracies,
            val_map_history, val_ndcg_history,
            per_class_recall_history, collapse_history,
            best_val_recall5_ref,
            train_dataset.label_mapping, exp_dir,
            tag=f"epoch{epoch + 1}",
            val_train_losses=val_train_losses,
            cat_vocab=train_dataset.cat_vocab,
            scaler=train_dataset.scaler,
            categorical_dims=categorical_dims,
            config_snapshot=config_snapshot,
        )

        # Periodic checkpoint every 5 epochs — keep only the latest one
        if epoch % 5 == 0:
            new_ckpt = f"{exp_dir}/fusion_encoder_epoch_{epoch+1}.pth"
            torch.save({
                'model_state_dict': encoder.state_dict(),
                'epoch': epoch + 1,
                'cat_vocab': train_dataset.cat_vocab,
                'scaler': train_dataset.scaler,
                'label_mapping': train_dataset.label_mapping,
                'categorical_dims': categorical_dims,
                'config': config_snapshot,
            }, new_ckpt)
            logger.info(f'Periodic checkpoint saved: {new_ckpt}')
            prev_periodic_epoch = epoch + 1 - 5
            if prev_periodic_epoch > 0:
                prev_ckpt = f"{exp_dir}/fusion_encoder_epoch_{prev_periodic_epoch}.pth"
                if os.path.exists(prev_ckpt):
                    os.remove(prev_ckpt)
                    logger.info(f'Deleted old checkpoint: {prev_ckpt}')

        if early_stopping(val_metrics['recall@5'], val_num):
            logger.info(f"Early stopping triggered at epoch {epoch + 1} (val #{val_num})")
            break

    except KeyboardInterrupt:
        logger.warning("Training interrupted — saving current state before exit...")
        interrupted_ckpt = os.path.join(exp_dir, "fusion_encoder_interrupted.pth")
        torch.save({
            'model_state_dict': encoder.state_dict(),
            'epoch': epoch + 1,
            'cat_vocab': train_dataset.cat_vocab,
            'scaler': train_dataset.scaler,
            'label_mapping': train_dataset.label_mapping,
            'categorical_dims': categorical_dims,
            'config': config_snapshot,
        }, interrupted_ckpt)
        _save_snapshot(tag=f"interrupted_epoch{epoch + 1}")
        logger.info(f"Interrupted checkpoint -> {interrupted_ckpt}")
        logger.info("Resume from best checkpoint or interrupted checkpoint.")
        raise  # re-raise so the outer loop can stop cleanly

    # Optional C3 confusion suite
    confusion_results = None
    confusion_pairs_path = config.get("confusion_pairs_path")
    if confusion_pairs_path and os.path.exists(confusion_pairs_path):
        logger.info(f"Running confusion suite from {confusion_pairs_path}")
        confusion_results = run_confusion_suite(
            encoder, tokenizer,
            {'config': {
                'categorical_cols': config['categorical_cols'],
                'numeric_cols': config['numeric_cols'],
                'text_cleaning': text_cleaning,
                'text_col': text_col,
            },
             'cat_vocab': train_dataset.cat_vocab,
             'scaler': train_dataset.scaler},
            confusion_pairs_path, device)
        logger.info(f"Confusion pass rate: {confusion_results['pass_rate']:.3f} ({confusion_results['n']} pairs)")

    # Save artifacts + logs, then patch in confusion_results (computed after the loop)
    _save_snapshot(tag="completed")
    if confusion_results is not None:
        _final_log_path = os.path.join(exp_dir, "logs", "training_logs.json")
        with open(_final_log_path) as f:
            _final_log = json.load(f)
        _final_log["confusion_results"] = confusion_results
        with open(_final_log_path, "w") as f:
            json.dump(_final_log, f, indent=2)

    # All plots are already up-to-date — written after each validation by _do_validation()

    logger.info(f"\nExperiment {exp_name} completed!")
    logger.info(f"Best Recall@5: {best_val_recall5_ref[0]:.4f} at epoch {early_stopping.best_epoch}")
    logger.info(f"Logs and plots saved in {exp_dir}")
    logger.info(f"Artifacts saved to: {_artifacts_path}")
    logger.info(f"Next steps — build index and run inference:")
    logger.info(f"  python run_inference.py --exp {exp_name} --csv <golden.csv> --input-csv <test.csv> --output-csv results_{exp_name}.csv")

    logger.remove(_log_sink_id)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple FusionEncoder experiments")
    parser.add_argument("--config", type=str, help="Path to config file (JSON or YAML)")
    parser.add_argument("--single", type=str, help="Single experiment config as JSON string")
    parser.add_argument("--format", type=str, default="json", choices=["json", "yaml"], help="Config file format")
    args = parser.parse_args()

    if args.config:
        if args.format == "yaml":
            with open(args.config, "r") as f: configs = yaml.safe_load(f)
        else:
            with open(args.config, "r") as f: configs = json.load(f)
        for config in configs: run_experiment(config)
    elif args.single:
        config = json.loads(args.single)
        run_experiment(config)
    else:
        logger.error("Please provide either --config or --single")
