import os
import math
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler
import json
import yaml
import argparse

from src.data_loader import TransactionDataset
from src.fusion_encoder import FusionEncoder
from src.plotting import plot_loss_curves, plot_grad_norms, plot_embedding_projection, plot_validation_curves, plot_collapse_metrics, plot_per_class_recall
from src.validation import EarlyStopping, print_validation_report, evaluate_validation_metrics


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
        print(f'Total Layers : {total_layers} | # Unfrozen Layers (top-down): {layers_to_unfreeze}')
        for param in encoder.bert.parameters(): param.requires_grad = False
        for i in range(total_layers - layers_to_unfreeze, total_layers):
            for param in encoder.bert.encoder.layer[i].parameters(): param.requires_grad = True
    elif strategy == "full":
        for param in encoder.bert.parameters(): param.requires_grad = True


def sample_triplets(embeddings, labels, margin=0.5, max_triplets=64,
                    mining_strategy='random'):
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)
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
            return torch.tensor(0.0, device=device, requires_grad=True)

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

        # Effective rank via SVD
        sample_for_svd = embeddings[:min(1000, embeddings.shape[0])]
        _, s, _ = torch.linalg.svd(sample_for_svd, full_matrices=False)
        threshold = 0.01 * s[0]
        effective_rank = (s > threshold).sum().item()

    return {
        'avg_cosine_similarity': avg_cos_sim,
        'dead_dimensions': dead_dims,
        'total_dimensions': total_dims,
        'effective_rank': effective_rank
    }


# ---------------------- Training  ----------------------
def run_experiment(config):
    # Gradient accumulation setup
    accumulation_steps = config.get("accumulation_steps", 1)
    base_batch_size = config.get("base_batch_size", 256)
    effective_batch_size = config["batch_size"] * accumulation_steps

    # Learning rate scaling
    if "base_lr" in config:
        batch_ratio = effective_batch_size / base_batch_size
        scaled_lr = config["base_lr"] * (batch_ratio ** 0.5)
        config["lr"] = scaled_lr
        print(f"LR Scaling: base_lr={config['base_lr']:.2e}, batch_ratio={batch_ratio:.2f}, scaled_lr={scaled_lr:.2e}")

    # Config flags with backward-compatible defaults
    loss_type = config.get("loss_type", "triplet")
    mining_strategy = config.get("mining_strategy", "random")
    pooling_strategy = config.get("pooling_strategy", "mean")
    scheduler_type = config.get("scheduler_type", "step")
    warmup_ratio = config.get("warmup_ratio", 0.05)
    use_projection_head = (loss_type == "supcon")
    use_pk_sampler = config.get("use_pk_sampler", False)
    pk_p = config.get("pk_p", 32)
    pk_k = config.get("pk_k", 8)

    exp_name = f"tagger_proj{config['text_proj_dim']}_final{config['final_dim']}_{config['freeze_strategy']}_bs{effective_batch_size}_lr{config['lr']:.2e}_val{config['val_split']}_{loss_type}"
    exp_dir = os.path.join("experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running experiment: {exp_name} on {device}")
    print(f"Batch config: per_device={config['batch_size']}, accumulation={accumulation_steps}, effective={effective_batch_size}")
    print(f"Loss: {loss_type} | Mining: {mining_strategy} | Pooling: {pooling_strategy} | Scheduler: {scheduler_type}")

    # Load data
    df = pd.read_csv(config["csv_path"])

    # Optional stratified subsample (used for smoke tests and fast iteration)
    sample_size = config.get("sample_size", None)
    if sample_size and len(df) > sample_size:
        df = (
            df.groupby(config["label_col"], group_keys=False)
            .apply(lambda x: x.sample(
                n=max(1, round(sample_size * len(x) / len(df))),
                random_state=42
            ))
            .reset_index(drop=True)
        )
        print(f"Stratified sample: using {len(df)} rows (target={sample_size})")

    # Stratified split
    val_split = config.get("val_split", 0.15)
    train_df, val_df = train_test_split(df,test_size=val_split,random_state=42,stratify=df[config["label_col"]])
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.sample(n=min(50000, len(val_df)), random_state=42).reset_index(drop=True)
    print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")

    tokenizer = BertTokenizer.from_pretrained(config["bert_model"])
    text_cleaning = config.get("text_cleaning", False)
    train_dataset = TransactionDataset(train_df, tokenizer, config["categorical_cols"], config["numeric_cols"], config["label_col"], text_cleaning=text_cleaning)
    val_dataset = TransactionDataset(val_df, tokenizer, config["categorical_cols"], config["numeric_cols"], config["label_col"], text_cleaning=text_cleaning)

    val_dataset.cat_vocab = train_dataset.cat_vocab
    val_dataset.scaler = train_dataset.scaler
    val_dataset.numeric_data = val_dataset.scaler.transform(val_df[config["numeric_cols"]])
    val_dataset.label_mapping = train_dataset.label_mapping

    num_workers = config.get("num_workers", 4)

    if use_pk_sampler:
        from src.triplet_sampler import PKSampler
        pk_sampler = PKSampler(train_dataset.labels.tolist(), p=pk_p, k=pk_k)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_sampler=pk_sampler,
            collate_fn=collate_fn, num_workers=num_workers,
            pin_memory=True, persistent_workers=True)
        print(f"PKSampler: P={pk_p} classes, K={pk_k} samples/class, batch_size={pk_p*pk_k}")
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
        use_projection_head=use_projection_head).to(device)

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
        print(f"SupCon Loss with temperature={temperature}")
    else:
        triplet_loss_fn = nn.TripletMarginLoss(margin=config["margin"])

    # Mixed precision training
    use_amp = config.get("use_amp", False) and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    if use_amp: print("Mixed precision training enabled (AMP)")

    # Early stopping setup - Using recall@5 as primary metric
    early_stopping = EarlyStopping(patience=config.get("patience", 5),min_delta=config.get("min_delta", 0.001), mode='max',  verbose=True)

    epoch_losses, step_losses, grad_norms = [], [], []
    val_losses, val_recall5, val_accuracies = [], [], []
    collapse_history = []
    best_val_recall5 = 0.0
    global_step = 0

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

                # Track gradient norms
                grad_norm = sum(p.grad.norm().item() for p in encoder.parameters() if p.grad is not None)
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

            step_losses.append(loss.item() * accumulation_steps)
            total_loss += loss.item() * accumulation_steps
            batch_count += 1

            if batch_idx % (8 * accumulation_steps) == 0:
                print(f"Epoch [{epoch+1}/{config['epochs']}], Batch [{batch_idx}], Loss: {loss.item() * accumulation_steps:.4f}")

        if not step_scheduler_per_batch:
            scheduler.step()

        avg_loss = total_loss / max(batch_count, 1)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{config['epochs']}] Train Loss: {avg_loss:.4f}")

        # VALIDATION PHASE
        print("Evaluating on validation set...")
        val_metrics = evaluate_validation_metrics(
            encoder, val_dataloader,
            nn.TripletMarginLoss(margin=config["margin"]),
            device, k_values=[1, 5, 10],
            margin=config["margin"], max_triplets=64)

        val_losses.append(val_metrics['val_loss'])
        val_recall5.append(val_metrics['recall@5'])
        val_accuracies.append(val_metrics['accuracy'])

        # Collapse detection
        if 'all_embeddings' in val_metrics:
            collapse = compute_collapse_metrics(val_metrics['all_embeddings'])
        else:
            # Compute on a single batch from validation
            encoder.eval()
            with torch.no_grad():
                sample_batch = next(iter(val_dataloader))
                sample_emb = encoder(
                    sample_batch['input_ids'].to(device),
                    sample_batch['attention_mask'].to(device),
                    sample_batch['categorical'].to(device),
                    sample_batch['numeric'].to(device))
                collapse = compute_collapse_metrics(sample_emb)
        collapse_history.append(collapse)
        print(f"  Collapse check: avg_cos_sim={collapse['avg_cosine_similarity']:.3f}, "
              f"dead_dims={collapse['dead_dimensions']}/{collapse['total_dimensions']}, "
              f"effective_rank={collapse['effective_rank']}")

        print_validation_report(val_metrics, epoch + 1)

        # Save best model based on recall@5
        if val_metrics['recall@5'] > best_val_recall5:
            best_val_recall5 = val_metrics['recall@5']
            best_model_path = f"{exp_dir}/fusion_encoder_best.pth"
            print(f'New best model! Saving to {best_model_path}')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_recall5': best_val_recall5,
                'val_metrics': val_metrics
            }, best_model_path)

        if epoch % 5 == 0:
            model_path = f"{exp_dir}/fusion_encoder_epoch_{epoch+1}.pth"
            print(f'Saving checkpoint at {model_path}')
            torch.save(encoder.state_dict(), model_path)

        if early_stopping(val_metrics['recall@5'], epoch + 1):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Serialize per-class recall history (convert int keys to str for JSON)
    per_class_recall_history = []
    for m in [val_metrics]:  # currently only last epoch; extend if stored per-epoch
        if 'per_class_recall1' in m:
            per_class_recall_history.append(
                {str(k): v for k, v in m['per_class_recall1'].items()})

    log_data = {
        "epoch_losses": epoch_losses,
        "step_losses": step_losses,
        "grad_norms": grad_norms,
        "val_losses": val_losses,
        "val_recall5": val_recall5,
        "val_accuracies": val_accuracies,
        "best_val_recall5": best_val_recall5,
        "best_epoch": early_stopping.best_epoch,
        "collapse_history": collapse_history,
        "per_class_recall_last_epoch": per_class_recall_history[-1] if per_class_recall_history else {},
        "config": {k: v for k, v in config.items() if isinstance(v, (str, int, float, bool, list))}
    }

    with open(os.path.join(exp_dir, "logs", "training_logs.json"), "w") as f:
        json.dump(log_data, f, indent=2)

    plot_loss_curves(log_data, os.path.join(exp_dir, "plots", "loss_curve.png"))
    plot_grad_norms(log_data, os.path.join(exp_dir, "plots", "grad_norms.png"))
    plot_validation_curves(log_data, os.path.join(exp_dir, "plots", "validation_metrics.png"))
    plot_collapse_metrics(log_data, os.path.join(exp_dir, "plots", "collapse_metrics.png"))

    # Per-class recall plot from last epoch
    if val_metrics.get('per_class_recall1'):
        label_mapping = train_dataset.label_mapping
        plot_per_class_recall(
            val_metrics['per_class_recall1'],
            os.path.join(exp_dir, "plots", "per_class_recall1.png"),
            label_mapping=label_mapping)

    # ---- Save training artifacts alongside model for inference pipeline ----
    _artifacts = {
        'cat_vocab': train_dataset.cat_vocab,
        'scaler': train_dataset.scaler,
        'label_mapping': train_dataset.label_mapping,
        'categorical_dims': categorical_dims,
        'config': {
            'categorical_cols': config['categorical_cols'],
            'numeric_cols': config['numeric_cols'],
            'label_col': config['label_col'],
            'bert_model': config['bert_model'],
            'text_proj_dim': config['text_proj_dim'],
            'final_dim': config['final_dim'],
            'dropout': config.get('dropout', 0.1),
            'num_categories': len(train_dataset.label_mapping),
            'text_cleaning': text_cleaning,
            'pooling_strategy': pooling_strategy,
        }
    }
    _artifacts_path = os.path.join(exp_dir, "training_artifacts.pkl")
    with open(_artifacts_path, 'wb') as f:
        pickle.dump(_artifacts, f)

    print(f"\nExperiment {exp_name} completed!")
    print(f"Best Recall@5: {best_val_recall5:.4f} at epoch {early_stopping.best_epoch}")
    print(f"Logs and plots saved in {exp_dir}")
    print(f"Artifacts saved to: {_artifacts_path}")
    print(f"\nNext steps — build index and run inference:")
    print(f"  make build-index EXP={exp_name}")
    print(f"  make infer      EXP={exp_name} INPUT_CSV=<your_file.csv> OUTPUT_CSV=results.csv")



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
        print("Please provide either --config or --single")
