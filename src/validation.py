import torch
import numpy as np
from sklearn.metrics import accuracy_score
from collections import defaultdict
from tqdm import tqdm

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(self, patience=5, min_delta=0.0, mode='min', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, current_score, epoch):
        """
        Call this method after each epoch with validation metric.
        Returns True if training should stop.
        """
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            return False

        # Check if current score is better
        if self.mode == 'min': is_better = current_score < (self.best_score - self.min_delta)
        else: is_better = current_score > (self.best_score + self.min_delta)

        if is_better:
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose: print(f"Validation metric improved to {current_score:.4f}")
        else:
            self.counter += 1
            if self.verbose: print(f"No improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose: print(f"Early stopping triggered! Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
                return True
        return False


def _sample_triplets_random(embeddings, labels, margin, max_triplets):
    """Random triplet sampling used for validation loss computation."""
    # fp32 cast — cdist under AMP with fp16 tensors is numerically unstable
    dist_matrix = torch.cdist(embeddings.float(), embeddings.float(), p=2)
    triplets = []
    batch_size = len(labels)
    for anchor_idx in range(batch_size):
        anchor_label = labels[anchor_idx]
        pos_candidates = (labels == anchor_label).nonzero(as_tuple=True)[0]
        pos_candidates = pos_candidates[pos_candidates != anchor_idx]
        if len(pos_candidates) == 0: continue
        neg_candidates = (labels != anchor_label).nonzero(as_tuple=True)[0]
        if len(neg_candidates) == 0: continue
        pos_idx = pos_candidates[torch.randint(0, len(pos_candidates), (1,)).item()]
        neg_idx = neg_candidates[torch.randint(0, len(neg_candidates), (1,)).item()]
        triplets.append((anchor_idx, pos_idx.item(), neg_idx.item()))
        if len(triplets) >= max_triplets: break
    return triplets


def evaluate_validation_metrics(encoder, dataloader, triplet_loss_fn, device,
                                k_values=[1, 5, 10], margin=0.5, max_triplets=64):
    """
    Compute validation loss, retrieval metrics, per-class Recall@1, and
    return embeddings for collapse detection — all in a single forward pass.

    Retrieval evaluation is capped at MAX_RETRIEVAL_EVAL samples to avoid the
    O(N^2) distance matrix OOM that occurs at large validation set sizes.
    """
    MAX_RETRIEVAL_EVAL = 5000

    encoder.eval()
    all_embeddings = []
    all_labels = []
    total_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            categorical = batch['categorical'].to(device)
            numeric = batch['numeric'].to(device)
            labels = batch['labels'].to(device)

            embeddings = encoder(input_ids, attention_mask, categorical, numeric)

            # Move to CPU immediately — avoids building a 10 GB N×N matrix on GPU later
            all_embeddings.append(embeddings.detach().cpu())
            all_labels.append(labels.cpu())

            triplets = _sample_triplets_random(embeddings, labels, margin, max_triplets)
            if triplets:
                anchor_emb = torch.stack([embeddings[a] for a, _, _ in triplets])
                pos_emb = torch.stack([embeddings[p] for _, p, _ in triplets])
                neg_emb = torch.stack([embeddings[n] for _, _, n in triplets])

                loss = triplet_loss_fn(anchor_emb, pos_emb, neg_emb)
                total_loss += loss.item()
                batch_count += 1

    avg_loss = total_loss / max(batch_count, 1)
    all_embeddings = torch.cat(all_embeddings, dim=0)   # [N, embedding_dim], CPU
    all_labels = torch.cat(all_labels, dim=0)           # [N], CPU

    # Subsample for retrieval evaluation to keep dist_matrix in memory
    # (50K × 50K × 4 bytes = 10 GB; 5K × 5K = 100 MB)
    if len(all_embeddings) > MAX_RETRIEVAL_EVAL:
        perm = torch.randperm(len(all_embeddings))[:MAX_RETRIEVAL_EVAL]
        eval_embeddings = all_embeddings[perm]
        eval_labels = all_labels[perm]
    else:
        eval_embeddings = all_embeddings
        eval_labels = all_labels

    dist_matrix = torch.cdist(eval_embeddings.float(), eval_embeddings.float(), p=2)  # [M, M], CPU
    recall_at_k = {k: [] for k in k_values}
    reciprocal_ranks = []
    correct_predictions = []
    average_precisions = []
    ndcg_scores = []
    NDCG_K = 10
    num_samples = len(eval_labels)

    # Per-class Recall@1 tracking
    per_class_correct = defaultdict(list)

    # Precompute log2 discount for nDCG
    log2_discount = 1.0 / np.log2(np.arange(2, NDCG_K + 2))

    for i in tqdm(range(num_samples), desc='Retrieval stats'):
        query_label = eval_labels[i].item()

        distances = dist_matrix[i].clone()
        distances[i] = float('inf')  # Exclude self

        sorted_indices = torch.argsort(distances)
        sorted_labels = eval_labels[sorted_indices]

        for k in k_values:
            top_k_labels = sorted_labels[:k]
            has_correct = (top_k_labels == query_label).any().item()
            recall_at_k[k].append(int(has_correct))

        # MRR: Find rank of first correct match
        correct_mask = (sorted_labels == query_label)
        if correct_mask.any():
            first_correct_rank = (correct_mask.nonzero(as_tuple=True)[0][0].item() + 1)
            reciprocal_ranks.append(1.0 / first_correct_rank)
        else:
            reciprocal_ranks.append(0.0)

        # Average Precision — use all retrieved hits
        relevance = correct_mask.numpy().astype(np.float32)
        num_relevant = relevance.sum()
        if num_relevant > 0:
            cum_hits = np.cumsum(relevance)
            ranks = np.arange(1, len(relevance) + 1)
            precisions_at_hits = (cum_hits / ranks) * relevance
            average_precisions.append(precisions_at_hits.sum() / num_relevant)
        else:
            average_precisions.append(0.0)

        # nDCG@10 with binary relevance (same-label=1, else=0)
        rel_at_k = relevance[:NDCG_K]
        dcg = (rel_at_k * log2_discount[:len(rel_at_k)]).sum()
        ideal_hits = int(min(num_relevant, NDCG_K))
        idcg = log2_discount[:ideal_hits].sum() if ideal_hits > 0 else 0.0
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

        pred_label = sorted_labels[0].item()
        is_correct = int(pred_label == query_label)
        correct_predictions.append(is_correct)
        per_class_correct[query_label].append(is_correct)

    # Aggregate per-class Recall@1
    per_class_recall1 = {
        label: float(np.mean(correct))
        for label, correct in per_class_correct.items()
    }

    metrics = {
        'val_loss': avg_loss,
        'accuracy': np.mean(correct_predictions),
        'mrr': np.mean(reciprocal_ranks),
        'map': float(np.mean(average_precisions)),
        f'ndcg@{NDCG_K}': float(np.mean(ndcg_scores)),
        'per_class_recall1': per_class_recall1,
        'all_embeddings': all_embeddings,  # full set (CPU) for collapse detection
        'eval_labels': eval_labels,         # retrieval-subsample labels, CPU
    }
    for k in k_values:
        metrics[f'recall@{k}'] = np.mean(recall_at_k[k])

    return metrics


def format_validation_report(metrics, epoch) -> str:
    lines = ["", "=" * 60, f"Validation Report - Epoch {epoch}", "=" * 60]
    lines.append(f"Accuracy:       {metrics['accuracy']:.4f}")
    lines.append(f"MRR:            {metrics['mrr']:.4f}")
    if 'map' in metrics:
        lines.append(f"MAP:            {metrics['map']:.4f}")
    if 'ndcg@10' in metrics:
        lines.append(f"nDCG@10:        {metrics['ndcg@10']:.4f}")
    for k in [1, 5, 10]:
        if f'recall@{k}' in metrics:
            lines.append(f"Recall@{k:2d}:      {metrics[f'recall@{k}']:.4f}")
    if 'val_loss' in metrics:
        lines.append(f"Val Loss:       {metrics['val_loss']:.4f}")
    if 'per_class_recall1' in metrics and metrics['per_class_recall1']:
        per_class = metrics['per_class_recall1']
        sorted_classes = sorted(per_class.items(), key=lambda x: x[1])
        lines.append("\nPer-class Recall@1 — worst 5:")
        for label, r in sorted_classes[:5]:
            lines.append(f"  class {label}: {r:.4f}")
        lines.append("Per-class Recall@1 — best 5:")
        for label, r in sorted_classes[-5:]:
            lines.append(f"  class {label}: {r:.4f}")
    lines.append("=" * 60)
    return "\n".join(lines)


def print_validation_report(metrics, epoch):
    print(format_validation_report(metrics, epoch))
