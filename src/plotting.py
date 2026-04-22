import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os
import torch

# ---------------------- Plot Loss Curves ----------------------
def plot_loss_curves(log_data, save_path):
    epoch_losses = log_data.get("epoch_losses", [])
    step_losses = log_data.get("step_losses", [])

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', label="Epoch Loss")
    plt.title("Epoch Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

    # Optional: Step loss curve
    step_path = save_path.replace("loss_curve.png", "step_loss_curve.png")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(step_losses) + 1), step_losses, label="Step Loss", alpha=0.7)
    plt.title("Step Loss Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(step_path)
    plt.close()

# ---------------------- Plot Gradient Norms ----------------------
def plot_grad_norms(log_data, save_path):
    grad_norms = log_data.get("grad_norms", [])
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(grad_norms) + 1), grad_norms, color='orange', alpha=0.8)
    plt.title("Gradient Norms Over Steps")
    plt.xlabel("Step")
    plt.ylabel("Gradient Norm")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# ---------------------- TSNE Embedding Visualization ----------------------
def plot_embedding_projection(embeddings, labels, save_path, perplexity=30, n_iter=1000):
    """
    embeddings: torch.Tensor or numpy array of shape [num_samples, embedding_dim]
    labels: numpy array of shape [num_samples]
    """
    if isinstance(embeddings, torch.Tensor): embeddings = embeddings.detach().cpu().numpy()

    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    reduced = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title("TSNE Projection of Embeddings")
    plt.savefig(save_path)
    plt.close()

# ---------------------- Plot Validation Metrics ----------------------
def plot_validation_curves(log_data, save_path):
    """
    Plot validation metrics over epochs including:
    - Train loss vs Validation loss
    - Recall@5
    - Accuracy
    """
    epoch_losses = log_data.get("epoch_losses", [])
    val_losses = log_data.get("val_losses", [])
    val_recall5 = log_data.get("val_recall5", [])
    val_accuracies = log_data.get("val_accuracies", [])

    epochs = range(1, len(epoch_losses) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Train vs Val Loss
    axes[0, 0].plot(epochs, epoch_losses, marker='o', label="Train Loss", color='blue')
    if val_losses:
        axes[0, 0].plot(epochs, val_losses, marker='s', label="Val Loss", color='red')
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training vs Validation Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot 2: Validation Recall@5
    if val_recall5:
        axes[0, 1].plot(epochs, val_recall5, marker='o', label="Recall@5", color='green')
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Recall@5")
        axes[0, 1].set_title("Validation Recall@5")
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[0, 1].set_ylim([0, 1.05])

    # Plot 3: Validation Accuracy
    if val_accuracies:
        axes[1, 0].plot(epochs, val_accuracies, marker='o', label="Accuracy", color='purple')
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].set_title("Validation Accuracy")
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].set_ylim([0, 1.05])

    # Plot 4: Combined metrics
    if val_recall5 and val_accuracies:
        axes[1, 1].plot(epochs, val_recall5, marker='o', label="Recall@5", color='green')
        axes[1, 1].plot(epochs, val_accuracies, marker='s', label="Accuracy", color='purple')
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Score")
        axes[1, 1].set_title("Combined Validation Metrics")
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ---------------------- Plot Collapse Metrics ----------------------
def plot_collapse_metrics(log_data, save_path):
    """
    Plot embedding collapse diagnostics over training epochs:
    - Average pairwise cosine similarity (collapse signal if > 0.9)
    - Effective rank of embedding matrix
    - Number of dead dimensions
    """
    collapse_history = log_data.get("collapse_history", [])
    if not collapse_history:
        return

    epochs = range(1, len(collapse_history) + 1)
    avg_cos = [c['avg_cosine_similarity'] for c in collapse_history]
    eff_rank = [c['effective_rank'] for c in collapse_history]
    dead_dims = [c['dead_dimensions'] for c in collapse_history]
    stable_rank = [c.get('stable_rank', float('nan')) for c in collapse_history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Average cosine similarity
    axes[0].plot(epochs, avg_cos, marker='o', color='red')
    axes[0].axhline(y=0.9, color='darkred', linestyle='--', alpha=0.6, label='Collapse threshold (0.9)')
    axes[0].axhline(y=0.2, color='green', linestyle='--', alpha=0.6, label='Healthy lower bound (0.2)')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Avg Pairwise Cosine Similarity")
    axes[0].set_title("Embedding Collapse: Cosine Similarity")
    axes[0].set_ylim([0, 1.05])
    axes[0].legend()
    axes[0].grid(True)

    # Effective rank + stable rank
    axes[1].plot(epochs, eff_rank, marker='o', color='blue', label='Effective rank (s>1%·s0)')
    axes[1].plot(epochs, stable_rank, marker='s', color='teal', label='Stable rank (sum(sᵢ²)/s0²)')
    axes[1].axhline(y=10, color='darkred', linestyle='--', alpha=0.6, label='Collapsed threshold (< 10)')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Rank")
    axes[1].set_title("Embedding Collapse: Rank Metrics")
    axes[1].legend()
    axes[1].grid(True)

    # Dead dimensions
    axes[2].plot(epochs, dead_dims, marker='o', color='orange')
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Dead Dimensions (var < 1e-4)")
    axes[2].set_title("Embedding Collapse: Dead Dimensions")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ---------------------- Plot Per-Class Recall@1 ----------------------
def plot_per_class_recall(per_class_recall1, save_path, label_mapping=None, top_n=40):
    """
    Bar chart of per-class Recall@1, sorted ascending.
    Shows at most top_n classes to keep the chart readable.

    Args:
        per_class_recall1: dict {label_code: recall_value}
        save_path: output png path
        label_mapping: dict {label_code: label_name} for readable axis labels
        top_n: max number of classes to display
    """
    if not per_class_recall1:
        return

    sorted_items = sorted(per_class_recall1.items(), key=lambda x: x[1])
    # Take worst half and best half if too many classes
    if len(sorted_items) > top_n:
        half = top_n // 2
        sorted_items = sorted_items[:half] + sorted_items[-half:]

    codes = [item[0] for item in sorted_items]
    recalls = [item[1] for item in sorted_items]

    if label_mapping:
        tick_labels = [label_mapping.get(c, str(c)) for c in codes]
    else:
        tick_labels = [str(c) for c in codes]

    fig_width = max(12, len(sorted_items) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    colors = ['red' if r < 0.5 else 'orange' if r < 0.8 else 'green' for r in recalls]
    ax.bar(range(len(recalls)), recalls, color=colors)
    ax.set_xticks(range(len(recalls)))
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylim([0, 1.05])
    ax.set_ylabel("Recall@1")
    ax.set_title(f"Per-Class Recall@1 (n={len(per_class_recall1)} classes, showing {len(sorted_items)})")
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='0.5')
    ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='0.8')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ---------------------- Plot Per-Class Recall History Heatmap ----------------------
def plot_per_class_recall_history(per_class_history, save_path, label_mapping=None, worst_k=30):
    """Heatmap: classes (worst-K averaged over epochs) × epochs → Recall@1.

    Args:
        per_class_history: list of {class_str: recall_float}, one per epoch.
        save_path: output path.
        label_mapping: dict {int_code: label_name} for readable y-axis.
        worst_k: show the worst-K classes (by final-epoch recall) for readability.
    """
    if not per_class_history:
        return

    all_classes = sorted({cls for ep in per_class_history for cls in ep.keys()})
    if not all_classes:
        return

    # Pick worst-K classes by last-epoch recall
    last = per_class_history[-1]
    sorted_cls = sorted(all_classes, key=lambda c: last.get(c, 0.0))
    pick = sorted_cls[:worst_k]

    matrix = np.array([[ep.get(c, np.nan) for ep in per_class_history] for c in pick])

    if label_mapping:
        y_labels = [label_mapping.get(int(c), c) if c.isdigit() or (c.startswith('-') and c[1:].isdigit()) else c for c in pick]
    else:
        y_labels = pick

    fig_h = max(6, len(pick) * 0.25)
    fig, ax = plt.subplots(figsize=(max(8, len(per_class_history) * 0.6), fig_h))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(len(per_class_history)))
    ax.set_xticklabels([f"e{i+1}" for i in range(len(per_class_history))], fontsize=8)
    ax.set_yticks(range(len(pick)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Epoch")
    ax.set_title(f"Per-Class Recall@1 over training (worst-{len(pick)})")
    plt.colorbar(im, ax=ax, label="Recall@1")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ---------------------- Plot Retrieval Metrics (MAP, nDCG) ----------------------
def plot_retrieval_metrics(log_data, save_path):
    """MAP and nDCG@10 over epochs."""
    val_map = log_data.get("val_map", [])
    val_ndcg = log_data.get("val_ndcg10", [])
    if not val_map and not val_ndcg:
        return
    epochs = range(1, max(len(val_map), len(val_ndcg)) + 1)

    plt.figure(figsize=(10, 6))
    if val_map:
        plt.plot(epochs, val_map, marker='o', label='MAP', color='navy')
    if val_ndcg:
        plt.plot(epochs, val_ndcg, marker='s', label='nDCG@10', color='darkorange')
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Retrieval Ranking Quality")
    plt.ylim([0, 1.05])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
