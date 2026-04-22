import random
import math
from torch.utils.data import Sampler


class TripletSampler(Sampler):
    def __init__(self,
                 labels,
                 num_triplets):
        """
        Args:
            labels: list or tensor of labels for all samples in the dataset.
            num_triplets: number of triplets to sample per epoch.
        """
        self.labels = labels
        self.num_triplets = num_triplets

        # Group indices by label
        self.label_to_indices = {}
        for idx, label in enumerate(labels): self.label_to_indices.setdefault(label, []).append(idx)

        self.unique_labels = list(self.label_to_indices.keys())

    def __iter__(self):
        triplets = []
        for _ in range(self.num_triplets):
            # Anchor label
            anchor_label = random.choice(self.unique_labels)
            positive_label = anchor_label
            negative_label = random.choice([l for l in self.unique_labels if l != anchor_label])

            # Sample indices
            anchor_idx = random.choice(self.label_to_indices[anchor_label])
            positive_idx = random.choice([i for i in self.label_to_indices[positive_label] if i != anchor_idx])
            negative_idx = random.choice(self.label_to_indices[negative_label])

            triplets.append((anchor_idx, positive_idx, negative_idx))

        return iter(triplets)

    def __len__(self):
        return self.num_triplets


class PKSampler(Sampler):
    """
    Balanced batch sampler that yields batches of exactly P classes × K samples.

    Each batch contains P randomly-chosen classes with K samples each,
    guaranteeing that every batch has meaningful positive/negative structure
    regardless of class frequency. This is a prerequisite for SupCon loss and
    helps with long-tailed distributions.

    Args:
        labels: list of integer labels for every sample in the dataset.
        p: number of classes per batch.
        k: number of samples per class per batch.
        drop_last: if True, drop the final incomplete batch.
    """

    def __init__(self, labels, p=32, k=8, drop_last=False):
        self.p = p
        self.k = k
        self.drop_last = drop_last

        # Group indices by label
        self.label_to_indices = {}
        for idx, label in enumerate(labels):
            self.label_to_indices.setdefault(label, []).append(idx)

        # Filter out classes with only 1 sample (cannot form a valid positive pair)
        self.unique_labels = [
            lbl for lbl, idxs in self.label_to_indices.items() if len(idxs) >= 2
        ]

        if len(self.unique_labels) < p:
            raise ValueError(
                f"PKSampler: p={p} but only {len(self.unique_labels)} classes have >= 2 samples. "
                f"Reduce p or the min-samples threshold."
            )

        # Estimate number of batches per epoch: cycle through all classes once
        self._num_batches = math.ceil(len(self.unique_labels) / p)

    def __iter__(self):
        # Shuffle class order at the start of each epoch
        shuffled_labels = self.unique_labels.copy()
        random.shuffle(shuffled_labels)

        batches = []
        for batch_start in range(0, len(shuffled_labels), self.p):
            batch_classes = shuffled_labels[batch_start: batch_start + self.p]
            if len(batch_classes) < self.p and self.drop_last:
                break

            batch_indices = []
            for cls in batch_classes:
                cls_indices = self.label_to_indices[cls]
                if len(cls_indices) >= self.k:
                    sampled = random.sample(cls_indices, self.k)
                else:
                    # Sample with replacement for small classes
                    sampled = random.choices(cls_indices, k=self.k)
                batch_indices.extend(sampled)

            random.shuffle(batch_indices)
            batches.append(batch_indices)

        return iter(batches)

    def __len__(self):
        return self._num_batches
