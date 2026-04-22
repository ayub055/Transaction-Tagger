"""
Enhanced Inference Pipeline for Transaction Tagging with RAG

This module implements a complete inference pipeline following the Amazon Science paper
"Cash booking with Retrieval Augmented LLM" approach:

1. Build golden record index (FAISS) with embeddings + metadata
2. Retrieve top-k similar transactions for any new transaction
3. Apply majority voting to assign final label
4. Return predictions with similar transaction details for debugging

Usage:
    # Step 1: Build golden record index from training data
    from src.inference_pipeline import GoldenRecordIndexer

    indexer = GoldenRecordIndexer(
        artifacts_path="training_artifacts/training_artifacts.pkl",
        model_path="experiments/model/fusion_encoder_best.pth"
    )
    indexer.build_index(
        csv_path="data/sample_txn.csv",
        output_path="golden_records.faiss"
    )

    # Step 2: Run inference
    from src.inference_pipeline import TransactionInferencePipeline

    pipeline = TransactionInferencePipeline(
        artifacts_path="training_artifacts/training_artifacts.pkl",
        model_path="experiments/model/fusion_encoder_best.pth",
        index_path="golden_records.faiss"
    )

    result = pipeline.predict(new_transaction, top_k=5)
    print(f"Predicted: {result['predicted_category']}")
    print(f"Confidence: {result['confidence']:.2%}")

    for similar in result['similar_transactions']:
        print(similar)
"""

import torch
import pandas as pd
import faiss
import numpy as np
import pickle
import os
from typing import Dict, List, Optional, Tuple
from collections import Counter
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm.auto import tqdm

from src.fusion_encoder import FusionEncoder
from src.data_loader import TransactionDataset, collate_fn, clean_narration


class GoldenRecordIndexer:
    """
    Builds and manages FAISS index of golden record transactions.

    Golden records are historical transactions with known labels that serve
    as the retrieval corpus for the RAG approach.
    """

    def __init__(
        self,
        artifacts_path: str,
        model_path: str,
        device: Optional[str] = None,
        use_fp16: bool = False
    ):
        """
        Initialize indexer with trained model and artifacts.

        Args:
            artifacts_path: Path to training artifacts (vocab, scaler, etc.)
            model_path: Path to trained encoder model
            device: Device for inference ('cuda', 'cpu', or None for auto)
            use_fp16: Enable FP16 half-precision inference (requires CUDA)
        """
        self.artifacts_path = artifacts_path
        self.model_path = model_path
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.use_fp16 = use_fp16 and (self.device.type == 'cuda')

        print(f"Initializing GoldenRecordIndexer on {self.device} (fp16={self.use_fp16})")
        self._load_artifacts()
        self._load_model()

    def _load_artifacts(self):
        """Load training artifacts."""
        print(f"Loading artifacts from {self.artifacts_path}...")
        with open(self.artifacts_path, 'rb') as f:
            artifacts = pickle.load(f)

        self.config = artifacts['config']
        self.categorical_cols = self.config['categorical_cols']
        self.numeric_cols = self.config['numeric_cols']
        self.label_col = self.config['label_col']
        self.cat_vocab = artifacts['cat_vocab']
        self.scaler = artifacts['scaler']
        self.label_mapping = artifacts['label_mapping']
        self.categorical_dims = artifacts['categorical_dims']
        self.text_cleaning = self.config.get('text_cleaning', False)
        self.pooling_strategy = self.config.get('pooling_strategy', 'mean')
        _tc = self.config.get('text_col', 'tran_partclr')
        self.text_col = [_tc] if isinstance(_tc, str) else list(_tc)

        self.tokenizer = BertTokenizer.from_pretrained(self.config['bert_model'])
        print(f"  ✓ Loaded {len(self.label_mapping)} label categories")

    def _load_model(self):
        """Load trained encoder model."""
        print(f"Loading model from {self.model_path}...")

        self.encoder = FusionEncoder(
            bert_model_name=self.config['bert_model'],
            categorical_dims=self.categorical_dims,
            numeric_dim=len(self.numeric_cols),
            text_proj_dim=self.config['text_proj_dim'],
            final_dim=self.config['final_dim'],
            p=self.config['dropout'],
            pooling_strategy=self.pooling_strategy,
            fusion_depth=self.config.get('fusion_depth', 1),
        )

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.encoder.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
        else:
            self.encoder.load_state_dict(checkpoint)
            print(f"Loaded model state dict")

        self.encoder.to(self.device)
        self.encoder.eval()
        if self.use_fp16:
            self.encoder = self.encoder.half()

    def build_index(
        self,
        csv_path: str,
        output_path: str,
        batch_size: int = 128,
        index_type: str = 'L2'
    ) -> Tuple[int, int]:
        """
        Build FAISS index from golden record CSV.

        Args:
            csv_path: Path to CSV with golden records
            output_path: Where to save the FAISS index
            batch_size: Batch size for encoding
            index_type: 'L2' or 'IP' (inner product)

        Returns:
            Tuple of (num_records, embedding_dim)
        """
        print(f"Building Golden Record Index")

        # Load data
        print(f"Loading golden records from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} records")

        MAX_GOLDEN_RECORDS = 200_000
        if len(df) > MAX_GOLDEN_RECORDS:
            print(f"  Capping to {MAX_GOLDEN_RECORDS} via stratified sample (from {len(df)})...")
            df = (
                df.groupby(self.label_col, group_keys=False)
                .apply(lambda x: x.sample(
                    n=max(1, round(MAX_GOLDEN_RECORDS * len(x) / len(df))),
                    random_state=42
                ))
                .reset_index(drop=True)
            )
            print(f"  Stratified sample size: {len(df)}")

        # Create dataset with training preprocessing (text_cleaning must match training)
        dataset = TransactionDataset(
            df, self.tokenizer,
            self.categorical_cols,
            self.numeric_cols,
            self.label_col,
            text_cleaning=self.text_cleaning,
            text_col=self.text_col,
        )

        # Use saved vocab and scaler from training
        dataset.cat_vocab = self.cat_vocab
        dataset.scaler = self.scaler
        dataset.label_mapping = self.label_mapping
        dataset.numeric_data = self.scaler.transform(df[self.numeric_cols])
        dataset.labels = df[self.label_col].astype('category').cat.codes

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False
        )

        # Encode all transactions
        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            with tqdm(total=len(df), desc="Encoding golden records", unit="txn") as pbar:
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    categorical = batch['categorical'].to(self.device)
                    numeric = batch['numeric'].to(self.device)

                    with autocast('cuda', enabled=self.use_fp16):
                        embeddings = self.encoder(input_ids, attention_mask, categorical, numeric)
                    all_embeddings.append(embeddings)
                    all_labels.append(batch['labels'].cpu())
                    pbar.update(len(batch['labels']))

        # Single PCIe transfer for the entire matrix; .float() handles FP16→FP32 for FAISS
        all_embeddings = torch.cat(all_embeddings).float().cpu().numpy().astype('float32')
        all_labels = torch.cat(all_labels).numpy()

        embedding_dim = all_embeddings.shape[1]
        print(f"  ✓ Generated {len(all_embeddings)} embeddings of dim {embedding_dim}")

        # Build FAISS index
        print(f"Building FAISS index (type: {index_type})...")
        if index_type == 'L2':
            index = faiss.IndexFlatL2(embedding_dim)
        elif index_type == 'IP':
            index = faiss.IndexFlatIP(embedding_dim)
        elif index_type == 'HNSW':
            # M=32: connections per node (higher = better recall, more memory)
            # efConstruction=200: build-time search depth (higher = better quality, slower build)
            index = faiss.IndexHNSWFlat(embedding_dim, 32)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 64
        elif index_type == 'IVF':
            n_lists = min(4096, len(all_embeddings) // 39)  # ~39 vectors per cluster
            quantiser = faiss.IndexFlatL2(embedding_dim)
            index = faiss.IndexIVFFlat(quantiser, embedding_dim, n_lists)
            print(f"  Training IVF index with {n_lists} clusters...")
            index.train(all_embeddings)
            index.nprobe = 64
        else:
            raise ValueError(f"Unknown index type: {index_type}. Choose from: L2, IP, HNSW, IVF")

        index.add(all_embeddings)
        print(f"  ✓ Index built with {index.ntotal} vectors")

        # Save index
        faiss.write_index(index, output_path)
        print(f"  ✓ Index saved to {output_path}")

        # Save metadata (transaction details + labels)
        metadata_path = output_path.replace('.faiss', '_metadata.pkl')
        transaction_metadata = df.to_dict('records')

        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'transaction_metadata': transaction_metadata,
                'labels': all_labels,
                'embeddings': all_embeddings  # Store for debugging if needed
            }, f)
        print(f"  ✓ Metadata saved to {metadata_path}")

        print(f"\n{'='*80}")
        print(f"Index Build Complete!")
        print(f"  Records: {len(all_embeddings)}")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Label categories: {len(self.label_mapping)}")
        print(f"{'='*80}\n")

        return len(all_embeddings), embedding_dim


class TransactionInferencePipeline:
    """
    Complete inference pipeline for transaction tagging using RAG approach.

    Pipeline steps:
    1. Encode new transaction using trained model
    2. Retrieve top-k similar transactions from FAISS index
    3. Apply majority voting on retrieved labels
    4. Return prediction + similar transactions for debugging
    """

    def __init__(
        self,
        artifacts_path: str,
        model_path: str,
        index_path: str,
        device: Optional[str] = None,
        use_fp16: bool = False
    ):
        """
        Initialize inference pipeline.

        Args:
            artifacts_path: Path to training artifacts
            model_path: Path to trained encoder
            index_path: Path to FAISS index
            device: Device for inference
            use_fp16: Enable FP16 half-precision inference (requires CUDA)
        """
        self.artifacts_path = artifacts_path
        self.model_path = model_path
        self.index_path = index_path
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.use_fp16 = use_fp16 and (self.device.type == 'cuda')

        print(f"\n{'='*80}")
        print(f"Initializing Transaction Inference Pipeline")
        print(f"{'='*80}")

        self._load_components()

        print(f"{'='*80}")
        print(f"Pipeline Ready!")
        print(f"  Device: {self.device}")
        print(f"  Index size: {self.index.ntotal} transactions")
        print(f"  Categories: {len(self.label_mapping)}")
        print(f"{'='*80}\n")

    def _load_components(self):
        """Load all pipeline components."""
        # Load artifacts
        print(f"[1/4] Loading artifacts...")
        with open(self.artifacts_path, 'rb') as f:
            artifacts = pickle.load(f)

        self.config = artifacts['config']
        self.categorical_cols = self.config['categorical_cols']
        self.numeric_cols = self.config['numeric_cols']
        self.label_col = self.config['label_col']
        self.cat_vocab = artifacts['cat_vocab']
        self.scaler = artifacts['scaler']
        self.label_mapping = artifacts['label_mapping']
        self.categorical_dims = artifacts['categorical_dims']
        self.text_cleaning = self.config.get('text_cleaning', False)
        self.pooling_strategy = self.config.get('pooling_strategy', 'mean')
        _tc = self.config.get('text_col', 'tran_partclr')
        self.text_col = [_tc] if isinstance(_tc, str) else list(_tc)
        self.tokenizer = BertTokenizer.from_pretrained(self.config['bert_model'])
        print(f"  ✓ Artifacts loaded")

        # Load model
        print(f"[2/4] Loading model...")
        self.encoder = FusionEncoder(
            bert_model_name=self.config['bert_model'],
            categorical_dims=self.categorical_dims,
            numeric_dim=len(self.numeric_cols),
            text_proj_dim=self.config['text_proj_dim'],
            final_dim=self.config['final_dim'],
            p=self.config['dropout'],
            pooling_strategy=self.pooling_strategy,
            fusion_depth=self.config.get('fusion_depth', 1),
        )

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.encoder.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.encoder.load_state_dict(checkpoint)

        self.encoder.to(self.device)
        self.encoder.eval()
        if self.use_fp16:
            self.encoder = self.encoder.half()
        print(f"  ✓ Model loaded (fp16={self.use_fp16})")

        # Load FAISS index
        print(f"[3/4] Loading FAISS index...")
        cpu_index = faiss.read_index(self.index_path)
        if self.device.type == 'cuda' and hasattr(faiss, 'StandardGpuResources'):
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            self._faiss_res = res   # keep reference — GC would corrupt the GPU index
            print(f"  ✓ Index moved to GPU ({self.index.ntotal} vectors)")
        else:
            self.index = cpu_index
            print(f"  ✓ Index loaded on CPU ({self.index.ntotal} vectors)")

        expected_dim = self.config['final_dim']
        if self.index.d != expected_dim:
            raise ValueError(
                f"Dimension mismatch: FAISS index has dim={self.index.d} but "
                f"current model outputs dim={expected_dim}. "
                f"Rebuild the golden record index with the current model."
            )

        # Load metadata
        print(f"[4/4] Loading metadata...")
        metadata_path = self.index_path.replace('.faiss', '_metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.transaction_metadata = metadata.get('transaction_metadata', [])
                self.golden_labels = metadata.get('labels', None)
                self.golden_embeddings = metadata.get('embeddings', None)
            print(f"  ✓ Metadata loaded ({len(self.transaction_metadata)} records)")
        else:
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    def encode_transaction(self, txn: Dict) -> np.ndarray:
        """
        Encode a single transaction into embedding vector.

        Args:
            txn: Dictionary with transaction features

        Returns:
            Embedding vector (1, embedding_dim)
        """
        # Text encoding — join configured text column(s), skip missing/NaN
        parts = [str(txn[c]) for c in self.text_col
                 if c in txn and txn[c] == txn[c] and str(txn[c]).lower() != 'nan']
        text = " ".join(parts) if parts else ""
        if self.text_cleaning:
            text = clean_narration(text)
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Categorical encoding
        categorical_indices = [
            self.cat_vocab[col].get(txn.get(col, ''), 0)
            for col in self.categorical_cols
        ]
        categorical = torch.tensor([categorical_indices], dtype=torch.long).to(self.device)

        # Numeric encoding
        numeric_values = [[txn.get(col, 0.0) for col in self.numeric_cols]]
        numeric = torch.tensor(
            self.scaler.transform(numeric_values),
            dtype=torch.float
        ).to(self.device)

        # Generate embedding
        with torch.no_grad():
            with autocast('cuda', enabled=self.use_fp16):
                embedding = self.encoder(
                    input_ids,
                    attention_mask,
                    categorical,
                    numeric
                ).float().cpu().numpy().astype('float32')

        return embedding

    def encode_batch(
        self,
        transactions: List[Dict],
        internal_batch_size: int = 256
    ) -> np.ndarray:
        """
        Encode a list of transactions into embedding vectors efficiently.

        Tokenizes all transactions in one call, then runs the encoder in
        chunks of internal_batch_size to avoid OOM. Returns float32 numpy
        array ready for FAISS search.

        Args:
            transactions: List of transaction dicts
            internal_batch_size: Chunk size for GPU encoder forward pass

        Returns:
            Embeddings array of shape (N, embedding_dim) in float32
        """
        N = len(transactions)
        all_embeddings = []

        with torch.no_grad():
            with tqdm(total=N, desc="Encoding transactions", unit="txn") as pbar:
                for start in range(0, N, internal_batch_size):
                    end = min(start + internal_batch_size, N)
                    chunk = transactions[start:end]

                    # Tokenize only this chunk
                    texts = []
                    for txn in chunk:
                        parts = [str(txn[c]) for c in self.text_col
                                 if c in txn and txn[c] == txn[c] and str(txn[c]).lower() != 'nan']
                        t = " ".join(parts) if parts else ""
                        if self.text_cleaning:
                            t = clean_narration(t)
                        texts.append(t)
                    encodings = self.tokenizer(
                        texts,
                        padding='max_length',
                        truncation=True,
                        max_length=128,
                        return_tensors='pt'
                    )

                    # Categorical
                    cat_indices = [
                        [self.cat_vocab[col].get(txn.get(col, ''), 0) for col in self.categorical_cols]
                        for txn in chunk
                    ]
                    categorical = torch.tensor(cat_indices, dtype=torch.long).to(self.device)

                    # Numeric
                    numeric_values = [[txn.get(col, 0.0) for col in self.numeric_cols] for txn in chunk]
                    numeric = torch.tensor(self.scaler.transform(numeric_values), dtype=torch.float).to(self.device)

                    input_ids = encodings['input_ids'].to(self.device)
                    attention_mask = encodings['attention_mask'].to(self.device)

                    with autocast('cuda', enabled=self.use_fp16):
                        embeddings = self.encoder(input_ids, attention_mask, categorical, numeric)
                    all_embeddings.append(embeddings)
                    pbar.update(end - start)

        # Single PCIe transfer; .float() handles FP16→FP32 for FAISS
        return torch.cat(all_embeddings).float().cpu().numpy().astype('float32')

    def retrieve_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve top-k similar transactions from index.

        Args:
            query_embedding: Query embedding (1, embedding_dim)
            top_k: Number of neighbors to retrieve

        Returns:
            Tuple of (distances, indices) arrays
        """
        distances, indices = self.index.search(query_embedding, top_k)
        return distances[0], indices[0]

    def majority_vote(self, retrieved_labels: np.ndarray) -> Tuple[int, float, Dict]:
        """
        Apply majority voting on retrieved labels.

        Args:
            retrieved_labels: Array of label codes from retrieved transactions

        Returns:
            Tuple of (predicted_label_code, confidence, vote_distribution)
        """
        label_counts = Counter(retrieved_labels)
        most_common_label, count = label_counts.most_common(1)[0]
        confidence = count / len(retrieved_labels)

        # Get vote distribution
        vote_dist = {
            self.label_mapping[label]: cnt
            for label, cnt in label_counts.items()
        }

        return most_common_label, confidence, vote_dist

    def predict(
        self,
        new_txn: Dict,
        top_k: int = 5,
        return_embeddings: bool = False
    ) -> Dict:
        """
        Predict category for new transaction using RAG approach.

        Args:
            new_txn: Transaction dictionary with required features
            top_k: Number of similar transactions to retrieve
            return_embeddings: Whether to include embeddings in output

        Returns:
            Dictionary with:
                - predicted_category: Final predicted label
                - confidence: Majority vote confidence
                - vote_distribution: How votes were distributed
                - top_k_labels: Labels of retrieved transactions
                - similar_transactions: List of retrieved transaction details
                - distances: Similarity distances
                - query_embedding: (optional) Query embedding
        """
        # Step 1: Encode new transaction
        query_emb = self.encode_transaction(new_txn)

        # Step 2: Retrieve similar transactions
        distances, indices = self.retrieve_similar(query_emb, top_k)

        # Step 3: Get labels of retrieved transactions
        retrieved_labels = self.golden_labels[indices]

        # Step 4: Majority voting
        pred_label_code, confidence, vote_dist = self.majority_vote(retrieved_labels)
        predicted_category = self.label_mapping[pred_label_code]

        # Step 5: Build similar transactions list
        similar_transactions = []
        for idx, distance in zip(indices, distances):
            txn_data = self.transaction_metadata[idx]
            similar_transactions.append({
                'index': int(idx),
                'transaction': {
                    'description': txn_data.get('tran_partclr', 'N/A'),
                    'amount': float(txn_data.get('tran_amt_in_ac', 0.0)),
                    'dr_cr': txn_data.get('dr_cr_indctor', 'N/A'),
                    'mode': txn_data.get('tran_mode', 'N/A'),
                    'sal_flag': txn_data.get('sal_flag', 'N/A'),
                    'merchant': txn_data.get('merchant', 'N/A'),
                    'date': txn_data.get('tran_date', 'N/A'),
                    'category': txn_data.get(self.label_col, 'N/A')
                },
                'similarity_distance': float(distance),
                'label': self.label_mapping[self.golden_labels[idx]]
            })

        # Build result
        result = {
            'predicted_category': predicted_category,
            'confidence': float(confidence),
            'vote_distribution': vote_dist,
            'top_k_labels': [self.label_mapping[label] for label in retrieved_labels],
            'similar_transactions': similar_transactions,
            'distances': distances.tolist()
        }

        if return_embeddings:
            result['query_embedding'] = query_emb

        return result

    def predict_batch(
        self,
        transactions: List[Dict],
        top_k: int = 5,
        batch_size: int = 256
    ) -> List[Dict]:
        """
        Predict categories for multiple transactions using batched GPU inference.

        Encodes all transactions in batches (single BERT pass per chunk),
        then runs a single batched FAISS search for all queries at once.
        Majority voting and result assembly run on CPU with no GPU ops.

        Args:
            transactions: List of transaction dictionaries
            top_k: Number of similar transactions per query
            batch_size: Internal chunk size for the encoder

        Returns:
            List of prediction result dicts (same schema as predict())
        """
        if not transactions:
            return []

        print(f"Encoding {len(transactions)} transactions (batch_size={batch_size})...")
        all_embeddings = self.encode_batch(transactions, internal_batch_size=batch_size)

        print(f"Running batched FAISS search (top_k={top_k})...")
        distances, indices = self.index.search(all_embeddings, top_k)  # (N, top_k) each

        results = []
        for i in tqdm(range(len(transactions)), desc="Assembling results", unit="txn"):
            retrieved_labels = self.golden_labels[indices[i]]
            pred_label_code, confidence, vote_dist = self.majority_vote(retrieved_labels)
            predicted_category = self.label_mapping[pred_label_code]

            similar_transactions = []
            for idx, distance in zip(indices[i], distances[i]):
                txn_data = self.transaction_metadata[idx]
                similar_transactions.append({
                    'index': int(idx),
                    'transaction': {
                        'description': txn_data.get('tran_partclr', 'N/A'),
                        'amount': float(txn_data.get('tran_amt_in_ac', 0.0)),
                        'dr_cr': txn_data.get('dr_cr_indctor', 'N/A'),
                        'mode': txn_data.get('tran_mode', 'N/A'),
                        'sal_flag': txn_data.get('sal_flag', 'N/A'),
                        'merchant': txn_data.get('merchant', 'N/A'),
                        'date': txn_data.get('tran_date', 'N/A'),
                        'category': txn_data.get(self.label_col, 'N/A')
                    },
                    'similarity_distance': float(distance),
                    'label': self.label_mapping[self.golden_labels[idx]]
                })

            results.append({
                'predicted_category': predicted_category,
                'confidence': float(confidence),
                'vote_distribution': vote_dist,
                'top_k_labels': [self.label_mapping[label] for label in retrieved_labels],
                'similar_transactions': similar_transactions,
                'distances': distances[i].tolist()
            })

        return results


def print_prediction_result(result: Dict, transaction: Dict, top_k: int = 5):
    """
    Pretty print prediction results.

    Args:
        result: Prediction result from pipeline
        transaction: Original transaction dict
        top_k: Number of similar transactions shown
    """
    print("\n" + "="*80)
    print("PREDICTION RESULT")
    print("="*80)
    print(f"Query Transaction:")
    _desc_key = next((k for k in ['tran_partclr', 'merchant', 'cleaned_merchant'] if k in transaction), None)
    print(f"  Description: {transaction.get(_desc_key, 'N/A') if _desc_key else 'N/A'}")
    print(f"  Amount: ${transaction.get('tran_amt_in_ac', 0):.2f}")
    print(f"  Mode: {transaction.get('tran_mode', 'N/A')}")
    print(f"  DR/CR: {transaction.get('dr_cr_indctor', 'N/A')}")

    print(f"\nPrediction:")
    print(f"  Category: {result['predicted_category']}")
    print(f"  Confidence: {result['confidence']:.2%} ({int(result['confidence']*top_k)}/{top_k} votes)")

    print(f"\nVote Distribution:")
    for category, count in sorted(
        result['vote_distribution'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"  {category}: {count} vote(s)")

    print("\n" + "="*80)
    print(f"TOP-{top_k} SIMILAR TRANSACTIONS")
    print("="*80)

    for i, similar in enumerate(result['similar_transactions'], 1):
        txn = similar['transaction']
        print(f"\n{i}. [{similar['label']}] | Distance: {similar['similarity_distance']:.4f}")
        print(f"   Description: {txn['description']}")
        print(f"   Amount: ${txn['amount']:.2f} | Mode: {txn['mode']} | DR/CR: {txn['dr_cr']}")
        if txn.get('merchant', 'N/A') != 'N/A':
            print(f"   Merchant: {txn['merchant']} | Date: {txn['date']}")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    """
    Example usage of the inference pipeline.
    """

    # Configuration
    ARTIFACTS_PATH = "training_artifacts/training_artifacts.pkl"
    MODEL_PATH = "experiments/tagger_proj256_final256_freeze-gradual_bs2048_lr5.66e-05/fusion_encoder_best.pth"
    CSV_PATH = "data/sample_txn.csv"
    INDEX_PATH = "golden_records.faiss"
    TOP_K = 5

    # ========================================================================
    # STEP 1: Build Golden Record Index (do this once)
    # ========================================================================
    print("STEP 1: Building Golden Record Index")
    print("This needs to be done once to create the retrieval corpus.\n")

    if not os.path.exists(INDEX_PATH):
        indexer = GoldenRecordIndexer(
            artifacts_path=ARTIFACTS_PATH,
            model_path=MODEL_PATH
        )

        indexer.build_index(
            csv_path=CSV_PATH,
            output_path=INDEX_PATH,
            batch_size=128
        )
    else:
        print(f"Index already exists at {INDEX_PATH}, skipping build.\n")

    # ========================================================================
    # STEP 2: Initialize Inference Pipeline
    # ========================================================================
    print("STEP 2: Initializing Inference Pipeline\n")

    pipeline = TransactionInferencePipeline(
        artifacts_path=ARTIFACTS_PATH,
        model_path=MODEL_PATH,
        index_path=INDEX_PATH
    )

    # ========================================================================
    # STEP 3: Run Inference on Sample Transactions
    # ========================================================================
    print("STEP 3: Running Inference\n")

    # Example transactions
    test_transactions = [
        {
            'tran_partclr': 'AMAZON PURCHASE ELECTRONICS',
            'tran_mode': 'ONLINE',
            'dr_cr_indctor': 'D',
            'sal_flag': 'N',
            'tran_amt_in_ac': 299.99
        },
        {
            'tran_partclr': 'SALARY CREDIT FROM EMPLOYER',
            'tran_mode': 'NEFT',
            'dr_cr_indctor': 'C',
            'sal_flag': 'Y',
            'tran_amt_in_ac': 5000.00
        },
        {
            'tran_partclr': 'ATM WITHDRAWAL CASH',
            'tran_mode': 'ATM',
            'dr_cr_indctor': 'D',
            'sal_flag': 'N',
            'tran_amt_in_ac': 200.00
        }
    ]

    # Predict for each transaction
    for i, txn in enumerate(test_transactions, 1):
        print(f"\n{'#'*80}")
        print(f"TRANSACTION {i}/{len(test_transactions)}")
        print(f"{'#'*80}")

        result = pipeline.predict(txn, top_k=TOP_K)
        print_prediction_result(result, txn, TOP_K)

    # ========================================================================
    # STEP 4: Batch Prediction (more efficient for multiple transactions)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: Batch Prediction Example")
    print("="*80 + "\n")

    batch_results = pipeline.predict_batch(test_transactions, top_k=TOP_K)

    print("Batch Prediction Summary:")
    for i, (txn, result) in enumerate(zip(test_transactions, batch_results), 1):
        print(f"{i}. '{txn['tran_partclr'][:50]}...'")
        print(f"   → {result['predicted_category']} (confidence: {result['confidence']:.2%})")

    print("\n" + "="*80)
    print("Inference Pipeline Demo Complete!")
    print("="*80)