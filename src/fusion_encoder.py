import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer
from src.data_loader import TransactionDataset, collate_fn
from src.triplet_sampler import TripletSampler


class FusionEncoder(nn.Module):
    def __init__(self,
                 bert_model_name='bert-base-uncased',
                 categorical_dims=None,
                 numeric_dim=1,
                 text_proj_dim=256,
                 final_dim=256,
                 lr = 0.1,
                 p = 0.1,
                 normalize_embeddings=True,
                 pooling_strategy='mean',
                 use_projection_head=False):
        """
        Args:
            bert_model_name: HuggingFace BERT model name.
            categorical_dims: list of vocab sizes for categorical features.
            numeric_dim: number of numeric features.
            text_proj_dim: dimension after projecting BERT output.
            final_dim: final embedding dimension.
            normalize_embeddings: whether to L2-normalize final embeddings.
            pooling_strategy: 'mean' for mean pooling over last_hidden_state,
                              'cls' for CLS token from last_hidden_state,
                              'pooler' for legacy pooler_output (not recommended).
            use_projection_head: if True, adds a projection head for contrastive
                                 training. The representation before the projection
                                 head is used at inference time.
        """
        super(FusionEncoder, self).__init__()

        # Text encoder (BERT)
        self.lr = lr
        self.p = p
        self.normalize_embeddings = normalize_embeddings
        self.pooling_strategy = pooling_strategy
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_hidden_dim = self.bert.config.hidden_size

        # Projection for text embedding
        self.text_proj = nn.Linear(bert_hidden_dim, text_proj_dim)

        # Embeddings for categorical features
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=vocab_size, embedding_dim=4)
            for vocab_size in categorical_dims])

        # Compute total input dimension for MLP
        cat_total_dim = len(categorical_dims) * 4
        mlp_input_dim = text_proj_dim + cat_total_dim + numeric_dim

        # Final MLP (representation head)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, final_dim),
            nn.ReLU(),
            nn.Dropout(self.p))

        # Projection head for contrastive learning
        self.use_projection_head = use_projection_head
        if use_projection_head:
            self.projection_head = nn.Sequential(
                nn.Linear(final_dim, final_dim),
                nn.ReLU(),
                nn.Linear(final_dim, final_dim))

    def _pool_bert(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if self.pooling_strategy == 'mean':
            token_embeddings = outputs.last_hidden_state  # (batch, seq_len, hidden)
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            summed = (token_embeddings * mask_expanded).sum(dim=1)
            counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
            return summed / counts
        elif self.pooling_strategy == 'cls':
            return outputs.last_hidden_state[:, 0, :]
        else:  # 'pooler' — legacy behaviour
            return outputs.pooler_output

    def forward(self, input_ids, attention_mask, categorical, numeric):
        # Text encoding
        text_embedding = self._pool_bert(input_ids, attention_mask)  # [batch, bert_hidden_dim]
        text_proj = self.text_proj(text_embedding)  # [batch, text_proj_dim]

        # Categorical encoding
        cat_embeds = [emb(categorical[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        cat_concat = torch.cat(cat_embeds, dim=1)  # [batch, cat_total_dim]

        # Numeric features (already standardized)
        # Fusion
        fused = torch.cat([text_proj, cat_concat, numeric], dim=1)
        representation = self.mlp(fused)  # [batch, final_dim]

        # L2 normalization for metric learning
        if self.normalize_embeddings:
            representation = F.normalize(representation, p=2, dim=1)

        # During training with projection head, return projected embeddings
        if self.use_projection_head and self.training:
            projected = self.projection_head(representation)
            if self.normalize_embeddings:
                projected = F.normalize(projected, p=2, dim=1)
            return projected

        return representation


if __name__ == "__main__":

    df = pd.read_csv('./data/sample_txn.csv')
    print(f'--- Data Read ---- {df.shape}')

    # Define columns
    categorical_cols = ['tran_mode', 'dr_cr_indctor', 'sal_flag']
    numeric_cols = ['tran_amt_in_ac']
    label_col = 'category'

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = TransactionDataset(df, tokenizer, categorical_cols, numeric_cols, label_col)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    # Initialize FusionEncoder
    categorical_dims = [len(dataset.cat_vocab[col]) for col in categorical_cols]
    print(categorical_dims, len(numeric_cols))
    encoder = FusionEncoder(categorical_dims=categorical_dims, numeric_dim=len(numeric_cols))

    batch = next(iter(dataloader))
    print()
    print("Batch Shapes:")
    print("input_ids:", batch['input_ids'].shape)
    print("attention_mask:", batch['attention_mask'].shape)
    print("categorical:", batch['categorical'].shape)
    print("numeric:", batch['numeric'].shape)
    print("labels:", batch['labels'].shape)
    print()

    # Forward pass
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    categorical = batch['categorical']
    numeric = batch['numeric']

    with torch.no_grad(): embeddings = encoder(input_ids, attention_mask, categorical, numeric)

    print("Encoder Output Shape:", embeddings.shape)
    print("Sample Embedding:", embeddings[0][:10])  # first 10 values of first embedding


    # Create TripletSampler
    labels = dataset.labels.tolist()
    sampler = TripletSampler(labels, num_triplets=1)
    triplet_indices = next(iter(sampler))  # (anchor_idx, positive_idx, negative_idx)

    anchor_idx, pos_idx, neg_idx = triplet_indices
    anchor = dataset[anchor_idx]
    positive = dataset[pos_idx]
    negative = dataset[neg_idx]

    # Prepare tensors for encoder
    def prepare(sample):
        return (sample['input_ids'].unsqueeze(0),
                sample['attention_mask'].unsqueeze(0),
                sample['categorical'].unsqueeze(0),
                sample['numeric'].unsqueeze(0))

    a_ids, a_mask, a_cat, a_num = prepare(anchor)
    p_ids, p_mask, p_cat, p_num = prepare(positive)
    n_ids, n_mask, n_cat, n_num = prepare(negative)

    with torch.no_grad():
        a_emb = encoder(a_ids, a_mask, a_cat, a_num)
        p_emb = encoder(p_ids, p_mask, p_cat, p_num)
        n_emb = encoder(n_ids, n_mask, n_cat, n_num)

    # Triplet Loss
    triplet_loss_fn = nn.TripletMarginLoss(margin=1.0)
    loss = triplet_loss_fn(a_emb, p_emb, n_emb)
    print(a_emb.shape, p_emb.shape, n_emb.shape)
    print(f"Triplet Loss: {loss.item():.4f}")
