import re
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer


# ---------------------- Text Cleaning ----------------------
_MONTH_PATTERN = r'\d{1,2}(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\d{0,4}'
_DATE_PATTERN = r'\d{2}/\d{2}/\d{2,4}'
_LONG_ALPHANUM = r'\b[A-Z0-9]{8,}\b'
_MULTI_SPACE = r'\s+'

def clean_narration(text):
    """Clean bank narration by removing noise tokens that carry no category signal."""
    if not isinstance(text, str):
        return ""
    text = text.upper()
    text = re.sub(_DATE_PATTERN, '', text)
    text = re.sub(_MONTH_PATTERN, '', text, flags=re.IGNORECASE)
    text = re.sub(_LONG_ALPHANUM, '', text)
    text = re.sub(_MULTI_SPACE, ' ', text).strip()
    return text


class TransactionDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 tokenizer,
                 categorical_cols,
                 numeric_cols,
                 label_col,
                 max_length=128,
                 text_cleaning=False,
                 text_col="tran_partclr"):
        """
        Args:
            df: pandas DataFrame containing transaction data.
            tokenizer: HuggingFace tokenizer for text encoding.
            categorical_cols: list of categorical feature column names.
            numeric_cols: list of numeric feature column names.
            label_col: column name for GL-account label.
            max_length: max token length for text.
            text_cleaning: whether to apply narration cleaning.
            text_col: column name (str) or list of column names to use as text input.
                      If a list, columns are joined with a space before tokenisation.
                      Examples:
                        "tran_partclr"                      — single column (default)
                        "merchant"                          — merchant name only
                        ["merchant", "tran_partclr"]        — merchant first, then narration
                        "cleaned_merchant"                  — pre-cleaned merchant column
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        self.label_col = label_col
        self.max_length = max_length
        self.text_cleaning = text_cleaning
        # Normalise to list internally — simpler __getitem__ logic
        self.text_col = [text_col] if isinstance(text_col, str) else list(text_col)

        # Build categorical vocab mapping with <UNK> token
        self.cat_vocab = {}
        for col in categorical_cols:
            unique_vals = df[col].unique()
            vocab = {'<UNK>': 0}  # Reserve index 0 for unknown values
            vocab.update({val: idx + 1 for idx, val in enumerate(unique_vals)})
            self.cat_vocab[col] = vocab

        # Standardize numeric features
        self.scaler = StandardScaler()
        self.numeric_data = self.scaler.fit_transform(df[numeric_cols])

        # Labels
        self.labels = df[label_col].astype('category').cat.codes
        self.label_mapping = dict(enumerate(df[label_col].astype('category').cat.categories))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Build text — join multiple columns with a space, skip missing/NaN values
        parts = [str(row[c]) for c in self.text_col
                 if c in row.index and row[c] == row[c] and str(row[c]).lower() != 'nan']
        text = " ".join(parts) if parts else ""
        if self.text_cleaning:
            text = clean_narration(text)
        encoding = self.tokenizer(text, padding='max_length', truncation=True,
                                  max_length=self.max_length, return_tensors='pt')

        categorical_indices = [self.cat_vocab[col].get(row[col], 0) for col in self.categorical_cols]
        numeric_features = torch.tensor(self.numeric_data[idx], dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # Metadata — include raw text columns and tracking fields
        metadata = {
            "text_input": text,
            "text_col": self.text_col,
            "dr_cr_indctor": row.get("dr_cr_indctor", None),
            "tran_amt_in_ac": row.get("tran_amt_in_ac", None),
            "label": row[self.label_col]}

        for col in self.text_col:
            if col in row.index:
                metadata[col] = row[col]

        # Add categorical columns to metadata dynamically
        for col in self.categorical_cols:
            if col in row.index:
                metadata[col] = row[col]

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "categorical": torch.tensor(categorical_indices, dtype=torch.long),
            "numeric": numeric_features,
            "label": label,
            "metadata": metadata}


def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    categorical = torch.stack([item['categorical'] for item in batch])
    numeric = torch.stack([item['numeric'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'categorical': categorical,
        'numeric': numeric,
        'labels': labels
    }

if __name__ == "__main__":
    df = pd.read_csv('sample_txn.csv')
    print(f'--- Data Read---- {df.shape}')

    # Define columns
    categorical_cols = ['tran_mode', 'dr_cr_indctor', 'sal_flag']
    numeric_cols = ['tran_amt_in_ac']
    label_col = 'category'

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = TransactionDataset(df, tokenizer, categorical_cols, numeric_cols, label_col)

    for i in range(3):
        sample = dataset[i]
        print(f"--- Sample {i} ---")
        print("Description Tokens:", sample['input_ids'][:10].tolist())  # first 10 token IDs
        print("Attention Mask:", sample['attention_mask'][:10].tolist())
        print("Categorical Indices:", sample['categorical'].tolist())
        print("Numeric Features:", sample['numeric'].tolist())
        print("Label:", sample['label'].item())
        print()

    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    # Fetch one batch
    batch = next(iter(dataloader))

    # Print batch shapes for verification
    print("Batch Shapes:")
    print("input_ids:", batch['input_ids'].shape)
    print("attention_mask:", batch['attention_mask'].shape)
    print("categorical:", batch['categorical'].shape)
    print("numeric:", batch['numeric'].shape)
    print("labels:", batch['labels'].shape)
