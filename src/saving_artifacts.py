import pandas as pd
import pickle
import json
import os
from transformers import BertTokenizer
from src.data_loader import TransactionDataset

csv_path = "./data/sample_txn.csv"
categorical_cols = ['tran_mode', 'dr_cr_indctor', 'sal_flag']
numeric_cols = ['tran_amt_in_ac']
label_col = 'category'
bert_model = 'bert-base-uncased'
text_proj_dim = 256
final_dim = 256
dropout = 0.2
text_cleaning = True
pooling_strategy = 'mean'

output_dir = "./training_artifacts"
os.makedirs(output_dir, exist_ok=True)

print("Loading training data...")
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} transactions")
print()

tokenizer = BertTokenizer.from_pretrained(bert_model)
dataset = TransactionDataset(df, tokenizer, categorical_cols, numeric_cols, label_col,
                             text_cleaning=text_cleaning)

transaction_metadata = df.to_dict('records')  # Store all transaction details

artifacts = {
    'cat_vocab': dataset.cat_vocab,
    'scaler': dataset.scaler,
    'label_mapping': dataset.label_mapping,
    'categorical_dims': [len(dataset.cat_vocab[col]) for col in categorical_cols],
    'transaction_metadata': transaction_metadata,
    'config': {
        'categorical_cols': categorical_cols,
        'numeric_cols': numeric_cols,
        'label_col': label_col,
        'bert_model': bert_model,
        'text_proj_dim': text_proj_dim,
        'final_dim': final_dim,
        'dropout': dropout,
        'num_categories': len(dataset.label_mapping),
        'text_cleaning': text_cleaning,
        'pooling_strategy': pooling_strategy,
    }
}

artifacts_path = os.path.join(output_dir, "training_artifacts.pkl")
with open(artifacts_path, 'wb') as f:
    pickle.dump(artifacts, f)

config_path = os.path.join(output_dir, "model_config.json")
config_dict = {
    'categorical_cols': categorical_cols,
    'numeric_cols': numeric_cols,
    'label_col': label_col,
    'bert_model': bert_model,
    'text_proj_dim': text_proj_dim,
    'final_dim': final_dim,
    'dropout': dropout,
    'categorical_dims': artifacts['categorical_dims'],
    'num_categories': len(dataset.label_mapping),
    'text_cleaning': text_cleaning,
    'pooling_strategy': pooling_strategy,
}
with open(config_path, 'w') as f:
    json.dump(config_dict, f, indent=2)

print(f"Artifacts saved to {artifacts_path}")
print(f"Config saved to {config_path}")
