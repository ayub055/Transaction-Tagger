"""Curated Confusion Test Suite (C3).

Evaluates whether the encoder ranks a known-positive closer to the query than a
known-confuser for each curated triple. Pair file is user-curated JSON:

[
  {
    "name": "salary_vs_p2p",
    "query":    {"tran_partclr": "...", "tran_mode": "NEFT", "dr_cr_indctor": "C", "sal_flag": "Y", "tran_amt_in_ac": 50000.0},
    "positive": {"tran_partclr": "...", ...},
    "confuser": {"tran_partclr": "...", ...}
  },
  ...
]

A pair passes iff cos(q, positive) > cos(q, confuser).
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from src.data_loader import clean_narration
from src.fusion_encoder import FusionEncoder


def _encode_one(encoder, tokenizer, artifacts, txn, device):
    cfg = artifacts['config']
    _tc = cfg.get('text_col', 'tran_partclr')
    text_cols = [_tc] if isinstance(_tc, str) else list(_tc)
    parts = [str(txn[c]) for c in text_cols
             if c in txn and txn[c] == txn[c] and str(txn[c]).lower() != 'nan']
    text = " ".join(parts) if parts else ""
    if cfg.get('text_cleaning', False):
        text = clean_narration(text)
    enc = tokenizer(text, padding='max_length', truncation=True,
                    max_length=128, return_tensors='pt')

    cat_vocab = artifacts['cat_vocab']
    cat_indices = [cat_vocab[col].get(txn.get(col, ''), 0)
                   for col in cfg['categorical_cols']]
    categorical = torch.tensor([cat_indices], dtype=torch.long).to(device)

    numeric_vals = [[txn.get(col, 0.0) for col in cfg['numeric_cols']]]
    numeric = torch.tensor(artifacts['scaler'].transform(numeric_vals),
                           dtype=torch.float).to(device)

    with torch.no_grad():
        emb = encoder(enc['input_ids'].to(device),
                      enc['attention_mask'].to(device),
                      categorical, numeric)
    return emb.squeeze(0).cpu()


def run_confusion_suite(encoder, tokenizer, artifacts, pairs_path, device):
    with open(pairs_path, 'r') as f:
        pairs = json.load(f)

    encoder.eval()
    results = []
    for i, pair in enumerate(pairs):
        q = _encode_one(encoder, tokenizer, artifacts, pair['query'], device)
        p = _encode_one(encoder, tokenizer, artifacts, pair['positive'], device)
        c = _encode_one(encoder, tokenizer, artifacts, pair['confuser'], device)

        cos_qp = F.cosine_similarity(q.unsqueeze(0), p.unsqueeze(0)).item()
        cos_qc = F.cosine_similarity(q.unsqueeze(0), c.unsqueeze(0)).item()
        passed = cos_qp > cos_qc

        results.append({
            'name': pair.get('name', f'pair_{i}'),
            'cos_query_positive': cos_qp,
            'cos_query_confuser': cos_qc,
            'margin': cos_qp - cos_qc,
            'passed': bool(passed),
        })

    pass_rate = float(np.mean([r['passed'] for r in results])) if results else 0.0
    return {'pass_rate': pass_rate, 'n': len(results), 'results': results}


def _build_encoder_from_artifacts(artifacts, model_path, device):
    cfg = artifacts['config']
    encoder = FusionEncoder(
        bert_model_name=cfg['bert_model'],
        categorical_dims=artifacts['categorical_dims'],
        numeric_dim=len(cfg['numeric_cols']),
        text_proj_dim=cfg['text_proj_dim'],
        final_dim=cfg['final_dim'],
        p=cfg.get('dropout', 0.1),
        pooling_strategy=cfg.get('pooling_strategy', 'mean'),
        fusion_depth=cfg.get('fusion_depth', 1),
    )
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    encoder.load_state_dict(state, strict=False)
    encoder.to(device)
    encoder.eval()
    return encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', required=True, help='Experiment directory containing training_artifacts.pkl and fusion_encoder_best.pth')
    parser.add_argument('--pairs', required=True, help='JSON file with curated pairs')
    parser.add_argument('--out', default=None, help='Optional JSON output path (defaults to {exp_dir}/logs/confusion_results.json)')
    args = parser.parse_args()

    with open(os.path.join(args.exp_dir, 'training_artifacts.pkl'), 'rb') as f:
        artifacts = pickle.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(artifacts['config']['bert_model'])
    encoder = _build_encoder_from_artifacts(
        artifacts, os.path.join(args.exp_dir, 'fusion_encoder_best.pth'), device)

    summary = run_confusion_suite(encoder, tokenizer, artifacts, args.pairs, device)
    print(f"Confusion suite: {summary['n']} pairs | pass_rate={summary['pass_rate']:.3f}")
    for r in summary['results']:
        status = 'PASS' if r['passed'] else 'FAIL'
        print(f"  [{status}] {r['name']:30s} margin={r['margin']:+.4f} (qp={r['cos_query_positive']:.3f} qc={r['cos_query_confuser']:.3f})")

    out_path = args.out or os.path.join(args.exp_dir, 'logs', 'confusion_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == '__main__':
    main()
