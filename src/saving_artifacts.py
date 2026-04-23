import argparse
import json
import os
import pickle
import re

import torch
import pandas as pd
from transformers import BertTokenizer

from src.data_loader import TransactionDataset

# Matches: tagger_proj256_final256_fd2_full_bs1024_lr4.00e-05_ext_cleaned_merchant_triplet
_EXP_RE = re.compile(
    r"tagger_proj(\d+)_final(\d+)_fd(\d+)"
    r"_(full|gradual|freeze)"
    r"_bs(\d+)_lr([^_]+)"
    r"_(ext|val[\d.]+)"
    r"_(.+)"           # text_tag  (may contain + for multi-col)
    r"_(triplet|supcon)$"
)

# Project-wide defaults — edit here if your schema ever changes
_DEFAULTS = {
    "categorical_cols": ["tran_mode", "dr_cr_indctor", "sal_flag"],
    "numeric_cols": ["tran_amt_in_ac"],
    "label_col": "category",
    "bert_model": "bert-base-uncased",
    "dropout": 0.1,
    "text_cleaning": True,
    "pooling_strategy": "mean",
}


def parse_exp_name(exp_dir: str) -> dict:
    name = os.path.basename(exp_dir.rstrip("/"))
    m = _EXP_RE.match(name)
    if not m:
        raise ValueError(
            f"Cannot parse experiment name: '{name}'\n"
            "Expected format: tagger_proj<N>_final<N>_fd<N>_<freeze>_bs<N>_lr<f>_<val>_<text>_<loss>"
        )
    text_tag = m.group(8)
    text_col = text_tag.split("+") if "+" in text_tag else text_tag
    return {
        "text_proj_dim": int(m.group(1)),
        "final_dim": int(m.group(2)),
        "fusion_depth": int(m.group(3)),
        "freeze_strategy": m.group(4),
        "text_col": text_col,
        "loss_type": m.group(9),
    }


def load_config(exp_dir: str) -> tuple[dict, str]:
    logs_path = os.path.join(exp_dir, "logs", "training_logs.json")
    if os.path.exists(logs_path):
        with open(logs_path) as f:
            data = json.load(f)
        if "config" in data:
            print(f"[INFO] Config loaded from training_logs.json")
            return data["config"], "logs"

    print(f"[WARN] training_logs.json missing or incomplete — parsing experiment name instead")
    parsed = parse_exp_name(exp_dir)
    cfg = {**_DEFAULTS, **parsed}
    return cfg, "name"


def parse_args():
    p = argparse.ArgumentParser(
        description="Rebuild training_artifacts.pkl from a finished or interrupted experiment."
    )
    p.add_argument("--exp_dir", required=True,
                   help="Path to the experiment directory.")
    p.add_argument("--csv", default=None,
                   help="CSV to fit vocab/scaler from. Required when training_logs.json is absent.")
    # Optional overrides (always respected when supplied)
    p.add_argument("--label_col", default=None)
    p.add_argument("--bert_model", default=None)
    p.add_argument("--pooling_strategy", default=None)
    p.add_argument("--no_text_cleaning", action="store_true")
    return p.parse_args()


def _try_load_from_checkpoint(exp_dir: str) -> dict | None:
    """Extract config + dataset state directly from a saved checkpoint (no CSV needed)."""
    for name in ("fusion_encoder_best.pth", "fusion_encoder_interrupted.pth"):
        path = os.path.join(exp_dir, name)
        if not os.path.exists(path):
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if all(k in ckpt for k in ("cat_vocab", "scaler", "label_mapping", "config")):
            print(f"[INFO] Dataset state loaded directly from checkpoint: {name}")
            return ckpt
    return None


def main():
    args = parse_args()

    # Fast path: checkpoint already carries everything — no CSV read needed
    ckpt_state = _try_load_from_checkpoint(args.exp_dir)
    if ckpt_state and not args.csv:
        artifacts = {
            "cat_vocab": ckpt_state["cat_vocab"],
            "scaler": ckpt_state["scaler"],
            "label_mapping": ckpt_state["label_mapping"],
            "categorical_dims": ckpt_state.get("categorical_dims", []),
            "transaction_metadata": [],
            "config": ckpt_state["config"],
        }
        artifacts_path = os.path.join(args.exp_dir, "training_artifacts.pkl")
        with open(artifacts_path, "wb") as f:
            pickle.dump(artifacts, f)
        config_path = os.path.join(args.exp_dir, "model_config.json")
        with open(config_path, "w") as f:
            json.dump({**artifacts["config"], "categorical_dims": artifacts["categorical_dims"]}, f, indent=2)
        print(f"Artifacts  -> {artifacts_path}  (extracted from checkpoint, no CSV read)")
        print(f"Config     -> {config_path}")
        return

    cfg, source = load_config(args.exp_dir)

    # Apply CLI overrides
    csv_path = args.csv or cfg.get("csv_path")
    if not csv_path:
        raise ValueError(
            "--csv is required when training_logs.json is missing "
            "(the CSV path cannot be inferred from the experiment name)."
        )
    if args.label_col:
        cfg["label_col"] = args.label_col
    if args.bert_model:
        cfg["bert_model"] = args.bert_model
    if args.pooling_strategy:
        cfg["pooling_strategy"] = args.pooling_strategy
    if args.no_text_cleaning:
        cfg["text_cleaning"] = False

    categorical_cols = cfg["categorical_cols"]
    numeric_cols = cfg["numeric_cols"]
    label_col = cfg["label_col"]
    bert_model = cfg.get("bert_model", "bert-base-uncased")
    text_proj_dim = cfg.get("text_proj_dim", 256)
    final_dim = cfg.get("final_dim", 256)
    dropout = cfg.get("dropout", 0.1)
    text_cleaning = cfg.get("text_cleaning", True)
    pooling_strategy = cfg.get("pooling_strategy", "mean")

    print(f"Experiment  : {args.exp_dir}")
    print(f"Config from : {source}")
    print(f"CSV         : {csv_path}")
    print(f"text_col    : {cfg.get('text_col', '(not parsed)')}")
    print(f"label_col   : {label_col}")
    print(f"pooling     : {pooling_strategy}  |  text_cleaning: {text_cleaning}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows")

    tokenizer = BertTokenizer.from_pretrained(bert_model)
    dataset = TransactionDataset(
        df, tokenizer, categorical_cols, numeric_cols, label_col,
        text_cleaning=text_cleaning,
    )

    artifacts = {
        "cat_vocab": dataset.cat_vocab,
        "scaler": dataset.scaler,
        "label_mapping": dataset.label_mapping,
        "categorical_dims": [len(dataset.cat_vocab[col]) for col in categorical_cols],
        "transaction_metadata": df.to_dict("records"),
        "config": {
            "categorical_cols": categorical_cols,
            "numeric_cols": numeric_cols,
            "label_col": label_col,
            "bert_model": bert_model,
            "text_proj_dim": text_proj_dim,
            "final_dim": final_dim,
            "dropout": dropout,
            "num_categories": len(dataset.label_mapping),
            "text_cleaning": text_cleaning,
            "pooling_strategy": pooling_strategy,
        },
    }

    artifacts_path = os.path.join(args.exp_dir, "training_artifacts.pkl")
    with open(artifacts_path, "wb") as f:
        pickle.dump(artifacts, f)

    config_path = os.path.join(args.exp_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump({**artifacts["config"], "categorical_dims": artifacts["categorical_dims"]}, f, indent=2)

    print(f"Artifacts  -> {artifacts_path}")
    print(f"Config     -> {config_path}")


if __name__ == "__main__":
    main()
