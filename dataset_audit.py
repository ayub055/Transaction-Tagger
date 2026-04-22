"""
Dataset Audit Script — TAGGER
Run before designing any sampling strategy.

Usage:
    python dataset_audit.py
    python dataset_audit.py --csv data/sample_txn.csv --label category
"""

import sys
import argparse
import numpy as np
import pandas as pd
from loguru import logger

# ── Loguru setup ────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    colorize=True,
    level="DEBUG",
)
logger.add(
    "dataset_audit.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="DEBUG",
    rotation="10 MB",
    encoding="utf-8",
)

SECTION = "=" * 60


def section(title: str) -> None:
    logger.info(SECTION)
    logger.info(f"  {title}")
    logger.info(SECTION)


# ── Section 1: Overview ─────────────────────────────────────────────────────
def audit_overview(df: pd.DataFrame, label_col: str) -> None:
    section("1. DATASET OVERVIEW")
    logger.info(f"Total rows            : {len(df):,}")
    logger.info(f"Total columns         : {len(df.columns)}")
    logger.info(f"Columns               : {list(df.columns)}")
    logger.info(f"Unique categories     : {df[label_col].nunique()}")

    if "tran_date" in df.columns:
        dates = pd.to_datetime(df["tran_date"], errors="coerce")
        logger.info(f"Date range            : {dates.min().date()} → {dates.max().date()}")
        logger.info(f"Null dates            : {dates.isna().sum():,}")
    else:
        logger.warning("Column 'tran_date' not found — skipping date range")

    if "cust_id" in df.columns:
        logger.info(f"Unique customers      : {df['cust_id'].nunique():,}")
    else:
        logger.warning("Column 'cust_id' not found — skipping customer stats")

    logger.info(f"Duplicate rows (all cols): {df.duplicated().sum():,}")


# ── Section 2: Class Distribution ───────────────────────────────────────────
def audit_class_distribution(df: pd.DataFrame, label_col: str) -> None:
    section("2. CLASS DISTRIBUTION")
    counts = df[label_col].value_counts()

    desc = counts.describe().round(1)
    for stat, val in desc.items():
        logger.info(f"  {stat:<10}: {val:,.1f}")

    n = len(counts)
    sorted_counts = counts.sort_values().values
    gini = (
        (2 * (sorted_counts * np.arange(1, n + 1)).sum())
        / (n * sorted_counts.sum())
        - (n + 1) / n
    )
    if gini > 0.6:
        logger.warning(f"Gini coefficient      : {gini:.3f}  ← SEVERE imbalance (>0.6)")
    elif gini > 0.4:
        logger.warning(f"Gini coefficient      : {gini:.3f}  ← moderate imbalance (>0.4)")
    else:
        logger.success(f"Gini coefficient      : {gini:.3f}  ← acceptable balance")

    logger.info("")
    logger.info("Class size thresholds:")
    logger.info(f"  < 50   samples : {(counts <    50).sum()} classes")
    logger.info(f"  < 100  samples : {(counts <   100).sum()} classes")
    logger.info(f"  < 500  samples : {(counts <   500).sum()} classes")
    logger.info(f"  < 1000 samples : {(counts <  1000).sum()} classes")
    logger.info(f"  > 10K  samples : {(counts > 10000).sum()} classes")
    logger.info(f"  > 50K  samples : {(counts > 50000).sum()} classes")

    logger.info("")
    logger.info("Top 10 classes by count:")
    for cat, cnt in counts.head(10).items():
        pct = cnt / len(df) * 100
        logger.info(f"  {cat:<40} {cnt:>8,}  ({pct:.1f}%)")

    logger.info("")
    logger.info("Bottom 10 classes by count:")
    for cat, cnt in counts.tail(10).items():
        pct = cnt / len(df) * 100
        logger.info(f"  {cat:<40} {cnt:>8,}  ({pct:.1f}%)")

    # Flag tiny classes
    tiny = counts[counts < 20]
    if len(tiny) > 0:
        logger.warning(
            f"{len(tiny)} classes have < 20 samples — PKSampler K=8 will sample "
            "almost entirely with replacement for these"
        )
        for cat, cnt in tiny.items():
            logger.warning(f"  {cat}: {cnt} samples")


# ── Section 3: Narration Characteristics ────────────────────────────────────
def audit_narration(df: pd.DataFrame, label_col: str, text_col: str) -> None:
    section("3. NARRATION CHARACTERISTICS")

    if text_col not in df.columns:
        logger.error(f"Text column '{text_col}' not found — skipping section")
        return

    df["_text_len"] = df[text_col].astype(str).str.split().str.len()
    desc = df["_text_len"].describe().round(1)

    logger.info("Token length distribution (word-split, pre-tokenisation):")
    for stat, val in desc.items():
        logger.info(f"  {stat:<10}: {val:.1f}")

    very_short = (df["_text_len"] <= 1).sum()
    if very_short > 0:
        logger.warning(
            f"{very_short:,} rows have <= 1 token after split — "
            "likely pure reference codes; text cleaning may leave them empty"
        )

    logger.info("")
    logger.info("Unique narrations per class (effective sample size for contrastive learning):")
    unique_narr = df.groupby(label_col)[text_col].nunique()
    total_narr  = df.groupby(label_col)[text_col].count()
    ratio       = (unique_narr / total_narr).round(3)

    summary = pd.DataFrame({
        "total_rows"        : total_narr,
        "unique_narrations" : unique_narr,
        "uniqueness_ratio"  : ratio,
    }).sort_values("total_rows")

    logger.info("Bottom 10 classes (fewest rows):")
    for cat, row in summary.head(10).iterrows():
        logger.info(
            f"  {cat:<40} rows={row['total_rows']:>6,}  "
            f"unique={row['unique_narrations']:>6,}  "
            f"ratio={row['uniqueness_ratio']:.3f}"
        )

    logger.info("Top 10 classes (most rows):")
    for cat, row in summary.tail(10).iterrows():
        logger.info(
            f"  {cat:<40} rows={row['total_rows']:>6,}  "
            f"unique={row['unique_narrations']:>6,}  "
            f"ratio={row['uniqueness_ratio']:.3f}"
        )

    low_unique = summary[summary["uniqueness_ratio"] < 0.1]
    if len(low_unique) > 0:
        logger.warning(
            f"{len(low_unique)} classes have uniqueness ratio < 0.1 — "
            "narrations are highly repetitive; oversampling these classes will not help"
        )
        for cat, row in low_unique.iterrows():
            logger.warning(
                f"  {cat:<40} ratio={row['uniqueness_ratio']:.3f}  "
                f"unique={row['unique_narrations']}"
            )
    else:
        logger.success("All classes have uniqueness ratio >= 0.1")

    df.drop(columns=["_text_len"], inplace=True)


# ── Section 4: Feature Discriminativeness ───────────────────────────────────
def audit_features(df: pd.DataFrame, label_col: str) -> None:
    section("4. FEATURE DISCRIMINATIVENESS")

    for feat in ["tran_mode", "dr_cr_indctor", "sal_flag"]:
        if feat not in df.columns:
            logger.warning(f"Column '{feat}' not found — skipping")
            continue

        missing = df[feat].isna().sum()
        vc = df[feat].value_counts()
        logger.info(f"{feat} — {df[feat].nunique()} unique values, {missing:,} missing:")
        for val, cnt in vc.items():
            pct = cnt / len(df) * 100
            logger.info(f"  {str(val):<20} {cnt:>8,}  ({pct:.1f}%)")

    if "tran_amt_in_ac" not in df.columns:
        logger.warning("Column 'tran_amt_in_ac' not found — skipping amount analysis")
        return

    logger.info("")
    logger.info("Amount stats per class (sorted by mean):")
    amt_stats = (
        df.groupby(label_col)["tran_amt_in_ac"]
        .agg(["mean", "median", "std", "min", "max"])
        .round(2)
        .sort_values("mean")
    )
    for cat, row in amt_stats.iterrows():
        logger.info(
            f"  {cat:<40}  mean={row['mean']:>12,.2f}  "
            f"median={row['median']:>12,.2f}  std={row['std']:>12,.2f}"
        )

    # Flag classes where amount is a strong discriminator
    overall_std = df["tran_amt_in_ac"].std()
    tight_classes = amt_stats[amt_stats["std"] < 0.1 * overall_std]
    if len(tight_classes) > 0:
        logger.info("")
        logger.info(
            f"{len(tight_classes)} classes have very tight amount std "
            f"(< 10% of global std={overall_std:,.0f}) — amount is a strong signal here:"
        )
        for cat in tight_classes.index:
            logger.info(f"  {cat}")


# ── Section 5: Customer Leakage Risk ────────────────────────────────────────
def audit_customer_leakage(df: pd.DataFrame, label_col: str) -> None:
    section("5. CUSTOMER LEAKAGE RISK")

    if "cust_id" not in df.columns:
        logger.warning("Column 'cust_id' not found — skipping leakage analysis")
        return

    txn_per_cust = df.groupby("cust_id").size()
    logger.info(f"Transactions per customer:")
    logger.info(f"  min    : {txn_per_cust.min():,}")
    logger.info(f"  median : {txn_per_cust.median():,.0f}")
    logger.info(f"  mean   : {txn_per_cust.mean():,.1f}")
    logger.info(f"  max    : {txn_per_cust.max():,}")

    top5_share = txn_per_cust.nlargest(5).sum() / len(df)
    if top5_share > 0.2:
        logger.warning(
            f"Top 5 customers account for {top5_share:.1%} of all transactions — "
            "dataset may be dominated by a few accounts"
        )
    else:
        logger.success(f"Top 5 customers account for {top5_share:.1%} of transactions — acceptable")

    logger.info("")
    logger.info("Classes dominated by a single customer (> 50% from 1 cust_id):")
    dominated = []
    for cat, grp in df.groupby(label_col):
        vc = grp["cust_id"].value_counts()
        top_share = vc.iloc[0] / len(grp)
        if top_share > 0.5:
            dominated.append((cat, vc.index[0], top_share, len(grp)))

    if dominated:
        for cat, cust, share, total in dominated:
            logger.warning(
                f"  {cat:<40} — {share:.1%} from cust_id={cust}  (class size={total:,})"
            )
        logger.warning(
            "These classes risk customer-level memorisation. "
            "Consider a customer-stratified train/val split."
        )
    else:
        logger.success("No class is dominated by a single customer (all < 50%)")

    # Check if a random split would leak customers across train/val
    unique_custs = df["cust_id"].nunique()
    avg_txn = txn_per_cust.mean()
    logger.info("")
    logger.info(
        f"Leakage note: with {unique_custs:,} customers and {avg_txn:.1f} txn/customer avg, "
        "a random train/val split WILL put the same customer in both sets. "
        "Use GroupShuffleSplit(groups=cust_id) to avoid this."
    )


# ── Section 6: Label Noise ───────────────────────────────────────────────────
def audit_label_noise(df: pd.DataFrame, label_col: str, text_col: str) -> None:
    section("6. LABEL NOISE — DUPLICATE ROWS WITH DIFFERENT LABELS")

    dupe_cols = [
        c for c in [text_col, "tran_mode", "dr_cr_indctor", "tran_amt_in_ac"]
        if c in df.columns
    ]
    logger.info(f"Checking for identical ({', '.join(dupe_cols)}) with different labels...")

    dupes = df[df.duplicated(subset=dupe_cols, keep=False)].copy()
    if len(dupes) == 0:
        logger.success("No duplicate feature rows found")
        return

    cross = (
        dupes.groupby(dupe_cols)[label_col]
        .nunique()
    )
    cross_noise = cross[cross > 1]

    logger.info(f"Duplicate feature rows total : {len(dupes):,}")
    if len(cross_noise) > 0:
        logger.warning(
            f"Rows with identical features but DIFFERENT labels: {len(cross_noise):,} groups"
        )
        logger.warning(
            "These are direct label noise candidates — "
            "the same transaction text+amount+mode is tagged differently"
        )
        # Show a few examples
        example_keys = cross_noise.head(5).index
        for key in example_keys:
            mask = True
            for col, val in zip(dupe_cols, key if isinstance(key, tuple) else [key]):
                mask = mask & (df[col] == val)
            rows = df[mask][[label_col] + dupe_cols].drop_duplicates()
            logger.warning(f"  Example conflict:")
            for _, r in rows.iterrows():
                logger.warning(f"    label={r[label_col]}  text='{str(r.get(text_col,''))[:60]}'")
    else:
        logger.success(
            f"{len(dupes):,} duplicate feature rows exist but all share the same label — no noise signal"
        )


# ── Section 7: Temporal Distribution ────────────────────────────────────────
def audit_temporal(df: pd.DataFrame, label_col: str) -> None:
    section("7. TEMPORAL DISTRIBUTION")

    if "tran_date" not in df.columns:
        logger.warning("Column 'tran_date' not found — skipping temporal analysis")
        return

    df = df.copy()
    df["_date"] = pd.to_datetime(df["tran_date"], errors="coerce")
    null_dates = df["_date"].isna().sum()
    if null_dates > 0:
        logger.warning(f"{null_dates:,} rows have unparseable dates")

    df["_month"] = df["_date"].dt.to_period("M")
    monthly = df.groupby("_month").size()

    logger.info("Monthly transaction volume:")
    for period, cnt in monthly.items():
        bar = "█" * min(40, int(cnt / monthly.max() * 40))
        logger.info(f"  {period}  {cnt:>8,}  {bar}")

    # Check for large monthly swings
    pct_change = monthly.pct_change().abs().dropna()
    big_swings = pct_change[pct_change > 0.5]
    if len(big_swings) > 0:
        logger.warning(
            f"{len(big_swings)} months show > 50% volume change month-over-month "
            "— possible concept drift or data collection gap:"
        )
        for period, chg in big_swings.items():
            logger.warning(f"  {period}: {chg:+.1%} change from prior month")

    # Check if top classes are stable across time
    logger.info("")
    logger.info("Class share stability across months (top 5 classes):")
    top5 = df[label_col].value_counts().head(5).index.tolist()
    monthly_class = (
        df[df[label_col].isin(top5)]
        .groupby(["_month", label_col])
        .size()
        .unstack(fill_value=0)
    )
    monthly_class_pct = monthly_class.div(monthly_class.sum(axis=1), axis=0).round(3)
    for cat in top5:
        if cat in monthly_class_pct.columns:
            col = monthly_class_pct[cat]
            logger.info(
                f"  {cat:<40}  min={col.min():.1%}  max={col.max():.1%}  "
                f"std={col.std():.3f}"
            )


# ── Section 8: PKSampler Readiness ──────────────────────────────────────────
def audit_pksampler_readiness(df: pd.DataFrame, label_col: str, pk_p: int = 32, pk_k: int = 8) -> None:
    section("8. PKSampler READINESS CHECK")

    counts = df[label_col].value_counts()
    eligible = counts[counts >= 2]
    eligible_k = counts[counts >= pk_k]

    logger.info(f"Configured: P={pk_p} classes per batch, K={pk_k} samples per class")
    logger.info(f"Classes with >= 2 samples (minimum for PKSampler) : {len(eligible)}")
    logger.info(f"Classes with >= K={pk_k} samples (no replacement) : {len(eligible_k)}")

    if len(eligible) < pk_p:
        logger.error(
            f"Only {len(eligible)} classes have >= 2 samples but P={pk_p} — "
            "PKSampler will raise ValueError. Reduce pk_p."
        )
    else:
        logger.success(f"P={pk_p} is satisfiable — {len(eligible)} eligible classes available")

    replacement_classes = eligible[eligible < pk_k]
    if len(replacement_classes) > 0:
        logger.warning(
            f"{len(replacement_classes)} classes will sample WITH REPLACEMENT "
            f"(have < K={pk_k} unique samples):"
        )
        for cat, cnt in replacement_classes.items():
            logger.warning(f"  {cat:<40} {cnt} samples  ({pk_k - cnt} duplicates per batch)")
    else:
        logger.success(f"All eligible classes have >= K={pk_k} samples — no replacement needed")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Dataset audit for TAGGER training")
    parser.add_argument("--csv",   default="data/sample_txn.csv", help="Path to transaction CSV")
    parser.add_argument("--label", default="category",            help="Label column name")
    parser.add_argument("--text",  default="tran_partclr",        help="Narration column name")
    parser.add_argument("--pk-p",  type=int, default=32,          help="PKSampler P value to check")
    parser.add_argument("--pk-k",  type=int, default=8,           help="PKSampler K value to check")
    args = parser.parse_args()

    logger.info(f"Loading dataset from: {args.csv}")
    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        logger.error(f"File not found: {args.csv}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        sys.exit(1)

    logger.success(f"Loaded {len(df):,} rows with columns: {list(df.columns)}")

    audit_overview(df, args.label)
    audit_class_distribution(df, args.label)
    audit_narration(df, args.label, args.text)
    audit_features(df, args.label)
    audit_customer_leakage(df, args.label)
    audit_label_noise(df, args.label, args.text)
    audit_temporal(df, args.label)
    audit_pksampler_readiness(df, args.label, pk_p=args.pk_p, pk_k=args.pk_k)

    logger.info(SECTION)
    logger.success("Audit complete. Full log saved to: dataset_audit.log")
    logger.info(SECTION)


if __name__ == "__main__":
    main()
