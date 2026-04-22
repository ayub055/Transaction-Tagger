"""
Run Inference Pipeline - Transaction Tagging with RAG
"""

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feat")

import argparse
import os
import pandas as pd
from src.inference_pipeline import (GoldenRecordIndexer, TransactionInferencePipeline, print_prediction_result)


def load_transactions_from_csv(csv_path):
    print(f"\nLoading transactions from: {csv_path}")

    if csv_path.endswith('.parquet'):
        df = pd.read_parquet(csv_path)
        # Fix known column name mismatches for parquet inputs
        if 'tran_amt' in df.columns and 'tran_amt_in_ac' not in df.columns:
            df = df.rename(columns={'tran_amt': 'tran_amt_in_ac'})
        if 'dr_cr_indctor' not in df.columns:
            df['dr_cr_indctor'] = 'D'
    else:
        df = pd.read_csv(csv_path)

    print(df.columns)
    required_columns = ['cust_id', 'tran_date', 'tran_partclr', 'dr_cr_indctor', 'tran_amt_in_ac', 'tran_mode']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns: raise ValueError(f"Missing required columns: {missing_columns}")
    
    print(f"Loaded {len(df)} transactions")
    print(f"Columns found: {list(df.columns)}")

    # Fill optional column before converting
    if 'sal_flag' not in df.columns:
        df['sal_flag'] = 'N'

    print(f"Converting {len(df)} rows to transaction records...")
    transactions = df[['cust_id', 'tran_date', 'tran_partclr', 'tran_mode', 'dr_cr_indctor', 'sal_flag', 'tran_amt_in_ac']].to_dict('records')
    print(f"Converted {len(transactions)} transactions")

    return transactions, df


def main():
    parser = argparse.ArgumentParser(description='Run transaction inference pipeline')
    parser.add_argument('--artifacts', default='training_artifacts/training_artifacts.pkl',
                        help='Path to training artifacts .pkl')
    parser.add_argument('--model', default=None,
                        help='Path to trained encoder .pth checkpoint (required)')
    parser.add_argument('--csv',default='data/sample_txn.csv',help='Path to CSV with golden records')
    parser.add_argument('--index', default='golden_records.faiss',
                        help='Path to FAISS index file')
    parser.add_argument('--top-k',type=int,default=5,help='Number of similar transactions to retrieve')
    parser.add_argument('--skip-build',action='store_true',help='Skip building index if it exists')
    parser.add_argument('--batch-size',type=int,default=512,help='Batch size for encoding during index building (default: 512 for speed)')
    parser.add_argument('--index-type',default='HNSW',choices=['L2', 'IP', 'HNSW', 'IVF'],help='FAISS index type (default: HNSW for speed)')
    parser.add_argument('--no-fp16', action='store_true', help='Disable FP16 precision (slower but may be more stable)')
    parser.add_argument('--input-csv',default=None,help='Path to input CSV file with transactions to tag (columns: cust_id, tran_date, tran_partlcr, dr_cr_indctor, tran_amt_in_ac, tran_mode)')
    parser.add_argument('--output-csv',default='inference_results.csv',help='Path to output CSV file for results (default: inference_results.csv)')

    args = parser.parse_args()

    if args.model is None:
        parser.error(
            "--model is required. Pass the path to fusion_encoder_best.pth "
            "from a training run, e.g.:\n"
            "  --model experiments/<exp_name>/fusion_encoder_best.pth"
        )

    # ========================================================================
    # STEP 1: Build Golden Record Index
    # ========================================================================
    if not args.skip_build or not os.path.exists(args.index):
        print("\n" + "="*80)
        print("BUILDING GOLDEN RECORD INDEX")
        print("="*80 + "\n")

        indexer = GoldenRecordIndexer(artifacts_path=args.artifacts,model_path=args.model,use_fp16=not args.no_fp16)
        indexer.build_index(csv_path=args.csv,output_path=args.index,batch_size=args.batch_size,index_type=args.index_type)

        print(f"\n✨ Optimizations enabled:")
        print(f"   - Index type: {args.index_type}")
        print(f"   - Batch size: {args.batch_size}")
        print(f"   - FP16: {not args.no_fp16}")
    else:
        print(f"\nSkipping index build (already exists at {args.index})")

    # ========================================================================
    # STEP 2: Initialize Pipeline
    # ========================================================================
    print("\n" + "="*80)
    print("INITIALIZING INFERENCE PIPELINE")
    print("="*80)

    pipeline = TransactionInferencePipeline(artifacts_path=args.artifacts,model_path=args.model,index_path=args.index,use_fp16=not args.no_fp16)

    # ========================================================================
    # STEP 3: Load Transactions and Run Predictions
    # ========================================================================
    
    if args.input_csv:
        # Load transactions from input CSV file
        print("LOADING TRANSACTIONS FROM CSV")
        test_transactions, input_df = load_transactions_from_csv(args.input_csv) 
        print(f"RUNNING PREDICTIONS ON {len(test_transactions)} TRANSACTIONS")
        
        batch_results = pipeline.predict_batch(test_transactions, top_k=args.top_k, batch_size=args.batch_size)
        
        print(f"Predictions complete for {len(batch_results)} transactions")
        
        # ========================================================================
        # STEP 4: Export Results
        # ========================================================================
        print("\n" + "="*80)
        print("EXPORTING RESULTS")
        print("="*80 + "\n")
        
        # Create results dataframe with all original columns plus predictions
        results_data = []
        for txn, result in zip(test_transactions, batch_results):
            results_data.append({
                'cust_id': txn['cust_id'],
                'tran_date': txn['tran_date'],
                'tran_partclr': txn['tran_partclr'],
                'tran_amt_in_ac': txn['tran_amt_in_ac'],
                'tran_mode': txn['tran_mode'],
                'dr_cr_indctor': txn['dr_cr_indctor'],
                'category': result['predicted_category'] if result['confidence'] > 0.69 else '',
                'confidence': result['confidence'],
                'top_k_matches': ', '.join(result['top_k_labels'][:3])
            })
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(args.output_csv, index=False)
        
        print(f"Results exported to: {args.output_csv}")
        print(f"Total predictions: {len(results_df)}")
        print(f"\nSample results (first 10 rows):")
        print(results_df.head(10).to_string(index=False))


    print("\n" + "="*80)
    print("INFERENCE COMPLETE!")
    print("="*80)
    print(f"\nFiles generated:")
    print(f"  - {args.index} (FAISS index)")
    print(f"  - {args.index.replace('.faiss', '_metadata.pkl')} (metadata)")
    print(f"  - {args.output_csv} (prediction results)")
    print("\n")


if __name__ == "__main__":
    main()