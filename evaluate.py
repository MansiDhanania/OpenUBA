"""
OpenUBA Evaluation Script
Evaluate pretrained models with CLI interface
"""
import argparse
import sys
from pathlib import Path
import time
from datetime import datetime
import joblib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import DATASETS, RESULTS_DIR, METRICS_DIR
from src.utils.data_loader import DataLoader
from src.utils.metrics import MetricsCalculator


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate OpenUBA pretrained models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a trained model on test data
  python evaluate.py --model trained_models/isolation_forest_session.pkl --dataset session
  
  # Evaluate with custom scaler
  python evaluate.py --model trained_models/my_model.pkl --dataset day --scaler trained_models/my_model_scaler.pkl
  
  # Evaluate without scaling
  python evaluate.py --model trained_models/model.pkl --dataset session --no-scaling
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model file (.pkl)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=list(DATASETS.keys()),
        help='Dataset to evaluate on'
    )
    
    parser.add_argument(
        '--scaler',
        type=str,
        default=None,
        help='Path to fitted scaler file (.pkl). If not provided, will look for matching scaler.'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output name for evaluation results (default: auto-generated)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.3,
        help='Test set size ratio (default: 0.3)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--no-scaling',
        action='store_true',
        help='Disable feature scaling'
    )
    
    parser.add_argument(
        '--full-dataset',
        action='store_true',
        help='Evaluate on full dataset instead of just test split'
    )
    
    return parser.parse_args()


def evaluate_model(args):
    """Main evaluation function"""
    
    print("="*80)
    print("OpenUBA - Model Evaluation")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model:       {args.model}")
    print(f"  Dataset:     {args.dataset}")
    print(f"  Test Size:   {args.test_size if not args.full_dataset else 'Full dataset'}")
    print(f"  Scaling:     {'Disabled' if args.no_scaling else 'Enabled'}")
    print(f"  Random Seed: {args.random_state}")
    print()
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return 1
    
    # Get dataset path
    dataset_path = DATASETS[args.dataset]
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        return 1
    
    # Load model
    print(f"Loading model from {model_path}...")
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return 1
    
    # Determine scaler path
    scaler_path = None
    if not args.no_scaling:
        if args.scaler:
            scaler_path = Path(args.scaler)
        else:
            # Try to find matching scaler
            base_name = model_path.stem
            potential_scaler = model_path.parent / f"{base_name}_scaler.pkl"
            if potential_scaler.exists():
                scaler_path = potential_scaler
                print(f"Found matching scaler: {scaler_path}")
    
    # Initialize data loader
    data_loader = DataLoader(
        test_size=args.test_size,
        random_state=args.random_state,
        scale=not args.no_scaling
    )
    
    # Load scaler if available
    if scaler_path and scaler_path.exists():
        try:
            data_loader.load_scaler(scaler_path)
        except Exception as e:
            print(f"WARNING: Failed to load scaler: {e}")
    
    # Load and prepare data
    print("\nLoading and preparing data...")
    start_time = time.time()
    
    try:
        if args.full_dataset:
            # Load full dataset
            data = data_loader.load_data(dataset_path)
            data = data_loader.preprocess_labels(data)
            
            X = data.drop('insider', axis=1)
            y = data['insider']
            
            if not args.no_scaling and data_loader.scaler is not None:
                X_scaled = data_loader.scaler.transform(X)
            else:
                X_scaled = X.values
            
            X_test = X_scaled
            y_test = y.values
            feature_names = X.columns.tolist()
        else:
            # Use train/test split
            X_train, X_test, y_train, y_test, feature_names = data_loader.prepare_data(
                dataset_path,
                scaler_type='standard'
            )
    except Exception as e:
        print(f"ERROR: Failed to load data: {e}")
        return 1
    
    data_load_time = time.time() - start_time
    print(f"Data preparation completed in {data_load_time:.2f} seconds")
    print(f"Evaluation samples: {len(X_test)}")
    
    # Make predictions
    print("\nRunning model predictions...")
    eval_start = time.time()
    
    try:
        y_pred = model.predict(X_test)
        
        # Convert Isolation Forest predictions if needed
        if hasattr(model, 'score_samples'):
            # This is an Isolation Forest
            y_pred = [1 if p == -1 else 0 for p in y_pred]
        
        # Try to get probabilities
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
            elif hasattr(model, 'score_samples'):
                # Isolation Forest anomaly scores
                import numpy as np
                scores = model.score_samples(X_test)
                y_pred_proba = 1 / (1 + np.exp(scores))
            else:
                y_pred_proba = None
        except:
            y_pred_proba = None
    except Exception as e:
        print(f"ERROR: Prediction failed: {e}")
        return 1
    
    eval_time = time.time() - eval_start
    print(f"Evaluation completed in {eval_time:.2f} seconds")
    
    # Calculate metrics
    print("\nCalculating performance metrics...")
    metrics_calc = MetricsCalculator()
    metrics = metrics_calc.calculate_metrics(y_test, y_pred, y_pred_proba)
    
    # Add evaluation metadata
    metrics['model_path'] = str(model_path)
    metrics['dataset'] = args.dataset
    metrics['eval_size'] = len(X_test)
    metrics['evaluation_time'] = eval_time
    metrics['feature_count'] = len(feature_names)
    metrics['timestamp'] = datetime.now().isoformat()
    
    # Print metrics
    model_name = model_path.stem.replace('_', ' ').title()
    metrics_calc.print_metrics(metrics, model_name=model_name)
    
    # Generate output filename
    if args.output:
        output_name = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_name = f"eval_{model_path.stem}_{args.dataset}_{timestamp}"
    
    # Save evaluation metrics
    eval_metrics_path = METRICS_DIR / f"{output_name}.json"
    try:
        metrics_calc.save_metrics(metrics, eval_metrics_path, model_name=output_name)
    except Exception as e:
        print(f"WARNING: Failed to save metrics: {e}")
    
    # Save evaluation summary
    summary_path = RESULTS_DIR / f"{output_name}_summary.txt"
    try:
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"OpenUBA Evaluation Summary - {output_name}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Model:           {model_path.name}\n")
            f.write(f"Dataset:         {args.dataset}\n")
            f.write(f"Eval Time:       {eval_time:.2f} seconds\n")
            f.write(f"Eval Samples:    {len(X_test)}\n")
            f.write(f"Features:        {len(feature_names)}\n\n")
            f.write("Performance Metrics:\n")
            f.write(f"  Accuracy:      {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
            f.write(f"  Precision:     {metrics['precision']:.4f}\n")
            f.write(f"  Recall:        {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score:      {metrics['f1_score']:.4f}\n")
            if metrics.get('roc_auc'):
                f.write(f"  ROC-AUC:       {metrics['roc_auc']:.4f}\n")
            f.write(f"\nEvaluated on:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"\nEvaluation summary saved to {summary_path}")
    except Exception as e:
        print(f"WARNING: Failed to save summary: {e}")
    
    print("\n" + "="*80)
    print(f"Evaluation completed successfully!")
    print(f"Metrics saved to: {eval_metrics_path}")
    print("="*80 + "\n")
    
    return 0


def main():
    """Main entry point"""
    args = parse_arguments()
    return evaluate_model(args)


if __name__ == '__main__':
    sys.exit(main())
