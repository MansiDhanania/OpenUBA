"""
OpenUBA Training Script
Train anomaly detection models with CLI interface
"""
import argparse
import sys
from pathlib import Path
import time
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import DATASETS, MODEL_CONFIGS, TRAINED_MODELS_DIR, METRICS_DIR
from src.models import get_model, get_available_models
from src.utils.data_loader import DataLoader
from src.utils.metrics import MetricsCalculator


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train OpenUBA anomaly detection models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train Isolation Forest on session data
  python train.py --model isolation_forest --dataset session
  
  # Train Logistic Regression with custom parameters
  python train.py --model logistic_regression --dataset session --test-size 0.2
  
  # Train SVC with custom output name
  python train.py --model svc --dataset day --output my_svc_model
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=get_available_models(),
        help='Model type to train'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=list(DATASETS.keys()),
        help='Dataset to use for training'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output name for saved model (default: model_dataset_timestamp)'
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
        '--scaler-type',
        type=str,
        choices=['standard', 'minmax'],
        default='standard',
        help='Type of scaler to use (default: standard)'
    )
    
    return parser.parse_args()


def train_model(args):
    """Main training function"""
    
    # Model display names
    model_display_names = {
        'isolation_forest': 'Isolation Forest',
        'logistic_regression': 'Logistic Regression',
        'svc': 'Support Vector Classifier',
        'lstm_autoencoder': 'LSTM Autoencoder',
        'lstm_gan': 'LSTM-GAN'
    }
    
    print("="*80)
    print("OpenUBA - Anomaly Detection Model Training")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model:       {model_display_names.get(args.model, args.model)}")
    print(f"  Dataset:     {args.dataset}")
    print(f"  Test Size:   {args.test_size}")
    print(f"  Scaling:     {'Disabled' if args.no_scaling else f'{args.scaler_type.capitalize()}'}")
    print(f"  Random Seed: {args.random_state}")
    print()
    
    # Get dataset path
    dataset_path = DATASETS[args.dataset]
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        return 1
    
    # Check if this is an LSTM model
    is_lstm_model = args.model in ['lstm_autoencoder', 'lstm_gan']
    
    # Initialize data loader based on model type
    if is_lstm_model:
        from src.utils.data_loader import LSTMDataLoader
        data_loader = LSTMDataLoader()
        print("Using LSTM-specific data loader...")
    else:
        data_loader = DataLoader(
            test_size=args.test_size,
            random_state=args.random_state,
            scale=not args.no_scaling
        )
    
    # Load and prepare data
    print("Loading and preparing data...")
    start_time = time.time()
    
    try:
        if is_lstm_model:
            # Get model configuration for LSTM-specific parameters
            model_config = MODEL_CONFIGS.get(args.model, {})
            model = get_model(args.model, **model_config)
            
            # LSTM models prepare their own data
            X_train, X_test, y_train, y_test = model.prepare_data(dataset_path)
            feature_names = model.feature_columns if model.feature_columns else []
        else:
            X_train, X_test, y_train, y_test, feature_names = data_loader.prepare_data(
                dataset_path,
                scaler_type=args.scaler_type
            )
    except Exception as e:
        print(f"ERROR: Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    data_load_time = time.time() - start_time
    print(f"Data preparation completed in {data_load_time:.2f} seconds")
    
    # Get model configuration
    model_config = MODEL_CONFIGS.get(args.model, {})
    model_display_name = model_display_names.get(args.model, args.model)
    
    # Initialize model (if not LSTM, since LSTM already initialized)
    if not is_lstm_model:
        print(f"\nInitializing {model_display_name} model...")
        try:
            model = get_model(args.model, **model_config)
        except Exception as e:
            print(f"ERROR: Failed to initialize model: {e}")
            return 1
    
    # Train model
    train_start = time.time()
    try:
        if is_lstm_model:
            # LSTM models need both X and y for training
            model.train(X_train, y_train, X_test, y_test)
        elif args.model == 'isolation_forest':
            # Isolation Forest doesn't use labels
            model.train(X_train)
        else:
            model.train(X_train, y_train)
    except Exception as e:
        print(f"ERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    train_time = time.time() - train_start
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Make predictions
    print("\nEvaluating model on test set...")
    try:
        if args.model == 'lstm_autoencoder':
            # LSTM Autoencoder predicts sequences, convert to anomaly scores
            pred_sequences = model.predict(X_test)
            # Calculate reconstruction error as anomaly score
            import numpy as np
            reconstruction_error = np.mean(np.abs(pred_sequences - y_test), axis=(1, 2))
            threshold = np.percentile(reconstruction_error, 90)
            y_pred = (reconstruction_error > threshold).astype(int)
            y_pred_proba = None
            print(f"Reconstruction error threshold (90th percentile): {threshold:.4f}")
        elif args.model == 'lstm_gan':
            # LSTM-GAN needs y_test for anomaly score calculation
            y_pred = model.predict(X_test, y_test)
            y_pred_proba = None
        else:
            y_pred = model.predict(X_test)
            # Get probabilities if available
            try:
                y_pred_proba = model.predict_proba(X_test)
            except:
                y_pred_proba = None
    except Exception as e:
        print(f"ERROR: Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Calculate metrics
    metrics_calc = MetricsCalculator()
    
    # For LSTM models, create pseudo-labels for evaluation
    if is_lstm_model:
        import numpy as np
        # Create pseudo labels based on first timestep of first feature
        y_test_eval = np.where(y_test[:, 0, 0] > 0.5, 1, 0)
    else:
        y_test_eval = y_test
    
    metrics = metrics_calc.calculate_metrics(y_test_eval, y_pred, y_pred_proba)
    
    # Add training metadata
    metrics['model_type'] = args.model
    metrics['model_name'] = model_display_name
    metrics['dataset'] = args.dataset
    metrics['train_size'] = len(X_train)
    metrics['test_size'] = len(X_test)
    metrics['training_time'] = train_time
    metrics['feature_count'] = len(feature_names)
    metrics['timestamp'] = datetime.now().isoformat()
    
    # Print metrics
    metrics_calc.print_metrics(metrics, model_name=model_display_name)
    
    # Generate output filename
    if args.output:
        output_name = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_name = f"{args.model}_{args.dataset}_{timestamp}"
    
    # Save model (use .h5 for LSTM models, .pkl for others)
    model_extension = '.h5' if is_lstm_model else '.pkl'
    model_path = TRAINED_MODELS_DIR / f"{output_name}{model_extension}"
    try:
        model.save_model(model_path)
    except Exception as e:
        print(f"WARNING: Failed to save model: {e}")
        import traceback
        traceback.print_exc()
    
    # Save scaler (only for non-LSTM models)
    if not is_lstm_model and not args.no_scaling and data_loader.scaler is not None:
        scaler_path = TRAINED_MODELS_DIR / f"{output_name}_scaler.pkl"
        try:
            data_loader.save_scaler(scaler_path)
        except Exception as e:
            print(f"WARNING: Failed to save scaler: {e}")
    
    # Save metrics
    metrics_path = METRICS_DIR / f"{output_name}_metrics.json"
    try:
        metrics_calc.save_metrics(metrics, metrics_path, model_name=output_name)
    except Exception as e:
        print(f"WARNING: Failed to save metrics: {e}")
    
    # Save training summary
    summary_path = TRAINED_MODELS_DIR / f"{output_name}_summary.txt"
    try:
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"OpenUBA Training Summary - {output_name}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Model Type:      {args.model}\n")
            f.write(f"Model Name:      {model_display_name}\n")
            f.write(f"Dataset:         {args.dataset}\n")
            f.write(f"Training Time:   {train_time:.2f} seconds\n")
            f.write(f"Train Samples:   {len(X_train)}\n")
            f.write(f"Test Samples:    {len(X_test)}\n")
            f.write(f"Features:        {len(feature_names)}\n\n")
            f.write("Performance Metrics:\n")
            f.write(f"  Accuracy:      {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
            f.write(f"  Precision:     {metrics['precision']:.4f}\n")
            f.write(f"  Recall:        {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score:      {metrics['f1_score']:.4f}\n")
            if metrics.get('roc_auc'):
                f.write(f"  ROC-AUC:       {metrics['roc_auc']:.4f}\n")
            f.write(f"\nTrained on:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"Training summary saved to {summary_path}")
    except Exception as e:
        print(f"WARNING: Failed to save summary: {e}")
    
    print("\n" + "="*80)
    print(f"Training completed successfully!")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print("="*80 + "\n")
    
    return 0


def main():
    """Main entry point"""
    args = parse_arguments()
    return train_model(args)


if __name__ == '__main__':
    sys.exit(main())
