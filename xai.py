"""
OpenUBA XAI (Explainable AI) Script
Generate LIME and SHAP explanations for all 5 models
"""
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import DATASETS, TRAINED_MODELS_DIR

# XAI Results Directory
XAI_RESULTS_DIR = Path("results/xai")
XAI_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate XAI explanations (LIME/SHAP) for OpenUBA models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate LIME explanation for Isolation Forest
  python xai.py --model isolation_forest --method lime --dataset session
  
  # Generate SHAP explanation for Logistic Regression
  python xai.py --method shap --model logistic_regression --dataset session
  
  # Generate LIME for LSTM Autoencoder with specific instance
  python xai.py --model lstm_autoencoder --method lime --dataset session --instance 10
  
  # Generate all XAI explanations for a model
  python xai.py --model svc --method all --dataset session
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['isolation_forest', 'logistic_regression', 'svc', 'lstm_autoencoder', 'lstm_gan'],
        help='Model type to explain'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        required=True,
        choices=['lime', 'shap', 'all'],
        help='XAI method to use'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=list(DATASETS.keys()),
        help='Dataset to use for explanation'
    )
    
    parser.add_argument(
        '--instance',
        type=int,
        default=0,
        help='Instance number for LIME explanation (default: 0)'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to trained model file (optional)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output name for saved plots (default: auto-generated)'
    )
    
    return parser.parse_args()


def generate_lime_explanation(model_type, dataset_path, instance_num, model_path=None):
    """Generate LIME explanation"""
    print(f"\n{'='*80}")
    print(f"Generating LIME Explanation")
    print(f"{'='*80}")
    print(f"Model: {model_type}")
    print(f"Instance: {instance_num}")
    print(f"Dataset: {dataset_path}")
    print()
    
    # Import appropriate LIME function based on model
    if model_type == 'isolation_forest':
        # Isolation Forest doesn't have LIME in your original files
        # Use SHAP instead or create a simple wrapper
        print("Note: Isolation Forest uses SHAP for explanations")
        return None, None
        
    elif model_type == 'logistic_regression':
        from limexailogreg import limeplotlogreg
        result = limeplotlogreg(str(dataset_path), instance_num)
        fig = plt.gcf()
        return fig, result
        
    elif model_type == 'svc':
        from limexaisvc import limeplotsvc
        result = limeplotsvc(str(dataset_path), instance_num)
        fig = plt.gcf()
        return fig, result
        
    elif model_type == 'lstm_autoencoder':
        from limexailstmautoencoder import limexailstmautoencoder
        fig, exp = limexailstmautoencoder(str(dataset_path), instance_num, model_path)
        return fig, exp
        
    elif model_type == 'lstm_gan':
        from limexailstmgan import limexailstmgan
        fig, exp = limexailstmgan(str(dataset_path), instance_num, model_path)
        return fig, exp
    
    return None, None


def generate_shap_explanation(model_type, dataset_path, model_path=None):
    """Generate SHAP explanation"""
    print(f"\n{'='*80}")
    print(f"Generating SHAP Explanation")
    print(f"{'='*80}")
    print(f"Model: {model_type}")
    print(f"Dataset: {dataset_path}")
    print()
    
    # Import appropriate SHAP function based on model
    if model_type == 'isolation_forest':
        from shapxaiisoforest import shapplotisoforest
        result = shapplotisoforest(str(dataset_path))
        fig = plt.gcf()
        return fig, result
        
    elif model_type == 'logistic_regression':
        from shapxailogreg import shapplotlogreg
        result = shapplotlogreg(str(dataset_path))
        fig = plt.gcf()
        return fig, result
        
    elif model_type == 'svc':
        from shapxaisvc import shapplotsvc
        result = shapplotsvc(str(dataset_path))
        fig = plt.gcf()
        return fig, result
        
    elif model_type == 'lstm_autoencoder':
        from shapxailstmautoencoder import shapxailstmautoencoder
        fig1, fig2, shap_values = shapxailstmautoencoder(str(dataset_path), model_path)
        return fig1, shap_values
        
    elif model_type == 'lstm_gan':
        from shapxailstmgan import shapxailstmgan
        fig1, fig2, shap_values = shapxailstmgan(str(dataset_path), model_path)
        return fig1, shap_values
    
    return None, None


def save_plot(fig, output_path):
    """Save plot to file"""
    if fig is not None:
        try:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Saved plot to: {output_path}")
        except Exception as e:
            print(f"[WARNING] Could not save plot: {e}")


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Model display names
    model_display_names = {
        'isolation_forest': 'Isolation Forest',
        'logistic_regression': 'Logistic Regression',
        'svc': 'Support Vector Classifier',
        'lstm_autoencoder': 'LSTM Autoencoder',
        'lstm_gan': 'LSTM-GAN'
    }
    
    print("="*80)
    print("OpenUBA - Explainable AI (XAI) Analysis")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model:    {model_display_names.get(args.model, args.model)}")
    print(f"  Method:   {args.method.upper()}")
    print(f"  Dataset:  {args.dataset}")
    if args.method == 'lime' or args.method == 'all':
        print(f"  Instance: {args.instance}")
    print()
    
    # Get dataset path
    dataset_path = DATASETS[args.dataset]
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        return 1
    
    # Generate output name
    if args.output:
        output_name = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_name = f"{args.model}_{args.method}_{args.dataset}_{timestamp}"
    
    # Generate explanations
    try:
        if args.method == 'lime' or args.method == 'all':
            fig, result = generate_lime_explanation(
                args.model,
                dataset_path,
                args.instance,
                args.model_path
            )
            
            if fig is not None:
                lime_output = XAI_RESULTS_DIR / f"{output_name}_lime.png"
                save_plot(fig, lime_output)
                plt.close(fig)
            
        if args.method == 'shap' or args.method == 'all':
            fig, result = generate_shap_explanation(
                args.model,
                dataset_path,
                args.model_path
            )
            
            if fig is not None:
                shap_output = XAI_RESULTS_DIR / f"{output_name}_shap.png"
                save_plot(fig, shap_output)
                plt.close(fig)
        
        print("\n" + "="*80)
        print("XAI Analysis completed successfully!")
        print(f"Results saved to: {XAI_RESULTS_DIR}/")
        print("="*80 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: XAI generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
