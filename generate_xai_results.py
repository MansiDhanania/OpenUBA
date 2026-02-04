"""
Generate all XAI results for comprehensive README
Creates LIME and SHAP explanations for all 5 models
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# XAI configurations for each model
XAI_CONFIGS = [
    # Model, Method, Dataset, Instance
    ('isolation_forest', 'shap', 'session', 0),
    ('logistic_regression', 'lime', 'session', 0),
    ('logistic_regression', 'shap', 'session', 0),
    ('svc', 'lime', 'session', 5),
    ('svc', 'shap', 'session', 0),
    ('lstm_autoencoder', 'lime', 'session', 0),
    ('lstm_autoencoder', 'shap', 'session', 0),
    ('lstm_gan', 'lime', 'session', 0),
    ('lstm_gan', 'shap', 'session', 0),
]


def run_xai_command(model, method, dataset, instance):
    """Run XAI command and capture result"""
    cmd = [
        'python', 'xai.py',
        '--model', model,
        '--method', method,
        '--dataset', dataset,
        '--instance', str(instance),
        '--output', f'{model}_{method}_{dataset}_readme'
    ]
    
    print(f"\n{'='*80}")
    print(f"Generating {method.upper()} for {model}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"[OK] Success: {model} - {method}")
            return True
        else:
            print(f"[FAIL] Failed: {model} - {method}")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] Timeout: {model} - {method} (took > 5 minutes)")
        return False
    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        return False


def main():
    """Generate all XAI results"""
    print("="*80)
    print("OpenUBA - Comprehensive XAI Results Generation")
    print("="*80)
    print(f"\nGenerating XAI explanations for {len(XAI_CONFIGS)} configurations")
    print(f"This may take several minutes...\n")
    
    results_dir = Path("results/xai")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    fail_count = 0
    
    start_time = datetime.now()
    
    for i, (model, method, dataset, instance) in enumerate(XAI_CONFIGS, 1):
        print(f"\n[{i}/{len(XAI_CONFIGS)}] Processing: {model} - {method}")
        
        success = run_xai_command(model, method, dataset, instance)
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Summary
    print("\n" + "="*80)
    print("XAI Generation Complete!")
    print("="*80)
    print(f"\nSummary:")
    print(f"  [OK] Successful: {success_count}/{len(XAI_CONFIGS)}")
    print(f"  [FAIL] Failed:     {fail_count}/{len(XAI_CONFIGS)}")
    print(f"  Duration:   {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"\nResults saved to: {results_dir.absolute()}")
    print("\nNext steps:")
    print("  1. Review generated plots in results/xai/")
    print("  2. Select best visualizations for README")
    print("  3. Add to README with markdown:")
    print("     ![XAI](results/xai/your_plot_name.png)")
    print("\n" + "="*80 + "\n")
    
    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
