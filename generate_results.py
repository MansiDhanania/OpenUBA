"""
OpenUBA Results Generation Script
Generate all visualizations and plots for README
"""
import argparse
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import METRICS_DIR, PLOTS_DIR, XAI_DIR

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def load_metrics_files(metrics_dir):
    """Load all metrics JSON files from directory"""
    metrics_dict = {}
    
    metrics_path = Path(metrics_dir)
    for json_file in metrics_path.glob('*_metrics.json'):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                model_name = data.get('model_name', json_file.stem.replace('_metrics', ''))
                metrics_dict[model_name] = data
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    return metrics_dict


def plot_model_comparison(metrics_dict, save_path):
    """Create model comparison bar chart"""
    if not metrics_dict:
        print("No metrics found for comparison")
        return
    
    # Extract metrics for comparison
    models = []
    accuracy = []
    precision = []
    recall = []
    f1 = []
    
    for model_name, metrics in metrics_dict.items():
        models.append(metrics.get('model_type', model_name).replace('_', ' ').title())
        accuracy.append(metrics.get('accuracy', 0) * 100)
        precision.append(metrics.get('precision', 0) * 100)
        recall.append(metrics.get('recall', 0) * 100)
        f1.append(metrics.get('f1_score', 0) * 100)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Model': models,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.2
    
    ax.bar(x - 1.5*width, df['Accuracy'], width, label='Accuracy', color='#2E86AB')
    ax.bar(x - 0.5*width, df['Precision'], width, label='Precision', color='#A23B72')
    ax.bar(x + 0.5*width, df['Recall'], width, label='Recall', color='#F18F01')
    ax.bar(x + 1.5*width, df['F1-Score'], width, label='F1-Score', color='#C73E1D')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Model comparison saved to {save_path}")
    plt.close()


def plot_confusion_matrices(metrics_dict, save_dir):
    """Create confusion matrix plots for each model"""
    if not metrics_dict:
        return
    
    for model_name, metrics in metrics_dict.items():
        if 'confusion_matrix' not in metrics:
            continue
        
        cm = np.array(metrics['confusion_matrix'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'],
                   ax=ax)
        
        model_display = metrics.get('model_type', model_name).replace('_', ' ').title()
        ax.set_title(f'Confusion Matrix - {model_display}', fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        
        # Add accuracy text
        accuracy = metrics.get('accuracy', 0)
        plt.text(0.5, -0.15, f"Accuracy: {accuracy:.2%}", 
                ha='center', transform=ax.transAxes, fontsize=11)
        
        plt.tight_layout()
        save_path = Path(save_dir) / f'confusion_matrix_{model_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
        plt.close()


def plot_metrics_table(metrics_dict, save_path):
    """Create a detailed metrics comparison table"""
    if not metrics_dict:
        return
    
    # Prepare data
    rows = []
    for model_name, metrics in metrics_dict.items():
        model_display = metrics.get('model_type', model_name).replace('_', ' ').title()
        
        row = {
            'Model': model_display,
            'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
            'Precision': f"{metrics.get('precision', 0):.4f}",
            'Recall': f"{metrics.get('recall', 0):.4f}",
            'F1-Score': f"{metrics.get('f1_score', 0):.4f}",
        }
        
        if 'roc_auc' in metrics and metrics['roc_auc']:
            row['ROC-AUC'] = f"{metrics['roc_auc']:.4f}"
        
        if 'training_time' in metrics:
            row['Train Time (s)'] = f"{metrics['training_time']:.2f}"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, len(df) * 0.8 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                    cellLoc='center', loc='center',
                    colWidths=[0.2] + [0.12] * (len(df.columns) - 1))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Model Performance Metrics', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Metrics table saved to {save_path}")
    plt.close()


def plot_class_distribution(metrics_dict, save_path):
    """Plot class distribution from metrics"""
    if not metrics_dict:
        return
    
    # Get a sample metrics to extract distribution
    sample_metrics = list(metrics_dict.values())[0]
    
    if 'confusion_matrix' in sample_metrics:
        cm = np.array(sample_metrics['confusion_matrix'])
        total_normal = cm[0].sum()
        total_anomaly = cm[1].sum()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        labels = ['Normal', 'Anomaly']
        sizes = [total_normal, total_anomaly]
        colors = ['#2E86AB', '#C73E1D']
        explode = (0.05, 0.05)
        
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, 
                                           colors=colors, autopct='%1.1f%%',
                                           shadow=True, startangle=90)
        
        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)
            autotext.set_fontweight('bold')
        
        ax.set_title('Dataset Class Distribution', fontsize=14, fontweight='bold', pad=20)
        
        # Add legend with counts
        legend_labels = [f'{label}: {size:,}' for label, size in zip(labels, sizes)]
        ax.legend(legend_labels, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution saved to {save_path}")
        plt.close()


def create_architecture_diagram(save_path):
    """Create system architecture diagram"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define components
    components = [
        {'name': 'Raw CERT\nDataset', 'pos': (1, 7), 'color': '#2E86AB'},
        {'name': 'Feature\nExtraction', 'pos': (3.5, 7), 'color': '#A23B72'},
        {'name': 'Data\nPreprocessing', 'pos': (6, 7), 'color': '#F18F01'},
        {'name': 'Model\nTraining', 'pos': (8.5, 7), 'color': '#C73E1D'},
        {'name': 'Isolation\nForest', 'pos': (2, 4), 'color': '#06A77D'},
        {'name': 'Logistic\nRegression', 'pos': (4, 4), 'color': '#06A77D'},
        {'name': 'SVC', 'pos': (6, 4), 'color': '#06A77D'},
        {'name': 'LSTM\nAutoencoder', 'pos': (8, 4), 'color': '#06A77D'},
        {'name': 'XAI\n(LIME/SHAP)', 'pos': (3.5, 1.5), 'color': '#D62246'},
        {'name': 'Anomaly\nDetection', 'pos': (6.5, 1.5), 'color': '#D62246'},
    ]
    
    # Draw components
    for comp in components:
        bbox = dict(boxstyle='round,pad=0.5', facecolor=comp['color'], 
                   edgecolor='black', linewidth=2, alpha=0.8)
        ax.text(comp['pos'][0], comp['pos'][1], comp['name'],
               ha='center', va='center', fontsize=10, fontweight='bold',
               bbox=bbox, color='white')
    
    # Draw arrows
    arrows = [
        ((1.5, 7), (3, 7)),
        ((4, 7), (5.5, 7)),
        ((6.5, 7), (8, 7)),
        ((8.5, 6.5), (2, 4.5)),
        ((8.5, 6.5), (4, 4.5)),
        ((8.5, 6.5), (6, 4.5)),
        ((8.5, 6.5), (8, 4.5)),
        ((3, 3.5), (3.5, 2)),
        ((5, 3.5), (6.5, 2)),
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.set_title('OpenUBA System Architecture', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Architecture diagram saved to {save_path}")
    plt.close()


def generate_summary_report(metrics_dict, save_path):
    """Generate a summary report text file"""
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("OpenUBA - Anomaly Detection System\n")
        f.write("Performance Summary Report\n")
        f.write("="*80 + "\n\n")
        
        if not metrics_dict:
            f.write("No metrics available.\n")
            return
        
        f.write(f"Total Models Evaluated: {len(metrics_dict)}\n\n")
        
        # Find best model
        best_model = None
        best_f1 = 0
        
        for model_name, metrics in metrics_dict.items():
            f1 = metrics.get('f1_score', 0)
            if f1 > best_f1:
                best_f1 = f1
                best_model = model_name
        
        if best_model:
            f.write(f"Best Performing Model: {best_model}\n")
            f.write(f"  F1-Score: {best_f1:.4f}\n")
            f.write(f"  Accuracy: {metrics_dict[best_model].get('accuracy', 0):.4f}\n\n")
        
        f.write("Detailed Results:\n")
        f.write("-" * 80 + "\n\n")
        
        for model_name, metrics in metrics_dict.items():
            model_display = metrics.get('model_type', model_name).replace('_', ' ').title()
            f.write(f"{model_display}:\n")
            f.write(f"  Accuracy:   {metrics.get('accuracy', 0):.4f} ({metrics.get('accuracy', 0)*100:.2f}%)\n")
            f.write(f"  Precision:  {metrics.get('precision', 0):.4f}\n")
            f.write(f"  Recall:     {metrics.get('recall', 0):.4f}\n")
            f.write(f"  F1-Score:   {metrics.get('f1_score', 0):.4f}\n")
            
            if 'roc_auc' in metrics and metrics['roc_auc']:
                f.write(f"  ROC-AUC:    {metrics['roc_auc']:.4f}\n")
            
            if 'training_time' in metrics:
                f.write(f"  Train Time: {metrics['training_time']:.2f}s\n")
            
            f.write("\n")
    
    print(f"Summary report saved to {save_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate OpenUBA result visualizations')
    parser.add_argument('--metrics-dir', type=str, default=str(METRICS_DIR),
                       help='Directory containing metrics JSON files')
    parser.add_argument('--output-dir', type=str, default=str(PLOTS_DIR),
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    print("="*80)
    print("OpenUBA - Results Generation")
    print("="*80)
    print(f"\nLoading metrics from: {args.metrics_dir}")
    
    # Load metrics
    metrics_dict = load_metrics_files(args.metrics_dir)
    
    if not metrics_dict:
        print("\nNo metrics files found. Please train models first using train.py")
        return 1
    
    print(f"Found {len(metrics_dict)} model results\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    print("Generating visualizations...")
    
    plot_model_comparison(metrics_dict, output_dir / 'model_comparison.png')
    plot_confusion_matrices(metrics_dict, output_dir)
    plot_metrics_table(metrics_dict, output_dir / 'metrics_table.png')
    plot_class_distribution(metrics_dict, output_dir / 'class_distribution.png')
    create_architecture_diagram(output_dir / 'architecture.png')
    generate_summary_report(metrics_dict, output_dir / 'summary_report.txt')
    
    print("\n" + "="*80)
    print("Results generation completed!")
    print(f"All visualizations saved to: {output_dir}")
    print("="*80 + "\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
