"""
Metrics calculation and evaluation utilities
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)
import json
from pathlib import Path


class MetricsCalculator:
    """Calculate and store model performance metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate comprehensive metrics"""
        # Convert predictions to binary if needed
        y_pred_binary = np.where(y_pred == -1, 1, y_pred) if -1 in y_pred else y_pred
        y_pred_binary = np.where(y_pred_binary == 1, 1, 0)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred_binary),
            'precision': precision_score(y_true, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true, y_pred_binary, zero_division=0),
            'f1_score': f1_score(y_true, y_pred_binary, zero_division=0),
        }
        
        # Add ROC-AUC if probability predictions available
        if y_pred_proba is not None:
            try:
                if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
                    # Binary classification with probabilities
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except:
                metrics['roc_auc'] = None
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_binary)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Calculate additional metrics from confusion matrix
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_positives'] = int(tp)
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Classification report
        target_names = ['normal', 'anomaly']
        cr = classification_report(
            y_true, y_pred_binary, 
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )
        metrics['classification_report'] = cr
        
        return metrics
    
    def print_metrics(self, metrics, model_name="Model"):
        """Print metrics in a formatted way"""
        print(f"\n{'='*60}")
        print(f"{model_name} Performance Metrics")
        print(f"{'='*60}")
        print(f"Accuracy:   {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision:  {metrics['precision']:.4f}")
        print(f"Recall:     {metrics['recall']:.4f}")
        print(f"F1-Score:   {metrics['f1_score']:.4f}")
        
        if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
            print(f"ROC-AUC:    {metrics['roc_auc']:.4f}")
        
        if 'specificity' in metrics:
            print(f"Specificity: {metrics['specificity']:.4f}")
        
        print(f"\nConfusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        print(f"                Predicted")
        print(f"              Normal  Anomaly")
        print(f"Actual Normal   {cm[0][0]:6d}  {cm[0][1]:6d}")
        print(f"       Anomaly  {cm[1][0]:6d}  {cm[1][1]:6d}")
        
        if 'true_positives' in metrics:
            print(f"\nTP: {metrics['true_positives']}, TN: {metrics['true_negatives']}, "
                  f"FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}")
        
        print(f"{'='*60}\n")
    
    def save_metrics(self, metrics, save_path, model_name="model"):
        """Save metrics to JSON file"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        metrics_json = self._convert_to_json_serializable(metrics)
        metrics_json['model_name'] = model_name
        
        with open(save_path, 'w') as f:
            json.dump(metrics_json, f, indent=4)
        
        print(f"Metrics saved to {save_path}")
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to Python native types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    def load_metrics(self, metrics_path):
        """Load metrics from JSON file"""
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return metrics
    
    def calculate_roc_data(self, y_true, y_pred_proba):
        """Calculate ROC curve data"""
        if y_pred_proba is None:
            return None
        
        try:
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
                fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
            else:
                fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            
            roc_auc = auc(fpr, tpr)
            
            return {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist(),
                'auc': float(roc_auc)
            }
        except:
            return None


def compare_models(metrics_dict):
    """
    Compare multiple models' metrics
    
    Args:
        metrics_dict: Dictionary with model names as keys and metrics as values
    
    Returns:
        DataFrame with comparison results
    """
    comparison_data = []
    
    for model_name, metrics in metrics_dict.items():
        row = {
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}",
        }
        
        if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
            row['ROC-AUC'] = f"{metrics['roc_auc']:.4f}"
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    return df
