"""
Support Vector Classifier Model Implementation
Author: Original implementation by project author
Adapted for CLI usage
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn import metrics
from pathlib import Path


class SVCModel:
    """Support Vector Classifier for anomaly detection - Original Implementation"""
    
    def __init__(self, kernel='linear', probability=True, random_state=16, **kwargs):
        """Initialize SVC model"""
        self.model = svm.SVC(kernel=kernel, probability=probability, random_state=random_state, **kwargs)
        self.model_name = "Support Vector Classifier"
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """Train the SVC model - Original logic with sampling for large datasets"""
        print(f"\nTraining {self.model_name}...")
        
        # Sample data if dataset is too large (SVC is O(n^2) to O(n^3))
        if len(X_train) > 10000:
            print(f"Dataset has {len(X_train)} samples. Sampling 10,000 for faster SVC training...")
            sample_indices = np.random.choice(len(X_train), size=10000, replace=False)
            X_train_sampled = X_train[sample_indices]
            y_train_sampled = y_train.iloc[sample_indices] if hasattr(y_train, 'iloc') else y_train[sample_indices]
            print(f"Training on {len(X_train_sampled)} sampled instances")
        else:
            X_train_sampled = X_train
            y_train_sampled = y_train
        
        self.model.fit(X_train_sampled, y_train_sampled)
        self.is_trained = True
        print(f"{self.model_name} training completed!")
    
    def predict(self, X_test):
        """Predict anomalies - Original logic"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        Y_pred = self.model.predict(X_test)
        return Y_pred
    
    def predict_proba(self, X_test):
        """Get prediction probabilities - Original logic"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_test)
            return probabilities
        else:
            raise ValueError("Model was not initialized with probability=True")
    
    def save_model(self, save_path):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, model_path):
        """Load trained model from disk"""
        self.model = joblib.load(model_path)
        self.is_trained = True
        print(f"Model loaded from {model_path}")
    
    def get_feature_importances(self):
        """Feature importances not directly available for SVC"""
        return None
    
    def plot_roc_curve(self, Y_test, Y_pred_proba, save_path=None):
        """Plot ROC curve - Original logic"""
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot the ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--', label='Baseline')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Support Vector Classifier')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            print(f"ROC curve saved to {save_path}")
        
        return roc_auc


# Legacy function for backward compatibility
def supportvectorclassifier(data):
    """
    Original function - kept for backward compatibility
    Performs complete training and evaluation pipeline
    """
    # Read the data
    data = pd.read_csv(data)

    # Label all insiders as '1'
    data_copy = data.copy()
    data_copy.loc[data['insider']==2, 'insider'] = 1
    data_copy.loc[data['insider']==3, 'insider'] = 1
    data = data_copy

    # Split data into training and testing sets
    from sklearn.model_selection import train_test_split
    # Initialize X and Y variables
    X = data.drop('insider', axis=1)
    Y = data['insider']
    # Training and Testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # Scale the dataset
    from sklearn.preprocessing import StandardScaler
    # Initialize the scaler
    scaler = StandardScaler()
    # Fit to dataset
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Support Vector Classifier
    # Initialize the model
    svc = svm.SVC(kernel='linear', probability=True)
    # Train the model
    svc.fit(X_train, Y_train)
    # Prediction
    Y_pred = svc.predict(X_test)

    # Accuracy
    acc = accuracy_score(Y_test, Y_pred)
    acc = acc*100

    # Confusion Matrix
    cm = metrics.confusion_matrix(Y_test, Y_pred)

    # Classification Report
    target_names = ['normal', 'anomaly'] # normal-0, insider-1
    cr = classification_report(Y_test, Y_pred, target_names=target_names)

    #  ROC Curve
    Y_pred_proba = svc.predict_proba(X_test)[::,1]
    def roc_curve_plot(Y_test, Y_pred_proba):
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_proba)
        roc_auc = auc(fpr, tpr)
        # Plot the ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--', label='Baseline')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        return(roc_auc)

    return acc, cm, cr, roc_curve_plot(Y_test, Y_pred_proba)