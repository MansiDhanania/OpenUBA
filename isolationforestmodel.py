"""
Isolation Forest Model Implementation
Author: Original implementation by project author
Adapted for CLI usage
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import metrics
from pathlib import Path


class IsolationForestModel:
    """Isolation Forest for anomaly detection - Original Implementation"""
    
    def __init__(self, random_state=16, contamination='auto', **kwargs):
        """Initialize Isolation Forest model"""
        self.model = IsolationForest(random_state=random_state, contamination=contamination, **kwargs)
        self.model_name = "Isolation Forest"
        self.is_trained = False
    
    def train(self, X_train, y_train=None):
        """
        Train the Isolation Forest model
        Original logic: Unsupervised learning (y_train not used)
        """
        print(f"\nTraining {self.model_name}...")
        # Isolation Forest doesn't use labels
        self.model.fit(X_train)
        self.is_trained = True
        print(f"{self.model_name} training completed!")
    
    def predict(self, X_test):
        """
        Predict anomalies
        Original logic: Convert -1 (outliers) to 1 (anomaly), 1 (inliers) to 0 (normal)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Get predictions: 1 for inliers, -1 for outliers
        Y_pred = self.model.predict(X_test)
        
        # Convert to binary: 0 for normal, 1 for anomaly
        Y_pred[Y_pred == 1] = 0
        Y_pred[Y_pred == -1] = 1
        
        return Y_pred
    
    def predict_proba(self, X_test):
        """Get anomaly scores (not true probabilities for Isolation Forest)"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Get anomaly scores (lower scores = more anomalous)
        scores = self.model.score_samples(X_test)
        # Convert to pseudo-probabilities (higher = more anomalous)
        scores_normalized = 1 / (1 + np.exp(scores))
        
        return scores_normalized
    
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
        """Feature importances not directly available for Isolation Forest"""
        return None


# Legacy function for backward compatibility
def isolationforest(data):
    """
    Original function - kept for backward compatibility
    Performs complete training and evaluation pipeline
    """
    import matplotlib.pyplot as plt
    
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

    # Isolation Forest
    # Initialize the model
    isoforest = IsolationForest(random_state=16)
    # Train the model
    isoforest.fit(X_train, Y_train)
    # Prediction
    Y_pred = isoforest.predict(X_test)
    Y_pred[Y_pred == 1] = 0
    Y_pred[Y_pred == -1] = 1

    # Accuracy
    acc = accuracy_score(Y_test, Y_pred)
    acc = acc*100

    # Confusion Matrix
    cm = metrics.confusion_matrix(Y_test, Y_pred)

    # Classification Report
    target_names = ['normal', 'anomaly'] # normal-0, insider-1
    cr = classification_report(Y_test, Y_pred, labels=np.arange(0,len(target_names),1), target_names=target_names)

    #  ROC Curve
    def roc_curve_plot():
        return('No ROC Curve for Isolation Forest')

    return acc, cm, cr, roc_curve_plot()