"""
Model factory for creating model instances
Uses original model implementations from project root
"""
import sys
from pathlib import Path

# Add project root to path to import original models
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from isolationforestmodel import IsolationForestModel
from logisticregressionmodel import LogisticRegressionModel
from supportvectorclassifiermodel import SVCModel
from lstmautoencodermodel import LSTMAutoencoderModel
from lstmganmodel import LSTMGANModel


def get_model(model_type, **kwargs):
    """
    Factory function to create model instances
    Uses original implementations by project author
    
    Args:
        model_type: Type of model ('isolation_forest', 'logistic_regression', 'svc', 
                    'lstm_autoencoder', 'lstm_gan')
        **kwargs: Additional parameters for the model
    
    Returns:
        model: Model instance
    """
    models = {
        'isolation_forest': IsolationForestModel,
        'logistic_regression': LogisticRegressionModel,
        'svc': SVCModel,
        'lstm_autoencoder': LSTMAutoencoderModel,
        'lstm_gan': LSTMGANModel
    }
    
    if model_type not in models:
        available = ', '.join(models.keys())
        raise ValueError(f"Unknown model type: {model_type}. Available: {available}")
    
    return models[model_type](**kwargs)


def get_available_models():
    """Get list of available model types"""
    return ['isolation_forest', 'logistic_regression', 'svc', 'lstm_autoencoder', 'lstm_gan']
