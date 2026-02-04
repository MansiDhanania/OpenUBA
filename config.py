"""
Configuration file for OpenUBA
Contains all project-wide settings and paths
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data directories
DATA_DIR = PROJECT_ROOT / "ExtractedData"
MIT_DATA_DIR = PROJECT_ROOT / "MITUEBADataset"

# Output directories
TRAINED_MODELS_DIR = PROJECT_ROOT / "trained_models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
METRICS_DIR = RESULTS_DIR / "metrics"
XAI_DIR = RESULTS_DIR / "xai"

# Dataset paths
DATASETS = {
    'session': DATA_DIR / 'sessionr4.2.csv',
    'day': DATA_DIR / 'dayr4.2.csv',
    'week': DATA_DIR / 'weekr4.2.csv',
    'session_time_120': DATA_DIR / 'sessiontime120r4.2.csv',
    'session_time_240': DATA_DIR / 'sessiontime240r4.2.csv',
    'session_nact_25': DATA_DIR / 'sessionnact25r4.2.csv',
    'session_nact_50': DATA_DIR / 'sessionnact50r4.2.csv',
    'mit_train': MIT_DATA_DIR / 'train_data.csv',
    'mit_test': MIT_DATA_DIR / 'A_test_data.csv'
}

# Model configurations
MODEL_CONFIGS = {
    'isolation_forest': {
        'random_state': 16,
        'contamination': 'auto'
    },
    'logistic_regression': {
        'solver': 'lbfgs',
        'max_iter': 1000,
        'random_state': 16
    },
    'svc': {
        'kernel': 'linear',
        'probability': True,
        'random_state': 16
    },
    'lstm_autoencoder': {
        'n_past': 100,
        'n_future': 5,
        'units': 100,
        'epochs': 5,
        'batch_size': 32
    },
    'lstm_gan': {
        'latent_dim': 100,
        'epochs': 5,
        'batch_size': 32
    }
}

# Training parameters
TRAIN_TEST_SPLIT = 0.3
RANDOM_STATE = 42
SCALE_DATA = True

# XAI parameters
SHAP_SAMPLES = 100
LIME_SAMPLES = 5000

# Create directories if they don't exist
for directory in [TRAINED_MODELS_DIR, RESULTS_DIR, PLOTS_DIR, METRICS_DIR, XAI_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
