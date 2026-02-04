# OpenUBA - Quick Start Guide

## Overview
OpenUBA is a complete anomaly detection system for insider threat detection with 5 ML/DL models and explainable AI. This guide will help you get started quickly.

## Installation

1. **Clone the repository** (or download the code)
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Workflow

### Step 1: Train Models

Train different models on your dataset:

```bash
# Train Isolation Forest (fast, unsupervised)
python train.py --model isolation_forest --dataset session

# Train Logistic Regression (fast, supervised)
python train.py --model logistic_regression --dataset session

# Train SVC (Support Vector Classifier - uses 10K sample)
python train.py --model svc --dataset session

# Train LSTM Autoencoder (deep learning, temporal patterns)
python train.py --model lstm_autoencoder --dataset session

# Train LSTM-GAN (deep learning, generative)
python train.py --model lstm_gan --dataset session
```

**Output:**
- Trained model saved to `trained_models/`
- Scaler saved to `trained_models/` (for traditional ML models)
- Metrics saved to `results/metrics/`
- Training summary saved to `trained_models/`

### Step 2: Generate XAI Explanations

Create explainability results for trained models:

```bash
# Generate LIME explanation for a specific instance
python xai.py --model isolation_forest --method lime --dataset session --instance 0

# Generate SHAP explanation
python xai.py --model logistic_regression --method shap --dataset session

# Generate both LIME and SHAP
python xai.py --model svc --method all --dataset session

# Batch generate all XAI results
python generate_xai_results.py
```

**Output:**
- LIME plots → `results/xai/lime_*.png`
- SHAP plots → `results/xai/shap_*.png`
- Explanation data → `results/xai/*_explanation.pkl`

### Step 3: Evaluate Models

Evaluate pretrained models on test data:

```bash
# Evaluate the model
python evaluate.py --model trained_models/isolation_forest_session_*.pkl --dataset session
```

**Output:**
- Evaluation metrics saved to `results/metrics/`
- Summary report saved to `results/`

## Command Reference

### Training Options

```bash
python train.py \
    --model {isolation_forest|logistic_regression|svc|lstm_autoencoder|lstm_gan} \
    --dataset {session|day|week|session_time_120|session_time_240} \
    [--output MODEL_NAME] \
    [--test-size 0.3] \
    [--random-state 42] \
    [--no-scaling] \
    [--scaler-type {standard|minmax}]
```

**Notes:**
- SVC automatically samples 10,000 instances for faster training on large datasets
- LSTM models automatically sample 100,000 rows and 50,000 sequences for memory efficiency
- LSTM models use 5 epochs by default (configured in config.py)

### XAI Options

```bash
python xai.py \
    --model {isolation_forest|logistic_regression|svc|lstm_autoencoder|lstm_gan} \
    --method {lime|shap|all} \
    --dataset {session|day|week|session_time_120|session_time_240} \
    [--instance INSTANCE_NUMBER] \
    [--output OUTPUT_NAME]
```

### Evaluation Options

```bash
python evaluate.py \
    --model PATH_TO_MODEL.pkl \
    --dataset {session|day|week|session_time_120|session_time_240} \
    [--scaler PATH_TO_SCALER.pkl] \
    [--output EVAL_NAME] \
    [--test-size 0.3] \
    [--random-state 42] \
    [--no-scaling] \
    [--full-dataset]
```

### Results Generation Options

```bash
# Generate XAI results for all models
python generate_xai_results.py
```

**Output:** Generates 9 XAI configurations (SHAP for Isolation Forest, LIME+SHAP for other 4 models)

## Complete Example

Here's a complete workflow to train all models and generate results:

```bash
# 1. Train all models
python train.py --model isolation_forest --dataset session --output isoforest_best
python train.py --model logistic_regression --dataset session --output logreg_best
python train.py --model svc --dataset session --output svc_best

# 2. Generate XAI explanations
python xai.py --model isolation_forest --method shap --dataset session
python xai.py --model logistic_regression --method shap --dataset session

# 3. Batch generate all XAI visualizations
python generate_xai_results.py

# 4. Check your results
# - View trained models in trained_models/
# - View metrics in results/metrics/
# - View XAI visualizations in results/xai/
```

## Understanding the Output

### Training Output Structure

```
trained_models/
├── model_name.pkl                 # Trained model
├── model_name_scaler.pkl          # Fitted scaler
└── model_name_summary.txt         # Training summary

results/metrics/
└── model_name_metrics.json        # Performance metrics
```

### Metrics JSON Format

```json
{
    "accuracy": 0.942,
    "precision": 0.895,
    "recall": 0.913,
    "f1_score": 0.904,
    "roc_auc": 0.927,
    "confusion_matrix": [[58234, 847], [145, 1523]],
    "model_type": "isolation_forest",
    "dataset": "session",
    "training_time": 2.34,
    "timestamp": "2026-02-04T10:30:00"
}
```

## Tips for Best Results

### 1. Dataset Selection
- **session**: Best for general anomaly detection (recommended)
- **day**: Aggregated daily patterns
- **week**: Long-term behavior patterns
- **session_time_120/240**: Time-windowed sessions

### 2. Model Selection
- **Isolation Forest**: Best overall performance, fast, no labels needed
- **Logistic Regression**: Interpretable, good baseline
- **SVC**: Good for complex patterns, slower training

### 3. Hyperparameter Tuning
Edit `config.py` to customize model parameters:

```python
MODEL_CONFIGS = {
    'isolation_forest': {
        'contamination': 0.01,  # Adjust based on expected anomaly ratio
        'random_state': 16
    }
}
```

### 4. For Demonstrations
1. Train 2-3 models for comparison
2. Run `python generate_xai_results.py` to create all XAI visualizations
3. Use images from `results/xai/` to explain model decisions
4. Share metrics JSON files to show quantitative results

## Troubleshooting

### Issue: "Dataset not found"
**Solution:** Make sure your dataset files are in `ExtractedData/` directory

### Issue: "Model file not found"
**Solution:** Use full path or relative path from project root:
```bash
python evaluate.py --model trained_models/your_model.pkl --dataset session
```

### Issue: "Import errors"
**Solution:** Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: "Scaler not found"
**Solution:** Either:
- Use `--no-scaling` flag
- Specify scaler path: `--scaler path/to/scaler.pkl`
- Ensure scaler was saved during training

## Next Steps

1. **Customize models**: Edit `config.py` and model classes in `src/models/`
2. **Add XAI**: Implement SHAP/LIME explanations (see legacy files for reference)
3. **Deploy**: Create a simple web interface or API
4. **Extend**: Add more models, datasets, or features

## Getting Help

- Check the main [README.md](README.md) for detailed documentation
- Review code comments in source files
- Examine example outputs in `results/` directory

---

**Happy detecting! 🚀**
