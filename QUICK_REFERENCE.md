# OpenUBA - Quick Reference Card

## Essential Commands

### Setup
```bash
pip install -r requirements.txt
```

### Train Models
```bash
# Traditional ML models (fast)
python train.py --model isolation_forest --dataset session
python train.py --model logistic_regression --dataset session
python train.py --model svc --dataset session

# Deep learning models (slower)
python train.py --model lstm_autoencoder --dataset session
python train.py --model lstm_gan --dataset session
```

### Generate XAI Explanations
```bash
# Individual model explanation
python xai.py --model isolation_forest --method shap --dataset session

# All models and methods
python generate_xai_results.py
```

### Evaluate Models
```bash
python evaluate.py --model trained_models/MODEL_NAME.pkl --dataset session
```

## File Locations

| Item | Location |
|------|----------|
| Trained models (ML) | `trained_models/*.pkl` |
| Trained models (LSTM) | `trained_models/*.h5` |
| Scalers | `trained_models/*_scaler.pkl` |
| Metrics | `results/metrics/*.json` |
| XAI Results | `results/xai/*.png` |
| Summaries | `trained_models/*_summary.txt` |

## Model Options

| Model | Flag | Best For | Speed |
|-------|------|----------|-------|
| Isolation Forest | `isolation_forest` | Unsupervised anomaly detection | Fast |
| Logistic Regression | `logistic_regression` | Interpretable baseline | Fast |
| SVC | `svc` | Complex patterns | Medium |
| LSTM Autoencoder | `lstm_autoencoder` | Temporal patterns | Slow |
| LSTM-GAN | `lstm_gan` | Generative anomaly detection | Slow |

## Dataset Options

| Dataset | Flag | Description |
|---------|------|-------------|
| Session | `session` | Full session data (recommended) |
| Day | `day` | Daily aggregated patterns |
| Week | `week` | Weekly patterns |
| Session Time 120 | `session_time_120` | 120-min windows |
| Session Time 240 | `session_time_240` | 240-min windows |

## Common Flags

| Flag | Description | Example |
|------|-------------|---------|
| `--output NAME` | Custom output name | `--output best_model` |
| `--test-size 0.3` | Test split ratio | `--test-size 0.2` |
| `--random-state 42` | Random seed | `--random-state 123` |
| `--no-scaling` | Disable scaling | `--no-scaling` |
| `--scaler-type` | Scaler type | `--scaler-type minmax` |

## Typical Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models
python train.py --model isolation_forest --dataset session
python train.py --model logistic_regression --dataset session
python train.py --model svc --dataset session

# 3. Generate XAI explanations
python generate_xai_results.py

# 4. Check results
# - trained_models/ - Model files
# - results/metrics/ - Performance metrics JSON
# - results/xai/ - XAI visualizations PNG
```

## Output Files

After training `model_name`:
```
trained_models/
├── model_name_session_*.pkl           # Trained model
├── model_name_session_*_scaler.pkl    # Fitted scaler
└── model_name_session_*_summary.txt   # Training summary

results/metrics/
└── model_name_session_*_metrics.json  # Performance metrics
```

After `generate_xai_results.py`:
```
results/xai/
├── isolation_forest_shap_session_readme_shap.png
├── logistic_regression_shap_session_readme_shap.png
├── lstm_autoencoder_shap_session_readme_shap.png
├── lstm_gan_lime_session_readme_lime.png
└── lstm_gan_shap_session_readme_shap.png
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | `pip install -r requirements.txt` |
| Dataset not found | Check `ExtractedData/` directory |
| Model not found | Use full path: `trained_models/model.pkl` |
| Scaler not found | Use `--no-scaling` or `--scaler path` |

## Configuration

Edit `config.py` to customize:
- Dataset paths
- Model hyperparameters
- Output directories
- Training parameters

## Documentation

| File | Purpose |
|------|---------|
| `README.md` | Complete documentation |
| `USAGE.md` | Quick start guide |
| `CONTRIBUTING.md` | Development guide |
| `PROJECT_SUMMARY.md` | Restructuring summary |
| `QUICK_REFERENCE.md` | This file |

## Help Commands

```bash
python train.py --help
python evaluate.py --help
python xai.py --help
python generate_xai_results.py --help
```

## Actual Performance Results

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| **Logistic Regression** | **99.81%** | **69.28%** | **31.74%** | **43.53%** | **1.68s** |
| **SVC** | **99.71%** | **37.16%** | **32.93%** | **34.92%** | **0.67s** |
| Isolation Forest | 91.43% | 0.92% | 32.93% | 1.78% | 0.49s |

## For Demonstrations

1. Train 2-3 models: `python train.py --model MODEL --dataset session`
2. Generate XAI: `python generate_xai_results.py`
3. Use XAI visualizations from `results/xai/`
4. Share metrics JSON files from `results/metrics/`
5. Explain model decisions with SHAP/LIME plots

---

**Keep this file handy for quick reference! 📌**
