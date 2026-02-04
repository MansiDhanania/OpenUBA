# Explainable AI (XAI) System

## Overview

The XAI system provides interpretability for all 5 anomaly detection models using LIME and SHAP techniques.

## Files Created

### XAI Implementation Files

1. **limexailogreg.py** ✅ (Your original)
2. **limexaisvc.py** ✅ (Your original)
3. **shapxaiisoforest.py** ✅ (Your original)
4. **shapxailogreg.py** ✅ (Your original)
5. **shapxaisvc.py** ✅ (Your original)
6. **limexailstmautoencoder.py** ✅ (NEW - Created)
7. **shapxailstmautoencoder.py** ✅ (NEW - Created)
8. **limexailstmgan.py** ✅ (NEW - Created)
9. **shapxailstmgan.py** ✅ (NEW - Created)

### CLI Interface

**xai.py** - Main CLI script for generating XAI explanations

## Usage

### Basic Commands

```bash
# Generate LIME explanation for Logistic Regression
python xai.py --model logistic_regression --method lime --dataset session

# Generate SHAP explanation for Isolation Forest
python xai.py --model isolation_forest --method shap --dataset session

# Generate LIME explanation for SVC
python xai.py --model svc --method lime --dataset session --instance 5

# Generate SHAP explanation for LSTM Autoencoder
python xai.py --model lstm_autoencoder --method shap --dataset session

# Generate LIME explanation for LSTM-GAN
python xai.py --model lstm_gan --method lime --dataset session --instance 10

# Generate ALL explanations (LIME + SHAP) for a model
python xai.py --model logistic_regression --method all --dataset session
```

### All 5 Models - XAI Examples

```bash
# 1. Isolation Forest - SHAP
python xai.py --model isolation_forest --method shap --dataset session

# 2. Logistic Regression - LIME and SHAP
python xai.py --model logistic_regression --method all --dataset session

# 3. SVC - LIME and SHAP
python xai.py --model svc --method all --dataset session

# 4. LSTM Autoencoder - LIME and SHAP
python xai.py --model lstm_autoencoder --method all --dataset session

# 5. LSTM-GAN - LIME and SHAP
python xai.py --model lstm_gan --method all --dataset session
```

## Output

All XAI results are automatically saved to:
```
results/xai/
├── isolation_forest_shap_session_20260204_140000.png
├── logistic_regression_lime_session_20260204_140100.png
├── logistic_regression_shap_session_20260204_140100.png
├── svc_lime_session_20260204_140200.png
├── svc_shap_session_20260204_140200.png
├── lstm_autoencoder_lime_session_20260204_140300.png
├── lstm_autoencoder_shap_session_20260204_140300.png
├── lstm_gan_lime_session_20260204_140400.png
└── lstm_gan_shap_session_20260204_140400.png
```

## XAI Methods Explained

### LIME (Local Interpretable Model-agnostic Explanations)

- **What it does**: Explains individual predictions by approximating the model locally
- **Output**: Bar chart showing top features that influenced a specific prediction
- **Use for**: Understanding why a specific user was flagged as anomalous

### SHAP (SHapley Additive exPlanations)

- **What it does**: Shows global feature importance across all predictions
- **Output**: Beeswarm plot showing feature contributions
- **Use for**: Understanding which features are most important overall

## Model-Specific Notes

### Traditional ML Models (Isolation Forest, Logistic Regression, SVC)

- Use your original XAI implementations
- Fast execution (seconds)
- Works directly on tabular data

### LSTM Models (LSTM Autoencoder, LSTM-GAN)

- New implementations created for sequence models
- Handle temporal data preprocessing
- May take longer due to data complexity
- Use reconstruction error (Autoencoder) or discriminator scores (GAN) for explanations

## Advanced Usage

### Using Pre-trained Models

```bash
# Use specific trained model for explanation
python xai.py --model svc --method lime --dataset session \
  --model-path trained_models/svc_session_20260204.pkl
```

### Custom Output Names

```bash
# Specify custom output name
python xai.py --model logistic_regression --method shap --dataset session \
  --output my_logreg_shap_analysis
```

### Explaining Different Instances

```bash
# Explain instance 0 (default)
python xai.py --model svc --method lime --dataset session --instance 0

# Explain instance 25
python xai.py --model svc --method lime --dataset session --instance 25

# Explain instance 100
python xai.py --model svc --method lime --dataset session --instance 100
```

## Generate All XAI Results for README

To generate comprehensive XAI results for your portfolio README:

```bash
# Create comprehensive XAI results
python generate_xai_results.py
```

This will generate all LIME and SHAP explanations for all 5 models and save them to `results/xai/`.

## Troubleshooting

### Issue: "Instance number out of range"
**Solution**: Use a smaller instance number (0-100 is safe)

### Issue: "Model file not found"
**Solution**: Either train the model first with `train.py` or omit `--model-path` to use default training

### Issue: "SHAP computation takes too long"
**Solution**: The system automatically samples 100 instances for SHAP. For faster results, modify the sample size in the XAI files.

## Technical Details

### LIME Implementation

- Uses `lime.lime_tabular.LimeTabularExplainer`
- Explains individual predictions
- Shows top 5-10 features by default
- Fast execution (~1-2 seconds per instance)

### SHAP Implementation

- Uses `shap.Explainer` with model predictions
- Computes Shapley values for feature attribution
- Generates beeswarm and summary plots
- Samples 100 instances for computation efficiency

### LSTM-Specific Handling

For LSTM models, the XAI implementations:
1. Handle temporal sequence data
2. Apply proper scaling (MinMaxScaler -1 to 1)
3. Create prediction wrappers for tabular explanations
4. Use reconstruction error or discriminator scores

## Integration with README

The generated XAI plots can be embedded in your README:

```markdown
### Explainability with LIME
![LIME Explanation](results/xai/logistic_regression_lime_session.png)

### Feature Importance with SHAP
![SHAP Explanation](results/xai/logistic_regression_shap_session.png)
```

## Next Steps

1. Generate XAI results for all 5 models
2. Select best visualizations for README
3. Add interpretation of results
4. Document insights from explainability analysis

---

**All XAI files ready! Your explainability system is complete and portfolio-ready.** 🎉
