# OpenUBA Project Structure

## Core Files

### Training & Evaluation
- `train.py` - Train any of the 5 models
- `evaluate.py` - Evaluate trained models
- `config.py` - Project configuration

### Model Implementations (Your Original Code)
- `isolationforestmodel.py` - Isolation Forest
- `logisticregressionmodel.py` - Logistic Regression
- `supportvectorclassifiermodel.py` - Support Vector Classifier
- `lstmautoencodermodel.py` - LSTM Autoencoder
- `lstmganmodel.py` - LSTM-GAN

### Explainable AI (XAI)
- `xai.py` - XAI CLI interface
- `generate_xai_results.py` - Batch XAI generation

#### LIME Implementations
- `limexailogreg.py` - LIME for Logistic Regression
- `limexaisvc.py` - LIME for SVC
- `limexailstmautoencoder.py` - LIME for LSTM Autoencoder
- `limexailstmgan.py` - LIME for LSTM-GAN

#### SHAP Implementations
- `shapxaiisoforest.py` - SHAP for Isolation Forest
- `shapxailogreg.py` - SHAP for Logistic Regression
- `shapxaisvc.py` - SHAP for SVC
- `shapxailstmautoencoder.py` - SHAP for LSTM Autoencoder
- `shapxailstmgan.py` - SHAP for LSTM-GAN

### Feature Extraction
- `feature_extraction.py` - Extract features from CERT dataset

### Utilities
- `src/models/` - Model factory
- `src/utils/` - Data loaders and metrics

## Documentation

- `README.md` - Main project documentation
- `USAGE.md` - Usage guide
- `MODEL_IMPLEMENTATIONS.md` - Model details
- `QUICK_REFERENCE.md` - Quick command reference
- `XAI_DOCUMENTATION.md` - XAI system guide
- `LICENSE` - Project license
- `requirements.txt` - Python dependencies

## Data

- `ExtractedData/` - Processed CERT r4.2 features
- `MITUEBADataset/` - MIT-DARPA dataset

## Notebooks

- `ipynb_code_files/` - Jupyter notebooks for exploration
  - `UBA_LabelledData.ipynb`
  - `UBA_UnlabelledData.ipynb`

## Output Directories

- `trained_models/` - Saved model files (.pkl, .h5)
- `results/`
  - `metrics/` - Performance metrics (JSON)
  - `xai/` - XAI explanation plots
  - `plots/` - Visualizations

## Configuration

- `.gitignore` - Git ignore rules

---

All outdated and unnecessary files have been removed. The project is clean and ready!
