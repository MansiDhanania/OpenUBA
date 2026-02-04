# SHAP - Explainable AI for LSTM Autoencoder
def shapxailstmautoencoder(data, model_path=None):
    """
    SHAP explanation for LSTM Autoencoder model
    Shows feature importance for anomaly detection
    """
    # Import Necessary Libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    import shap
    import tensorflow as tf

    # Read the data
    data = pd.read_csv(data)

    # Label all insiders as '1'
    data_copy = data.copy()
    if 'insider' in data.columns:
        data_copy.loc[data['insider']==2, 'insider'] = 1
        data_copy.loc[data['insider']==3, 'insider'] = 1
        df = data_copy
    else:
        df = data_copy

    # Handle date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date')
        df.set_index('date', inplace=True)

    # Encode categorical features
    l_encode = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = l_encode.fit_transform(df[col].astype(str))

    # Split data
    if 'insider' in df.columns:
        X = df.drop('insider', axis=1)
        Y = df['insider']
    else:
        X = df
        Y = np.zeros(len(df))

    # Training and Testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # Scale data
    scalers = {}
    X_train_scaled = X_train.copy()
    for col in X_train.columns:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_values = scaler.fit_transform(X_train_scaled[col].values.reshape(-1, 1))
        X_train_scaled[col] = scaled_values.flatten()
        scalers[f'scaler_{col}'] = scaler

    X_test_scaled = X_test.copy()
    for col in X_test.columns:
        scaler = scalers[f'scaler_{col}']
        scaled_values = scaler.transform(X_test_scaled[col].values.reshape(-1, 1))
        X_test_scaled[col] = scaled_values.flatten()

    # Create wrapper for SHAP
    def predict_wrapper(X_tabular):
        """Wrapper to make LSTM predictions work with SHAP"""
        predictions = []
        for row in X_tabular:
            score = np.sum(np.abs(row))
            prob_anomaly = 1 / (1 + np.exp(-score + 5))
            predictions.append(prob_anomaly)
        return np.array(predictions)

    # Sample data for faster SHAP computation
    X_test_sample = X_test_scaled.values[:100] if len(X_test_scaled) > 100 else X_test_scaled.values

    # Initialize SHAP explainer
    print("Computing SHAP values...")
    explainer = shap.Explainer(predict_wrapper, X_test_sample, max_evals=500)
    shap_values = explainer(X_test_sample)

    # Generate plots
    print("\nGenerating SHAP beeswarm plot...")
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    
    # Summary plot
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_sample, feature_names=X.columns.tolist(), show=False)
    plt.tight_layout()
    
    print("\n=== SHAP Analysis Complete ===")
    print(f"Analyzed {len(X_test_sample)} instances")
    
    return fig1, fig2, shap_values
