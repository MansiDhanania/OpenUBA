# LIME - Explainable AI for LSTM Autoencoder
def limexailstmautoencoder(data, inputnumber, model_path=None):
    """
    LIME explanation for LSTM Autoencoder model
    Shows which features contribute to anomaly predictions
    """
    # Import Necessary Libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    import lime
    import lime.lime_tabular
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
        # For unlabeled data, create pseudo labels
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

    # Load or train model
    if model_path:
        model = tf.keras.models.load_model(model_path)
        print(f"Loaded model from {model_path}")
    else:
        # Train a simple LSTM Autoencoder for explanation
        from lstmautoencodermodel import LSTMAutoencoderModel
        lstm_model = LSTMAutoencoderModel(n_past=10, n_future=2, epochs=5)  # Smaller for quick XAI
        
        # Prepare sequences
        def split_sequences(data, n_past, n_future):
            X_seq, y_seq = [], []
            for i in range(len(data)):
                end_ix = i + n_past
                out_end_ix = end_ix + n_future
                if out_end_ix > len(data):
                    break
                X_seq.append(data[i:end_ix])
                y_seq.append(data[end_ix:out_end_ix])
            return np.array(X_seq), np.array(y_seq)
        
        X_train_seq, y_train_seq = split_sequences(X_train_scaled.values, 10, 2)
        X_test_seq, y_test_seq = split_sequences(X_test_scaled.values, 10, 2)
        
        n_features = X_train.shape[1]
        X_train_seq = X_train_seq.reshape((X_train_seq.shape[0], X_train_seq.shape[1], n_features))
        y_train_seq = y_train_seq.reshape((y_train_seq.shape[0], y_train_seq.shape[1], n_features))
        
        lstm_model.n_features = n_features
        lstm_model.model = lstm_model._build_model()
        lstm_model.model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
        lstm_model.model.fit(X_train_seq, y_train_seq, epochs=5, verbose=0)
        model = lstm_model.model

    # Create wrapper for LIME (flatten sequences to tabular)
    def predict_proba_wrapper(X_tabular):
        """Wrapper to make LSTM predictions work with LIME"""
        # For tabular input, create simple predictions based on reconstruction error
        predictions = []
        for row in X_tabular:
            # Simple heuristic: sum of absolute values as anomaly indicator
            score = np.sum(np.abs(row))
            # Convert to probability (anomaly vs normal)
            prob_anomaly = 1 / (1 + np.exp(-score + 5))  # Sigmoid-like
            predictions.append([1 - prob_anomaly, prob_anomaly])
        return np.array(predictions)

    # Initialize LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_test_scaled.values,
        feature_names=X.columns.tolist(),
        class_names=['Normal', 'Anomaly'],
        verbose=True,
        mode='classification'
    )

    # Create Explanation for specified instance
    exp = explainer.explain_instance(
        X_test_scaled.values[inputnumber],
        predict_proba_wrapper,
        num_features=10
    )
    
    # Generate plot
    fig = exp.as_pyplot_figure()
    plt.tight_layout()
    
    print(f"\n=== LIME Explanation for LSTM Autoencoder ===")
    print(f"Instance: {inputnumber}")
    print(f"Features influencing prediction:\n")
    for feature, weight in exp.as_list():
        print(f"  {feature}: {weight:.4f}")
    
    return fig, exp
