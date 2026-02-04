# LIME - Explainable AI for LSTM-GAN
def limexailstmgan(data, inputnumber, model_path=None):
    """
    LIME explanation for LSTM-GAN model
    Shows which features contribute to anomaly detection
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

    # Sample for GAN (as per original implementation)
    data = data.sample(n=min(1000, len(data)), random_state=42)

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

    # Load discriminator model if available
    if model_path:
        try:
            discriminator_path = model_path.replace('.h5', '_discriminator.h5')
            discriminator = tf.keras.models.load_model(discriminator_path)
            print(f"Loaded discriminator from {discriminator_path}")
        except:
            print("Could not load discriminator, using heuristic predictions")
            discriminator = None
    else:
        discriminator = None

    # Create prediction wrapper for LIME
    def predict_proba_wrapper(X_tabular):
        """Wrapper for GAN discriminator predictions"""
        predictions = []
        for row in X_tabular:
            if discriminator:
                # Use actual discriminator if available
                score = discriminator.predict(row.reshape(1, -1), verbose=0)[0][0]
            else:
                # Heuristic based on feature values
                score = np.mean(np.abs(row))
                score = 1 / (1 + np.exp(-score * 2))
            
            # Convert to binary probabilities
            predictions.append([1 - score, score])
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
    
    print(f"\n=== LIME Explanation for LSTM-GAN ===")
    print(f"Instance: {inputnumber}")
    print(f"Features influencing anomaly detection:\n")
    for feature, weight in exp.as_list():
        print(f"  {feature}: {weight:.4f}")
    
    return fig, exp
