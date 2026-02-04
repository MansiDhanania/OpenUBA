"""
LSTM-GAN Model Implementation
Author: Original implementation by project author
Adapted for CLI usage
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.stats import gaussian_kde


class LSTMGANModel:
    """LSTM-GAN for anomaly detection - Original Implementation"""
    
    def __init__(self, n_past=100, n_future=5, latent_dim=100, epochs=5, batch_size=32, **kwargs):
        """Initialize LSTM-GAN model"""
        self.n_past = n_past
        self.n_future = n_future
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.generator = None
        self.discriminator = None
        self.model_name = "LSTM-GAN"
        self.is_trained = False
        self.scalers = {}
        self.feature_columns = None
        self.n_features = None
        self.anomaly_scores = None
    
    def _split_data_series(self, series):
        """Split time series into sequences - Original logic"""
        X, y = list(), list()
        for start_window in range(len(series)):
            past_end = start_window + self.n_past
            future_end = past_end + self.n_future
            if future_end > len(series):
                break
            past = series[start_window:past_end, :]
            future = series[past_end:future_end, :]
            X.append(past)
            y.append(future)
        return np.array(X), np.array(y)
    
    def _build_generator(self):
        """Build LSTM Generator - Original architecture"""
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(self.n_past, self.latent_dim)))
        model.add(tf.keras.layers.LSTM(100))
        model.add(tf.keras.layers.Dense(self.n_features * self.n_future, activation='tanh'))
        model.add(tf.keras.layers.Reshape((self.n_future, self.n_features)))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
        return model
    
    def _build_discriminator(self):
        """Build LSTM Discriminator - Original architecture"""
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(self.n_future, self.n_features)))
        model.add(tf.keras.layers.LSTM(100))
        model.add(tf.keras.layers.Dense(50))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        return model
    
    def prepare_data(self, data_path, date_column='date', sample_size=1000):
        """Prepare data for LSTM-GAN - Original data preparation logic"""
        # Read and sample
        df = pd.read_csv(data_path)
        df = df.sample(n=sample_size, random_state=42)
        
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.sort_values(by=date_column)
            df.set_index(date_column, inplace=True)
        
        # Encode categorical features
        l_encode = LabelEncoder()
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = l_encode.fit_transform(df[col].astype(str))
        
        # Split into train/test
        train_size = int(0.7 * len(df))
        df_train, df_test = df[:train_size], df[train_size:]
        
        # Scale data - Original scaling approach
        train_scaled = df_train.copy()
        for col in df_train.columns:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaled_values = scaler.fit_transform(train_scaled[col].values.reshape(-1, 1))
            train_scaled[col] = scaled_values.flatten()
            self.scalers[f'scaler_{col}'] = scaler
        
        test_scaled = df_test.copy()
        for col in df_test.columns:
            scaler = self.scalers[f'scaler_{col}']
            scaled_values = scaler.transform(test_scaled[col].values.reshape(-1, 1))
            test_scaled[col] = scaled_values.flatten()
        
        self.feature_columns = df_train.columns.tolist()
        self.n_features = len(self.feature_columns)
        
        # Create sequences
        X_train, y_train = self._split_data_series(train_scaled.values)
        X_test, y_test = self._split_data_series(test_scaled.values)
        
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], self.n_features))
        y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], self.n_features))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], self.n_features))
        y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], self.n_features))
        
        print(f"LSTM-GAN sequences prepared - Train: {X_train.shape}, Test: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train, X_test=None, y_test=None):
        """Train the LSTM-GAN model - Original training logic"""
        print(f"\nTraining {self.model_name}...")
        
        # Sample data if dataset is too large to fit in memory
        if len(X_train) > 50000:
            print(f"Dataset has {len(X_train)} sequences. Sampling 50,000 for memory efficiency...")
            sample_indices = np.random.choice(len(X_train), size=50000, replace=False)
            X_train = X_train[sample_indices]
            y_train = y_train[sample_indices]
            print(f"Training on {len(X_train)} sampled sequences")
        
        # Build models
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        
        # Compile discriminator - Original configuration
        self.discriminator.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0008),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Combined GAN model
        self.discriminator.trainable = False
        gan_input = tf.keras.layers.Input(shape=(self.n_past, self.latent_dim))
        generated_sequence = self.generator(gan_input)
        gan_output = self.discriminator(generated_sequence)
        self.gan = tf.keras.models.Model(gan_input, gan_output)
        self.gan.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0004),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train - Original training loop
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            # Get batch count
            batch_count = X_train.shape[0] // self.batch_size
            
            for batch_idx in range(batch_count):
                # Real data
                start_idx = batch_idx * self.batch_size
                end_idx = start_idx + self.batch_size
                real_sequences = y_train[start_idx:end_idx]
                
                # Generate fake data
                noise = np.random.normal(0, 1, (self.batch_size, self.n_past, self.latent_dim))
                fake_sequences = self.generator.predict(noise, verbose=0)
                
                # Train discriminator (need to recompile when changing trainable)
                self.discriminator.trainable = True
                self.discriminator.compile(
                    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0008),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                d_loss_real = self.discriminator.train_on_batch(real_sequences, np.ones((self.batch_size, 1)))
                d_loss_fake = self.discriminator.train_on_batch(fake_sequences, np.zeros((self.batch_size, 1)))
                d_loss = 0.5 * np.add(d_loss_real[0], d_loss_fake[0])
                
                # Train generator
                self.discriminator.trainable = False
                noise = np.random.normal(0, 1, (self.batch_size, self.n_past, self.latent_dim))
                g_loss = self.gan.train_on_batch(noise, np.ones((self.batch_size, 1)))
                
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{batch_count} - D Loss: {d_loss:.4f}, G Loss: {g_loss[0]:.4f}")
        
        self.is_trained = True
        print(f"{self.model_name} training completed!")
    
    def predict(self, X_test, y_test):
        """Generate anomaly scores - Original logic"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Get discriminator predictions for all sequences
        self.anomaly_scores = self.discriminator.predict(y_test)
        
        # Generate pseudo labels (top 10% as anomalies)
        threshold = np.percentile(self.anomaly_scores, 90)
        y_pred = (self.anomaly_scores > threshold).astype(int).flatten()
        
        print(f"Anomaly threshold (90th percentile): {threshold:.4f}")
        print(f"Detected anomalies: {np.sum(y_pred)} out of {len(y_pred)}")
        
        return y_pred
    
    def save_model(self, save_path):
        """Save trained models to disk"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save generator and discriminator separately
        base_path = save_path.with_suffix('')
        generator_path = f"{base_path}_generator.h5"
        discriminator_path = f"{base_path}_discriminator.h5"
        
        self.generator.save(generator_path)
        self.discriminator.save(discriminator_path)
        print(f"Models saved: {generator_path}, {discriminator_path}")
    
    def load_model(self, model_path):
        """Load trained models from disk"""
        base_path = Path(model_path).with_suffix('')
        generator_path = f"{base_path}_generator.h5"
        discriminator_path = f"{base_path}_discriminator.h5"
        
        self.generator = tf.keras.models.load_model(generator_path)
        self.discriminator = tf.keras.models.load_model(discriminator_path)
        self.is_trained = True
        print(f"Models loaded from {generator_path}, {discriminator_path}")
    
    def get_feature_importances(self):
        """Feature importances not directly available for LSTM-GAN"""
        return None


# LSTM-GAN Model - Original function preserved for backward compatibility

def lstmganmodel(data):
    """
    Original function - kept for backward compatibility
    Performs complete training and evaluation pipeline
    """
    # Import Necessary Libraries
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import matplotlib.pyplot as plt

    # Read the data
    df = pd.read_csv(data)

    # Sort according to date
    df = df.sample(1000, random_state=200)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date')
    df.set_index('date', inplace=True)

    # Encode the data
    from sklearn.preprocessing import LabelEncoder
    l_encode = LabelEncoder()
    for col in df.columns.values:
        df[col] = l_encode.fit_transform(df[col])

    # Split into training and testing sets
    df_train,df_test = df[1:round(0.7*len(df))], df[round(0.7*len(df)):]

    # Split data according to specified window
    def split_data_series(series, n_past, n_future):
        X, y = list(), list()
        for start_window in range(len(series)):
            past_end = start_window + n_past
            future_end = past_end + n_future
            if future_end > len(series):
                break
            past, future = series[start_window:past_end, :], series[past_end:future_end, :]
            X.append(past)
            y.append(future)
        return np.array(X), np.array(y)

    # Scale the data and reshape
    from sklearn.preprocessing import MinMaxScaler
    train = df_train
    scalers={}
    for i in df_train.columns:
        scaler = MinMaxScaler(feature_range=(-1,1))
        s_s = scaler.fit_transform(train[i].values.reshape(-1,1))
        s_s=np.reshape(s_s,len(s_s))
        scalers['scaler_'+ i] = scaler
        train.loc[:, (i)] = s_s
    test = df_test
    for i in df_train.columns:
        scaler = scalers['scaler_'+i]
        s_s = scaler.transform(test[i].values.reshape(-1,1))
        s_s=np.reshape(s_s,len(s_s))
        scalers['scaler_'+i] = scaler
        test.loc[:, (i)] = s_s

    # Define parameters
    n_past = 100 # use past values
    n_future = 5 # predict future values
    n_features = len(df.columns.values) # number of features

    # Split into X and Y and reshape into 3D
    X_train, y_train = split_data_series(train.values,n_past, n_future)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],n_features))
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], n_features))
    X_test, y_test = split_data_series(test.values,n_past, n_future)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],n_features))
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], n_features))

    latent_dim = 100

    # Create generator
    def generator():
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(100, input_shape=(n_past, latent_dim)))
        model.add(tf.keras.layers.LSTM(100))
        model.add(tf.keras.layers.Dense(n_features*n_future, activation='tanh'))
        model.add(tf.keras.layers.Reshape((n_future, n_features)))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
        return model

    # Create discriminator
    def discriminator():
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(n_future, n_features)))
        model.add(tf.keras.layers.LSTM(100))
        model.add(tf.keras.layers.Dense(50))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        return model

    # Creating Generator
    gen = generator()

    # Creating discriminator
    dis = discriminator()
    dis.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0008), loss='binary_crossentropy')

    # Combining them into a GAN
    dis.trainable = False
    ganInput = tf.keras.Input(shape=(n_past, latent_dim))
    x = gen(ganInput)
    ganOutput = dis(x)
    gan = tf.keras.Model(inputs=ganInput, outputs=ganOutput)
    gan.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0004), loss='binary_crossentropy')

    # Training the GAN
    epochs = 5
    batchSize = 32
    batchCount = X_train.shape[0] // batchSize

    for e in range(1, epochs+1):
        print ('Epoch {}'.format(e))
        for _ in range(batchCount):
            # Real data
            seq = np.random.randint(0, X_train.shape[0], size=batchSize)
            sequencesBatch = y_train[seq]

            # Generate fake data
            noise = np.random.normal(0, 1, size=[batchSize, n_past, latent_dim])
            generatedSequences = gen.predict(noise)

            # Train discriminator
            dis.trainable = True
            dlossReal = dis.train_on_batch(sequencesBatch, np.ones((batchSize, 1)))
            dlossFake = dis.train_on_batch(generatedSequences, np.zeros((batchSize, 1)))
            dLoss = 0.5 * np.add(dlossReal, dlossFake)

            # Train generator
            dis.trainable = False
            noise = np.random.normal(0, 1, size=[batchSize, n_past, latent_dim])
            gLoss = gan.train_on_batch(noise, np.ones((batchSize, 1)))

    # Using it for anomaly detection
    # Let's extract anomaly scores for all data points
    test_size = X_test.shape[0]
    anomaly_scores = dis.predict(y_test)

    # Create pseudo labels on basis of scores being greater or less than 90 percentile
    threshold = np.percentile(anomaly_scores, 90)
    y_pred = anomaly_scores > threshold
    y_pred = y_pred.astype(int)

    # Plot ROC Curve
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import auc
    y_true_binary = np.where(y_test[:, 0, 0] > 0.5, 1, 0)
    y_pred_binary = y_pred.flatten()
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred_binary)
    roc_auc = auc(fpr, tpr)
    print('Area Under Curve: ', roc_auc)
    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--', label='Baseline')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    # Plot Kernel Density estimate
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(anomaly_scores.flatten())
    x_vals = np.linspace(anomaly_scores.min(), anomaly_scores.max(), 1000)
    density = kde(x_vals)
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, density, label='KDE of Anomaly Scores')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.2f}')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title('Kernel Density Estimate of Anomaly Scores')
    plt.legend()
    plt.grid(True)
    plt.show()

    return roc_auc
