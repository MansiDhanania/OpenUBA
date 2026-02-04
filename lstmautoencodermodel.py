"""
LSTM Autoencoder Model Implementation
Author: Original implementation by project author
Adapted for CLI usage
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, classification_report


class LSTMAutoencoderModel:
    """LSTM Autoencoder for anomaly detection - Original Implementation"""
    
    def __init__(self, n_past=100, n_future=5, units=100, epochs=5, batch_size=1000, **kwargs):
        """Initialize LSTM Autoencoder model"""
        self.n_past = n_past
        self.n_future = n_future
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.model_name = "LSTM Autoencoder"
        self.is_trained = False
        self.scalers = {}
        self.feature_columns = None
        self.n_features = None
    
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
    
    def _build_model(self):
        """Build LSTM 2-layer AutoEncoder - Original architecture"""
        # Encoder
        encoder_inputs = tf.keras.layers.Input(shape=(self.n_past, self.n_features))
        encoder_l1 = tf.keras.layers.LSTM(self.units, return_sequences=True, return_state=True)
        encoder_outputs1 = encoder_l1(encoder_inputs)
        encoder_states1 = encoder_outputs1[1:]
        encoder_l2 = tf.keras.layers.LSTM(self.units, return_state=True)
        encoder_outputs2 = encoder_l2(encoder_outputs1[0])
        encoder_states2 = encoder_outputs2[1:]
        
        # Decoder
        decoder_inputs = tf.keras.layers.RepeatVector(self.n_future)(encoder_outputs2[0])
        decoder_l1 = tf.keras.layers.LSTM(self.units, return_sequences=True)(decoder_inputs, initial_state=encoder_states1)
        decoder_l2 = tf.keras.layers.LSTM(self.units, return_sequences=True)(decoder_l1, initial_state=encoder_states2)
        decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.n_features))(decoder_l2)
        
        # Add Encoder and Decoder
        model = tf.keras.models.Model(encoder_inputs, decoder_outputs2)
        return model
    
    def prepare_data(self, data_path, date_column='date'):
        """Prepare data for LSTM - Original data preparation logic"""
        # Read and sort by date
        df = pd.read_csv(data_path)
        
        # Sample data early if dataset is too large (memory constraint)
        if len(df) > 100000:
            print(f"Dataset has {len(df)} rows. Sampling 100,000 for memory efficiency...")
            df = df.sample(n=100000, random_state=42)
            print(f"Using {len(df)} sampled rows")
        
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
        
        print(f"LSTM sequences prepared - Train: {X_train.shape}, Test: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train, X_test=None, y_test=None):
        """Train the LSTM Autoencoder model - Original training logic"""
        print(f"\nTraining {self.model_name}...")
        
        # Sample data if dataset is too large to fit in memory
        if len(X_train) > 50000:
            print(f"Dataset has {len(X_train)} sequences. Sampling 50,000 for memory efficiency...")
            sample_indices = np.random.choice(len(X_train), size=50000, replace=False)
            X_train = X_train[sample_indices]
            y_train = y_train[sample_indices]
            print(f"Training on {len(X_train)} sampled sequences")
        
        # Build model
        self.model = self._build_model()
        
        # Compile - Original configuration
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.Huber()
        )
        
        # Callbacks - Original setup
        reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)
        
        # Train
        validation_data = (X_test, y_test) if X_test is not None and y_test is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            validation_data=validation_data,
            batch_size=self.batch_size,
            verbose=1,
            callbacks=[reduce_lr]
        )
        
        self.is_trained = True
        print(f"{self.model_name} training completed!")
    
    def predict(self, X_test):
        """Predict sequences - Original logic"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        pred = self.model.predict(X_test)
        return pred
    
    def save_model(self, save_path):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save Keras model
        if save_path.suffix != '.h5':
            save_path = save_path.with_suffix('.h5')
        
        self.model.save(save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, model_path):
        """Load trained model from disk"""
        self.model = tf.keras.models.load_model(model_path)
        self.is_trained = True
        print(f"Model loaded from {model_path}")
    
    def get_feature_importances(self):
        """Feature importances not directly available for LSTM"""
        return None


# LSTM AUTO-ENCODER MODEL - Original function preserved for backward compatibility

def lstmautoencoder(data):
    """
    Original function - kept for backward compatibility
    Performs complete training and evaluation pipeline
    """
    # Import Necessary Libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import seaborn as sns
    from sklearn import preprocessing

    # Read the data
    df = pd.read_csv(data)

    # Sort according to date
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

    # LSTM 2-layer AutoEncoder
    def lstmmodel():
        # Encoder
        encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
        encoder_l1 = tf.keras.layers.LSTM(100,return_sequences = True, return_state=True)
        encoder_outputs1 = encoder_l1(encoder_inputs)
        encoder_states1 = encoder_outputs1[1:]
        encoder_l2 = tf.keras.layers.LSTM(100, return_state=True)
        encoder_outputs2 = encoder_l2(encoder_outputs1[0])
        encoder_states2 = encoder_outputs2[1:]
        # Decoder
        decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs2[0])
        decoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
        decoder_l2 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_l1,initial_state = encoder_states2)
        decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(decoder_l2)
        # Add Encoder and Decoder
        model = tf.keras.models.Model(encoder_inputs,decoder_outputs2)
        return model

    # Training features
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
    tensorboard = TensorBoard(log_dir = 'logs')
    checkpoint = ModelCheckpoint("lstmautoencodermodel.h5", monitor = "val_loss", save_best_only = True, mode = "auto", verbose = 1)

    # Train model according to specified features
    reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)
    model = lstmmodel()
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber())
    history = model.fit(X_train,y_train,epochs=10,validation_data=(X_test,y_test),batch_size=1000,verbose=0,callbacks=[tensorboard, checkpoint, reduce_lr])

    # Plot for Loss
    def plot_loss(history):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(["loss","val_loss"])
        plt.title('Loss Vs Val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

    # Prediction
    pred = model.predict(X_test)

    # Scaling predicted data
    from sklearn.preprocessing import StandardScaler
    for index,i in enumerate(df_train.columns):
        scaler = StandardScaler()
        scaler = scaler.fit(pred[:,:,index].reshape(-1,1))
        pred[:,:,index] = scaler.inverse_transform(pred[:,:,index])
        y_train[:,:,index] = scaler.inverse_transform(y_train[:,:,index])
        y_test[:,:,index] = scaler.inverse_transform(y_test[:,:,index])

    # Mean Absolute and Mean Squared Error
    def get_mae_mse(pred, y_test):
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        for index,i in enumerate(df_train.columns):
            print(i)
            for j in range(1,6):
                print("Day ",j,":")
                print("MAE: ", mean_absolute_error(y_test[:,j-1,index],pred[:,j-1,index]),end=", ")
                print("MSE: ", mean_squared_error(y_test[:,j-1,index],pred[:,j-1,index]),end=", ")
            print()
            print()

    # Classification metrics
    import tensorflow as tf
    from tensorflow.keras.metrics import Accuracy, Precision, Recall
    from sklearn.metrics import confusion_matrix, classification_report

    def get_classification_metrics(y_true, y_pred):

        # Calculate the confusion matrix
        confusion_matrix = tf.math.confusion_matrix(y_true, y_pred)

        # Convert continuous target values to binary values using a threshold
        y_true_binary = np.where(y_true > 0.5, 1, 0)
        y_pred_binary = np.where(y_pred > 0.5, 1, 0)

        # Calculate the accuracy
        accuracy = tf.keras.metrics.Accuracy()
        accuracy.update_state(y_true_binary, y_pred_binary)
        accuracy_value = accuracy.result().numpy()
        print('Accuracy: ', accuracy_value)

        # Create a classification report
        report = classification_report(y_true_binary, y_pred_binary, output_dict=True)

        # Calculate other classification metrics, using False class for anomalies
        precision = report['1']['precision']
        recall = report['1']['recall']
        f1 = report['1']['f1-score']
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 score:", f1)

        # ROC Curve
        from sklearn.metrics import roc_curve
        from sklearn.metrics import roc_auc_score
        from sklearn.metrics import auc
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

        return confusion_matrix

    # Get Classification Report for each column
    def get_classification_report(pred, y_test):
        for index,i in enumerate(df_train.columns):
            if i=='activity':
                print(i)
                for j in range(1,6):
                    if j==1:
                        print("Day ",j,":")
                        print("Confusion Matrix: ", get_classification_metrics(y_test[:,j-1,index],pred[:,j-1,index]), end=", ")

    return plot_loss(history), get_mae_mse(pred, y_test), get_classification_report(pred, y_test)
