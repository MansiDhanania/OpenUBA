"""
Data loading and preprocessing utilities
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from pathlib import Path


class DataLoader:
    """Handles data loading and preprocessing"""
    
    def __init__(self, test_size=0.3, random_state=42, scale=True):
        self.test_size = test_size
        self.random_state = random_state
        self.scale = scale
        self.scaler = None
        
    def load_data(self, data_path):
        """Load dataset from CSV"""
        print(f"Loading data from {data_path}...")
        data = pd.read_csv(data_path)
        print(f"Loaded {len(data)} samples with {len(data.columns)} features")
        return data
    
    def preprocess_labels(self, data):
        """Convert all insider labels to binary (0: normal, 1: insider)"""
        data_copy = data.copy()
        if 'insider' in data_copy.columns:
            # Convert all insider types (2, 3) to 1
            data_copy.loc[data['insider'] == 2, 'insider'] = 1
            data_copy.loc[data['insider'] == 3, 'insider'] = 1
        return data_copy
    
    def split_data(self, data, target_column='insider'):
        """Split data into train/test sets"""
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"Class distribution - Train: {dict(y_train.value_counts())}, Test: {dict(y_test.value_counts())}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_data(self, X_train, X_test, scaler_type='standard'):
        """Scale features using StandardScaler or MinMaxScaler"""
        if not self.scale:
            return X_train, X_test
            
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Data scaled using {scaler_type} scaler")
        
        return X_train_scaled, X_test_scaled
    
    def save_scaler(self, save_path):
        """Save the fitted scaler"""
        if self.scaler is not None:
            joblib.dump(self.scaler, save_path)
            print(f"Scaler saved to {save_path}")
    
    def load_scaler(self, scaler_path):
        """Load a previously fitted scaler"""
        self.scaler = joblib.load(scaler_path)
        print(f"Scaler loaded from {scaler_path}")
        return self.scaler
    
    def prepare_data(self, data_path, target_column='insider', scaler_type='standard'):
        """Complete data preparation pipeline"""
        # Load data
        data = self.load_data(data_path)
        
        # Preprocess labels
        data = self.preprocess_labels(data)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(data, target_column)
        
        # Scale data
        X_train_scaled, X_test_scaled = self.scale_data(X_train, X_test, scaler_type)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns.tolist()


class LSTMDataLoader(DataLoader):
    """Specialized data loader for LSTM models"""
    
    def __init__(self, n_past=100, n_future=5, **kwargs):
        super().__init__(**kwargs)
        self.n_past = n_past
        self.n_future = n_future
    
    def split_sequences(self, series):
        """Split time series into sequences for LSTM"""
        X, y = [], []
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
    
    def prepare_lstm_data(self, data_path, date_column='date'):
        """Prepare data specifically for LSTM models"""
        # Load and sort by date
        df = pd.read_csv(data_path)
        
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.sort_values(by=date_column)
            df.set_index(date_column, inplace=True)
        
        # Encode categorical features
        from sklearn.preprocessing import LabelEncoder
        l_encode = LabelEncoder()
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = l_encode.fit_transform(df[col].astype(str))
        
        # Split into train/test
        train_size = int(0.7 * len(df))
        df_train, df_test = df[:train_size], df[train_size:]
        
        # Scale data
        scalers = {}
        train_scaled = df_train.copy()
        for col in df_train.columns:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaled_values = scaler.fit_transform(train_scaled[col].values.reshape(-1, 1))
            train_scaled[col] = scaled_values.flatten()
            scalers[f'scaler_{col}'] = scaler
        
        test_scaled = df_test.copy()
        for col in df_test.columns:
            scaler = scalers[f'scaler_{col}']
            scaled_values = scaler.transform(test_scaled[col].values.reshape(-1, 1))
            test_scaled[col] = scaled_values.flatten()
        
        # Create sequences
        X_train, y_train = self.split_sequences(train_scaled.values)
        X_test, y_test = self.split_sequences(test_scaled.values)
        
        n_features = len(df.columns)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
        y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], n_features))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
        y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], n_features))
        
        print(f"LSTM sequences prepared - Train: {X_train.shape}, Test: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, scalers
