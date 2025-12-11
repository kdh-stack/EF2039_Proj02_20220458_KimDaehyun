import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class SolarDataset(Dataset):
    """
    Custom Dataset class to load and preprocess the Kaggle 'Solar Radiation' dataset.
    It constructs time-series sequences (past N hours) to predict future solar radiation.
    """
    def __init__(self, csv_file_path, window_size=24, is_train=True, train_ratio=0.8):
        """
        Args:
            csv_file_path (str): Path to the CSV data file (e.g., 'data/SolarEnergy.csv').
            window_size (int): Length of the look-back period for training (default: 24 hours).
            is_train (bool): Flag to indicate training mode (True) or testing mode (False).
            train_ratio (float): Ratio of data used for training (default: 0.8).
        """
        self.window_size = window_size
        
        # 1. Data Loading
        # Load the raw data into a Pandas DataFrame.
        raw_df = pd.read_csv(csv_file_path)
        
        # 2. Data Sorting
        # Sort by UNIXTime in ascending order to ensure correct time-series sequence.
        raw_df = raw_df.sort_values('UNIXTime').reset_index(drop=True)
        
        # 3. Feature Selection
        # Selected features: Radiation, Temperature, Pressure, Humidity, WindDirection, Speed
        feature_columns = ['Radiation', 'Temperature', 'Pressure', 'Humidity', 'WindDirection(Degrees)', 'Speed']
        self.data = raw_df[feature_columns].values.astype(np.float32)
        
        # 4. Data Normalization
        # Scale data to the range [0, 1] as LSTMs are sensitive to the magnitude of input data.
        self.scaler = MinMaxScaler()
        self.data_normalized = self.scaler.fit_transform(self.data)
        
        # 5. Train/Test Split
        # Split data sequentially (Shuffle=False) to preserve the temporal order.
        split_index = int(len(self.data_normalized) * train_ratio)
        
        if is_train:
            self.data_processed = self.data_normalized[:split_index]
        else:
            self.data_processed = self.data_normalized[split_index:]
            
    def __len__(self):
        # Total samples = Total time steps - Window size
        # Example: If total data is 100 and window is 24, available samples are 76.
        return len(self.data_processed) - self.window_size

    def __getitem__(self, idx):
        """
        Retrieves a single sample of input (x) and target (y) for the model.
        """
        # Input (x): Data sequence from 'idx' to 'idx + window_size' (Past 24 hours)
        x_window = self.data_processed[idx : idx + self.window_size]
        
        # Target (y): 'Radiation' value at