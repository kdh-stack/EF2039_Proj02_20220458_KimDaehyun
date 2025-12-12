import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class SolarDataset(Dataset):
    """
    Custom Dataset class for loading Year-based Solar Radiation Data.
    - Loads specific CSV files (e.g., '2017.csv', '2018.csv').
    - Combines date columns ('Year', 'Month', etc.) into datetime objects.
    - Prepares input features and the target variable (Real DNI) for the LSTM model.
    """
    def __init__(self, csv_file_path, window_size=96):
        """
        Args:
            csv_file_path (str): Relative path to the CSV file (e.g., 'data/2017.csv').
            window_size (int): Length of the look-back period (default: 96 => 15min X 96 = 24 hours).
        """
        self.window_size = window_size
        
        # 1. Data Loading
        try:
            # Load the CSV file into a Pandas DataFrame
            raw_df = pd.read_csv(csv_file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ File not found at '{csv_file_path}'. Please check the 'data' directory.")

        # 2. Date Parsing
        # The new dataset has separate columns: 'Year', 'Month', 'Day', 'Hour', 'Minute'.
        # We combine them into a single 'datetime' object for sorting.
        try:
            raw_df['datetime'] = pd.to_datetime(raw_df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
        except Exception as e:
            print(f"⚠️ Error parsing dates. Please check column names in the CSV file. Found: {raw_df.columns}")
            raise e

        # 3. Sort by Time
        # Ensure the data is in chronological order (Past -> Future).
        raw_df = raw_df.sort_values('datetime').reset_index(drop=True)
        
        # 4. Feature Selection
        # We select specific columns based on the dataset structure.
        # Target: 'DNI' (Direct Normal Irradiance) - Column L in your screenshot
        # Inputs: Basic weather metrics (Temperature, Pressure, Humidity, Wind Speed)
        
        target_col = 'DNI'
        input_cols = ['Temperature', 'Pressure', 'Relative Humidity', 'Wind Speed']
        
        # Combine target and inputs into a single list
        selected_cols = [target_col] + input_cols
        
        # Validation: Check if these columns actually exist in the CSV
        for col in selected_cols:
            if col not in raw_df.columns:
                raise ValueError(f"❌ Column '{col}' not found in CSV. Available columns: {list(raw_df.columns)}")

        # Select columns and drop rows with missing values (NaN)
        raw_df = raw_df[selected_cols].dropna()

        # Convert to NumPy array (Float32 for PyTorch compatibility)
        self.data = raw_df[selected_cols].values.astype(np.float32)

        # 5. Normalization
        # Scale all features to the [0, 1] range for stable LSTM training.
        self.scaler = MinMaxScaler()
        self.data_normalized = self.scaler.fit_transform(self.data)
        
        self.data_processed = self.data_normalized

    def __len__(self):
        # Total samples = Total data length - Window size
        # Ensures we don't go out of bounds when creating sequences.
        return max(0, len(self.data_processed) - self.window_size)

    def __getitem__(self, idx):
        """
        Returns a single sample (Input Sequence, Target Value).
        """
        # Input (x): Sequence of past 'window_size' hours
        # Includes all features (DNI history + Weather history)
        x_window = self.data_processed[idx : idx + self.window_size]
        
        # Target (y): DNI value at the NEXT time step (Future 1 hour)
        # Note: 'DNI' is at index 0 because 'target_col' was first in 'selected_cols'
        y_target = self.data_processed[idx + self.window_size, 0] 
        
        # Convert to PyTorch Tensors
        return torch.tensor(x_window), torch.tensor(y_target)