import torch
import torch.nn as nn

class DNIPredictionLSTM(nn.Module):
    """
    LSTM-based Neural Network for Short-term Direct Normal Irradiance (DNI) Forecasting.
    This model captures temporal dependencies in weather data to predict future solar radiation.
    """
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=1, num_layers=2, dropout_prob=0.2):
        """
        Args:
            input_dim (int): Number of input features (e.g., Radiation, Temp, Humidity, etc. = 6).
            hidden_dim (int): Number of features in the hidden state (e.g., 64).
            output_dim (int): Number of output values (e.g., 1 for Radiation).
            num_layers (int): Number of stacked LSTM layers.
            dropout_prob (float): Dropout probability to prevent overfitting.
        """
        super(DNIPredictionLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 1. LSTM Layer
        # batch_first=True ensures input shape is (batch_size, seq_length, features)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob
        )

        # 2. Fully Connected Layer (Regressor)
        # Maps the final hidden state of the LSTM to the target output value.
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input sequence data. 
                              Shape: (batch_size, sequence_length, input_dim)
        
        Returns:
            torch.Tensor: Predicted value. 
                          Shape: (batch_size, output_dim)
        """
        # Initialize hidden state (h0) and cell state (c0) 
        # (PyTorch automatically initializes them to zeros if not provided, so we skip explicit init)

        # Pass input through LSTM layers
        # lstm_out shape: (batch_size, sequence_length, hidden_dim)
        lstm_out, (hn, cn) = self.lstm(x)

        # Select the output of the LAST time step for prediction
        # We are interested in the final state after processing the entire sequence.
        last_time_step_out = lstm_out[:, -1, :]

        # Pass through the linear layer to get the final prediction
        prediction = self.fc(last_time_step_out)

        return prediction