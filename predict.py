import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import custom modules
from src.dataset import SolarDataset
from src.model import DNIPredictionLSTM

def predict_future():
    """
    Loads the trained LSTM model and performs inference on the test dataset.
    Visualizes the comparison between Actual values and AI Predicted values.
    """
    
    # 1. Configuration
    BATCH_SIZE = 1     # Predict one sample at a time for clear visualization
    WINDOW_SIZE = 24   # Input sequence length (Past 24 hours)
    HIDDEN_DIM = 64
    DATA_PATH = 'data/SolarEnergy.csv'
    MODEL_PATH = 'models/best_dni_model.pth'
    
    # Check device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Prediction running on: {device}")

    # 2. Prepare Test Dataset
    # Set is_train=False to load data that was NOT used during training.
    print("Loading test data...")
    test_dataset = SolarDataset(DATA_PATH, window_size=WINDOW_SIZE, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Initialize Model & Load Weights
    model = DNIPredictionLSTM(input_dim=6, hidden_dim=HIDDEN_DIM, output_dim=1, num_layers=2).to(device)
    
    try:
        # Load the saved model weights (map_location ensures it works on CPU even if trained on GPU)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("✅ Pre-trained model loaded successfully!")
    except FileNotFoundError:
        print("❌ Model file not found. Please run 'main.py' to train the model first.")
        return

    # Set model to evaluation mode (Disables Dropout)
    model.eval()

    # 4. Inference Loop
    actuals = []
    predictions = []
    
    print("🔮 Predicting future radiation...")
    
    with torch.no_grad(): # Disable gradient calculation for faster inference
        for i, (inputs, targets) in enumerate(test_loader):
            # Limit visualization to the first 100 hours for clarity
            if i >= 100: break 
            
            inputs = inputs.to(device)
            
            # Forward pass (Get prediction from AI)
            output = model(inputs)
            
            # Store results (Move to CPU and convert to scalar)
            predictions.append(output.item())
            actuals.append(targets.item())

    # 5. Inverse Transformation (Denormalization)
    # The model predicts values in range [0, 1]. We need to convert them back to original W/m².
    # We use the scaler parameters from the dataset: min_ and scale_ (range).
    
    # Radiation is the 0-th feature in our dataset
    rad_min = test_dataset.scaler.data_min_[0]
    rad_scale = test_dataset.scaler.data_range_[0]

    # Formula: Original_Value = (Normalized_Value * Scale) + Min
    real_predictions = [ (x * rad_scale) + rad_min for x in predictions ]
    real_actuals = [ (x * rad_scale) + rad_min for x in actuals ]

    # 6. Visualization
    plt.figure(figsize=(12, 6))
    
    # Plot Actual Data
    plt.plot(real_actuals, label='Actual Data (Real)', color='grey', alpha=0.7, linewidth=2)
    
    # Plot AI Prediction
    plt.plot(real_predictions, label='AI Prediction', color='red', linestyle='--', linewidth=2)
    
    plt.title('CSP DNI Forecasting: AI vs Real (First 100 Hours of Test Data)')
    plt.xlabel('Time (Hours)')
    plt.ylabel('Solar Radiation (W/m²)')
    plt.legend()
    plt.grid(True)
    
    # Save the result graph
    save_path = 'models/prediction_result.png'
    plt.savefig(save_path)
    print(f"📊 Prediction graph saved to '{save_path}'")
    
    # Print a sample comparison
    print(f"   Sample Prediction (Hour 10): AI Predicted {real_predictions[10]:.2f} W/m² / Actual {real_actuals[10]:.2f} W/m²")

if __name__ == '__main__':
    predict_future()