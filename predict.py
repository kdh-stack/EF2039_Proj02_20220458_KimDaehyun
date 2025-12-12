import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import custom modules
from src.dataset import SolarDataset
from src.model import DNIPredictionLSTM

def predict_future():
    """
    Evaluates the trained model on 2018 data (OOT Validation).
    Visualizes 'Actual DNI' vs 'AI Predicted DNI' for verification.
    
    [Updated for 15-minute intervals]
    - Window Size: 96 steps (Past 24 hours)
    - Visualization: X-axis converted to Hours
    """
    
    # 1. Configuration
    BATCH_SIZE = 1
    # ⭐️ Critical: 24 hours * 4 steps/hour = 96 steps
    WINDOW_SIZE = 96    
    HIDDEN_DIM = 64
    
    TEST_FILE = 'data/2018.csv'
    MODEL_PATH = 'models/best_dni_model.pth'
    
    # Visualization Period: 400 steps (Approx. 4 days = 100 hours)
    STEPS_TO_VISUALIZE = 400

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Prediction running on: {device}")

    # 2. Load Test Data
    print(f"📂 Loading Test Data from: {TEST_FILE}")
    try:
        # Ensure 'window_size' matches the training configuration (96)
        test_dataset = SolarDataset(TEST_FILE, window_size=WINDOW_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 3. Initialize Model
    # Input dimension is 5 (DNI, Temp, Pressure, Humidity, Wind Speed)
    model = DNIPredictionLSTM(input_dim=5, hidden_dim=HIDDEN_DIM, output_dim=1, num_layers=2).to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("✅ Trained model weights loaded successfully.")
    except FileNotFoundError:
        print("❌ Model file not found. Did you retrain 'main.py' with WINDOW_SIZE=96?")
        return

    model.eval()

    # 4. Inference Loop
    actuals = []
    predictions = []
    
    print(f"🔮 Predicting 2018 DNI (First {STEPS_TO_VISUALIZE} steps)...")
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            # Stop after visualizing the specified number of steps
            if i >= STEPS_TO_VISUALIZE: break 
            
            inputs = inputs.to(device)
            output = model(inputs)
            
            predictions.append(output.item())
            actuals.append(targets.item())

    # 5. Denormalize (Convert 0~1 range back to W/m²)
    # We use the scaler parameters from the dataset
    dni_min = test_dataset.scaler.data_min_[0]
    dni_scale = test_dataset.scaler.data_range_[0]

    real_predictions = [ (x * dni_scale) + dni_min for x in predictions ]
    real_actuals = [ (x * dni_scale) + dni_min for x in actuals ]

    # 6. Visualization
    # Convert step indices to Hours (1 step = 15 mins = 0.25 hours)
    time_axis = np.arange(0, STEPS_TO_VISUALIZE) * 0.25

    plt.figure(figsize=(14, 6))
    
    # Plot Actual Data (Grey line)
    plt.plot(time_axis, real_actuals, label='Actual DNI (2018)', color='grey', alpha=0.7, linewidth=2)
    
    # Plot AI Prediction (Red dashed line)
    plt.plot(time_axis, real_predictions, label='AI Prediction', color='red', linestyle='--', linewidth=1.5)
    
    plt.title(f'Real DNI Forecasting: AI vs Actual (First {int(STEPS_TO_VISUALIZE*0.25)} Hours of 2018)')
    plt.xlabel('Time (Hours)') 
    plt.ylabel('DNI (W/m²)')
    plt.legend()
    plt.grid(True)
    
    # Set X-axis ticks every 12 hours for better readability
    plt.xticks(np.arange(0, time_axis[-1]+1, 12))

    save_path = 'models/prediction_result.png'
    plt.savefig(save_path)
    print(f"📊 Prediction graph saved to '{save_path}'")

if __name__ == '__main__':
    predict_future()