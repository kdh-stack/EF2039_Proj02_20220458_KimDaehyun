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
    Visualizes 'Actual DNI' vs 'AI Predicted DNI'.
    """
    
    # 1. Configuration
    BATCH_SIZE = 1
    WINDOW_SIZE = 24
    HIDDEN_DIM = 64
    
    # Use 2018 Data for Testing (Unseen future data)
    TEST_FILE = 'data/2018.csv'
    MODEL_PATH = 'models/best_dni_model.pth'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Prediction running on: {device}")

    # 2. Load Test Data
    print(f"📂 Loading Test Data: {TEST_FILE}")
    try:
        test_dataset = SolarDataset(TEST_FILE, window_size=WINDOW_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 3. Initialize Model
    # Important: Input dim is now 5 (DNI, Temp, Press, Hum, Wind)
    model = DNIPredictionLSTM(input_dim=5, hidden_dim=HIDDEN_DIM, output_dim=1, num_layers=2).to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("✅ Trained model weights loaded.")
    except FileNotFoundError:
        print("❌ Model file not found. Run main.py first.")
        return

    model.eval()

    # 4. Inference Loop
    actuals = []
    predictions = []
    
    print("🔮 Predicting 2018 DNI...")
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            if i >= 100: break # Visualize first 100 hours
            
            inputs = inputs.to(device)
            output = model(inputs)
            
            predictions.append(output.item())
            actuals.append(targets.item())

    # 5. Denormalize
    dni_min = test_dataset.scaler.data_min_[0]
    dni_scale = test_dataset.scaler.data_range_[0]

    real_predictions = [ (x * dni_scale) + dni_min for x in predictions ]
    real_actuals = [ (x * dni_scale) + dni_min for x in actuals ]

    # 6. Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(real_actuals, label='Actual DNI (2018)', color='grey', alpha=0.7, linewidth=2)
    plt.plot(real_predictions, label='AI Prediction', color='red', linestyle='--', linewidth=2)
    
    plt.title('Real DNI Forecasting: AI vs Actual (2018 Data)')
    plt.xlabel('Time (Hours)')
    plt.ylabel('DNI (W/m²)')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('models/prediction_result.png')
    print("📊 Prediction graph saved to 'models/prediction_result.png'")

if __name__ == '__main__':
    predict_future()