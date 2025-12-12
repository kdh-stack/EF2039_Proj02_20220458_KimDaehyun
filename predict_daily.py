import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os

from src.dataset import SolarDataset
from src.model import DNIPredictionLSTM

def suggest_operation_schedule():
    """
    CSP Operation Scheduler (Corrected for 15-minute intervals).
    - Data Interval: 15 minutes
    - Steps per Day: 96 steps (24 hours * 4)
    """
    
    # 1. Configuration
    BATCH_SIZE = 1
    WINDOW_SIZE = 96   # ⭐️ 24hour = 96 steps
    HIDDEN_DIM = 64
    TEST_FILE = 'data/2018.csv'
    MODEL_PATH = 'models/best_dni_model.pth'
    
    TURBINE_THRESHOLD = 400.0
    
    # If you want to see data from other seasons, change your starting point
    # Calculate the start point near June 1st:
    # 151day(month 1~5) * 24hour * 4step = about 14,500 step
    START_STEP = 14500 
    DAYS_TO_SCAN = 3    # 3일치만 자세히 봅시다
    TOTAL_STEPS = DAYS_TO_SCAN * 96 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 CSP Scheduler running on: {device}")

    # 2. Load Data & Model
    test_dataset = SolarDataset(TEST_FILE, window_size=WINDOW_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = DNIPredictionLSTM(input_dim=5, hidden_dim=HIDDEN_DIM, output_dim=1, num_layers=2).to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print("❌ Model not found. Did you retrain with WINDOW_SIZE=96?")
        return

    model.eval()

    # 3. Collect Predictions
    print(f"🔮 Scanning summer days (Starting step: {START_STEP})...")
    all_predictions = []
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            # Jump to summer data
            if i < START_STEP: continue
            if len(all_predictions) >= TOTAL_STEPS: break
            
            inputs = inputs.to(device)
            output = model(inputs)
            all_predictions.append(output.item())

    # 4. Denormalize
    dni_min = test_dataset.scaler.data_min_[0]
    dni_scale = test_dataset.scaler.data_range_[0]
    real_predictions = np.array([ (x * dni_scale) + dni_min for x in all_predictions ])

    # 5. Analyze & Plot
    steps_per_day = 96
    days_analyzed = len(real_predictions) // steps_per_day
    
    print("\n" + "="*60)
    print("🏭 CSP PLANT OPERATION REPORT (Summer Simulation)")
    print("="*60)

    for day in range(days_analyzed):
        # 96data(1day)
        day_data = real_predictions[day*steps_per_day : (day+1)*steps_per_day]
        
        # Create x-axis to convert 15-minute data to time axis (0-24h)
        time_axis = np.linspace(0, 24, steps_per_day) 
        
        max_dni = np.max(day_data)
        valid_indices = np.where(day_data > TURBINE_THRESHOLD)[0]
        
        status = "✅ GO" if len(valid_indices) > 0 else "⛔ NO-GO"
        print(f"📅 [Day {day+1}] Max DNI: {max_dni:.2f} W/m² -> {status}")

        if len(valid_indices) > 0:
            # Start/End Time Calculation (Index * 15 mins / 60 mins)
            start_hour = valid_indices[0] * 15 / 60
            end_hour = valid_indices[-1] * 15 / 60
            duration = end_hour - start_hour
            
            print(f"   ⚙️ Run Turbine: {start_hour:.1f}h ~ {end_hour:.1f}h ({duration:.1f} hours)")

            plt.figure(figsize=(10, 6))
            plt.plot(time_axis, day_data, label='Predicted DNI', color='orange', linewidth=2)
            plt.axhline(y=TURBINE_THRESHOLD, color='red', linestyle='--', label=f'Threshold ({int(TURBINE_THRESHOLD)})')
            
            # Fill Area
            plt.fill_between(time_axis, day_data, TURBINE_THRESHOLD, 
                             where=(day_data > TURBINE_THRESHOLD), 
                             color='yellow', alpha=0.3, label='Operation Window')

            plt.title(f'CSP Schedule: Summer Operational Day')
            plt.xlabel('Time (Hour of Day)')
            plt.ylabel('DNI (W/m²)')
            plt.xticks(np.arange(0, 25, 2)) # x-axis 2hour spacing
            plt.legend()
            plt.grid(True)
            
            save_path = f'models/daily_schedule_summer_day_{day+1}.png'
            plt.savefig(save_path)
            print(f"   📊 Graph saved to '{save_path}'")

if __name__ == '__main__':
    suggest_operation_schedule()