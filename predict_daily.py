import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os

from src.dataset import SolarDataset
from src.model import DNIPredictionLSTM

def suggest_operation_schedule():
    """
    CSP Operation Scheduler for 2018 Data.
    - Scans the next 15 days (360 hours).
    - Identifies 'GO' days (DNI > Threshold).
    - Generates an operational report and graph.
    """
    
    # 1. Configuration
    BATCH_SIZE = 1
    WINDOW_SIZE = 24
    HIDDEN_DIM = 64
    TEST_FILE = 'data/2018.csv' # Target Year
    MODEL_PATH = 'models/best_dni_model.pth'
    
    TURBINE_THRESHOLD = 400.0  # CSP Operation Threshold (W/m²)
    HOURS_TO_PREDICT = 360     # Scan 15 days
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 CSP Scheduler running on: {device}")

    # 2. Load Data & Model
    test_dataset = SolarDataset(TEST_FILE, window_size=WINDOW_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Input dim = 5
    model = DNIPredictionLSTM(input_dim=5, hidden_dim=HIDDEN_DIM, output_dim=1, num_layers=2).to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        return

    model.eval()

    # 3. Collect Predictions
    print(f"🔮 Scanning next {HOURS_TO_PREDICT} hours (15 days) of 2018...")
    all_predictions = []
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            if i >= HOURS_TO_PREDICT: break
            inputs = inputs.to(device)
            output = model(inputs)
            all_predictions.append(output.item())

    # 4. Denormalize
    dni_min = test_dataset.scaler.data_min_[0]
    dni_scale = test_dataset.scaler.data_range_[0]
    real_predictions = np.array([ (x * dni_scale) + dni_min for x in all_predictions ])

    # 5. Analyze & Find Best Day
    days_to_analyze = len(real_predictions) // 24
    
    # Logic to find the best day to plot
    best_day_idx = -1
    max_dni_all_time = -1.0
    
    for day in range(days_to_analyze):
        day_data = real_predictions[day*24 : (day+1)*24]
        day_max = np.max(day_data)
        if day_max > max_dni_all_time:
            max_dni_all_time = day_max
            best_day_idx = day

    # 6. Report & Plotting
    print("\n" + "="*60)
    print("🏭 CSP PLANT OPERATION REPORT (2018)")
    print("="*60)
    
    plot_done = False

    for day in range(days_to_analyze):
        day_data = real_predictions[day*24 : (day+1)*24]
        valid_hours = np.where(day_data > TURBINE_THRESHOLD)[0]
        max_dni = np.max(day_data)
        
        status = "✅ GO" if len(valid_hours) > 0 else "⛔ NO-GO"
        print(f"📅 [Day {day+1}] Max DNI: {max_dni:.2f} W/m² -> {status}")

        # Plot condition: If it's a valid day OR it's the best day found
        if (len(valid_hours) > 0 or day == best_day_idx) and not plot_done:
            
            if len(valid_hours) > 0:
                title_suffix = "Operational Day"
                fill_color = 'yellow'
                file_suffix = f"day_{day+1}"
            else:
                title_suffix = "Best Available Day (Below Threshold)"
                fill_color = 'gray'
                file_suffix = "best_effort"

            plt.figure(figsize=(10, 6))
            plt.plot(day_data, label='Predicted DNI', color='orange', linewidth=2)
            plt.axhline(y=TURBINE_THRESHOLD, color='red', linestyle='--', label=f'Threshold ({int(TURBINE_THRESHOLD)})')
            
            if len(valid_hours) > 0:
                 plt.fill_between(range(24), day_data, TURBINE_THRESHOLD, 
                                 where=(day_data > TURBINE_THRESHOLD), 
                                 color=fill_color, alpha=0.3, label='Operation Window')
            else:
                plt.fill_between(range(24), day_data, 0, color=fill_color, alpha=0.1)

            plt.title(f'CSP Schedule: {title_suffix}')
            plt.xlabel('Time (Hour of Day)')
            plt.ylabel('DNI (W/m²)')
            plt.legend()
            plt.grid(True)
            
            save_path = f'models/daily_schedule_{file_suffix}.png'
            plt.savefig(save_path)
            print(f"   📊 Graph saved to '{save_path}'")
            plot_done = True # Plot only one graph

    print("="*60)

if __name__ == '__main__':
    suggest_operation_schedule()