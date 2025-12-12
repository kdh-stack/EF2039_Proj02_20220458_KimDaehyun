import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import sys

# Import custom modules
from src.dataset import SolarDataset
from src.model import DNIPredictionLSTM

def get_user_input():
    """
    Get start period and duration from user input via terminal.
    """
    print("\n" + "="*50)
    print("🎮 CSP Simulation Configuration")
    print("="*50)
    print("Select the starting season for simulation:")
    print("  1. Spring (March ~)")
    print("  2. Summer (June ~)")
    print("  3. Autumn (September ~)")
    print("  4. Winter (January ~)")
    print("  5. Custom Start Step")
    
    choice = input("👉 Enter your choice (1-5): ")
    
    # 1 Day = 96 Steps (15 min intervals)
    # Approx start steps for each season
    if choice == '1':
        start_step = 5760   # March 1st approx
        season_name = "Spring"
    elif choice == '2':
        start_step = 14500  # June 1st approx
        season_name = "Summer"
    elif choice == '3':
        start_step = 23328  # Sept 1st approx
        season_name = "Autumn"
    elif choice == '4':
        start_step = 0      # Jan 1st
        season_name = "Winter"
    elif choice == '5':
        start_step = int(input("👉 Enter custom start step (0~35000): "))
        season_name = "Custom"
    else:
        print("❌ Invalid choice. Defaulting to Summer.")
        start_step = 14500
        season_name = "Summer"

    try:
        days = int(input("👉 How many days to scan? (e.g., 3): "))
    except:
        days = 3
        print("⚠️ Invalid input. Defaulting to 3 days.")

    return start_step, days, season_name

def suggest_operation_schedule():
    """
    CSP Operation Scheduler with User Input.
    """
    
    # 1. Get User Input
    START_STEP, DAYS_TO_SCAN, SEASON_NAME = get_user_input()
    
    # Configuration
    BATCH_SIZE = 1
    WINDOW_SIZE = 96
    HIDDEN_DIM = 64
    TEST_FILE = 'data/2018.csv'
    MODEL_PATH = 'models/best_dni_model.pth'
    TURBINE_THRESHOLD = 400.0
    
    TOTAL_STEPS = DAYS_TO_SCAN * 96
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 CSP Scheduler running on: {device}")
    print(f"📋 Simulation Target: {SEASON_NAME} for {DAYS_TO_SCAN} days")

    # 2. Load Data & Model
    try:
        test_dataset = SolarDataset(TEST_FILE, window_size=WINDOW_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    model = DNIPredictionLSTM(input_dim=5, hidden_dim=HIDDEN_DIM, output_dim=1, num_layers=2).to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print("❌ Model not found. Please train first.")
        return

    model.eval()

    # 3. Collect Predictions
    print(f"🔮 Scanning data starting from step {START_STEP}...")
    all_predictions = []
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            # Skip until start step
            if i < START_STEP: continue
            # Stop after collecting enough data
            if len(all_predictions) >= TOTAL_STEPS: break
            
            inputs = inputs.to(device)
            output = model(inputs)
            all_predictions.append(output.item())
            
            # Progress indicator (simple)
            if len(all_predictions) % 96 == 0:
                print(f"   ... Processed {len(all_predictions)//96} / {DAYS_TO_SCAN} days")

    # 4. Denormalize
    dni_min = test_dataset.scaler.data_min_[0]
    dni_scale = test_dataset.scaler.data_range_[0]
    real_predictions = np.array([ (x * dni_scale) + dni_min for x in all_predictions ])

    # 5. Analyze & Plot
    steps_per_day = 96
    days_analyzed = len(real_predictions) // steps_per_day
    
    print("\n" + "="*60)
    print(f"🏭 CSP PLANT OPERATION REPORT ({SEASON_NAME})")
    print("="*60)

    for day in range(days_analyzed):
        day_data = real_predictions[day*steps_per_day : (day+1)*steps_per_day]
        time_axis = np.linspace(0, 24, steps_per_day)
        
        max_dni = np.max(day_data)
        valid_indices = np.where(day_data > TURBINE_THRESHOLD)[0]
        
        status = "✅ GO" if len(valid_indices) > 0 else "⛔ NO-GO"
        print(f"📅 [Day {day+1}] Max DNI: {max_dni:.2f} W/m² -> {status}")

        # Always plot for user verification
        plt.figure(figsize=(10, 6))
        plt.plot(time_axis, day_data, label='Predicted DNI', color='orange', linewidth=2)
        plt.axhline(y=TURBINE_THRESHOLD, color='red', linestyle='--', label=f'Threshold ({int(TURBINE_THRESHOLD)})')
        
        if len(valid_indices) > 0:
            start_hour = valid_indices[0] * 15 / 60
            end_hour = valid_indices[-1] * 15 / 60
            duration = end_hour - start_hour
            print(f"   ⚙️ Run Turbine: {start_hour:.1f}h ~ {end_hour:.1f}h ({duration:.1f} hours)")
            
            plt.fill_between(time_axis, day_data, TURBINE_THRESHOLD, 
                             where=(day_data > TURBINE_THRESHOLD), 
                             color='yellow', alpha=0.3, label='Operation Window')
            title_suffix = "Operational"
        else:
            plt.fill_between(time_axis, day_data, 0, color='gray', alpha=0.1)
            title_suffix = "Non-Operational"

        plt.title(f'CSP Schedule: {SEASON_NAME} Day {day+1} ({title_suffix})')
        plt.xlabel('Time (Hour of Day)')
        plt.ylabel('DNI (W/m²)')
        plt.xticks(np.arange(0, 25, 2))
        plt.legend()
        plt.grid(True)
        
        save_path = f'models/daily_schedule_{SEASON_NAME.lower()}_day_{day+1}.png'
        
        # if don't have models folder, generate
        if not os.path.exists('models'):
            os.makedirs('models')
            
        plt.savefig(save_path)
        print(f"   📊 Graph saved to '{save_path}'")

if __name__ == '__main__':
    suggest_operation_schedule()