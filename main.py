import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# Import custom modules
from src.dataset import SolarDataset
from src.model import DNIPredictionLSTM

def train_model():
    """
    Main execution function for training the DNI Forecasting Model.
    - Training Data: 2017 Historical Weather Data
    - Validation Data: 2018 Future Weather Data (OOT Validation)
    """
    
    # 1. Hyperparameters & Configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 20        # Number of training iterations
    WINDOW_SIZE = 24   # Look-back period (Past 24 hours)
    HIDDEN_DIM = 64    # LSTM hidden state dimension
    
    # File Paths for Train/Test Split by Year
    TRAIN_FILE = 'data/2017.csv'
    TEST_FILE = 'data/2018.csv'
    
    MODEL_SAVE_PATH = 'models/best_dni_model.pth'
    PLOT_SAVE_PATH = 'models/training_result.png'
    
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training started on device: {device}")

    # 2. Data Preparation
    # Load separate files for training and validation to ensure no data leakage.
    print(f"📂 Loading Training Data from: {TRAIN_FILE}")
    try:
        train_dataset = SolarDataset(TRAIN_FILE, window_size=WINDOW_SIZE)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        print(f"❌ Error loading train file: {e}")
        return

    print(f"📂 Loading Validation Data from: {TEST_FILE}")
    try:
        test_dataset = SolarDataset(TEST_FILE, window_size=WINDOW_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    except Exception as e:
        print(f"❌ Error loading test file: {e}")
        return

    # 3. Model Initialization
    # input_dim = 5 (DNI, Temperature, Pressure, Relative Humidity, Wind Speed)
    model = DNIPredictionLSTM(
        input_dim=5, 
        hidden_dim=HIDDEN_DIM, 
        output_dim=1, 
        num_layers=2
    ).to(device)

    # 4. Loss Function and Optimizer
    criterion = nn.MSELoss() # Mean Squared Error for regression tasks
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Lists to store loss history
    train_losses = []
    val_losses = []

    # 5. Training Loop
    print("⚡ Start Training...")
    for epoch in range(EPOCHS):
        model.train() # Set model to training mode
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Reshape targets to match output shape (batch_size, 1)
            targets = targets.view(-1, 1)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate average training loss for this epoch
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # 6. Validation Loop
        # Evaluate performance on 2018 data (Unseen data)
        model.eval() # Set model to evaluation mode
        val_running_loss = 0.0
        
        with torch.no_grad(): # Disable gradient calculation
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.view(-1, 1)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item()
        
        epoch_val_loss = val_running_loss / len(test_loader)
        val_losses.append(epoch_val_loss)

        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {epoch_train_loss:.5f}, Val Loss (2018): {epoch_val_loss:.5f}")

    # 7. Save the Trained Model
    if not os.path.exists('models'):
        os.makedirs('models')
        
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Model saved to '{MODEL_SAVE_PATH}'")

    # 8. Visualize Training Results
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss (2017)')
    plt.plot(val_losses, label='Validation Loss (2018)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('DNI Forecasting: Train(2017) vs Test(2018)')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(PLOT_SAVE_PATH)
    print(f"📊 Training graph saved to '{PLOT_SAVE_PATH}'")

if __name__ == '__main__':
    train_model()