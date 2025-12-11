import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# Import our custom modules
from src.dataset import SolarDataset
from src.model import DNIPredictionLSTM

def train_model():
    """
    Main function to execute the training loop for the CSP DNI Forecasting Model.
    """
    
    # 1. Hyperparameters & Configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 20        # Number of times to iterate through the entire dataset
    WINDOW_SIZE = 24   # Look-back period (24 hours)
    HIDDEN_DIM = 64
    DATA_PATH = 'data/SolarEnergy.csv' # Path to your dataset
    MODEL_SAVE_PATH = 'models/best_dni_model.pth'
    
    # Check for GPU availability (Use CUDA if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # 2. Data Preparation
    print("Loading data...")
    train_dataset = SolarDataset(DATA_PATH, window_size=WINDOW_SIZE, is_train=True)
    test_dataset = SolarDataset(DATA_PATH, window_size=WINDOW_SIZE, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Model Initialization
    model = DNIPredictionLSTM(
        input_dim=6, 
        hidden_dim=HIDDEN_DIM, 
        output_dim=1, 
        num_layers=2
    ).to(device)

    # 4. Loss Function and Optimizer
    criterion = nn.MSELoss() # Mean Squared Error for Regression
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Lists to store loss history for plotting
    train_losses = []
    val_losses = []

    # 5. Training Loop
    print("Start Training...")
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

        # Calculate average training loss
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # 6. Validation Loop
        model.eval() # Set model to evaluation mode
        val_running_loss = 0.0
        
        with torch.no_grad(): # Disable gradient calculation for validation
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.view(-1, 1)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item()
        
        epoch_val_loss = val_running_loss / len(test_loader)
        val_losses.append(epoch_val_loss)

        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {epoch_train_loss:.5f}, Val Loss: {epoch_val_loss:.5f}")

    # 7. Save the Trained Model
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # 8. Plotting Training Results
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('DNI Forecasting Training Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('models/training_result.png')
    print("Training graph saved to models/training_result.png")

if __name__ == '__main__':
    train_model()