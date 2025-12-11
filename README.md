# CSP DNI Forecasting Model (LSTM)

This project develops a **Short-term DNI (Direct Normal Irradiance) Forecasting Model** for optimizing CSP (Concentrated Solar Power) plant operations.
The model uses an LSTM (Long Short-Term Memory) network implemented from scratch using PyTorch.

## 📂 Project Structure
- `data/`: Contains raw weather and solar datasets (not included in repo due to size).
- `src/`: Source codes for data loading, preprocessing, and model definition.
- `models/`: Directory to save trained model weights.
- `main.py`: Main execution script for training and validation.

## 📊 Dataset
The model relies on the **"Solar Radiation Prediction Dataset"** from Kaggle.
* **Download:** [Solar Radiation Prediction Dataset](https://www.kaggle.com/datasets/dronio/SolarEnergy)
* **Instruction:** Download the `SolarEnergy.csv` file and place it in the `data/` directory.