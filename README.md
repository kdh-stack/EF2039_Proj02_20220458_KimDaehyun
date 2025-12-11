# CSP DNI Forecasting Model (LSTM)

This project develops a **Short-term DNI (Direct Normal Irradiance) Forecasting Model** for optimizing CSP (Concentrated Solar Power) plant operations.
The model uses an LSTM (Long Short-Term Memory) network implemented from scratch using PyTorch.

## 📌 Project Overview
* **Objective:** Predict solar radiation 1 hour into the future based on the past 24 hours of weather data.
* **Model:** Stacked LSTM (2 Layers) implemented in PyTorch.
* **Application:** Thermal Energy Storage (TES) management in CSP plants.

## 📂 Directory Structure

CSP_DNI_Project/ ├── data/ # Dataset directory (contains SolarEnergy.csv) ├── models/ # Saved models and result graphs │ ├── best_dni_model.pth # Trained model weights │ ├── training_result.png │ └── prediction_result.png ├── src/ # Source code modules │ ├── dataset.py # Data loading & Sliding window preprocessing │ └── model.py # LSTM architecture definition ├── main.py # Training execution script ├── predict.py # Inference & Visualization script ├── requirements.txt # Python dependencies └── README.md # Project documentation

## 📊 Dataset
This project uses the **"Solar Radiation Prediction Dataset"** from Kaggle.
* **Source:** [Kaggle Solar Energy Dataset](https://www.kaggle.com/datasets/dronio/SolarEnergy)
* **Features Used:** `Radiation`, `Temperature`, `Pressure`, `Humidity`, `WindDirection`, `Speed`.
* **Preprocessing:**
    * **Normalization:** MinMax Scaling (0~1).
    * **Sliding Window:** Input sequence length = 24 hours.
    * **Data Split:** Chronological Split (Train 80% / Test 20%).
        * *Why?* To prevent data leakage in time-series forecasting, the data is split by time order instead of random shuffling.

## 🚀 How to Run

### 1. Prerequisites
Install the required Python packages.
```bash
pip install -r requirements.txt
```
### 2. Data Setup
Download `SolarEnergy.csv` from the Kaggle link above.

Place the file inside the `data/` directory.

### 3. Training
Train the LSTM model from scratch.
```bash
python main.py
```
- This script trains the model for 20 epochs.
- Saves the best weights to `models/best_dni_model.pth`.
- Generates the training loss curve.

### 4. Prediction & Evaluation
Evaluate the model on the test dataset (unseen future data).
```bash
python predict.py
```
- Visualizes the comparison between AI Predictions and Actual Data.
- Saves the result graph to `models/prediction_result.png`.

## 📈 Results
### 1. Training Curve
The model shows stable convergence, with both training and validation loss decreasing effectively.

### 2. Prediction Performance (Test Data)
The AI model accurately follows the diurnal patterns and fluctuations of solar radiation on unseen test data.

## 🛠 Tech Stack
- Language: Python 3.x
- Framework: PyTorch
- Libraries: Pandas, NumPy, Matplotlib, Scikit-learn