# CSP DNI Forecasting & Operation Optimization Model ☀️

This project implements a **Short-term Direct Normal Irradiance (DNI) Forecasting Model** using a Long Short-Term Memory (LSTM) network.
It is designed to optimize the operation of **Concentrated Solar Power (CSP)** plants by predicting solar energy availability in **15-minute intervals** and suggesting optimal turbine schedules.

> **Key Update:** The project has evolved to use **Out-of-Time (OOT) Validation**, training on 2017 historical data and validating on unseen 2018 data to ensure robust real-world performance.

---

## 📌 Project Overview
* **Objective:** Predict future DNI based on the past 24 hours (96 steps) of weather data.
* **Architecture:** Stacked LSTM (2 Layers) implemented in PyTorch.
* **Resolution:** 15-minute intervals (High-resolution forecasting).
* **Application:** * Thermal Energy Storage (TES) management.
    * Turbine Start/Stop decision support (Threshold: 400 W/m²).

## 📊 Dataset & Validation Strategy
We use the **Solar Radiation Dataset** containing specific meteorological parameters for **2017 and 2018**.

* **Source:** [Kaggle Solar Radiation Dataset](https://www.kaggle.com/datasets/ibrahimkiziloklu/solar-radiation-dataset)
* **Data Split (OOT Validation):**
    * **Training Set:** Year **2017** (Historical Learning)
    * **Test Set:** Year **2018** (Future Prediction)
    * *Rationale:* Instead of a random split, we use strictly chronological splitting to prevent data leakage and simulate real-world forecasting scenarios.
* **Input Features (5 Variables):**
    * `DNI` (Direct Normal Irradiance) - **Target Variable**
    * `Temperature`, `Pressure`, `Relative Humidity`, `Wind Speed`
* **Preprocessing:**
    * MinMax Normalization (0~1).
    * **Sliding Window:** Input sequence length = **96 steps** (Past 24 hours).

## 📂 Directory Structure
```text
EF2039_Proj02_20220458_KimDaehyun/
CSP_DNI_Project/
├── data/                  # Dataset directory
│   ├── 2017.csv           # Training Data
│   └── 2018.csv           # Test Data (Unseen)
├── models/                # Saved models and analysis graphs
│   ├── best_dni_model.pth # Trained weights (on 2017 data)
│   ├── training_result.png
│   ├── prediction_result.png
│   └── daily_schedule_summer_day_X.png
├── src/                   # Source code modules
│   ├── dataset.py         # OOT Data loading & Preprocessing
│   └── model.py           # LSTM Architecture
├── main.py                # Training Script (2017 -> 2018)
├── predict.py             # Accuracy Evaluation Script
├── predict_daily.py       # CSP Operation Scheduler (Summer/Winter Scan)
├── requirements.txt       # Dependencies
└── README.md              # Project Documentation
```

## 🚀 How to Run

### 1. Prerequisites
Install the required Python packages.
```bash
pip install -r requirements.txt
```
### 2. Data Setup
Download `2017.csv`&`2018.csv` from the Kaggle link above.

Place the file inside the `data/` directory.

### 3. Training (OOT Validation)
Train the model on 2017 data and validate against 2018 data.
```bash
python main.py
```
- This script trains the model for 20 epochs.
- Saves the best weights to `models/best_dni_model.pth`.
- Generates the training loss curve.
- Window Size: 96 (15-min intervals * 24 hours).

### 4. Prediction & Accuracy Check
Evaluate how well the model predicts the 2018 DNI.
```bash
python predict.py
```
- Visualizes the comparison between AI Predictions and Actual Data.
- Saves the result graph to `models/prediction_result.png`.

### 5. 4. Operational Scheduling (Interactive Simulation)
Run the interactive scheduler to generate turbine operation plans for specific seasons or custom periods.
```bash
python predict_daily.py
```
- Logic: Finds days where DNI > 400 W/m² (Turbine Threshold).
- Output: Generates a daily report and highlights the Operating Window in yellow.
- Interactive Mode: The program will ask you to select a season (Spring, Summer, Autumn, Winter) and the duration (number of days).
- Output: Generates daily status reports and visualizes the Operating Window (Yellow area) in `models/`.

## 📈 Results
### 1. Training Performance
The model successfully learned the weather patterns of 2017 and generalized well to 2018 data without overfitting.

### 2. Forecasting Accuracy (2018 Test Data)
The AI model accurately tracks the diurnal cycle and fluctuations of DNI in 15-minute intervals.

### 3. CSP Operation Schedule (Summer Simulation)
Below is an example of the AI-generated schedule for a sunny summer day in 2018. The yellow area indicates the optimal window for turbine operation.

## 🛠 Tech Stack
- Language: Python 3.x
- Framework: PyTorch
- Libraries: Pandas, NumPy, Matplotlib, Scikit-learn
- Methodology: Time-Series Forecasting, LSTM, Sliding Window

## 💡 Reference: Simulation Start Steps
When using the **Custom Start Step** mode in `predict_daily.py`, refer to the table below to select a specific starting date (based on 15-minute intervals).

| Season | Approx. Date | Start Step (Index) | Description |
| :--- | :--- | :--- | :--- |
| **Winter** | Jan 1st | **0** | Start of the year (Cloudy/Short days) |
| **Spring** | Mar 1st | **5,760** | Increasing DNI trend |
| **Summer** | Jun 1st | **14,500** | Peak DNI (Best for CSP generation) |
| **Autumn** | Sep 1st | **23,328** | Decreasing DNI trend |
| **Winter** | Dec 1st | **32,064** | End of the year |

> **Calculation Logic:** $1 \text{ Day} = 24 \text{ Hours} \times 4 \text{ Steps} = 96 \text{ Steps}$