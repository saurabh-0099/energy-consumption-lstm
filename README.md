# ⚡ Sustainable Energy Consumption Prediction (LSTM)

Predicting residential energy usage from smart-meter time-series data using a
Long Short-Term Memory (LSTM) neural network, trained on the **UCI Individual
Household Electric Power Consumption** dataset.

## 📌 Problem Statement

> Predict residential energy usage based on smart meter time-series data.
> **Dataset:** UCI Household Power Consumption

## 🧠 Approach

1. **Data cleaning** — parse the raw minute-level readings, handle missing
   values (`?` markers), and resample to hourly averages.
2. **Feature engineering** — add calendar features (`hour`, `day of week`,
   `month`) alongside the raw electrical measurements (active/reactive power,
   voltage, intensity, sub-metering 1–3).
3. **Sequence generation** — build sliding windows of the last 24 hours to
   predict the next hour's `Global_active_power`.
4. **Modeling** — a stacked LSTM network (LSTM → Dropout → LSTM → Dropout →
   Dense → Dense) trained with early stopping and learning-rate scheduling.
5. **Evaluation** — RMSE, MAE, R², and MAPE on a chronologically held-out test
   set, plus visual plots of predicted vs. actual consumption.

## 📁 Project Structure

```
energy-consumption-lstm/
├── data/                          # place the raw UCI dataset here (not committed)
├── models/                        # trained model + scaler get saved here
├── outputs/                       # generated plots + metrics.json
├── src/
│   ├── config.py                  # all hyperparameters & paths
│   ├── data_preprocessing.py      # cleaning, resampling, sequence building
│   ├── model.py                   # LSTM architecture
│   ├── train.py                   # training loop
│   ├── evaluate.py                # test-set evaluation + plots
│   └── predict.py                 # forecast the next hour from latest data
├── main.py                        # runs the full pipeline end-to-end
├── requirements.txt
└── README.md
```

## 📦 Dataset Setup

This repo does **not** include the raw dataset (it's ~130MB, too large for
git). Download it manually:

1. Go to the UCI dataset page:
   https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
2. Download `household_power_consumption.zip` and unzip it.
3. Place `household_power_consumption.txt` inside the `data/` folder of this
   project, so the path is:
   `energy-consumption-lstm/data/household_power_consumption.txt`

## 🛠️ Setup (VS Code / local machine)

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/energy-consumption-lstm.git
cd energy-consumption-lstm

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add the dataset (see "Dataset Setup" above), then run the full pipeline
python main.py
```

Running `main.py` will:
- Build `data/hourly_power_consumption.csv` from the raw file
- Train the LSTM and save the best model to `models/lstm_energy_model.keras`
- Evaluate on the test set, printing RMSE/MAE/R²/MAPE
- Save `outputs/training_loss.png` and `outputs/predictions_vs_actual.png`
- Print a next-hour energy consumption forecast

### Run steps individually

```bash
python src/data_preprocessing.py   # build the processed hourly dataset
python src/train.py                # train the LSTM
python src/evaluate.py             # evaluate + generate plots
python src/predict.py              # forecast the next hour
```

## 📊 Results

| Metric | Value |
|--------|-------|
| RMSE   | 0.4685 kW |
| MAE    | 0.3256 kW |
| R²     | 0.5859 |
| MAPE   | 40.54% |

**Training loss curve:**

![Training Loss](outputs/training_loss.png)

**Predicted vs. actual energy consumption (test set):**

![Predictions vs Actual](outputs/predictions_vs_actual.png)

**Interpretation:** The model explains ~59% of the variance in hourly energy
consumption (R² = 0.59), with a typical prediction error of ~0.33 kW (MAE).
The relatively high MAPE (40.5%) is largely driven by hours with very low
actual consumption (e.g. overnight), where even small absolute errors
translate into large percentage errors — a known limitation of MAPE rather
than a sign the model is unreliable. RMSE and MAE are more representative of
real-world performance here.
## 🔧 Tech Stack

- Python 3.10+
- TensorFlow / Keras (LSTM model)
- pandas, NumPy (data processing)
- scikit-learn (scaling, metrics)
- Matplotlib (visualization)

## 📈 Future Improvements

- Multi-step forecasting (predict several hours/days ahead)
- Add weather data as an external regressor
- Try GRU / Transformer-based architectures for comparison
- Deploy as a simple Streamlit/Flask dashboard for live predictions

## 📄 License

This project is released under the MIT License — feel free to use and adapt
it for coursework or personal projects.
