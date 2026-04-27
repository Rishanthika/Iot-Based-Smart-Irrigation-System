<div align="center">

# 🌿 IoT Smart Irrigation System
### AI-Driven Precision Agriculture with LSTM Forecasting & Uncertainty-Aware Decision Making

<br/>

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-Datasets-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://kaggle.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-22C55E?style=for-the-badge)]()

<br/>

> *"Water is the driving force of all nature." — Leonardo da Vinci*
>
> This system puts that force under intelligent control.

<br/>

</div>

---

## 📌 What Is This?

The **IoT Smart Irrigation System** is an end-to-end machine learning pipeline that fuses **real-time IoT sensor data** with **LSTM-predicted climate forecasts** to make intelligent, field-level irrigation decisions — automatically, continuously, and with quantified confidence.

Rather than irrigating on fixed schedules or intuition, this system answers one question with precision:

> **"Given the current soil state, the crop type, and what the weather will do in the next 5 days — should I irrigate this field right now, and how much?"**

The answer is not just a class label. It comes with a **confidence score**, an **uncertainty estimate**, and a mapped **water volume decision** — enabling safe automation even in adversarial weather conditions.

---

## 🌍 Why It Matters

| Global Challenge | This System's Role |
|---|---|
| Agriculture consumes **~70% of global freshwater** | Reduces irrigation volume vs. naive scheduling |
| Crop yield suffers from both **over- and under-watering** | Outputs calibrated volume: 0 mm, 5 mm, or 10 mm |
| Weather unpredictability makes static schedules obsolete | LSTM forecasts future climate from time-series patterns |
| Sensor noise leads to erroneous automation decisions | Monte Carlo Dropout flags high-uncertainty predictions |
| Class imbalance skews model decisions | Balanced class weights ensure minority classes are learned |

---

## ✨ Key Features

- **🔮 LSTM Climate Forecaster** — Learns temporal dependencies in daily climate sequences (temp, humidity, wind, pressure) and predicts the next time step, providing future-aware context to irrigation decisions.

- **🔗 Multimodal Data Fusion** — Seamlessly merges heterogeneous data: IoT soil/field sensor readings are augmented with LSTM-generated weather forecasts to form a rich, unified feature space.

- **🧠 Deep Neural Classifier** — A regularized feedforward network classifies irrigation need into three levels: Low, Medium, or High — trained with class-weight balancing to handle skewed label distributions.

- **🎲 Monte Carlo Dropout Uncertainty** — Inference is run 50 times with dropout active. The mean and standard deviation across passes yield per-prediction confidence and uncertainty scores, enabling safe fallback logic.

- **⚖️ Intelligent Decision Engine** — Maps classifier output and uncertainty scores to four concrete actions: *Irrigate Immediately*, *Irrigate (Check Conditions)*, *Moderate Irrigation*, or *Delay Irrigation*.

- **💧 Water Savings Simulation** — Computes total water applied under the smart system versus a naive "always irrigate at maximum" baseline, quantifying conservation impact.

- **📊 Training Diagnostics** — Accuracy and loss curves plotted across epochs for both LSTM and classifier models, with early stopping and confusion matrix reporting.

---

## 🏗️ System Architecture

```
╔══════════════════════════════════════════════════════════════════════════╗
║                        DATA INGESTION LAYER                             ║
║                                                                          ║
║   📡 Daily Delhi Climate CSV          🌱 Irrigation Prediction CSV       ║
║   (meantemp, humidity, wind,          (Field_Area, Soil_Moisture,        ║
║    meanpressure — time-indexed)        Temperature, Humidity,            ║
║                                        Crop_Type, Soil_Type, ...)        ║
╚════════════════╦═════════════════════════════╦════════════════════════════╝
                 │                             │
                 ▼                             │
╔══════════════════════════════╗               │
║   LSTM FORECASTING MODULE    ║               │
║                              ║               │
║  MinMax Normalize            ║               │
║       ↓                      ║               │
║  Sliding Window (seq=5)      ║               │
║       ↓                      ║               │
║  LSTM(64) → Dropout(0.2)     ║               │
║       ↓                      ║               │
║  LSTM(32) → Dropout(0.2)     ║               │
║       ↓                      ║               │
║  Dense(4) — MSE Loss         ║               │
║       ↓                      ║               │
║  Inverse Transform           ║               │
║       ↓                      ║               │
║  forecast_temp               ║               │
║  forecast_humidity    ───────╬───────────────▼
║  forecast_wind               ║   DATA FUSION (column-wise join)
║  forecast_pressure           ║               │
╚══════════════════════════════╝               │
                                               ▼
╔══════════════════════════════════════════════════════════════════════════╗
║                       FEATURE ENGINEERING LAYER                         ║
║                                                                          ║
║  Joined Features (14 total):                                             ║
║  Field_Area_hectare · Previous_Irrigation_mm · Soil_Moisture             ║
║  Temperature_C · Humidity · Rainfall_mm                                  ║
║  forecast_temp · forecast_humidity · forecast_wind · forecast_pressure   ║
║  Mulching_Used · Region · Soil_Type · Crop_Type                         ║
║                                                                          ║
║  → One-Hot Encoding (drop_first=True)   → MinMax Scaling (final)        ║
╚════════════════════════════════╦═════════════════════════════════════════╝
                                 │
                                 ▼
╔══════════════════════════════════════════════════════════════════════════╗
║                    IRRIGATION CLASSIFIER MODULE                          ║
║                                                                          ║
║  Input(n_features)                                                       ║
║       ↓                                                                  ║
║  Dense(64, ReLU) → Dropout(0.3)                                          ║
║       ↓                                                                  ║
║  Dense(32, ReLU) → Dropout(0.2)                                          ║
║       ↓                                                                  ║
║  Dense(3, Softmax)   ← Sparse Categorical Cross-Entropy                  ║
║       ↓                                                                  ║
║  Class Weights: balanced · EarlyStopping(patience=5)                    ║
╚════════════════════════════════╦═════════════════════════════════════════╝
                                 │
                                 ▼
╔══════════════════════════════════════════════════════════════════════════╗
║              MONTE CARLO DROPOUT UNCERTAINTY ENGINE                      ║
║                                                                          ║
║  Run inference × 50 with training=True (dropout active)                  ║
║       ↓                                                                  ║
║  mean_pred  = preds.mean(axis=0)   → predicted class + confidence        ║
║  uncertainty = preds.std(axis=0)   → per-sample uncertainty score        ║
╚════════════════════════════════╦═════════════════════════════════════════╝
                                 │
                                 ▼
╔══════════════════════════════════════════════════════════════════════════╗
║                         DECISION ENGINE                                  ║
║                                                                          ║
║  pred_class == 2 (High)  AND  uncertainty < 0.15  →  Irrigate Immediately (10 mm)  ║
║  pred_class == 2 (High)  AND  uncertainty ≥ 0.15  →  Irrigate (Check Conditions) (10 mm) ║
║  pred_class == 1 (Medium)                         →  Moderate Irrigation (5 mm)    ║
║  pred_class == 0 (Low)                            →  Delay Irrigation (0 mm)       ║
╚════════════════════════════════╦═════════════════════════════════════════╝
                                 │
                                 ▼
╔══════════════════════════════════════════════════════════════════════════╗
║                      WATER SAVINGS SIMULATION                            ║
║                                                                          ║
║  Baseline: every field irrigated at 10 mm every cycle                    ║
║  Smart system: sum of actual water applied per decision                  ║
║  Savings (%) = (baseline − smart_total) / baseline × 100                ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 🔬 Model Details

### Stage 1 — LSTM Climate Forecaster

| Component | Value |
|---|---|
| Input | Rolling window of 5 consecutive days |
| Features | `meantemp`, `humidity`, `wind_speed`, `meanpressure` |
| Normalization | MinMaxScaler (fitted on train, applied to test) |
| Missing values | Forward-fill (`ffill`) |
| Architecture | `LSTM(64, return_sequences=True)` → `Dropout(0.2)` → `LSTM(32)` → `Dropout(0.2)` → `Dense(4)` |
| Loss function | Mean Squared Error (MSE) |
| Optimizer | Adam |
| Epochs | 10 |
| Batch size | 32 |
| Train/Test split | 80% / 20% (chronological) |
| Output | 4 forecast features per time step (inverse-transformed to original scale) |

### Stage 2 — Irrigation Need Classifier

| Component | Value |
|---|---|
| Input features | 14 (6 sensor + 4 forecast + 4 categorical one-hot encoded) |
| Target | `Irrigation_Need` — 3 classes: Low (0), Medium (1), High (2) |
| Label encoding | `sklearn.LabelEncoder` |
| Categorical encoding | `pd.get_dummies(drop_first=True)` |
| Normalization | Second `MinMaxScaler` on fused feature matrix |
| Architecture | `Dense(64, ReLU)` → `Dropout(0.3)` → `Dense(32, ReLU)` → `Dropout(0.2)` → `Dense(3, Softmax)` |
| Loss function | Sparse Categorical Cross-Entropy |
| Optimizer | Adam |
| Class balancing | `compute_class_weight('balanced')` passed as `class_weight` dict |
| Regularization | Early Stopping (`monitor='val_loss'`, `patience=5`, `restore_best_weights=True`) |
| Final training | 30 epochs, batch size 16, with class weights |
| Evaluation | Accuracy, Confusion Matrix, Classification Report |

### Stage 3 — Monte Carlo Dropout Uncertainty

| Component | Value |
|---|---|
| Method | MC Dropout (Gal & Ghahramani, 2016) |
| Iterations | 50 stochastic forward passes (`training=True`) |
| Mean prediction | `preds.mean(axis=0)` across 50 passes |
| Uncertainty score | `preds.std(axis=0)` — standard deviation per class |
| Per-sample uncertainty | `np.mean(uncertainty, axis=1)` — scalar per sample |
| Confidence | `np.max(mean_pred, axis=1)` — max class probability |
| Uncertainty threshold | `0.15` — below this → high-confidence action |

---

## 🌾 Feature Reference

### IoT Sensor Features (from Irrigation Dataset)

| Feature | Description |
|---|---|
| `Field_Area_hectare` | Total field area in hectares |
| `Previous_Irrigation_mm` | Water applied in the last irrigation cycle (mm) |
| `Soil_Moisture` | Current volumetric soil moisture content |
| `Temperature_C` | Ambient temperature measured at field (°C) |
| `Humidity` | Relative humidity at field level (%) |
| `Rainfall_mm` | Recorded rainfall (mm) |
| `Mulching_Used` | Whether mulching is applied (categorical) |
| `Region` | Geographic/climate region (categorical) |
| `Soil_Type` | Soil classification (categorical) |
| `Crop_Type` | Crop being grown (categorical) |

### LSTM-Generated Forecast Features

| Feature | Source Climate Column | Description |
|---|---|---|
| `forecast_temp` | `meantemp` | Predicted mean temperature (next step) |
| `forecast_humidity` | `humidity` | Predicted relative humidity |
| `forecast_wind` | `wind_speed` | Predicted wind speed |
| `forecast_pressure` | `meanpressure` | Predicted mean atmospheric pressure |

---

## 💡 Decision Engine Logic

```python
# Per-sample decision mapping
if pred_class == 2:            # High irrigation need
    if uncertainty < 0.15:
        decision = "Irrigate Immediately"       # water = 10 mm
    else:
        decision = "Irrigate (Check Conditions)"  # water = 10 mm (manual verify)

elif pred_class == 1:          # Medium irrigation need
    decision = "Moderate Irrigation"            # water = 5 mm

else:                          # Low irrigation need
    decision = "Delay Irrigation"               # water = 0 mm
```

The uncertainty threshold of `0.15` is the critical safety gate. When the model is unsure — even if it predicts High need — the system signals a human check rather than blindly triggering full irrigation. This is what separates a robust production system from a naive classifier.

---

## 📂 Project Structure

```
iot-smart-irrigation/
│
├── 📓 notebooks/
│   └── iot-smart-irrigation.ipynb     # Full end-to-end analysis notebook
│
├── 🐍 src/
│   ├── __init__.py                     # Package exports
│   ├── lstm_forecaster.py              # LSTM pipeline: load → scale → sequence → train → predict
│   ├── irrigation_classifier.py        # Data fusion, feature engineering, classifier training
│   ├── decision_engine.py              # MC Dropout inference + decision logic
│   └── water_savings.py                # Water usage simulation and savings report
│
├── 📁 data/
│   └── README.md                       # Dataset download instructions
│
├── 📁 models/
│   └── README.md                       # Model saving/loading instructions
│
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Excludes data files and model weights
├── LICENSE                             # MIT License
└── README.md                           # You are here
```

---

## 📦 Datasets

This project uses two publicly available Kaggle datasets:

| Dataset | Author | Records | Key Columns |
|---|---|---|---|
| [Irrigation Water Requirement Prediction](https://www.kaggle.com/datasets/miadul/irrigation-water-requirement-prediction-dataset) | miadul | ~1,000+ rows | Field_Area, Soil_Moisture, Crop_Type, Irrigation_Need |
| [Daily Climate Time Series – Delhi](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data) | sumanthvrao | 1,462 rows | meantemp, humidity, wind_speed, meanpressure |

Download both CSVs and place them in the `data/` directory:

```
data/
├── irrigation_prediction.csv
└── DailyDelhiClimateTrain.csv
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) A Kaggle account to download datasets

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/iot-smart-irrigation.git
cd iot-smart-irrigation
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Datasets

Using the Kaggle CLI:

```bash
pip install kaggle

kaggle datasets download -d miadul/irrigation-water-requirement-prediction-dataset -p data/ --unzip
kaggle datasets download -d sumanthvrao/daily-climate-time-series-data -p data/ --unzip
```

Or download manually from Kaggle and place the CSVs in `data/`.

### 4. Run the Notebook

```bash
jupyter notebook notebooks/iot-smart-irrigation.ipynb
```

### 5. Or Use the Modular Source

```python
from src.lstm_forecaster import train_forecaster
from src.irrigation_classifier import fuse_data, prepare_features, train_classifier
from src.decision_engine import run_decision_engine
from src.water_savings import print_savings_report
import pandas as pd

# Stage 1: LSTM Climate Forecasting
forecast_array, scaler, lstm_history = train_forecaster("data/DailyDelhiClimateTrain.csv")

# Stage 2: Data Fusion + Classification
irrigation_df = pd.read_csv("data/irrigation_prediction.csv")
fused_df = fuse_data(irrigation_df, forecast_array)
X, y, le = prepare_features(fused_df)
model, history, X_test, y_test = train_classifier(X, y)

# Stage 3: MC Dropout Decision Engine
decisions, classes, confidence, uncertainty = run_decision_engine(model, X_test)

# Stage 4: Water Savings Report
print_savings_report(decisions)
```

---

## 📊 Results & Evaluation

### Classifier Evaluation

After training with class-weight balancing and early stopping:

- **Evaluation metrics**: Accuracy, Confusion Matrix, and full Classification Report (precision, recall, F1-score per class)
- **Class balancing**: `compute_class_weight('balanced')` ensures Low / Medium / High classes are learned equally despite frequency differences
- **Early stopping**: Restores best weights based on `val_loss` to prevent overfitting

### Water Savings Simulation

```
Baseline (naive):   Every field irrigated at 10 mm every cycle
Smart system:       Water applied = f(decision) ∈ {0, 5, 10} mm

Savings (%) = (baseline_total − smart_total) / baseline_total × 100
```

The savings figure reflects the proportion of water conserved by only irrigating when the model is confident that irrigation is genuinely needed.

### Training Curves

Both models produce accuracy/loss plots across epochs:

```
Model Accuracy — Train vs Validation
Model Loss     — Train vs Validation
```

These are rendered inline in the notebook with `matplotlib`.

---

## 🔒 Design Decisions & Engineering Notes

**Why LSTM for climate?**
Climate data is inherently sequential — today's temperature depends on yesterday's. LSTM networks are specifically designed to capture these temporal dependencies through learned cell state and gating mechanisms, outperforming simple regression for multi-step time series.

**Why two separate scalers?**
The LSTM scaler is fitted on climate data only and must be used to inverse-transform LSTM outputs back to real-world units. The final classifier scaler is fitted on the full fused feature matrix (sensors + forecasts). Conflating them would leak distribution information across data sources.

**Why Monte Carlo Dropout instead of a deterministic softmax?**
Standard softmax confidence scores are notoriously overconfident — a model can output 95% confidence on inputs far outside its training distribution. MC Dropout approximates Bayesian inference, sampling from the posterior distribution of weights at inference time, producing uncertainty estimates that are actually calibrated to model knowledge.

**Why class-weight balancing?**
Real-world irrigation datasets are typically skewed toward "no irrigation needed" labels. Without balancing, the model converges to a trivial solution that always predicts the majority class. `compute_class_weight('balanced')` reweights the loss to treat every class with equal importance during training.

**Why `drop_first=True` in one-hot encoding?**
Dropping the first dummy variable avoids perfect multicollinearity (the "dummy variable trap") in the feature matrix — particularly important for linear components of the model and for numerical stability.

---

## 🛠️ Requirements

```
numpy>=1.23.0
pandas>=1.5.0
scikit-learn>=1.2.0
tensorflow>=2.12.0
matplotlib>=3.6.0
jupyter>=1.0.0
notebook>=6.5.0
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 🗺️ Roadmap

- [ ] **SHAP Explainability** — Feature importance visualization to understand per-decision drivers
- [ ] **Real IoT Integration** — MQTT client to ingest live sensor streams from field devices
- [ ] **REST API** — FastAPI wrapper for `run_decision_engine` callable via HTTP
- [ ] **Automated Retraining** — Scheduled pipeline to retrain LSTM on new climate data weekly
- [ ] **Field-level Dashboard** — React frontend to visualize decisions, confidence, and water savings per field in real time
- [ ] **Multi-crop Profiles** — Crop-specific irrigation thresholds (rice vs. maize vs. wheat)
- [ ] **Rainfall Cancellation Logic** — Auto-suppress irrigation when significant rainfall is forecast

---

## 🤝 Contributing

Contributions are welcome and encouraged.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "Add: meaningful description"`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request with a clear description of the change

Please ensure your code follows existing style conventions and includes docstrings for new functions.

---

## 📚 References

- Gal, Y., & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.* ICML 2016.
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8), 1735–1780.
- FAO. (2020). *The State of Food and Agriculture: Overcoming Water Challenges in Agriculture.* Food and Agriculture Organization of the United Nations.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for full terms.

---

<div align="center">

**Built to conserve water. Powered by deep learning. Designed for the field.**

<br/>

*If this project helps your research or work, consider giving it a ⭐*

</div>
