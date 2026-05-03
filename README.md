# Wind Velocity Forecasting for Rocket Launch Sites — ISRO

> Multi-model deep learning system for multi-horizon wind velocity prediction at India's primary spaceport (SDSC-SHAR, Sriharikota), built to support launch window optimization.

Built during an internship at **ISRO — Satish Dhawan Space Centre SHAR**, Range Operations division, under the supervision of **Ram Senthil C** (Sci/Eng-SF, Mission Computers, SCOF/RO).

---

## Overview

Wind velocity forecasting is mission-critical for rocket launch operations — strong or gusty winds can destabilize vehicles during ascent, deviate trajectories, or endanger surrounding areas. Launch teams at SDSC-SHAR require reliable multi-hour forecasts to identify safe launch windows and enforce go/no-go criteria.

This project builds and benchmarks multiple deep learning architectures for wind velocity prediction across 1hr, 2hr, and 3hr forecast horizons, with a focus on practical deployment on the SHAR Computer Facility (SCOF) HPC infrastructure.

---

## Results

### Model Comparison — 1 Hour Forecast Horizon

| Model | R² Score | RMSE (m/s) | MAE (m/s) |
|:---:|:---:|:---:|:---:|
| ARIMA (baseline) | 0.7134 | 0.8823 | 0.6241 |
| GRU | 0.8602 | 0.5890 | 0.4012 |
| **LSTM** | **0.8877** | **0.5374** | **0.3651** |
| BiLSTM | 0.8791 | 0.5512 | 0.3788 |
| LSTM + Attention | 0.8843 | 0.5427 | 0.3702 |

### Best Model (LSTM) Across Horizons

| Forecast Horizon | R² Score | RMSE (m/s) |
|:---:|:---:|:---:|
| **1 hour** | **0.8877** | **0.5374** |
| 2 hours | 0.7512 | 0.6922 |
| 3 hours | 0.6710 | 0.7960 |

Performance degrades with longer horizons — consistent with increasing atmospheric uncertainty. The vanilla LSTM outperformed attention-augmented variants on this dataset, likely due to the relatively low-dimensional feature space where attention overhead doesn't justify the added complexity.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│            Raw Meteorological Data                   │
│     10-min intervals: wind velocity, temperature,    │
│     pressure, humidity, wind direction                │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│              Data Preprocessing                      │
│  • Missing value interpolation (3-hr window)         │
│  • IQR-based spike detection & removal               │
│  • Min-Max normalization [0, 1]                      │
│  • Multivariate feature matrix construction          │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│             Sequence Generation                      │
│  Sliding window → (X, y) pairs                       │
│  70% train / 20% val / 10% test                      │
│  Window sizes: 6 (1hr), 12 (2hr), 18 (3hr)          │
└────────────────────────┬────────────────────────────┘
                         │
            ┌────────────┼────────────┬──────────┐
            ▼            ▼            ▼          ▼
     ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐
     │  ARIMA   │ │   LSTM   │ │ BiLSTM   │ │  GRU   │
     │(baseline)│ │(100 unit)│ │(2x50 u.) │ │(100 u.)│
     └────┬─────┘ └────┬─────┘ └────┬─────┘ └───┬────┘
          │             │            │            │
          └─────────────┴────────────┴────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│            Evaluation & Comparison                   │
│  • R², RMSE, MAE per model per horizon               │
│  • Actual vs Predicted visualizations                │
│  • Residual analysis & error distributions           │
│  • Statistical significance testing                  │
└─────────────────────────────────────────────────────┘
```

---

## Key Technical Details

- **Data**: Meteorological time series at 10-minute intervals with multivariate features (wind velocity, temperature, pressure, humidity)
- **Missing data**: Linear interpolation + forward/backward fill within 3-hour windows; IQR-based spike detection and removal
- **Feature engineering**: Multivariate input matrix, lag features, rolling statistics (mean, std over configurable windows)
- **Models benchmarked**: ARIMA (statistical baseline), GRU, LSTM, Bidirectional LSTM, LSTM with Attention
- **Training**: Adam optimizer, MSE loss, early stopping on validation loss, batch size 1024
- **Multi-horizon**: Separate models for 1hr, 2hr, 3hr forecasts with corresponding sequence window sizes

---

## Tech Stack

`TensorFlow` `Keras` `Python` `NumPy` `Pandas` `Scikit-learn` `Matplotlib` `statsmodels`

---

## Project Structure

```
isro-wind-forecasting/
├── src/
│   ├── data_preprocessing.py      # Download, clean, spike removal
│   ├── feature_engineering.py     # Multivariate features, sequences
│   ├── models/
│   │   ├── arima_baseline.py      # ARIMA statistical baseline
│   │   ├── lstm_model.py          # Vanilla LSTM
│   │   ├── bilstm_model.py        # Bidirectional LSTM
│   │   ├── gru_model.py           # GRU
│   │   └── attention_lstm.py      # LSTM + Attention mechanism
│   ├── train.py                   # Unified training loop
│   ├── evaluate.py                # Metrics, comparison tables
│   └── visualize.py               # Plotting utilities
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory data analysis
│   ├── 02_model_comparison.ipynb  # Side-by-side benchmarks
│   └── 03_error_analysis.ipynb    # Residuals, distributions
├── configs/
│   └── model_config.yaml
├── results/
│   ├── forecast_plots/
│   ├── comparison_tables/
│   └── metrics.json
└── README.md
```

---

## Setup

```bash
git clone https://github.com/rohitharavindra08/isro-wind-forecasting.git
cd isro-wind-forecasting
pip install -r requirements.txt

# Train all models (1-hour horizon)
python src/train.py --horizon 1 --models lstm bilstm gru arima

# Train specific model
python src/train.py --horizon 1 --models lstm --epochs 100 --batch_size 1024

# Evaluate & compare
python src/evaluate.py --horizon 1 --compare all

# Run all horizons
python src/train.py --horizon 1 2 3 --models lstm
```

---

## Acknowledgments

Built at **ISRO — Satish Dhawan Space Centre SHAR** (Sriharikota, Andhra Pradesh), Range Operations division. Supervised by **Ram Senthil C**, Deputy Manager, Mission Computers, SCOF/RO.

