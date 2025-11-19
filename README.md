README.md — NAFM Power Consumption Forecasting
 Conditional Normalizing Flow for Multivariate Power Consumption Forecasting

(Baseline VAR vs Neural Autoregressive Flow Model — NAFM)

This project builds a probabilistic time-series forecasting model using a Conditional Masked Autoregressive Flow (MAF) to predict next-step power consumption across 3 industrial zones.
A traditional Vector Autoregression (VAR) model is used as a baseline.

The goal is to compare:

Point Forecast Accuracy (MSE)

Uncertainty Estimation Quality (Coverage)

Prediction Interval Widths

Forecast Visualization

Neural Flow vs Classical VAR

The project uses a multivariate time series dataset containing:

Temperature

Humidity

Wind speed

Diffuse solar flux

Power consumption in 3 zones

Project Structure
NAFM-PowerForecasting/
│
├── NAFM_Project.ipynb                 # Full Colab notebook (end-to-end)
├── README.md                          # Project documentation
├── requirements.txt                   # Dependencies
├── .gitignore                         # Ignore checkpoints & large data
│
├── data/
│   └── sample_powerconsumption.csv    # Optional (sample only)
│
├── models/
│   └── nafm_model_stable.pth          # Trained Conditional Normalizing Flow
│
├── results/
│   ├── baseline_forecast.csv          # VAR baseline predictions
│   ├── nafm_forecast.csv              # NAFM probabilistic predictions
│   └── metrics_summary.csv            # MSE, coverage, interval width
│
└── plots/
    └── forecast_zone1.png             # Forecast comparison plot (Zone 1)

Methodology Overview
Load & preprocess the time series

Parse datetime

Standardize numeric features

Create multi-step windows

Split into train-test sets

Baseline: Vector Autoregression (VAR)

Train a classical VAR(1) model

Generate point forecasts

Compute standard deviation for intervals

Save baseline forecast CSV

Main Model: Conditional Masked Autoregressive Flow (NAFM)

A Normalizing Flow is trained to model:

P(Y(t) | X(t − input_window : t))

Key features:

Conditional MAF architecture

Normalization of inputs and outputs

Multi-layer autoregressive transforms

Learned probabilistic output distribution

Monte Carlo sampling (200 samples per step)

Evaluation Metrics

For each target variable:

Mean Squared Error (MSE)

Coverage Probability
(True value lies within prediction interval)

Interval Width
(Narrow intervals = confident predictions)

Visualization

True vs Baseline vs NAFM forecasts

Confidence intervals (5% – 95%)

Zone1 forecast plot saved in /plots/

Key Results Summary

NAFM significantly outperforms the baseline VAR model:

Variable	MSE Baseline	MSE NAFM	Coverage Baseline	Coverage NAFM
Zone1	50.9M	48.5M	5.6%	92.6%
Zone2	37.9M	1.77M	3.8%	93.0%
Zone3	62.0M	2.12M	1.7%	93.0%
Interpretation:

NAFM is far more accurate (10×–20× lower MSE for Zone 2 & 3)

Baseline VAR intervals fail completely (coverage < 6%)

NAFM intervals are well-calibrated (~93% coverage)

Flow model captures realistic uncertainty

 How to Run the Project
Option 1: Run on Google Colab (Recommended)

Upload repository to Colab

Mount Google Drive

Install dependencies

Run each cell in NAFM_Project.ipynb

Option 2: Local Environment
pip install -r requirements.txt


Then open notebook:

jupyter notebook NAFM_Project.ipynb

Dependencies

See: requirements.txt

Key packages:

PyTorch

nflows

pandas

numpy

statsmodels

scikit-learn

matplotlib

Dataset Source

Power consumption & environmental variable dataset from Kaggle:
"Household Power Consumption with Weather"
(Or custom uploaded dataset depending on usage)

License

This project is for academic and research use.

Acknowledgements

Normalizing Flows (nflows library)

Vector Autoregression (statsmodels)

Google Colab GPU/CPU support