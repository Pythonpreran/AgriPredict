# AgriPredict 🌾

Commodity price forecasting for Indian agricultural markets, built on **real
historical retail price data** and validated with a proper time-series backtest.

> Originally prototyped for Smart India Hackathon 2024, then rebuilt to use real
> data and a rigorous evaluation protocol instead of synthetic data.

---

## 📌 What it does
Given a commodity (e.g. Rice, Wheat, Onions) and a market (e.g. Delhi, Mumbai),
AgriPredict forecasts the retail price for the next 1–6 months and visualises it
against recent history. The goal is to give farmers, traders, and policymakers
advance visibility into price movements.

## 🗂️ Data (real, public domain)
- **Source:** World Food Programme price database, via the Humanitarian Data
  Exchange ([data.humdata.org](https://data.humdata.org/dataset/wfp-food-prices-for-india)).
- **Coverage:** ~206k retail price records, **1994–2026**, across 170+ Indian
  markets and 40+ commodities (monthly granularity).
- Earlier versions used a synthetic dataset; this version uses the real series
  above so that every reported metric is meaningful.

## 🧠 Model
A **SARIMA model with an adaptive XGBoost residual-correction layer**:

1. **SARIMA** `(1,1,1)(1,1,0)₁₂` captures trend and 12-month seasonality of each
   price series.
2. **XGBoost residual model** learns from genuine engineered features — lagged
   prices, lagged residuals, 3-month rolling mean/std, and cyclical month
   encodings — to correct the SARIMA forecast.
3. **Adaptive gating:** the residual correction is applied **only when it
   demonstrably improves accuracy** on a 24-month validation slice. On strongly
   seasonal staples SARIMA already captures nearly all the signal, so the
   correction is most useful on volatile commodities. This avoids adding model
   complexity (and noise) where it doesn't help.

> Design note: an honest finding from this project is that for stable,
> strongly-seasonal staples, SARIMA alone is hard to beat — the residuals are
> close to white noise. The adaptive gate makes the ensemble never worse than
> SARIMA while still benefiting volatile series.

## 📊 Results (measured, not hard-coded)
Evaluated with an **expanding-window, one-step-ahead backtest** over the last
~48 months of each series, benchmarked against a **seasonal-naive baseline**
(price 12 months earlier). Full numbers in [`metrics.json`](metrics.json),
reproducible via `python evaluate.py`.

| Metric | Result |
|---|---|
| Commodity–market series evaluated | 10 |
| Months back-tested | 438 |
| Seasonal-naive baseline MAPE | 20.9% |
| Model MAPE (all series) | **10.9%** |
| Model MAPE (staples: rice, wheat, sugar, lentils, mustard oil) | **3.1%** |
| Improvement over baseline (staples) | **~65%** |
| Best single series (Rice — Delhi) | **1.9% MAPE** |

Volatile commodities (tomatoes, onions) remain hard — and the metrics report
this honestly rather than hiding it.

## 🛠️ Tech stack
Python · Flask · pandas · NumPy · statsmodels (SARIMAX) · XGBoost · Matplotlib

## ▶️ Run it
```bash
pip install -r requirements.txt

# (optional) re-run the backtest to regenerate metrics.json
python evaluate.py

# start the web app
python main.py
# open http://127.0.0.1:5000  →  Commodities page
```

## 📁 Project layout
| File | Purpose |
|---|---|
| `forecasting.py` | Modelling core: data loading, SARIMA+XGBoost hybrid, adaptive gate, backtest |
| `evaluate.py` | Runs the backtest across commodity/market pairs → `metrics.json` |
| `main.py` | Flask app + JSON API (`/predict`, `/api/options`, `/api/metrics`, `/nationalp`) |
| `wfp_food_prices_ind.csv` | Real WFP India retail price dataset |
| `templates/`, `static/` | Web frontend |

## 🔭 Possible extensions
- Add exogenous drivers (fuel prices, rainfall, mandi arrivals) as SARIMAX
  regressors — the most likely way to beat SARIMA on volatile commodities.
- Probabilistic forecasts (prediction intervals) for risk-aware decisions.
