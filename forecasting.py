"""
AgriPredict — forecasting core.

A SARIMA + XGBoost hybrid for monthly commodity-price forecasting on REAL
Indian retail price data (WFP / data.humdata.org, public domain).

Design
------
1. SARIMA models trend + 12-month seasonality of the price series.
2. XGBoost models the SARIMA residuals using genuine engineered features
   (lagged prices, rolling statistics, lagged residuals, cyclical month
   encodings). This is the fix for the original version, where XGBoost only
   saw a time index and therefore added almost nothing.
3. Evaluation uses an expanding-window, one-step-ahead backtest — the correct
   protocol for time series — and is benchmarked against a seasonal-naive
   baseline. Every reported metric is measured, not hard-coded.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

DATA_FILE = "wfp_food_prices_ind.csv"

SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL = (1, 1, 0, 12)
N_LAGS = 3


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_dataset(path: str = DATA_FILE) -> pd.DataFrame:
    """Load and clean the WFP India retail price dataset."""
    df = pd.read_csv(path, skiprows=[1])  # second row is an HXL tag row
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["date", "price"])
    df = df[df["pricetype"] == "Retail"]
    return df


def get_series(df: pd.DataFrame, commodity: str, market: str) -> pd.Series:
    """Return a clean monthly price series for one commodity/market."""
    sub = df[(df["commodity"] == commodity) & (df["market"] == market)]
    if sub.empty:
        raise ValueError(f"No data for {commodity} @ {market}")
    s = sub.set_index("date")["price"].sort_index()
    # collapse duplicate months, regularise to month-start frequency
    s = s.groupby(pd.Grouper(freq="MS")).mean()
    s = s.interpolate(limit_direction="both")
    return s.dropna()


def available_pairs(df: pd.DataFrame, min_points: int = 180):
    """List (commodity, market) pairs with enough history to model."""
    g = df.groupby(["commodity", "market"]).size()
    return g[g >= min_points].sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# Feature engineering for the residual model
# --------------------------------------------------------------------------- #
def _make_features(price: pd.Series, resid: pd.Series) -> pd.DataFrame:
    """Build genuine features for predicting SARIMA residuals."""
    feat = pd.DataFrame(index=price.index)
    for lag in range(1, N_LAGS + 1):
        feat[f"price_lag{lag}"] = price.shift(lag)
        feat[f"resid_lag{lag}"] = resid.shift(lag)
    feat["roll_mean3"] = price.shift(1).rolling(3).mean()
    feat["roll_std3"] = price.shift(1).rolling(3).std()
    month = price.index.month
    feat["month_sin"] = np.sin(2 * np.pi * month / 12)
    feat["month_cos"] = np.cos(2 * np.pi * month / 12)
    return feat


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = actual - pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mask = actual != 0
    mape = float(np.mean(np.abs(err[mask] / actual[mask])) * 100)
    return {"rmse": rmse, "mae": mae, "mape": mape}


# --------------------------------------------------------------------------- #
# Backtest: expanding-window one-step-ahead
# --------------------------------------------------------------------------- #
def _fit_residual_model(series: pd.Series, res) -> XGBRegressor:
    """Fit the XGBoost residual model on a fitted SARIMA result."""
    fitted = res.predict(start=series.index[0], end=series.index[-1])
    resid = series - fitted
    feat = _make_features(series, resid).dropna()
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3,
                       subsample=0.9, colsample_bytree=0.9, random_state=42)
    xgb.fit(feat, resid.loc[feat.index])
    return xgb


def _residual_gate_helps(train: pd.Series, val_months: int = 24, margin: float = 0.03) -> bool:
    """
    Decide whether residual correction actually improves accuracy, judged on a
    held-out validation slice at the tail of the training window. Returns True
    only if the hybrid beats SARIMA-only by at least `margin` on validation
    (adaptive correction — avoids adding noise to already-clean forecasts).
    """
    if len(train) < val_months + 72:
        return False  # not enough history to trust the gate; prefer plain SARIMA
    inner_train = train.iloc[:-val_months]
    val = train.iloc[-val_months:]
    try:
        res = SARIMAX(inner_train, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL,
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        xgb = _fit_residual_model(inner_train, res)
    except Exception:
        return False

    hist_p, hist_r = inner_train.copy(), (inner_train - res.predict(
        start=inner_train.index[0], end=inner_train.index[-1]))
    s_err, h_err = [], []
    for ts, actual in val.items():
        s_fc = float(res.forecast(steps=1).iloc[0])
        tmp_p = pd.concat([hist_p, pd.Series([np.nan], index=[ts])])
        tmp_r = pd.concat([hist_r, pd.Series([np.nan], index=[ts])])
        frow = _make_features(tmp_p, tmp_r).iloc[[-1]]
        r_hat = 0.0 if frow.isnull().any(axis=1).iloc[0] else float(xgb.predict(frow)[0])
        s_err.append(abs(actual - s_fc))
        h_err.append(abs(actual - (s_fc + r_hat)))
        hist_p.loc[ts] = actual
        hist_r.loc[ts] = actual - s_fc
        res = res.append([actual], refit=False)
    return float(np.mean(h_err)) < (1.0 - margin) * float(np.mean(s_err))


def backtest(series: pd.Series, test_frac: float = 0.2, max_test: int = 48) -> dict:
    """
    Expanding-window, one-step-ahead backtest comparing:
      - seasonal-naive baseline (value from 12 months earlier)
      - SARIMA only
      - SARIMA + XGBoost hybrid (with adaptive residual gating)
    Returns measured metrics for each.
    """
    n = len(series)
    n_test = min(max_test, max(12, int(n * test_frac)))
    train = series.iloc[: n - n_test]

    # Adaptive gate: only apply residual correction if it helps on validation.
    use_residual = _residual_gate_helps(train)

    # Fit SARIMA once on the training window.
    model = SARIMAX(train, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL,
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)

    # In-sample residuals on the training window -> train the residual model.
    fitted_train = res.predict(start=train.index[0], end=train.index[-1])
    resid_train = train - fitted_train
    xgb = _fit_residual_model(train, res)

    # Walk forward over the test window, one step at a time.
    history_price = train.copy()
    history_resid = resid_train.copy()

    sarima_preds, hybrid_preds, snaive_preds, actuals = [], [], [], []

    for t in range(n - n_test, n):
        ts = series.index[t]
        actual = series.iloc[t]

        # SARIMA one-step forecast
        sarima_fc = float(res.forecast(steps=1).iloc[0])

        # Residual-model feature row built from history known at forecast time
        tmp_price = pd.concat([history_price, pd.Series([np.nan], index=[ts])])
        tmp_resid = pd.concat([history_resid, pd.Series([np.nan], index=[ts])])
        frow = _make_features(tmp_price, tmp_resid).iloc[[-1]]
        if not use_residual or frow.isnull().any(axis=1).iloc[0]:
            resid_hat = 0.0
        else:
            resid_hat = float(xgb.predict(frow)[0])

        hybrid_fc = sarima_fc + resid_hat

        # Seasonal-naive baseline: value 12 months earlier
        snaive = float(series.iloc[t - 12]) if t - 12 >= 0 else float(history_price.iloc[-1])

        sarima_preds.append(sarima_fc)
        hybrid_preds.append(hybrid_fc)
        snaive_preds.append(snaive)
        actuals.append(actual)

        # Update histories with the realised actual, then feed SARIMA the obs.
        realised_resid = actual - sarima_fc
        history_price.loc[ts] = actual
        history_resid.loc[ts] = realised_resid
        res = res.append([actual], refit=False)

    return {
        "n_train": len(train),
        "n_test": n_test,
        "seasonal_naive": _metrics(actuals, snaive_preds),
        "sarima": _metrics(actuals, sarima_preds),
        "hybrid": _metrics(actuals, hybrid_preds),
    }


# --------------------------------------------------------------------------- #
# Forecasting future values (used by the web app)
# --------------------------------------------------------------------------- #
def forecast_future(series: pd.Series, horizon: int = 6) -> pd.Series:
    """Fit on the full series and forecast `horizon` months ahead (hybrid)."""
    use_residual = _residual_gate_helps(series)
    model = SARIMAX(series, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL,
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)

    fitted = res.predict(start=series.index[0], end=series.index[-1])
    resid = series - fitted
    xgb = _fit_residual_model(series, res)

    sarima_fc = res.forecast(steps=horizon)

    # Recursive residual correction over the horizon.
    hist_price = series.copy()
    hist_resid = resid.copy()
    corrected = []
    for i, ts in enumerate(sarima_fc.index):
        tmp_price = pd.concat([hist_price, pd.Series([np.nan], index=[ts])])
        tmp_resid = pd.concat([hist_resid, pd.Series([np.nan], index=[ts])])
        frow = _make_features(tmp_price, tmp_resid).iloc[[-1]]
        if not use_residual or frow.isnull().any(axis=1).iloc[0]:
            resid_hat = 0.0
        else:
            resid_hat = float(xgb.predict(frow)[0])
        val = float(sarima_fc.iloc[i]) + resid_hat
        corrected.append(val)
        hist_price.loc[ts] = val
        hist_resid.loc[ts] = resid_hat
    return pd.Series(corrected, index=sarima_fc.index)
