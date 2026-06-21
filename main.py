"""
AgriPredict — Flask web app.

Serves SARIMA(+adaptive XGBoost) commodity-price forecasts trained on REAL
Indian retail price data (World Food Programme, via data.humdata.org —
public domain). See forecasting.py for the modelling core and evaluate.py
for the backtest that produced metrics.json.
"""

import json
import os
import base64
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template, request, jsonify

import forecasting as fc

app = Flask(__name__)

# --------------------------------------------------------------------------- #
# Load data once at startup and precompute selectable options.
# --------------------------------------------------------------------------- #
print("Loading dataset ...")
DF = fc.load_dataset()
PAIRS = fc.available_pairs(DF, min_points=150)

# commodity -> sorted list of markets with enough history
OPTIONS = {}
for (commodity, market), _ in PAIRS.items():
    OPTIONS.setdefault(commodity, []).append(market)
for c in OPTIONS:
    OPTIONS[c] = sorted(OPTIONS[c])
OPTIONS = dict(sorted(OPTIONS.items()))

_FORECAST_CACHE = {}  # (commodity, market) -> dict


def _load_metrics():
    path = os.path.join(os.path.dirname(__file__), "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# --------------------------------------------------------------------------- #
# Page routes
# --------------------------------------------------------------------------- #
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/prediction")
def prediction():
    return render_template("commodities.html")


@app.route("/commodities")
def commodities():
    return render_template("commodities.html")


@app.route("/supply")
def supply():
    return render_template("disaster.html")


@app.route("/disaster")
def disaster():
    return render_template("disaster.html")


@app.route("/aboutus")
def aboutus():
    return "About Us page coming soon..."


@app.route("/features")
def features():
    return "Features page coming soon..."


@app.route("/my_account")
def my_account():
    return "My Account page coming soon..."


# --------------------------------------------------------------------------- #
# API routes
# --------------------------------------------------------------------------- #
@app.route("/api/options")
def api_options():
    """Return the available commodities and their markets."""
    return jsonify(OPTIONS)


@app.route("/api/metrics")
def api_metrics():
    """Return the measured backtest metrics."""
    return jsonify(_load_metrics())


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    commodity = data.get("commodity")
    market = data.get("market") or data.get("center")

    if commodity not in OPTIONS or market not in OPTIONS.get(commodity, []):
        return jsonify({"error": "Invalid commodity or market"}), 400

    key = (commodity, market)
    if key in _FORECAST_CACHE:
        return jsonify(_FORECAST_CACHE[key])

    series = fc.get_series(DF, commodity, market)
    forecast = fc.forecast_future(series, horizon=6)

    # Build chart: history (last 3 yrs) + forecast.
    recent = series.iloc[-36:]
    plt.figure(figsize=(11, 5.5))
    plt.plot(recent.index, recent.values, color="#1f6feb",
             marker="o", markersize=3, linewidth=1.8, label="Actual price")
    plt.plot(forecast.index, forecast.values, color="#d1495b",
             marker="s", markersize=4, linewidth=2, linestyle="--",
             label="Forecast (6 months)")
    plt.title(f"{commodity} — {market}", fontsize=15, weight="bold")
    plt.xlabel("Date"); plt.ylabel("Price (INR per unit)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(); plt.tight_layout()
    img = BytesIO()
    plt.savefig(img, format="png", dpi=90)
    plt.close()
    img.seek(0)
    chart = base64.b64encode(img.getvalue()).decode("utf-8")

    result = {
        "currentPrice": round(float(series.iloc[-1]), 2),
        "oneMonthPrediction": round(float(forecast.iloc[0]), 2),
        "threeMonthPrediction": round(float(forecast.iloc[2]), 2),
        "sixMonthPrediction": round(float(forecast.iloc[5]), 2),
        "unit": "INR",
        "chartData": {"chart": chart},
    }
    _FORECAST_CACHE[key] = result
    return jsonify(result)


@app.route("/nationalp", methods=["POST"])
def national_prediction():
    """National average historical trend for a commodity (across all markets)."""
    data = request.get_json(force=True)
    commodity = data.get("commodity")
    if commodity not in OPTIONS:
        return jsonify({"error": "Invalid commodity"}), 400

    sub = DF[DF["commodity"] == commodity]
    national = (sub.set_index("date")["price"]
                .groupby(pd.Grouper(freq="MS")).mean()
                .interpolate(limit_direction="both").dropna())

    plt.figure(figsize=(11, 5))
    plt.plot(national.index, national.values, color="#1f6feb", linewidth=1.6)
    plt.title(f"{commodity} — National Average (Retail)", fontsize=14, weight="bold")
    plt.xlabel("Date"); plt.ylabel("Price (INR per unit)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    img = BytesIO()
    plt.savefig(img, format="png", dpi=90)
    plt.close()
    img.seek(0)
    chart = base64.b64encode(img.getvalue()).decode("utf-8")

    return jsonify({
        "currentPrice": round(float(national.iloc[-1]), 2),
        "chartData": {"chart": "data:image/png;base64," + chart},
    })


if __name__ == "__main__":
    print(f"Loaded {len(OPTIONS)} commodities, "
          f"{sum(len(v) for v in OPTIONS.values())} commodity-market series.")
    app.run(debug=False)
