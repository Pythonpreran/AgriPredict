"""
Evaluate the AgriPredict hybrid model on real Indian retail price data.

Runs an expanding-window, one-step-ahead backtest for a set of major
commodity/market series and reports honest, measured metrics plus the
improvement of the SARIMA+XGBoost hybrid over a seasonal-naive baseline.

Results are written to metrics.json so the web app and README can cite them.
"""

import json
import numpy as np
import pandas as pd

import forecasting as fc

# Major commodities x liquid markets with long histories.
PAIRS = [
    ("Rice", "Mumbai"),
    ("Rice", "Delhi"),
    ("Wheat", "Chennai"),
    ("Wheat", "Jaipur"),
    ("Sugar", "Delhi"),
    ("Onions", "Mumbai"),
    ("Potatoes", "Delhi"),
    ("Oil (mustard)", "Kolkata"),
    ("Tomatoes", "Chennai"),
    ("Lentils (masur)", "Delhi"),
]


def main():
    df = fc.load_dataset()
    rows = []
    for commodity, market in PAIRS:
        try:
            s = fc.get_series(df, commodity, market)
            if len(s) < 120:
                print(f"skip {commodity}@{market} (only {len(s)} pts)")
                continue
            r = fc.backtest(s)
            rows.append({
                "commodity": commodity,
                "market": market,
                "n_train": r["n_train"],
                "n_test": r["n_test"],
                "baseline_mape": r["seasonal_naive"]["mape"],
                "sarima_mape": r["sarima"]["mape"],
                "hybrid_mape": r["hybrid"]["mape"],
                "hybrid_rmse": r["hybrid"]["rmse"],
                "hybrid_mae": r["hybrid"]["mae"],
            })
            print(f"{commodity:16s} {market:10s} | "
                  f"baseline MAPE {r['seasonal_naive']['mape']:5.2f}%  "
                  f"SARIMA {r['sarima']['mape']:5.2f}%  "
                  f"hybrid {r['hybrid']['mape']:5.2f}%  "
                  f"(RMSE {r['hybrid']['rmse']:.2f})")
        except Exception as e:
            print(f"error {commodity}@{market}: {e}")

    res = pd.DataFrame(rows)
    if res.empty:
        print("No results.")
        return

    baseline = res["baseline_mape"].mean()
    sarima = res["sarima_mape"].mean()
    hybrid = res["hybrid_mape"].mean()
    impr_vs_baseline = (baseline - hybrid) / baseline * 100
    impr_vs_sarima = (sarima - hybrid) / sarima * 100

    # Staple grains/essentials: the stable, high-signal series.
    staples = res[res["commodity"].isin(
        ["Rice", "Wheat", "Sugar", "Lentils (masur)", "Oil (mustard)"])]
    staple_mape = staples["hybrid_mape"].mean()
    staple_baseline = staples["baseline_mape"].mean()

    summary = {
        "n_series": int(len(res)),
        "total_months_evaluated": int(res["n_test"].sum()),
        "mean_baseline_mape": round(baseline, 2),
        "mean_sarima_mape": round(sarima, 2),
        "mean_hybrid_mape": round(hybrid, 2),
        "hybrid_improvement_over_baseline_pct": round(impr_vs_baseline, 1),
        "hybrid_improvement_over_sarima_pct": round(impr_vs_sarima, 1),
        "staples_mean_mape": round(staple_mape, 2),
        "staples_baseline_mape": round(staple_baseline, 2),
        "staples_improvement_pct": round((staple_baseline - staple_mape) / staple_baseline * 100, 1),
        "best_hybrid_mape": round(res["hybrid_mape"].min(), 2),
        "per_series": rows,
    }

    with open("metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== SUMMARY (measured on real data) =====")
    print(f"Series evaluated         : {summary['n_series']}")
    print(f"Months back-tested       : {summary['total_months_evaluated']}")
    print(f"Seasonal-naive MAPE      : {summary['mean_baseline_mape']}%")
    print(f"Model MAPE (all)         : {summary['mean_hybrid_mape']}%")
    print(f"Model MAPE (staples)     : {summary['staples_mean_mape']}%")
    print(f"Improvement vs baseline  : {summary['hybrid_improvement_over_baseline_pct']}% (all), "
          f"{summary['staples_improvement_pct']}% (staples)")
    print("Wrote metrics.json")


if __name__ == "__main__":
    main()
