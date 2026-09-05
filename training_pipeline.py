"""
TRAINING PIPELINE
Run this after you have data in the Feature Store (i.e. after running
backfill_historical.py at least once). It will:
  1. Pull all historical features from Hopsworks
  2. Build 3 targets: AQI 24h / 48h / 72h ahead ("next 3 days")
  3. Train a Ridge Regression, a Random Forest, and a small neural net
  4. Evaluate all three with RMSE / MAE / R^2
  5. Save the best model (by average R^2 across horizons) to the
     Hopsworks Model Registry, and also locally to ./saved_model/

Run it with:
    python training_pipeline.py
"""

import os
import json
import shutil
import joblib
import numpy as np
import pandas as pd
import hopsworks

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import HOPSWORKS_API_KEY, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION, MODEL_NAME, HORIZONS_HOURS

FEATURE_COLUMNS = [
    "temperature", "humidity", "pressure", "wind_speed",
    "co", "no2", "o3", "so2", "pm10",
    "hour", "day", "month", "day_of_week",
    "aqi", "aqi_change_rate",
]
TARGET_PREFIX = "target_aqi_h"


def load_features() -> pd.DataFrame:
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()
    df = df.sort_values("date").reset_index(drop=True)
    return df, project


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Because data is hourly, 'h hours ahead' = shift the AQI column
    up by h rows."""
    df = df.copy()
    for h in HORIZONS_HOURS:
        df[f"{TARGET_PREFIX}{h}"] = df["aqi"].shift(-h)
    target_cols = [f"{TARGET_PREFIX}{h}" for h in HORIZONS_HOURS]
    df = df.dropna(subset=target_cols)
    return df


def evaluate(y_true, y_pred, label):
    results = {}
    for i, h in enumerate(HORIZONS_HOURS):
        rmse = mean_squared_error(y_true[:, i], y_pred[:, i]) ** 0.5
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        results[f"{h}h"] = {"RMSE": round(rmse, 2), "MAE": round(mae, 2), "R2": round(r2, 3)}
        print(f"  [{label}] +{h}h -> RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.3f}")
    return results


def main():
    print("Loading features from Hopsworks...")
    df, project = load_features()
    df = build_targets(df)

    if len(df) < 30:
        print(f"WARNING: only {len(df)} usable rows. Run backfill_historical.py "
              f"with more days, or wait for the hourly pipeline to collect more data.")

    target_cols = [f"{TARGET_PREFIX}{h}" for h in HORIZONS_HOURS]
    X = df[FEATURE_COLUMNS].fillna(0)
    y = df[target_cols].values

    # Time-based split: train on the earlier 80%, test on the most recent 20%
    # (never randomly shuffle time series data)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    all_results = {}
    models = {}

    print("\nTraining Ridge Regression...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    all_results["Ridge"] = evaluate(y_test, ridge.predict(X_test), "Ridge")
    models["Ridge"] = ridge

    print("\nTraining Random Forest...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    all_results["RandomForest"] = evaluate(y_test, rf.predict(X_test), "RandomForest")
    models["RandomForest"] = rf

    print("\nTraining Neural Network (MLP)...")
    nn = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42)
    nn.fit(X_train, y_train)
    all_results["NeuralNet"] = evaluate(y_test, nn.predict(X_test), "NeuralNet")
    models["NeuralNet"] = nn

    # Pick the best model by average R2 across all 3 horizons
    def avg_r2(name):
        return np.mean([all_results[name][f"{h}h"]["R2"] for h in HORIZONS_HOURS])

    best_name = max(all_results, key=avg_r2)
    best_model = models[best_name]
    print(f"\nBest model: {best_name} (avg R2 = {avg_r2(best_name):.3f})")

    # Save everything locally first
    os.makedirs("saved_model", exist_ok=True)
    joblib.dump(best_model, "saved_model/model.pkl")
    with open("saved_model/feature_columns.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f)
    with open("saved_model/metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)
    with open("saved_model/best_model_name.txt", "w") as f:
        f.write(best_name)
    print("Saved model + metrics to ./saved_model/")

    # Push to Hopsworks Model Registry
    print("Uploading to Hopsworks Model Registry...")
    mr = project.get_model_registry()
    model_meta = mr.python.create_model(
        name=MODEL_NAME,
        metrics={f"avg_r2": float(avg_r2(best_name))},
        description=f"Best model: {best_name}. Predicts AQI 24h/48h/72h ahead.",
    )
    model_meta.save("saved_model")
    print("Done. Model registered in Hopsworks Model Registry as "
          f"'{MODEL_NAME}'.")


if __name__ == "__main__":
    main()
