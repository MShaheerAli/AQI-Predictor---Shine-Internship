"""
DASHBOARD
Run locally with:
    streamlit run app.py
Or deploy for free at https://share.streamlit.io by pointing it at your
GitHub repo (see README.md).
"""

import json
import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import hopsworks
import shap

from config import (
    CITY_NAME, HOPSWORKS_API_KEY, MODEL_NAME, HORIZONS_HOURS, HAZARD_THRESHOLD,
)
from utils import fetch_current_weather, fetch_current_pollution, add_time_features, add_derived_features

st.set_page_config(page_title=f"{CITY_NAME} AQI Forecast", page_icon="🌤️", layout="centered")


@st.cache_resource(show_spinner="Connecting to Hopsworks and loading the latest model...")
def load_model():
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    mr = project.get_model_registry()
    model_obj = mr.get_best_model(MODEL_NAME, "avg_r2", "max")
    model_dir = model_obj.download()
    model = joblib.load(f"{model_dir}/model.pkl")
    with open(f"{model_dir}/feature_columns.json") as f:
        feature_columns = json.load(f)
    with open(f"{model_dir}/metrics.json") as f:
        metrics = json.load(f)
    with open(f"{model_dir}/best_model_name.txt") as f:
        model_name = f.read().strip()
    return model, feature_columns, metrics, model_name


def get_latest_row():
    weather = fetch_current_weather()
    pollution = fetch_current_pollution()
    row = {
        "city": CITY_NAME,
        "date": datetime.datetime.utcnow(),
        **weather,
        **pollution,
    }
    df = pd.DataFrame([row])
    df = add_time_features(df)
    df = add_derived_features(df)
    return df


def aqi_label(value: float) -> str:
    if value <= 50: return "Good"
    if value <= 100: return "Moderate"
    if value <= 150: return "Unhealthy for Sensitive Groups"
    if value <= 200: return "Unhealthy"
    if value <= 300: return "Very Unhealthy"
    return "Hazardous"


st.title(f"🌤️ {CITY_NAME} — 3-Day AQI Forecast")
st.caption("Pearls AQI Predictor — serverless ML pipeline (Hopsworks + scikit-learn + Streamlit)")

try:
    model, feature_columns, metrics, model_name = load_model()
    latest = get_latest_row()

    X = latest[feature_columns].fillna(0)
    preds = model.predict(X)[0]  # array of 3 values, one per horizon

    st.subheader("Current conditions")
    c1, c2, c3 = st.columns(3)
    c1.metric("Current AQI", f"{latest['aqi'].iloc[0]:.0f}", aqi_label(latest['aqi'].iloc[0]))
    c2.metric("PM2.5 (µg/m³)", f"{latest['pm2_5'].iloc[0]:.1f}")
    c3.metric("Temperature (°C)", f"{latest['temperature'].iloc[0]:.1f}")

    st.subheader("Forecast — next 3 days")
    days = [f"+{h//24}d" for h in HORIZONS_HOURS]
    forecast_df = pd.DataFrame({"Day": days, "Predicted AQI": preds})
    fig, ax = plt.subplots()
    ax.plot(forecast_df["Day"], forecast_df["Predicted AQI"], marker="o", linewidth=2)
    ax.axhline(HAZARD_THRESHOLD, color="red", linestyle="--", label="Hazard threshold")
    ax.set_ylabel("AQI")
    ax.legend()
    st.pyplot(fig)

    if (preds >= HAZARD_THRESHOLD).any():
        worst_day = days[int(np.argmax(preds))]
        st.error(
            f"⚠️ ALERT: predicted AQI reaches hazardous levels "
            f"({preds.max():.0f}) around {worst_day}. Consider limiting outdoor activity."
        )
    else:
        st.success("No hazardous AQI levels predicted in the next 3 days.")

    st.subheader("Model performance")
    st.caption(f"Best model selected during training: **{model_name}**")
    metrics_rows = []
    for m_name, horizons in metrics.items():
        for h_label, vals in horizons.items():
            metrics_rows.append({"Model": m_name, "Horizon": h_label, **vals})
    st.dataframe(pd.DataFrame(metrics_rows), hide_index=True)

    st.subheader("Why did the model predict this? (SHAP)")
    try:
        explainer = shap.Explainer(model.predict, X)
        shap_values = explainer(X)
        fig2, ax2 = plt.subplots()
        # explain the first (24h) horizon's output only —
        # shap.plots.bar needs single-output values, not the full
        # (features, horizons) array the multi-output model produces
        shap.plots.bar(shap_values[0, :, 0], show=False)
        st.pyplot(fig2)
    except Exception as e:
        st.info(f"SHAP explanation unavailable for this model type ({e}).")

except Exception as e:
    st.error(
        "Could not load the model or live data. Make sure you've run "
        "feature_pipeline.py, backfill_historical.py and training_pipeline.py "
        f"at least once, and that your API keys are set correctly.\n\nDetails: {e}"
    )
