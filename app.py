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
import plotly.graph_objects as go
import hopsworks
import shap

from config import (
    CITY_NAME, HOPSWORKS_API_KEY, MODEL_NAME, HORIZONS_HOURS, HAZARD_THRESHOLD,
    FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION,
)
from utils import fetch_current_weather, fetch_current_pollution, add_time_features, add_derived_features

st.set_page_config(page_title=f"{CITY_NAME} AQI Forecast", page_icon="🌤️", layout="centered")

# ---------------------------------------------------------------------
# AQI reference scale — standard US EPA bands, each with a display
# color, a light "chip" background, and a short actionable health tip.
# Used to color KPI badges, table cells, and the scale/chart bands so
# the same categorization appears consistently everywhere on the page.
# ---------------------------------------------------------------------
AQI_BANDS = [
    {"low": 0, "high": 50, "label": "Good", "color": "#2E7D32", "bg": "#E8F5E9",
     "advice": "Air quality is satisfactory. Enjoy normal outdoor activities."},
    {"low": 51, "high": 100, "label": "Moderate", "color": "#B8860B", "bg": "#FFF8E1",
     "advice": "Acceptable air quality. Unusually sensitive people should consider reducing prolonged outdoor exertion."},
    {"low": 101, "high": 150, "label": "Unhealthy for Sensitive Groups (children, elderly, asthma)",
     "color": "#E65100", "bg": "#FFF3E0",
     "advice": "Children, older adults, and people with respiratory issues should limit prolonged outdoor exertion."},
    {"low": 151, "high": 200, "label": "Unhealthy", "color": "#C62828", "bg": "#FFEBEE",
     "advice": "Everyone may begin to experience health effects. Consider wearing a mask outdoors and limiting exertion."},
    {"low": 201, "high": 300, "label": "Very Unhealthy", "color": "#6A1B9A", "bg": "#F3E5F5",
     "advice": "Health alert: avoid outdoor exertion. Run an air purifier indoors if you have one."},
    {"low": 301, "high": 500, "label": "Hazardous", "color": "#7B0000", "bg": "#FBE9E7",
     "advice": "Health emergency: everyone should avoid all outdoor exertion and stay indoors with air filtration."},
]


def get_aqi_band(aqi: float) -> dict:
    aqi = max(0.0, min(500.0, float(aqi)))
    for band in AQI_BANDS:
        if band["low"] <= aqi <= band["high"]:
            return band
    return AQI_BANDS[-1]


def aqi_badge_html(aqi: float, size: str = "0.85rem") -> str:
    band = get_aqi_band(aqi)
    return (
        f'<span style="background:{band["bg"]}; color:{band["color"]}; '
        f'padding:3px 10px; border-radius:999px; font-weight:600; '
        f'font-size:{size}; white-space:nowrap;">{band["label"]}</span>'
    )


# ---------------------------------------------------------------------
# Global styling — light CSS tweaks layered on top of Streamlit's own
# theme variables (so it still adapts to light/dark mode) rather than
# fighting it with a full custom theme.
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 900px; }
    h1, h2, h3 { letter-spacing: -0.01em; }
    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128,128,128,0.18);
        border-radius: 12px;
        padding: 0.9rem 1rem 0.7rem;
    }
    div[data-testid="stMetricLabel"] { font-size: 0.85rem; opacity: 0.75; }
    .kpi-caption { font-size: 0.8rem; opacity: 0.7; margin-top: -0.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


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


@st.cache_resource(show_spinner="Loading historical data for SHAP background...")
def load_shap_background(feature_columns, n=150):
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()[feature_columns].dropna()
    if len(df) > n:
        df = df.sample(n, random_state=42)
    return df.reset_index(drop=True)


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


def primary_pollutant(row: pd.Series) -> tuple[str, float]:
    """Picks whichever pollutant is closest to (or over) its own rough
    'of concern' reference level, as a simple relative-severity heuristic
    (not an official sub-index)."""
    refs = {
        "PM2.5": (row["pm2_5"], 150.0),
        "PM10": (row["pm10"], 250.0),
        "Ozone (O3)": (row["o3"], 180.0),
        "NO2": (row["no2"], 200.0),
        "SO2": (row["so2"], 500.0),
        "CO": (row["co"], 10000.0),
    }
    name, (value, ref) = max(refs.items(), key=lambda kv: kv[1][0] / kv[1][1])
    return name, value


def confidence_label(avg_r2: float) -> tuple[str, str]:
    if avg_r2 >= 0.7:
        return "High", "#2E7D32"
    if avg_r2 >= 0.3:
        return "Moderate", "#B8860B"
    if avg_r2 >= 0:
        return "Low", "#E65100"
    return "Very Low", "#C62828"


def render_aqi_scale(current_aqi: float):
    """A horizontal, color-coded legend of the 6 standard AQI bands,
    with the band containing the current reading visually raised and
    outlined so it's obvious at a glance where today's air quality sits."""
    active_low = get_aqi_band(current_aqi)["low"]
    chips = []
    for band in AQI_BANDS:
        is_active = band["low"] == active_low
        border = f'3px solid {band["color"]}'
        lift = "transform: translateY(-5px); box-shadow: 0 3px 8px rgba(0,0,0,0.15);" if is_active else ""
        marker = "▼" if is_active else "&nbsp;"
        range_label = f'{band["low"]}-{band["high"]}' if band["high"] < 500 else f'{band["low"]}+'
        chips.append(
            f'<div style="flex:1; text-align:center; min-width:0;">'
            f'<div style="font-size:0.9rem; line-height:1.1; color:{band["color"] if is_active else "transparent"};">{marker}</div>'
            f'<div style="background:{band["color"]}; color:white; padding:5px 2px; '
            f'font-size:0.68rem; font-weight:700; border-radius:8px 8px 0 0; '
            f'border:{border}; border-bottom:none; {lift}">{range_label}</div>'
            f'<div style="background:{band["bg"]}; color:#222; padding:4px 3px; min-height:3.6em; '
            f'font-size:0.62rem; line-height:1.15; border-radius:0 0 8px 8px; '
            f'border:{border}; border-top:none; {lift}">{band["label"]}</div>'
            f'</div>'
        )
    st.markdown(
        f'<div style="display:flex; gap:4px; margin: 0.4rem 0 1.1rem;">{"".join(chips)}</div>',
        unsafe_allow_html=True,
    )


def render_forecast_table(forecast_times, preds, rmse_by_horizon):
    rows = []
    for t, p, rmse in zip(forecast_times, preds, rmse_by_horizon):
        band = get_aqi_band(p)
        rows.append({
            "Date / time": t.strftime("%b %d, %I %p"),
            "Predicted AQI": round(float(p), 1),
            "± RMSE": round(float(rmse), 1),
            "Category": band["label"],
        })
    df = pd.DataFrame(rows)

    def style_category(val):
        band = next(b for b in AQI_BANDS if b["label"] == val)
        return f'background-color:{band["bg"]}; color:{band["color"]}; font-weight:600; border-radius:6px;'

    styled = (
        df.style
        .applymap(style_category, subset=["Category"])
        .format({"Predicted AQI": "{:.1f}", "± RMSE": "±{:.1f}"})
    )
    st.dataframe(styled, hide_index=True, use_container_width=True)


def render_forecast_chart(forecast_times, preds, rmse_by_horizon, hazard_threshold):
    upper = np.clip(preds + rmse_by_horizon, 0, 500)
    lower = np.clip(preds - rmse_by_horizon, 0, 500)
    categories = [get_aqi_band(p)["label"] for p in preds]

    # zoom the y-axis to the data + threshold instead of the full 0-500
    # scale, so the shaded AQI bands provide context without flattening
    # the actual forecast line into an unreadable sliver
    y_top = max(float(upper.max()), hazard_threshold) * 1.25
    y_range = [0, y_top]

    fig = go.Figure()

    # Shaded AQI-category bands behind the forecast line, so it's
    # visually obvious the moment the trend crosses into a worse zone.
    for band in AQI_BANDS:
        fig.add_hrect(
            y0=band["low"], y1=min(band["high"], 500),
            fillcolor=band["color"], opacity=0.07, line_width=0,
        )

    # Rough uncertainty band using each horizon's historical RMSE —
    # labeled honestly as an error margin, not a true statistical CI.
    fig.add_trace(go.Scatter(
        x=list(forecast_times) + list(forecast_times)[::-1],
        y=list(upper) + list(lower)[::-1],
        fill="toself", fillcolor="rgba(31,119,180,0.15)",
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
        showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=forecast_times, y=preds, mode="lines+markers",
        line=dict(color="#1f77b4", width=3),
        marker=dict(size=11, color="#1f77b4", line=dict(color="white", width=1)),
        customdata=np.stack([categories, rmse_by_horizon], axis=-1),
        hovertemplate=(
            "<b>%{x|%b %d, %I %p}</b><br>"
            "Predicted AQI: %{y:.0f} (± %{customdata[1]:.0f})<br>"
            "Category: %{customdata[0]}<extra></extra>"
        ),
        name="Forecast",
    ))

    fig.add_hline(
        y=hazard_threshold, line_dash="dash", line_color="#C62828", line_width=1.5,
        annotation_text="Hazard threshold", annotation_position="top left",
        annotation_font_color="#C62828",
    )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=25, b=10),
        height=340,
        showlegend=False,
        yaxis=dict(title="AQI", gridcolor="rgba(128,128,128,0.15)", zeroline=False, range=y_range),
        xaxis=dict(title=None, gridcolor="rgba(128,128,128,0)"),
        hoverlabel=dict(bgcolor="white", font_color="#222"),
    )
    st.plotly_chart(fig, use_container_width=True)


def style_metrics_table(metrics_rows, best_model_name):
    df = pd.DataFrame(metrics_rows)
    df.insert(0, "Selected", np.where(df["Model"] == best_model_name, "✓", ""))

    def style_r2(val):
        if val >= 0.5:
            color = "#2E7D32"
        elif val >= 0:
            color = "#B8860B"
        elif val >= -1:
            color = "#E65100"
        else:
            color = "#C62828"
        return f"color:{color}; font-weight:700;"

    def style_selected_row(row):
        is_best = row["Model"] == best_model_name
        return ["background-color: rgba(31,119,180,0.08)" if is_best else "" for _ in row]

    styled = (
        df.style
        .apply(style_selected_row, axis=1)
        .applymap(style_r2, subset=["R2"])
        .format({"RMSE": "{:.2f}", "MAE": "{:.2f}", "R2": "{:.3f}"})
    )
    st.dataframe(styled, hide_index=True, use_container_width=True)


st.title(f"🌤️ {CITY_NAME} — 3-Day AQI Forecast")
st.caption("Pearls AQI Predictor — serverless ML pipeline (Hopsworks + scikit-learn + Streamlit)")

try:
    model, feature_columns, metrics, model_name = load_model()
    latest = get_latest_row()

    X = latest[feature_columns].fillna(0)
    preds = model.predict(X)[0]  # array of 3 values, one per horizon
    row = latest.iloc[0]
    current_aqi = float(row["aqi"])
    band = get_aqi_band(current_aqi)

    trend_delta = preds[0] - current_aqi
    poll_name, poll_value = primary_pollutant(row)
    avg_r2 = float(np.mean([metrics[model_name][f"{h}h"]["R2"] for h in HORIZONS_HOURS]))
    conf_text, conf_color = confidence_label(avg_r2)
    rmse_by_horizon = np.array([metrics[model_name][f"{h}h"]["RMSE"] for h in HORIZONS_HOURS])

    # ---------------- KPI summary cards ----------------
    st.subheader("Current conditions")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Current AQI", f"{current_aqi:.0f}")
        st.markdown(aqi_badge_html(current_aqi, "0.72rem"), unsafe_allow_html=True)
    with k2:
        st.metric("24h Trend", f"{preds[0]:.0f}", delta=f"{trend_delta:+.0f}", delta_color="inverse")
        st.markdown('<div class="kpi-caption">vs. current AQI</div>', unsafe_allow_html=True)
    with k3:
        st.metric("Primary Pollutant", poll_name)
        st.markdown(f'<div class="kpi-caption">{poll_value:.1f} µg/m³</div>', unsafe_allow_html=True)
    with k4:
        st.metric("Model Confidence", conf_text)
        st.markdown(f'<div class="kpi-caption">avg R² = {avg_r2:.2f}</div>', unsafe_allow_html=True)

    # ---------------- AQI reference scale + health advice ----------------
    st.subheader("AQI reference scale")
    render_aqi_scale(current_aqi)
    st.markdown(
        f'<div style="background:{band["bg"]}; border-left:4px solid {band["color"]}; '
        f'padding:0.7rem 1rem; border-radius:8px; margin-bottom:1rem;">'
        f'<b style="color:{band["color"]};">{band["label"]}</b> — {band["advice"]}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ---------------- Forecast chart + table ----------------
    st.subheader("Forecast — next 3 days")
    now = row["date"] if isinstance(row["date"], (pd.Timestamp, datetime.datetime)) else datetime.datetime.utcnow()
    forecast_times = [pd.Timestamp(now) + pd.Timedelta(hours=h) for h in HORIZONS_HOURS]
    render_forecast_chart(forecast_times, preds, rmse_by_horizon, HAZARD_THRESHOLD)
    render_forecast_table(forecast_times, preds, rmse_by_horizon)

    if (preds >= HAZARD_THRESHOLD).any():
        days_labels = [f"+{h // 24}d" for h in HORIZONS_HOURS]
        worst_day = days_labels[int(np.argmax(preds))]
        st.error(
            f"⚠️ ALERT: predicted AQI reaches hazardous levels "
            f"({preds.max():.0f}) around {worst_day}. Consider limiting outdoor activity."
        )
    else:
        st.success("No hazardous AQI levels predicted in the next 3 days.")

    # ---------------- Model performance ----------------
    st.subheader("Model performance")
    st.caption(f"Best model selected during training: **{model_name}**")
    metrics_rows = []
    for m_name, horizons in metrics.items():
        for h_label, vals in horizons.items():
            metrics_rows.append({"Model": m_name, "Horizon": h_label, **vals})
    style_metrics_table(metrics_rows, model_name)

    # ---------------- SHAP explainability ----------------
    st.subheader("Why did the model predict this? (SHAP)")
    try:
        background = load_shap_background(feature_columns)
        if len(background) < 30:
            st.info(
                f"Not enough historical data yet for a SHAP background sample "
                f"({len(background)} rows available, need at least 30). Run "
                f"backfill_historical.py or wait for more hourly data to collect."
            )
        else:
            # the model predicts 3 horizons at once; explain the +24h output
            # (index 0) specifically, against a sample of historical
            # conditions rather than the single live row being explained
            predict_24h = lambda data: model.predict(data)[:, 0]
            explainer = shap.Explainer(predict_24h, background)
            shap_values = explainer(X)
            st.caption(
                "Feature contributions to the **+24h** AQI forecast, relative "
                f"to a background of {len(background)} historical hours."
            )
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            fig2.patch.set_alpha(0)
            ax2.patch.set_alpha(0)
            shap.plots.bar(shap_values[0], show=False)
            st.pyplot(fig2)
    except Exception as e:
        st.info(f"SHAP explanation unavailable right now ({e}).")

except Exception as e:
    st.error(
        "Could not load the model or live data. Make sure you've run "
        "feature_pipeline.py, backfill_historical.py and training_pipeline.py "
        f"at least once, and that your API keys are set correctly.\n\nDetails: {e}"
    )
