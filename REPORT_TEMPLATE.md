# Pearls AQI Predictor — Project Report

**Name:** Hassan
**City predicted:** Lahore, Pakistan
**Date:** [fill in]

## 1. Overview
[2-3 sentences: what the system does — fetches live weather + pollution
data hourly, engineers features, trains models to forecast AQI 24/48/72
hours ahead, and serves predictions through a live dashboard.]

## 2. Architecture
[Paste the architecture diagram from the assignment brief. Describe the
4 stages: Feature pipeline -> Feature Store -> Training pipeline ->
Model Registry -> Web app, and how GitHub Actions automates the first
two on a schedule.]

## 3. Data sources
- OpenWeather Air Pollution API (CO, NO2, O3, SO2, PM2.5, PM10)
- OpenWeather Current Weather API (temperature, humidity, pressure, wind)
- Open-Meteo Historical Weather Archive (used only for backfilling past weather)

## 4. Feature engineering
[List the features: hour, day, month, day_of_week, aqi, aqi_change_rate,
plus the 6 raw pollutant/weather readings. Explain the AQI conversion:
PM2.5 concentration converted to the 0-500 US EPA AQI scale.]

## 5. Exploratory Data Analysis
[Paste 1-2 charts here — e.g. AQI over time, AQI by hour-of-day.
You can generate these quickly with:
  df.plot(x="date", y="aqi")   -- after loading the feature group into pandas]

## 6. Model training & evaluation
[Paste the metrics table produced by training_pipeline.py
(saved_model/metrics.json). One row per model per horizon, with RMSE,
MAE, R². State which model was selected as best and why.]

| Model | Horizon | RMSE | MAE | R² |
|---|---|---|---|---|
| Ridge | +24h | | | |
| Ridge | +48h | | | |
| Ridge | +72h | | | |
| RandomForest | +24h | | | |
| RandomForest | +48h | | | |
| RandomForest | +72h | | | |
| NeuralNet | +24h | | | |
| NeuralNet | +48h | | | |
| NeuralNet | +72h | | | |

## 7. Feature importance (SHAP)
[Screenshot the SHAP bar chart from the dashboard here. 1-2 sentences on
which features matter most, e.g. "PM2.5 and the previous AQI reading
were the strongest predictors of next-day AQI."]

## 8. Automation (CI/CD)
[Screenshot your GitHub Actions "Actions" tab showing the two scheduled
workflows (hourly feature pipeline, daily training pipeline) and at
least one successful run of each.]

## 9. Dashboard
[Screenshot the live Streamlit app showing the 3-day forecast chart and
the hazard alert banner.]

## 10. Limitations & what I'd improve with more time
[Be honest: e.g. limited backfill window due to time constraints,
coarse hourly granularity, single-city scope, simple hazard threshold
rather than pollutant-specific alerts, etc.]

## Links
- GitHub repo: [paste link]
- Live dashboard: [paste link]
