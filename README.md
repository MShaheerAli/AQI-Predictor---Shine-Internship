# Pearls AQI Predictor — Lahore

A serverless, fully automated machine learning system that forecasts Lahore's
Air Quality Index (AQI) 24, 48, and 72 hours ahead. Live weather and
pollution data is collected hourly, stored in a feature store, used to
retrain models daily, and served through a public dashboard — with zero
manual intervention once deployed.

- **Live dashboard:** https://mshaheerali-aqi-predictor---shine-internship-app-i8ffsa.streamlit.app/
- **GitHub repo:** https://github.com/MShaheerAli/AQI-Predictor---Shine-Internship

---

## What this project does

1. **Collects data hourly** — live temperature, humidity, pressure, wind
   speed (OpenWeather) and pollutant concentrations: CO, NO2, O3, SO2,
   PM2.5, PM10 (OpenWeather Air Pollution API) for Lahore, Pakistan.
2. **Engineers features** — converts PM2.5 into the standard 0–500 US EPA
   AQI scale, adds time-based features (hour, day, month, day of week),
   and computes the hour-over-hour AQI change rate.
3. **Stores everything in a feature store** (Hopsworks) so the hourly
   collector and the daily trainer share one consistent source of truth.
4. **Trains 3 models daily** — Ridge Regression, Random Forest, and a
   small neural network (MLP) — each predicting AQI at +24h, +48h, and
   +72h, and automatically keeps whichever model scores best.
5. **Registers the best model** in the Hopsworks Model Registry, versioned
   and tagged with its own metrics.
6. **Serves a live dashboard** (Streamlit) showing the current AQI, a
   3-day forecast chart, a hazard alert, the metrics for all 3 models,
   and a SHAP explanation of what's driving the forecast.
7. **Runs unattended** via two scheduled GitHub Actions workflows — an
   hourly feature collector and a daily retrainer — so the whole pipeline
   keeps itself up to date without anyone running a script by hand.

---

## Architecture

```
OpenWeather API ─┐
                  ├─> feature_pipeline.py ──> Hopsworks Feature Store
Open-Meteo API ───┘         (hourly, via GitHub Actions cron)
                                    │
                                    ▼
                          backfill_historical.py
                          (one-time, ~45 days of history)
                                    │
                                    ▼
                          training_pipeline.py ──> Hopsworks Model Registry
                          (daily, via GitHub Actions cron)      │
                                                                 ▼
                                                            app.py (Streamlit)
                                                          live public dashboard
```

- **Feature Store / Model Registry:** [Hopsworks](https://www.hopsworks.ai/) —
  free serverless MLOps platform; project `aqi_predictor9048`.
- **Models:** scikit-learn only (Ridge, RandomForestRegressor,
  MLPRegressor) — deliberately lightweight, no TensorFlow/PyTorch, so the
  dashboard deploys cleanly on free hosting.
- **Dashboard:** [Streamlit](https://streamlit.io/), deployed on
  Streamlit Community Cloud.
- **Automation:** GitHub Actions — `feature_pipeline.yml` (hourly cron)
  and `training_pipeline.yml` (daily cron), both also manually
  triggerable via `workflow_dispatch`.

---

## Repo layout

| File | Purpose |
|---|---|
| `config.py` | City/coordinates, API keys (from `.env`), feature store & model registry names, forecast horizons, hazard threshold |
| `utils.py` | API fetchers (live + historical), AQI conversion, feature engineering |
| `feature_pipeline.py` | Fetches current conditions, pushes one row to the feature store (runs hourly) |
| `backfill_historical.py` | One-time pull of ~45 days of historical weather + pollution to seed the feature store |
| `training_pipeline.py` | Loads all features, trains 3 models per horizon, saves the best one to the Model Registry (runs daily) |
| `app.py` | Streamlit dashboard — loads the latest registered model and live data, shows forecast + SHAP explanation |
| `.github/workflows/` | The two scheduled GitHub Actions pipelines |
| `saved_model/` | Local copy of the last-trained model + metrics (git-ignored) |
| `REPORT_TEMPLATE.md` | Assignment report template |

---

## Current model performance

From the most recent training run (`saved_model/metrics.json`):

| Model | Horizon | RMSE | MAE | R² |
|---|---|---|---|---|
| Ridge | +24h | 40.15 | 34.93 | -1.515 |
| Ridge | +48h | 27.70 | 20.37 | -0.386 |
| Ridge | +72h | 32.26 | 27.37 | -1.573 |
| Random Forest | +24h | 30.64 | 26.56 | -0.465 |
| Random Forest | +48h | 26.93 | 23.47 | -0.310 |
| Random Forest | +72h | 46.89 | 36.41 | -4.437 |
| **Neural Net (selected)** | +24h | 29.58 | 22.82 | -0.365 |
| **Neural Net (selected)** | +48h | 23.64 | 20.03 | -0.009 |
| **Neural Net (selected)** | +72h | 23.82 | 20.89 | -0.403 |

The Neural Net was selected (avg R² = -0.259 across horizons). All R²
values are currently negative, meaning the models perform worse than
predicting the historical average — see **Known limitations** below.

Feature store: **1,034 rows** spanning **2026-07-22 to present**, growing
by one row every hour via the automated pipeline.

---

## Setup — running it yourself

### 1. Accounts
- [OpenWeather](https://openweathermap.org/) account + API key (new keys
  take up to ~1 hour to activate).
- [Hopsworks](https://app.hopsworks.ai/) account, with a project created
  (any name).
- A GitHub repo (public) to host the code and run the Actions workflows.

### 2. Local environment

> **Important:** `hopsworks` does not yet support Python 3.14. Use
> **Python 3.10–3.13** — 3.11 or 3.12 is recommended.

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

cp .env.example .env         # then fill in your two API keys
pip install -r requirements.txt
```

### 3. Build the data + model

Run in order, waiting for each to finish:

```bash
python feature_pipeline.py       # pushes one live row to the feature store
python backfill_historical.py    # pulls ~45 days of history
python training_pipeline.py      # trains 3 models, registers the best one
```

### 4. Run the dashboard locally

```bash
streamlit run app.py
```

### 5. Deploy

- Push to GitHub, add `OPENWEATHER_API_KEY` and `HOPSWORKS_API_KEY` as
  **repository secrets** (Settings → Secrets and variables → Actions).
- Trigger both workflows once manually (Actions tab → Run workflow) to
  confirm they succeed.
- Deploy `app.py` on [share.streamlit.io](https://share.streamlit.io/),
  adding the same two secrets under Advanced settings.
  **On Streamlit Community Cloud, set the Python version to 3.11** in the
  app's Settings — the platform defaults to 3.14, which will fail to
  install `hopsworks`.

---

## Build log — issues hit and how they were fixed

This project's setup surfaced several real-world environment
compatibility problems, documented here since they aren't obvious from
the code alone:

1. **Python 3.14 incompatibility.** The `hopsworks` SDK doesn't support
   Python 3.14 yet — pip silently falls back to years-old versions that
   crash with `ModuleNotFoundError: No module named 'imp'` (a module
   removed in Python 3.12+). Fixed by installing Python 3.12 and running
   everything in a dedicated virtual environment.

2. **`twofish` has no Windows wheel.** `pyjks` (a transitive dependency
   of `hopsworks`, used only for Java KeyStore/BKS support that this
   project never touches) depends on `twofish`, which has no prebuilt
   wheel and needs a full C++ compiler toolchain to build from source on
   Windows. Rather than installing several gigabytes of Visual Studio
   Build Tools, a tiny stub `twofish` package was installed locally to
   satisfy the dependency without the real (unused) implementation.

3. **Feature group defaulted to Delta Lake format.** `get_or_create_feature_group`
   defaults to `time_travel_format="DELTA"`, which requires the `delta`
   Python package. Switched to `time_travel_format="HUDI"` (Hopsworks'
   classic format), which needs no extra local dependency for simple
   inserts.

4. **Schema type mismatch on `pressure`.** The first row (from live
   OpenWeather data) had a whole-number `pressure` reading, locking the
   feature group's schema to `bigint`. The historical backfill source
   (Open-Meteo) returns `pressure` as a decimal, causing a schema
   conflict on insert. Fixed by explicitly casting `pressure` (and other
   numeric weather fields) to `float` at the source, and recreating the
   feature group with the corrected schema.

5. **SHAP crashed on the multi-output model.** `shap.plots.bar()` can
   only plot a single output's feature importances, but the model
   predicts 3 horizons at once. Fixed by explaining the +24h output
   specifically.

6. **SHAP values were all exactly `+0`.** The dashboard was explaining
   the live data row using *itself* as the only background/reference
   sample — mathematically valid, but meaningless (nothing to compare
   against). Fixed by sampling up to 150 real historical rows from the
   feature store as the SHAP background.

7. **Streamlit Community Cloud also defaults to Python 3.14.** A
   `runtime.txt` pin was tried first but is no longer honored by the
   platform; the fix is setting the Python version directly in the app's
   **Settings → Python version** dropdown in the Streamlit Cloud
   dashboard.

---

## Known limitations

- **Negative R² scores.** With ~45 days of hourly data and no lag/rolling
  features beyond the immediate AQI change rate, the models currently
  perform worse than a naive average predictor. More historical data and
  richer lag features would likely improve this substantially.
- **Single city.** Hardcoded to Lahore (`config.py` — coordinates are
  the only thing that would need to change for another city).
- **Coarse hourly granularity** and a single fixed hazard threshold
  (AQI ≥ 150) rather than pollutant-specific alert levels.

---

## Tech stack

Python · pandas · scikit-learn · Streamlit · SHAP · Hopsworks
(Feature Store + Model Registry) · GitHub Actions · OpenWeather API ·
Open-Meteo API
