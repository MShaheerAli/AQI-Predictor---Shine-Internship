"""
Shared settings for the whole project.
Change CITY_NAME / LAT / LON here if you ever want to predict AQI for a
different city — everything else in the project reads from this file.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads the .env file so os.environ has your keys

# ---- City ----
CITY_NAME = "Lahore"
LAT = 31.5204
LON = 74.3587

# ---- API keys (set these in a .env file, see .env.example) ----
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")
HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY", "")

# ---- Hopsworks feature store / model registry names ----
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = "aqi_feature_view"
MODEL_NAME = "aqi_forecast_model"

# ---- Forecast horizons (in hours) ----
# We predict AQI 24h, 48h and 72h ahead == "next 3 days"
HORIZONS_HOURS = [24, 48, 72]

# ---- Hazard alert threshold (US EPA AQI scale, 0-500) ----
HAZARD_THRESHOLD = 150  # "Unhealthy" and above triggers a dashboard alert
