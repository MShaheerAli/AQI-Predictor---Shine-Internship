"""
BACKFILL SCRIPT — run this ONCE (not on a schedule).

It pulls the last N days of pollution history (OpenWeather) and weather
history (Open-Meteo, no key needed), merges them, engineers the same
features the live pipeline creates, and bulk-inserts everything into the
Hopsworks Feature Store. This gives your training pipeline enough rows
to actually train a model on, without waiting weeks for the hourly
pipeline to accumulate data on its own.

Usage:
    python backfill_historical.py            (defaults to last 45 days)
    python backfill_historical.py 60         (last 60 days)
"""

import sys
import datetime
import pandas as pd
import hopsworks

from config import (
    CITY_NAME,
    HOPSWORKS_API_KEY,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
)
from utils import (
    fetch_historical_pollution,
    fetch_historical_weather,
    add_time_features,
    add_derived_features,
)

DEFAULT_DAYS = 45


def backfill(days: int) -> pd.DataFrame:
    end = datetime.datetime.utcnow()
    start = end - datetime.timedelta(days=days)

    print(f"Fetching pollution history: {start.date()} to {end.date()} ...")
    pollution_df = fetch_historical_pollution(
        int(start.timestamp()), int(end.timestamp())
    )

    print("Fetching weather history (Open-Meteo) ...")
    weather_df = fetch_historical_weather(
        start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    )

    # Both dataframes are hourly — round to the hour and merge on that.
    pollution_df["date"] = pd.to_datetime(pollution_df["date"]).dt.floor("h")
    weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.floor("h")

    merged = pd.merge(pollution_df, weather_df, on="date", how="inner")
    merged["city"] = CITY_NAME

    merged = add_time_features(merged)
    merged = add_derived_features(merged)
    merged = merged.dropna(subset=["pm2_5", "temperature"])

    print(f"Built {len(merged)} historical rows.")
    return merged


def push_to_feature_store(df: pd.DataFrame):
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Hourly weather + air pollution features for AQI forecasting",
        primary_key=["city", "date"],
        event_time="date",
        online_enabled=False,
        time_travel_format="HUDI",
    )
    fg.insert(df, write_options={"wait_for_job": True})
    print(f"Inserted {len(df)} historical rows into '{FEATURE_GROUP_NAME}'.")


if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DAYS
    try:
        df = backfill(days)
        push_to_feature_store(df)
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)
