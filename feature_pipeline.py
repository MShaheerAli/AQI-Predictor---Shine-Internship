"""
FEATURE PIPELINE
Run this script and it will:
  1. Fetch the current weather + air pollution for the city in config.py
  2. Turn that into a feature row (time features + derived AQI features)
  3. Push that row into your Hopsworks Feature Store

This is meant to run once per hour. Locally you run it by hand with:
    python feature_pipeline.py
In production it runs automatically via .github/workflows/feature_pipeline.yml
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
    fetch_current_weather,
    fetch_current_pollution,
    add_time_features,
    add_derived_features,
)


def build_feature_row() -> pd.DataFrame:
    weather = fetch_current_weather()
    pollution = fetch_current_pollution()

    row = {
        "city": CITY_NAME,
        "date": datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0),
        **weather,
        **pollution,
    }
    df = pd.DataFrame([row])
    df = add_time_features(df)
    df = add_derived_features(df)  # note: change-rate will be 0 for a single row,
    # the feature group as a whole is what carries the real history
    return df


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
    print(f"Inserted {len(df)} row(s) into feature group "
          f"'{FEATURE_GROUP_NAME}' v{FEATURE_GROUP_VERSION}.")


if __name__ == "__main__":
    try:
        df = build_feature_row()
        print("Fetched row:")
        print(df.to_string(index=False))
        push_to_feature_store(df)
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)
