"""
Small reusable helper functions shared by the other scripts.
You should not need to edit this file.
"""

import datetime
import requests
import pandas as pd

from config import OPENWEATHER_API_KEY, LAT, LON

# ---------------------------------------------------------------------
# AQI conversion: OpenWeather gives raw pollutant concentrations
# (ug/m3). We convert PM2.5 into the standard 0-500 US EPA AQI scale,
# because that's the number people actually recognise as "AQI"
# (the same scale used by apps like AQICN / IQAir).
# ---------------------------------------------------------------------
_PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]


def pm25_to_aqi(pm25: float) -> float:
    """Convert a PM2.5 concentration (ug/m3) into a 0-500 US AQI value."""
    if pm25 is None:
        return None
    pm25 = max(0.0, float(pm25))
    for c_low, c_high, i_low, i_high in _PM25_BREAKPOINTS:
        if c_low <= pm25 <= c_high:
            return round(
                (i_high - i_low) / (c_high - c_low) * (pm25 - c_low) + i_low, 1
            )
    return 500.0  # anything worse than the last breakpoint is capped at 500


# ---------------------------------------------------------------------
# Live data fetchers (used by the hourly feature pipeline)
# ---------------------------------------------------------------------
def fetch_current_weather() -> dict:
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": LAT, "lon": LON, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    return {
        "temperature": float(data["main"]["temp"]),
        "humidity": data["main"]["humidity"],
        "pressure": float(data["main"]["pressure"]),
        "wind_speed": float(data["wind"]["speed"]),
    }


def fetch_current_pollution() -> dict:
    url = "https://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": LAT, "lon": LON, "appid": OPENWEATHER_API_KEY}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    entry = r.json()["list"][0]
    comps = entry["components"]
    return {
        "co": comps.get("co"),
        "no2": comps.get("no2"),
        "o3": comps.get("o3"),
        "so2": comps.get("so2"),
        "pm2_5": comps.get("pm2_5"),
        "pm10": comps.get("pm10"),
        "openweather_aqi_index": entry["main"]["aqi"],  # coarse 1-5 scale
    }


# ---------------------------------------------------------------------
# Historical data fetchers (used only by the one-time backfill script)
# ---------------------------------------------------------------------
def fetch_historical_pollution(start_ts: int, end_ts: int) -> pd.DataFrame:
    """start_ts / end_ts are unix timestamps (seconds)."""
    url = "https://api.openweathermap.org/data/2.5/air_pollution/history"
    params = {
        "lat": LAT,
        "lon": LON,
        "start": start_ts,
        "end": end_ts,
        "appid": OPENWEATHER_API_KEY,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    rows = []
    for entry in r.json().get("list", []):
        comps = entry["components"]
        rows.append(
            {
                "date": datetime.datetime.utcfromtimestamp(entry["dt"]),
                "co": comps.get("co"),
                "no2": comps.get("no2"),
                "o3": comps.get("o3"),
                "so2": comps.get("so2"),
                "pm2_5": comps.get("pm2_5"),
                "pm10": comps.get("pm10"),
                "openweather_aqi_index": entry["main"]["aqi"],
            }
        )
    return pd.DataFrame(rows)


def fetch_historical_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Uses Open-Meteo's free historical weather archive (no API key needed).
    start_date / end_date format: 'YYYY-MM-DD'
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
        "timezone": "UTC",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    hourly = r.json()["hourly"]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(hourly["time"]),
            "temperature": hourly["temperature_2m"],
            "humidity": hourly["relative_humidity_2m"],
            "pressure": hourly["surface_pressure"],
            "wind_speed": hourly["wind_speed_10m"],
        }
    )
    return df


# ---------------------------------------------------------------------
# Feature engineering (used by both the live pipeline and the backfill)
# ---------------------------------------------------------------------
def add_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["hour"] = df[date_col].dt.hour
    df["day"] = df[date_col].dt.day
    df["month"] = df[date_col].dt.month
    df["day_of_week"] = df[date_col].dt.dayofweek
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the AQI value itself + the hour-over-hour AQI change rate."""
    df = df.copy()
    df = df.sort_values("date")
    df["aqi"] = df["pm2_5"].apply(pm25_to_aqi)
    df["aqi_change_rate"] = df["aqi"].diff().fillna(0)
    return df
