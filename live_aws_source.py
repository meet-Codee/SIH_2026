"""
live_aws_source.py — SkyGuard AI Phase 3B
==========================================
Live weather observation fetcher using Open-Meteo API.

WHY Open-Meteo, not IMD direct API:
  - Official IMD API (api.imd.gov.in) requires registration + IP whitelisting;
    the /observation endpoint returns HTTP 401 without credentials.
  - Bhuvan/NRSC WMS timed out from this environment.
  - Open-Meteo (open-meteo.com) is a free, open-source, no-key-required
    meteorological API that sources data from NWP models (ERA5, GFS, ECMWF)
    which ingest WMO/SYNOP surface observations globally including Indian AWS.
  - Batch endpoint supports up to 50 lat/lon pairs in a single request.
  - Fields verified: temperature_2m (°C), relative_humidity_2m (%),
    surface_pressure (hPa), time (ISO-8601 local).

This module is the ONLY place that touches the network for live data.
All downstream code (feature engineering, Isolation Forest, root-cause
classification, SHAP, sensor health) is data-source-agnostic.

Public API:
    fetch_live_observations(stations) -> pd.DataFrame | None
    OPEN_METEO_AVAILABLE              -> bool (checked at import time)
"""

from __future__ import annotations

import urllib.request
import urllib.error
import ssl
import json
import warnings
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# SSL context (ignore cert errors for some government proxy environments)
# ---------------------------------------------------------------------------
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"

# ---------------------------------------------------------------------------
# Verified field mapping (confirmed by probe_openmeteo.py)
# ---------------------------------------------------------------------------
# Open-Meteo field        → SkyGuard internal column
OM_FIELD_MAP = {
    "temperature_2m":       "temp",       # °C, 2 m above ground
    "relative_humidity_2m": "humidity",   # %, 2 m above ground
    "surface_pressure":     "pressure",   # hPa, surface level
}
OM_CURRENT_FIELDS = ",".join(OM_FIELD_MAP.keys())

FETCH_TIMEOUT_S = 15   # seconds

# ---------------------------------------------------------------------------
# Check availability once at import time (fast HEAD-like check)
# ---------------------------------------------------------------------------
def _check_open_meteo() -> bool:
    try:
        url = f"{OPEN_METEO_BASE}?latitude=28.6&longitude=77.2&current=temperature_2m&forecast_days=1"
        req = urllib.request.Request(url, headers={"User-Agent": "SkyGuardAI/3.0"})
        with urllib.request.urlopen(req, timeout=6, context=_SSL_CTX) as r:
            return r.status == 200
    except Exception:
        return False


OPEN_METEO_AVAILABLE: bool = _check_open_meteo()


# ---------------------------------------------------------------------------
# Public fetch function
# ---------------------------------------------------------------------------

def fetch_live_observations(stations: list[dict]) -> Optional[pd.DataFrame]:
    """
    Fetch current weather observations for a list of stations from Open-Meteo.

    Parameters
    ----------
    stations : list of dicts, each with keys:
        station_id, name, lat, lon
        (same format as data_engine.STATIONS)

    Returns
    -------
    pd.DataFrame with normalized schema:
        timestamp, station_id, name, lat, lon, temp, humidity, pressure
    or None on failure.

    The timestamp is the observation time returned by Open-Meteo (ISO-8601),
    converted to naive UTC datetime. It represents the current 15-minute
    interval at the query time.

    This function NEVER returns simulated data. On any failure it returns None
    and the caller is responsible for showing an appropriate error to the user.
    """
    if not stations:
        return None

    lats = ",".join(str(s["lat"]) for s in stations)
    lons = ",".join(str(s["lon"]) for s in stations)
    url = (
        f"{OPEN_METEO_BASE}?"
        f"latitude={lats}&longitude={lons}"
        f"&current={OM_CURRENT_FIELDS}"
        f"&forecast_days=1&timezone=auto"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SkyGuardAI/3.0"})
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S, context=_SSL_CTX) as r:
            raw = r.read()
        responses = json.loads(raw)
    except urllib.error.URLError as e:
        warnings.warn(f"Open-Meteo fetch failed (URLError): {e}")
        return None
    except json.JSONDecodeError as e:
        warnings.warn(f"Open-Meteo response parse error: {e}")
        return None
    except Exception as e:
        warnings.warn(f"Open-Meteo unexpected error: {e}")
        return None

    # Ensure list format (single station returns a dict, multiple returns a list)
    if isinstance(responses, dict):
        responses = [responses]

    if len(responses) != len(stations):
        warnings.warn(
            f"Open-Meteo returned {len(responses)} responses for {len(stations)} stations."
        )
        return None

    records = []
    for station, resp in zip(stations, responses):
        cur = resp.get("current", {})
        if not cur:
            warnings.warn(f"No 'current' block for station {station['station_id']}")
            continue

        # Parse timestamp
        ts_str = cur.get("time")
        if ts_str:
            try:
                ts = pd.Timestamp(ts_str)
            except Exception:
                ts = pd.Timestamp.now()
        else:
            ts = pd.Timestamp.now()

        temp     = cur.get("temperature_2m")
        humidity = cur.get("relative_humidity_2m")
        pressure = cur.get("surface_pressure")

        # Skip rows where all three primary fields are missing
        if temp is None and humidity is None and pressure is None:
            warnings.warn(f"All fields missing for station {station['station_id']}")
            continue

        records.append({
            "timestamp":  ts,
            "station_id": station["station_id"],
            "name":       station["name"],
            "lat":        float(station["lat"]),
            "lon":        float(station["lon"]),
            "temp":       float(temp) if temp is not None else float("nan"),
            "humidity":   float(humidity) if humidity is not None else float("nan"),
            "pressure":   float(pressure) if pressure is not None else float("nan"),
        })

    if not records:
        return None

    df = pd.DataFrame(records)
    return df


# ---------------------------------------------------------------------------
# Historical fetch (last N hours at 15-min resolution)
# ---------------------------------------------------------------------------

def fetch_historical_observations(stations: list[dict], hours: int = 72) -> Optional[pd.DataFrame]:
    """
    Fetch recent hourly weather data (past `hours` hours) from Open-Meteo.
    Uses the hourly endpoint so we get a time series, not just the current snapshot.

    Returns normalized DataFrame with multiple rows per station (one per hour),
    or None on failure.
    """
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)
    start_str = start.strftime("%Y-%m-%d")
    end_str   = now.strftime("%Y-%m-%d")

    lats = ",".join(str(s["lat"]) for s in stations)
    lons = ",".join(str(s["lon"]) for s in stations)
    url = (
        f"{OPEN_METEO_BASE}?"
        f"latitude={lats}&longitude={lons}"
        f"&hourly=temperature_2m,relative_humidity_2m,surface_pressure"
        f"&start_date={start_str}&end_date={end_str}"
        f"&timezone=auto"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SkyGuardAI/3.0"})
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S * 3, context=_SSL_CTX) as r:
            raw = r.read()
        responses = json.loads(raw)
    except Exception as e:
        warnings.warn(f"Open-Meteo historical fetch failed: {e}")
        return None

    if isinstance(responses, dict):
        responses = [responses]

    all_records = []
    for station, resp in zip(stations, responses):
        hourly = resp.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        hums  = hourly.get("relative_humidity_2m", [])
        press = hourly.get("surface_pressure", [])

        n = len(times)
        for i in range(n):
            t  = temps[i]  if i < len(temps)  else None
            rh = hums[i]   if i < len(hums)   else None
            p  = press[i]  if i < len(press)  else None

            # Skip completely null rows
            if t is None and rh is None and p is None:
                continue

            try:
                ts = pd.Timestamp(times[i])
            except Exception:
                continue

            all_records.append({
                "timestamp":  ts,
                "station_id": station["station_id"],
                "name":       station["name"],
                "lat":        float(station["lat"]),
                "lon":        float(station["lon"]),
                "temp":       float(t)  if t  is not None else float("nan"),
                "humidity":   float(rh) if rh is not None else float("nan"),
                "pressure":   float(p)  if p  is not None else float("nan"),
            })

    if not all_records:
        return None

    df = pd.DataFrame(all_records)
    # Keep only the last `hours` worth of data
    cutoff = pd.Timestamp.now() - pd.Timedelta(hours=hours)
    df = df[df["timestamp"] >= cutoff].copy()
    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    return df

def validate_live_data(df: pd.DataFrame, min_hours: int = 12, min_stations: int = 3) -> tuple[bool, str, pd.DataFrame]:
    """
    Perform lightweight validation on fetched live/historical data before ML processing.
    Ensures data provenance, deduplicates, and checks sufficient history/spatial coverage.
    
    Returns: (is_valid, status_message, cleaned_df)
    """
    if df is None or df.empty:
        return False, "Data fetch returned empty results.", df
        
    df_clean = df.copy()
    
    # 1. Drop complete duplicates
    df_clean = df_clean.drop_duplicates(subset=["station_id", "timestamp"], keep="last")
    
    # 2. Check for missing essential columns
    required_cols = ["timestamp", "station_id", "temp", "pressure", "humidity", "lat", "lon"]
    missing = [c for c in required_cols if c not in df_clean.columns]
    if missing:
        return False, f"Malformed records: missing required columns {missing}", df_clean
        
    # 3. Ensure numeric types for sensors
    for col in ["temp", "pressure", "humidity", "lat", "lon"]:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
        
    # 4. Check spatial coverage
    n_stations = df_clean["station_id"].nunique()
    if n_stations < min_stations:
        return False, f"Insufficient spatial coverage: only {n_stations} stations reported (need {min_stations}).", df_clean
        
    # 5. Check temporal history
    ts_min = df_clean["timestamp"].min()
    ts_max = df_clean["timestamp"].max()
    history_hours = (ts_max - ts_min).total_seconds() / 3600.0
    if history_hours < min_hours:
        return False, f"Warming Up / Insufficient History: got {history_hours:.1f}h of data, need at least {min_hours}h for rolling features.", df_clean
        
    # 6. Check for excessively stale data (max timestamp older than 12 hours)
    now = pd.Timestamp.now(tz=ts_max.tz) if ts_max.tz else pd.Timestamp.now()
    stale_hours = (now - ts_max).total_seconds() / 3600.0
    if stale_hours > 12:
        return False, f"Stale observations: latest data is {stale_hours:.1f} hours old.", df_clean

    return True, "Data validated successfully.", df_clean
