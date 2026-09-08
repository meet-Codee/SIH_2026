"""
data_engine.py — SkyGuard AI Phase 2
======================================
Generates a realistic multi-station Indian AWS network dataset.

Key improvements over Phase 1:
  - Station individuality: each station has its own amplitude, phase, noise level
  - Coastal/arid/inland climate characteristics
  - Realistic multi-variable correlations
  - 10 injected fault scenarios covering all required anomaly types
  - Spatially coherent genuine weather event (staggered onset, different magnitudes)
  - Explicit ground-truth dataframe returned alongside sensor data
  - Ground truth is NEVER passed to the ML pipeline — used only for evaluation

Injected anomalies (ground truth):
  AWS-001 Delhi       h=14        Sensor Spike / Fault         (+15.5°C, neighbours normal)
  AWS-002 Mumbai      h=36-42     Genuine Weather Event        (monsoon surge, spatial coherent)
  AWS-003 Chennai     h=28-34     Frozen / Stuck Sensor        (all 3 vars stuck for 7h)
  AWS-004 Kolkata     h=48-60     Calibration Drift            (temp drifts +0.4°C/h)
  AWS-006 Jaipur      h=20-22     Communication Failure        (NaN readings)
  AWS-007 Hyderabad   h=55        Multivariate Inconsistency   (T+H+P physically inconsistent)
  AWS-010 Ahmedabad   h=30        Sensor Spike / Fault         (temperature drop -12°C)
  AWS-009 Patna       h=45        Sensor Spike / Fault         (humidity spike +35%)
  AWS-011 Lucknow     h=58-65     Calibration Drift            (pressure drifts +0.6 hPa/h)
  AWS-008 Pune        h=36-42     Genuine Weather Event        (same monsoon surge, weaker)
  AWS-005 Bhopal      —           Normal reference             (no faults injected)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Station registry — each station has climate-specific parameters
#
# amplitude_temp:  ± degrees C in daily cycle (larger for arid/continental)
# phase_offset:    hours after 06:00 when temperature peaks (12–15 h typical)
# noise_temp:      σ of Gaussian sensor noise (°C)
# amplitude_hum:   ± % humidity swing in daily cycle
# climate:         used for narrative context
# ---------------------------------------------------------------------------
STATIONS = [
    {
        "station_id": "AWS-001", "name": "Delhi",       "lat": 28.61, "lon": 77.21, "wmo_id": "42182",
        "base_temp": 36.0, "base_hum": 52.0, "base_pres": 1006.0,
        "amplitude_temp": 7.0, "phase_offset": 2.0,   "noise_temp": 0.4,
        "amplitude_hum": 12.0, "noise_hum": 1.5, "noise_pres": 0.18,
        "climate": "semi-arid",
    },
    {
        "station_id": "AWS-002", "name": "Mumbai",      "lat": 19.08, "lon": 72.88, "wmo_id": "43003",
        "base_temp": 30.0, "base_hum": 82.0, "base_pres": 1008.0,
        "amplitude_temp": 3.5, "phase_offset": 1.5,   "noise_temp": 0.3,
        "amplitude_hum": 8.0,  "noise_hum": 2.0, "noise_pres": 0.12,
        "climate": "coastal-tropical",
    },
    {
        "station_id": "AWS-003", "name": "Chennai",     "lat": 13.08, "lon": 80.27, "wmo_id": "43279",
        "base_temp": 32.0, "base_hum": 76.0, "base_pres": 1007.0,
        "amplitude_temp": 4.0, "phase_offset": 2.5,   "noise_temp": 0.3,
        "amplitude_hum": 9.0,  "noise_hum": 2.2, "noise_pres": 0.13,
        "climate": "coastal-tropical",
    },
    {
        "station_id": "AWS-004", "name": "Kolkata",     "lat": 22.57, "lon": 88.36, "wmo_id": "42809",
        "base_temp": 32.0, "base_hum": 80.0, "base_pres": 1007.0,
        "amplitude_temp": 5.5, "phase_offset": 1.8,   "noise_temp": 0.35,
        "amplitude_hum": 10.0, "noise_hum": 1.8, "noise_pres": 0.15,
        "climate": "humid-subtropical",
    },
    {
        "station_id": "AWS-005", "name": "Bhopal",      "lat": 23.26, "lon": 77.40, "wmo_id": "42667",
        "base_temp": 31.0, "base_hum": 65.0, "base_pres": 1005.0,
        "amplitude_temp": 6.0, "phase_offset": 2.0,   "noise_temp": 0.38,
        "amplitude_hum": 11.0, "noise_hum": 1.6, "noise_pres": 0.16,
        "climate": "inland",
    },
    {
        "station_id": "AWS-006", "name": "Jaipur",      "lat": 26.91, "lon": 75.79, "wmo_id": "42348",
        "base_temp": 35.0, "base_hum": 47.0, "base_pres": 1005.0,
        "amplitude_temp": 8.0, "phase_offset": 2.2,   "noise_temp": 0.45,
        "amplitude_hum": 14.0, "noise_hum": 1.4, "noise_pres": 0.19,
        "climate": "arid",
    },
    {
        "station_id": "AWS-007", "name": "Hyderabad",   "lat": 17.38, "lon": 78.47, "wmo_id": "43128",
        "base_temp": 29.0, "base_hum": 70.0, "base_pres": 1007.0,
        "amplitude_temp": 5.0, "phase_offset": 2.0,   "noise_temp": 0.32,
        "amplitude_hum": 10.0, "noise_hum": 1.7, "noise_pres": 0.14,
        "climate": "semi-arid",
    },
    {
        "station_id": "AWS-008", "name": "Pune",        "lat": 18.52, "lon": 73.86, "wmo_id": "43063",
        "base_temp": 27.0, "base_hum": 76.0, "base_pres": 1008.0,
        "amplitude_temp": 4.5, "phase_offset": 1.8,   "noise_temp": 0.30,
        "amplitude_hum": 9.0,  "noise_hum": 1.9, "noise_pres": 0.13,
        "climate": "inland-western",
    },
    {
        "station_id": "AWS-009", "name": "Patna",       "lat": 25.59, "lon": 85.14, "wmo_id": "42492",
        "base_temp": 33.0, "base_hum": 80.0, "base_pres": 1006.0,
        "amplitude_temp": 6.5, "phase_offset": 1.5,   "noise_temp": 0.40,
        "amplitude_hum": 11.0, "noise_hum": 2.0, "noise_pres": 0.17,
        "climate": "humid",
    },
    {
        "station_id": "AWS-010", "name": "Ahmedabad",   "lat": 23.03, "lon": 72.57, "wmo_id": "42647",
        "base_temp": 33.0, "base_hum": 57.0, "base_pres": 1006.0,
        "amplitude_temp": 7.5, "phase_offset": 2.3,   "noise_temp": 0.42,
        "amplitude_hum": 13.0, "noise_hum": 1.5, "noise_pres": 0.18,
        "climate": "arid",
    },
    {
        "station_id": "AWS-011", "name": "Lucknow",     "lat": 26.85, "lon": 80.95, "wmo_id": "42369",
        "base_temp": 33.0, "base_hum": 71.0, "base_pres": 1006.0,
        "amplitude_temp": 6.0, "phase_offset": 1.9,   "noise_temp": 0.38,
        "amplitude_hum": 11.5, "noise_hum": 1.7, "noise_pres": 0.16,
        "climate": "inland",
    },
    {
        "station_id": "AWS-012", "name": "Bhubaneswar", "lat": 20.30, "lon": 85.82, "wmo_id": "42971",
        "base_temp": 31.0, "base_hum": 82.0, "base_pres": 1007.0,
        "amplitude_temp": 4.5, "phase_offset": 2.1,   "noise_temp": 0.33,
        "amplitude_hum": 9.5,  "noise_hum": 2.1, "noise_pres": 0.14,
        "climate": "coastal",
    },
    {
        "station_id": "AWS-013", "name": "Bengaluru Urban", "lat": 12.97, "lon": 77.59, "wmo_id": "43295",
        "base_temp": 28.0, "base_hum": 65.0, "base_pres": 920.0,
        "amplitude_temp": 4.0, "phase_offset": 2.0,   "noise_temp": 0.30,
        "amplitude_hum": 10.0, "noise_hum": 1.5, "noise_pres": 0.15,
        "climate": "temperate-plateau",
    },
]


def generate_dataset(hours: int = 72, seed: int = 42):
    """
    Generate a synthetic multi-station weather dataset with ground truth.

    Parameters
    ----------
    hours : int — number of simulated hours (rows per station)
    seed  : int — NumPy random seed for reproducibility

    Returns
    -------
    sensor_df    : pd.DataFrame
        Raw sensor observations as an AWS would report them.
        Columns: timestamp, station_id, name, lat, lon, temp, humidity, pressure
        NaN where communication failure was injected.
        Anomalous values where faults were injected.
        NO ground-truth columns — the ML pipeline must detect anomalies.

    ground_truth_df : pd.DataFrame
        Hidden labels for evaluation ONLY. Never pass to the ML pipeline.
        Columns: timestamp, station_id, is_anomaly_gt, true_root_cause
    """
    rng = np.random.default_rng(seed)
    start = datetime(2026, 9, 1, 0, 0, 0)
    timestamps = [start + timedelta(hours=h) for h in range(hours)]

    records = []

    for stn_i, stn in enumerate(STATIONS):
        sid          = stn["station_id"]
        base_temp    = stn["base_temp"]
        base_hum     = stn["base_hum"]
        base_pres    = stn["base_pres"]
        amp_t        = stn["amplitude_temp"]
        phase        = stn["phase_offset"]   # hours after 06:00 when peak occurs
        noise_t      = stn["noise_temp"]
        amp_h        = stn["amplitude_hum"]
        noise_h      = stn["noise_hum"]
        noise_p      = stn["noise_pres"]

        # Per-station random stream (derived from seed + station index)
        st_rng = np.random.default_rng(seed + stn_i * 1000)

        # Slow multi-day pressure trend (random ±1 hPa drift over 72 h)
        pressure_trend_slope = st_rng.uniform(-0.015, 0.015)  # hPa/h

        for h, ts in enumerate(timestamps):
            hour_of_day = h % 24

            # ----------------------------------------------------------------
            # Temperature: daily sine cycle with station-specific amplitude and phase
            # Peak occurs at (6 + phase + 8) = 14-17:00 depending on station
            # ----------------------------------------------------------------
            temp = (base_temp
                    + amp_t * np.sin((hour_of_day - 6 - phase) * np.pi / 12)
                    + st_rng.normal(0, noise_t))

            # ----------------------------------------------------------------
            # Humidity: negatively correlated with temp cycle, station-specific amplitude
            # Also includes a random slow drift component
            # ----------------------------------------------------------------
            hum = (base_hum
                   - amp_h * np.sin((hour_of_day - 6 - phase) * np.pi / 12)
                   + st_rng.normal(0, noise_h))
            hum = float(np.clip(hum, 5, 100))

            # ----------------------------------------------------------------
            # Pressure: semi-diurnal variation + slow trend + noise
            # ----------------------------------------------------------------
            pres = (base_pres
                    + 1.2 * np.sin(hour_of_day * np.pi / 12)
                    + 0.4 * np.sin(hour_of_day * np.pi / 6)
                    + pressure_trend_slope * h
                    + st_rng.normal(0, noise_p))

            records.append({
                "timestamp":  ts,
                "station_id": sid,
                "name":       stn["name"],
                "lat":        stn["lat"],
                "lon":        stn["lon"],
                "temp":       round(float(temp), 2),
                "humidity":   round(float(hum),  2),
                "pressure":   round(float(pres), 2),
            })

    df = pd.DataFrame(records)

    # -----------------------------------------------------------------------
    # Ground truth: initially all Normal
    # -----------------------------------------------------------------------
    gt = df[["timestamp", "station_id"]].copy()
    gt["is_anomaly_gt"]  = False
    gt["true_root_cause"] = "Normal"

    # -----------------------------------------------------------------------
    # INJECT ANOMALIES
    # Sensor observations are modified; ground truth is updated in parallel.
    # -----------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 1. AWS-001 Delhi — Sensor Spike at hour 14
    #    Large sudden temp jump. Neighbours unaffected.
    # ------------------------------------------------------------------
    _inject(df, gt, "AWS-001", [timestamps[14]],
            temp_delta=+15.5,
            true_root_cause="Sensor Spike / Fault")

    # ------------------------------------------------------------------
    # 2. AWS-002 Mumbai + AWS-008 Pune — Genuine Monsoon Surge h=36-42
    #    Spatially coherent event. Mumbai hit first (h=36), Pune 1h later (h=37).
    #    Different magnitudes at each station.
    #    Chennai is slightly affected (partial signal, further away).
    # ------------------------------------------------------------------
    for h in range(36, 43):
        _inject(df, gt, "AWS-002", [timestamps[h]],
                temp_delta=-4.5, hum_delta=+18.0, pres_delta=-7.0,
                true_root_cause="Genuine Weather Event (not a fault)")

    for h in range(37, 43):
        _inject(df, gt, "AWS-008", [timestamps[h]],
                temp_delta=-2.5, hum_delta=+12.0, pres_delta=-4.5,
                true_root_cause="Genuine Weather Event (not a fault)")

    # Partial/weaker signal at Chennai (not labelled anomaly in ground truth — genuine boundary)
    for h in range(37, 43):
        _inject(df, gt, "AWS-003", [timestamps[h]],
                hum_delta=+6.0, pres_delta=-2.0,
                true_root_cause=None)  # No GT label — weak boundary signal

    # ------------------------------------------------------------------
    # 3. AWS-003 Chennai — Frozen Sensor h=28-34 (7 consecutive hours)
    # ------------------------------------------------------------------
    frozen_row = df[(df["station_id"] == "AWS-003") & (df["timestamp"] == timestamps[28])]
    if not frozen_row.empty:
        ft = float(frozen_row["temp"].iloc[0])
        fh = float(frozen_row["humidity"].iloc[0])
        fp = float(frozen_row["pressure"].iloc[0])
        for h in range(28, 35):
            mask = (df["station_id"] == "AWS-003") & (df["timestamp"] == timestamps[h])
            df.loc[mask, ["temp", "humidity", "pressure"]] = ft, fh, fp
            gt.loc[(gt["station_id"] == "AWS-003") & (gt["timestamp"] == timestamps[h]),
                   ["is_anomaly_gt", "true_root_cause"]] = True, "Frozen / Stuck Sensor"

    # ------------------------------------------------------------------
    # 4. AWS-004 Kolkata — Temperature Calibration Drift h=48-60
    #    Sensor over-reads by 0.4°C per hour cumulatively.
    # ------------------------------------------------------------------
    for i, h in enumerate(range(48, 61)):
        _inject(df, gt, "AWS-004", [timestamps[h]],
                temp_delta=(i + 1) * 0.4,
                true_root_cause="Calibration Drift")

    # ------------------------------------------------------------------
    # 5. AWS-006 Jaipur — Communication Failure h=20-22 (NaN)
    # ------------------------------------------------------------------
    for h in range(20, 23):
        mask = (df["station_id"] == "AWS-006") & (df["timestamp"] == timestamps[h])
        df.loc[mask, ["temp", "humidity", "pressure"]] = np.nan
        gt.loc[(gt["station_id"] == "AWS-006") & (gt["timestamp"] == timestamps[h]),
               ["is_anomaly_gt", "true_root_cause"]] = True, "Communication Failure"

    # ------------------------------------------------------------------
    # 6. AWS-007 Hyderabad — Multivariate Inconsistency h=55
    #    Temp spikes high while pressure drops sharply and humidity collapses.
    #    This combination is physically implausible — sensor fault.
    # ------------------------------------------------------------------
    _inject(df, gt, "AWS-007", [timestamps[55]],
            temp_delta=+9.0, hum_delta=-32.0, pres_delta=-10.0,
            true_root_cause="Multivariate Inconsistency")

    # ------------------------------------------------------------------
    # 7. AWS-010 Ahmedabad — Temperature Drop h=30
    #    Sudden large negative temperature excursion. Neighbours normal.
    # ------------------------------------------------------------------
    _inject(df, gt, "AWS-010", [timestamps[30]],
            temp_delta=-12.0,
            true_root_cause="Sensor Spike / Fault")

    # ------------------------------------------------------------------
    # 8. AWS-009 Patna — Humidity Spike h=45
    #    Humidity shoots to near 100% while temp and pressure remain normal.
    #    Spatially isolated (neighbours not affected).
    # ------------------------------------------------------------------
    _inject(df, gt, "AWS-009", [timestamps[45]],
            hum_delta=+35.0,
            true_root_cause="Sensor Spike / Fault")

    # ------------------------------------------------------------------
    # 9. AWS-011 Lucknow — Pressure Calibration Drift h=58-65
    #    Pressure sensor slowly reads too high.
    # ------------------------------------------------------------------
    for i, h in enumerate(range(58, 66)):
        _inject(df, gt, "AWS-011", [timestamps[h]],
                pres_delta=(i + 1) * 0.6,
                true_root_cause="Calibration Drift")

    # ------------------------------------------------------------------
    # 10. AWS-005 Bhopal — Normal reference station (no faults injected)
    #     Kept clean to provide a pure normal baseline for evaluation.
    # ------------------------------------------------------------------

    # Sort both dataframes consistently
    df = df.sort_values(["timestamp", "station_id"]).reset_index(drop=True)
    gt = gt.sort_values(["timestamp", "station_id"]).reset_index(drop=True)

    return df, gt


def _inject(df: pd.DataFrame, gt: pd.DataFrame, sid: str, ts_list: list, *,
            temp_delta: float = 0.0, hum_delta: float = 0.0, pres_delta: float = 0.0,
            true_root_cause: str | None = None) -> None:
    """
    In-place helper: apply sensor-reading deltas and update ground truth.

    true_root_cause=None means the event is a partial/boundary signal
    and should not be included in ground-truth evaluation labels.
    """
    for ts in ts_list:
        mask = (df["station_id"] == sid) & (df["timestamp"] == ts)
        if temp_delta != 0.0:
            df.loc[mask, "temp"]     = df.loc[mask, "temp"]     + temp_delta
        if hum_delta != 0.0:
            df.loc[mask, "humidity"] = df.loc[mask, "humidity"] + hum_delta
            df.loc[mask, "humidity"] = df.loc[mask, "humidity"].clip(0, 100)
        if pres_delta != 0.0:
            df.loc[mask, "pressure"] = df.loc[mask, "pressure"] + pres_delta

        if true_root_cause is not None:
            gt_mask = (gt["station_id"] == sid) & (gt["timestamp"] == ts)
            gt.loc[gt_mask, "is_anomaly_gt"]  = True
            gt.loc[gt_mask, "true_root_cause"] = true_root_cause


def neighbor_map(top_k: int = 4) -> dict:
    """
    Spatial proximity graph: each station → list of nearest station IDs.

    Uses Euclidean distance on (lat, lon) degrees.
    top_k increased to 4 for richer spatial consistency analysis.

    Returns
    -------
    dict[str, list[str]]
    """
    nbrs = {}
    for stn in STATIONS:
        distances = []
        for other in STATIONS:
            if other["station_id"] == stn["station_id"]:
                continue
            d = ((stn["lat"] - other["lat"]) ** 2 + (stn["lon"] - other["lon"]) ** 2) ** 0.5
            distances.append((other["station_id"], d))
        distances.sort(key=lambda x: x[1])
        nbrs[stn["station_id"]] = [s for s, _ in distances[:top_k]]
    return nbrs


def get_station_info() -> pd.DataFrame:
    """Return station metadata as a DataFrame (useful for UI)."""
    return pd.DataFrame([
        {k: v for k, v in stn.items()
         if k in ("station_id", "name", "lat", "lon", "climate", "base_temp", "base_hum", "base_pres")}
        for stn in STATIONS
    ])


def build_neighbor_map_from_df(sensor_df: pd.DataFrame, top_k: int = 5, max_radius: float = 6.0) -> dict:
    """
    Build a spatial neighbour map from any sensor DataFrame that has
    station_id, lat, lon columns.  Uses Euclidean distance on degrees.
    Restricts neighbours to within `max_radius` (approx 600km) to ensure
    geographic proximity / regional relevance.

    Returns
    -------
    dict[str, list[tuple[str, float]]] 
        Each station_id → list of up to top_k nearest station_ids within radius.
        Returns just the list of station IDs for compatibility.
    """
    stations = (
        sensor_df.groupby("station_id")[["lat", "lon"]]
        .first()
        .dropna()
        .reset_index()
    )
    nbrs: dict = {sid: [] for sid in sensor_df["station_id"].unique()}
    for _, row in stations.iterrows():
        sid = row["station_id"]
        lat1, lon1 = row["lat"], row["lon"]
        dists = []
        for _, other in stations.iterrows():
            osid = other["station_id"]
            if osid == sid:
                continue
            d = ((lat1 - other["lat"]) ** 2 + (lon1 - other["lon"]) ** 2) ** 0.5
            if d <= max_radius:
                dists.append((d, osid))
        dists.sort()
        nbrs[sid] = [s for _, s in dists[:top_k]]
    return nbrs

