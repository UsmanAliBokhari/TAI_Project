"""
TAI Project — Data Cleaning & Preprocessing
============================================
Cleans and merges the Vlinder observation data and NWP ensemble forecast data
into a single dataset ready for ML model training.

Steps:
  1. Load & parse Vlinder observations
  2. Detect and remove flatline periods (station downtime)
  3. Resample Vlinder to 6-hourly, matching NWP valid_times
  4. Convert Vlinder temperature °C → K
  5. Load NWP forecast data
  6. Aggregate 50 ensemble members per (valid_time, leadtime) into summary statistics
  7. Engineer additional features (leadtime, hour-of-day, day-of-year)
  8. Merge on valid_time, dropping rows that coincide with Vlinder downtime
  9. Chronological train / val / test split (no data leakage)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from sklearn.model_selection import train_test_split


# ── Configuration ────────────────────────────────────────────────────────────

VLINDER_FILE   = "Vlinder_VUB_Etterbeek_all.csv"
NWP_FILE       = "NWP_input.csv"

# Flatline detection: flag a run as downtime if the temperature is
# constant for more than this many consecutive 5-minute readings.
# 12 readings = 1 hour. We use 12 to avoid flagging short genuine plateaus.
FLATLINE_MIN_CONSECUTIVE = 12   # ≥ 12 × 5 min = ≥ 60 min flat → downtime

# Resampling: 6-hourly windows anchored to 00:00 UTC
RESAMPLE_FREQ = "6h"

# Train / val / test split dates (adjust to match your data coverage)
VAL_START  = "2024-10-01"
TEST_START = "2025-02-01"


# ── 1. Load & parse Vlinder ───────────────────────────────────────────────────

print("Loading Vlinder observations …")
vlinder = pd.read_csv(VLINDER_FILE, sep=",", parse_dates=False)


# The file already has a pre-built UTC timestamp column; use it directly.
# Fall back to combining Datum + Tijd if the column is missing or all-NaN.
if "time_utc" in vlinder.columns and vlinder["time_utc"].notna().any():
    vlinder["timestamp"] = pd.to_datetime(vlinder["time_utc"], utc=True)
else:
    vlinder["timestamp"] = pd.to_datetime(
        vlinder["Datum"].astype(str) + " " + vlinder["Tijd (UTC)"].astype(str),
        dayfirst=False, utc=True
    )

# Keep only the columns we need
vlinder = vlinder[["timestamp", "Temperatuur"]].rename(columns={"Temperatuur": "obs_temp_C"})
vlinder = vlinder.sort_values("timestamp").reset_index(drop=True)

print(f"  Vlinder: {len(vlinder):,} rows spanning "
      f"{vlinder['timestamp'].min()} → {vlinder['timestamp'].max()}")


# ── 2. Detect flatline periods ────────────────────────────────────────────────
# When the station is down it keeps repeating the last valid reading.
# We flag any run of ≥ FLATLINE_MIN_CONSECUTIVE identical temperature values.

print("\nDetecting flatline (station downtime) periods …")

# Mark rows where the temperature is identical to the previous row
vlinder["temp_change"] = vlinder["obs_temp_C"].diff().ne(0)

# Assign a group ID to each consecutive run of identical values
vlinder["run_id"] = vlinder["temp_change"].cumsum()

# Count how long each run is
run_lengths = vlinder.groupby("run_id")["run_id"].transform("count")

# Flag rows that belong to a run long enough to be downtime
vlinder["is_flatline"] = run_lengths >= FLATLINE_MIN_CONSECUTIVE

n_flatline = vlinder["is_flatline"].sum()
pct = 100 * n_flatline / len(vlinder)
print(f"  Flagged {n_flatline:,} readings ({pct:.1f}%) as flatline/downtime")

# ── Optional: visualise flatlines for inspection ─────────────────────────────
# Plot a 4-week window so the flatlines are clearly visible as horizontal lines.

fig, ax = plt.subplots(figsize=(14, 3))
ok  = vlinder[~vlinder["is_flatline"]]
bad = vlinder[ vlinder["is_flatline"]]
ax.plot(ok["timestamp"],  ok["obs_temp_C"],  color="#1D9E75", lw=0.5, label="Valid")
ax.scatter(bad["timestamp"], bad["obs_temp_C"], color="#D85A30", s=1, label="Flatline (downtime)")
ax.set_title("Vlinder temperature — flatline detection")
ax.set_ylabel("°C")
ax.legend(markerscale=8, fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("flatline_detection.png", dpi=150)
plt.close()
print("  Saved flatline_detection.png")


# ── 3. Resample to 6-hourly, masking downtime windows ────────────────────────
# We want one row per 6-hour window, aligned to 00:00 / 06:00 / 12:00 / 18:00 UTC.
# Any 6-hour window that contains *any* flatline readings is discarded entirely,
# because we cannot trust the mean temperature for that window.

print("\nResampling Vlinder to 6-hourly …")

vlinder = vlinder.set_index("timestamp")

# Count total and flatline readings per 6h window
total_per_window    = vlinder["obs_temp_C"].resample(RESAMPLE_FREQ, origin="start_day").count()
flatline_per_window = vlinder["is_flatline"].resample(RESAMPLE_FREQ, origin="start_day").sum()

# Mean temperature for each window (using only valid readings)
valid_only = vlinder.loc[~vlinder["is_flatline"], "obs_temp_C"]
mean_per_window = valid_only.resample(RESAMPLE_FREQ, origin="start_day").mean()

vlinder_6h = pd.DataFrame({
    "obs_temp_C":        mean_per_window,
    "n_readings":        total_per_window,
    "n_flatline":        flatline_per_window,
})

# Drop windows where any flatline was present OR where too few readings remain
vlinder_6h["valid_window"] = (
    (vlinder_6h["n_flatline"] == 0) &
    (vlinder_6h["n_readings"] >= 6)    # at least 6 × 5-min readings in the window
)

n_discarded = (~vlinder_6h["valid_window"]).sum()
print(f"  6-hourly windows: {len(vlinder_6h):,} total, "
      f"{n_discarded:,} discarded (flatline or too few readings)")

vlinder_clean = vlinder_6h[vlinder_6h["valid_window"]].copy()

# ── 4. Convert °C → K ─────────────────────────────────────────────────────────
vlinder_clean["obs_temp_K"] = vlinder_clean["obs_temp_C"] + 273.15
vlinder_clean.index.name = "valid_time"
vlinder_clean = vlinder_clean.reset_index()

print(f"  Clean Vlinder (6-hourly): {len(vlinder_clean):,} rows")


# ── 5. Load NWP forecast data ─────────────────────────────────────────────────

print("\nLoading NWP forecast data …")
nwp = pd.read_csv(NWP_FILE)

# Parse datetimes
nwp["model_time"] = pd.to_datetime(nwp["model_time"], dayfirst=False, utc=True)
nwp["valid_time"]  = pd.to_datetime(nwp["valid_time"],  dayfirst=False, utc=True)

# Drop leadtime = 0 (no input features available at initiation time, per assignment)
nwp = nwp[nwp["leadtime_hours"] > 0].copy()
nwp = nwp[nwp["leadtime_hours"] <= 168].copy()

print(f"  NWP: {len(nwp):,} rows (leadtimes 6–168h, 50 members each)")


# ── 6. Aggregate ensemble members ────────────────────────────────────────────
# For each (model_time, valid_time) pair, compute summary statistics across the
# 50 members for every meteorological variable.
# This collapses 50 rows → 1 row per forecast timestep.

print("\nAggregating 50 ensemble members per forecast timestep …")

# Columns to aggregate (all except index/meta columns)
meta_cols = ["model_time", "leadtime_hours", "valid_time", "member",
             "latitude", "longitude"]
feature_cols = [c for c in nwp.columns if c not in meta_cols]

def ensemble_agg(group):
    """Return mean, std, 10th and 90th percentile for each variable."""
    stats = {}
    for col in feature_cols:
        vals = group[col].dropna()
        stats[f"{col}_mean"] = vals.mean()
        stats[f"{col}_std"]  = vals.std()
        stats[f"{col}_q10"]  = vals.quantile(0.10)
        stats[f"{col}_q90"]  = vals.quantile(0.90)
    return pd.Series(stats)

nwp_agg = (
    nwp
    .groupby(["model_time", "leadtime_hours", "valid_time"])
    .apply(ensemble_agg, include_groups=False)
    .reset_index()
)

print(f"  Aggregated: {len(nwp_agg):,} rows, {len(nwp_agg.columns)} columns")


# ── 7. Feature engineering ────────────────────────────────────────────────────
# These temporal features help the model learn diurnal and seasonal patterns.

nwp_agg["hour_of_day"] = nwp_agg["valid_time"].dt.hour
nwp_agg["day_of_year"] = nwp_agg["valid_time"].dt.dayofyear

# Encode cyclically so that e.g. hour 23 and hour 0 are close in feature space
nwp_agg["hour_sin"] = np.sin(2 * np.pi * nwp_agg["hour_of_day"] / 24)
nwp_agg["hour_cos"] = np.cos(2 * np.pi * nwp_agg["hour_of_day"] / 24)
nwp_agg["doy_sin"]  = np.sin(2 * np.pi * nwp_agg["day_of_year"]  / 365)
nwp_agg["doy_cos"]  = np.cos(2 * np.pi * nwp_agg["day_of_year"]  / 365)

print("  Added temporal features: hour_sin/cos, doy_sin/cos, leadtime_hours")


# ── 8. Merge NWP and Vlinder on valid_time ───────────────────────────────────
# Inner join: only keep timesteps where we have BOTH a clean observation
# and a corresponding forecast.  This automatically excludes any NWP rows
# whose valid_time fell in a Vlinder downtime window.

print("\nMerging NWP and Vlinder on valid_time …")

dataset = nwp_agg.merge(
    vlinder_clean[["valid_time", "obs_temp_K", "obs_temp_C"]],
    on="valid_time",
    how="inner"
)

print(f"  Merged dataset: {len(dataset):,} rows")

# Drop any remaining NaN rows (e.g. from NWP missing values)
before = len(dataset)
dataset = dataset.dropna()
print(f"  After dropping NaN rows: {len(dataset):,} "
      f"({before - len(dataset):,} dropped)")


# ── 9. Chronological train / val / test split ─────────────────────────────────
# CRITICAL: We sort by valid_time and split by date — never shuffle.
# Shuffling would allow future observations to leak into the training set.

print("\nSplitting into train / val / test …")

dataset = dataset.sort_values("valid_time").reset_index(drop=True)

# val_start  = pd.Timestamp(VAL_START,  tz="UTC")
# test_start = pd.Timestamp(TEST_START, tz="UTC")

# train = dataset[dataset["valid_time"] <  val_start]
# val   = dataset[(dataset["valid_time"] >= val_start) & (dataset["valid_time"] < test_start)]
# test  = dataset[dataset["valid_time"] >= test_start]

# Step 1: carve out the test set (last 20%)
train_val, test = train_test_split(dataset, test_size=0.20, shuffle=False)

# Step 2: split the remainder into train and val (val = 20% of total = 25% of train_val)
train, val = train_test_split(train_val, test_size=0.25, shuffle=False)

print(f"  Train : {len(train):,} rows  ({train['valid_time'].min().date()} → "
      f"{train['valid_time'].max().date()})")
print(f"  Val   : {len(val):,} rows  ({val['valid_time'].min().date()} → "
      f"{val['valid_time'].max().date()})")
print(f"  Test  : {len(test):,} rows  ({test['valid_time'].min().date()} → "
      f"{test['valid_time'].max().date()})")

# ── Save processed datasets ───────────────────────────────────────────────────

os.makedirs("processed_data", exist_ok=True)
train.to_csv("processed_data/train.csv", index=False)
val.to_csv(  "processed_data/val.csv",   index=False)
test.to_csv( "processed_data/test.csv",  index=False)
dataset.to_csv("processed_data/full_dataset.csv", index=False)

print("\nSaved to processed_data/")
print("  train.csv, val.csv, test.csv, full_dataset.csv")
print("\nDone. Data is ready for feature selection and modelling.")
