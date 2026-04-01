"""
TAI Project — Data Exploration & Quality Checks
================================================
Run this AFTER data_cleaning.py has produced processed_data/

Checks:
  1. Basic shape and dtypes of the final dataset
  2. Missing value audit
  3. Target variable (obs_temp_K) distribution and time coverage
  4. Flatline periods visualised on the raw Vlinder data
  5. NWP vs Vlinder temperature comparison (model bias)
  6. Feature correlation with the target
  7. Train / val / test split sanity check (no overlap, correct sizes)
  8. Outlier detection on the target
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ── Load processed data ───────────────────────────────────────────────────────

print("=" * 60)
print("Loading processed datasets …")
print("=" * 60)

train = pd.read_csv("processed_data/train.csv")
val   = pd.read_csv("processed_data/val.csv")
test  = pd.read_csv("processed_data/test.csv")
full  = pd.read_csv("processed_data/full_dataset.csv")

for df in [train, val, test, full]:
    df["valid_time"]  = pd.to_datetime(df["valid_time"],  utc=True)
    df["model_time"]  = pd.to_datetime(df["model_time"],  utc=True)
Path("exploration_plots").mkdir(exist_ok=True)


# ── 1. Basic shape and dtypes ─────────────────────────────────────────────────

print("\n── 1. Dataset shapes ──────────────────────────────────────")
print(f"  Full    : {full.shape[0]:>6,} rows × {full.shape[1]} columns")
print(f"  Train   : {train.shape[0]:>6,} rows")
print(f"  Val     : {val.shape[0]:>6,} rows")
print(f"  Test    : {test.shape[0]:>6,} rows")
print(f"  Train % : {100*len(train)/len(full):.1f}%")
print(f"  Val   % : {100*len(val)/len(full):.1f}%")
print(f"  Test  % : {100*len(test)/len(full):.1f}%")

print("\n── Column dtypes ──────────────────────────────────────────")
print(full.dtypes.to_string())


# ── 2. Missing value audit ────────────────────────────────────────────────────

print("\n── 2. Missing values ──────────────────────────────────────")
missing = full.isnull().sum()
missing_pct = 100 * missing / len(full)
missing_report = pd.DataFrame({"count": missing, "pct": missing_pct})
missing_report = missing_report[missing_report["count"] > 0].sort_values("count", ascending=False)

if missing_report.empty:
    print("  No missing values found — data is complete.")
else:
    print(f"  {len(missing_report)} columns have missing values:")
    print(missing_report.to_string())


# ── 3. Target variable analysis ───────────────────────────────────────────────

print("\n── 3. Target variable (obs_temp_K) ────────────────────────")
target = full["obs_temp_K"]
target_C = full["obs_temp_C"]

print(f"  Min    : {target_C.min():.2f} °C  ({target.min():.2f} K)")
print(f"  Max    : {target_C.max():.2f} °C  ({target.max():.2f} K)")
print(f"  Mean   : {target_C.mean():.2f} °C  ({target.mean():.2f} K)")
print(f"  Std    : {target_C.std():.2f} °C")
print(f"  Median : {target_C.median():.2f} °C")

# Time coverage
print(f"\n  Date range : {full['valid_time'].min().date()} → {full['valid_time'].max().date()}")
print(f"  Total days : {(full['valid_time'].max() - full['valid_time'].min()).days}")

# Leadtime distribution
print(f"\n  Leadtime hours present: {sorted(full['leadtime_hours'].unique())}")


# ── Plot 1: Target over time + distribution ───────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(full["valid_time"], target_C, lw=0.5, color="#1D9E75", alpha=0.7)
axes[0].set_title("Observed temperature over time (Vlinder91)")
axes[0].set_ylabel("Temperature (°C)")
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.autofmt_xdate()

axes[1].hist(target_C, bins=60, color="#1D9E75", edgecolor="none", alpha=0.85)
axes[1].set_title("Distribution of observed temperature")
axes[1].set_xlabel("Temperature (°C)")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("exploration_plots/01_target_variable.png", dpi=150)
plt.close()
print("\n  Saved: exploration_plots/01_target_variable.png")


# ── 4. Flatline check on raw Vlinder ─────────────────────────────────────────

print("\n── 4. Checking raw Vlinder for flatlines ──────────────────")
try:
    raw = pd.read_csv("Vlinder_VUB_Etterbeek_all.csv", sep=",", parse_dates=False)
    raw["timestamp"] = pd.to_datetime(raw["time_utc"], utc=True)
    raw = raw.sort_values("timestamp").reset_index(drop=True)

    raw["temp_change"] = raw["Temperatuur"].diff().ne(0)
    raw["run_id"] = raw["temp_change"].cumsum()
    run_lengths = raw.groupby("run_id")["run_id"].transform("count")
    raw["is_flatline"] = run_lengths >= 12

    # Find the actual downtime periods so we can report them
    flatline_runs = raw[raw["is_flatline"]].groupby("run_id")["timestamp"].agg(["min", "max"])
    flatline_runs["duration_h"] = (flatline_runs["max"] - flatline_runs["min"]).dt.total_seconds() / 3600
    flatline_runs = flatline_runs.sort_values("duration_h", ascending=False)

    print(f"  Total downtime readings  : {raw['is_flatline'].sum():,}")
    print(f"  Number of downtime runs  : {len(flatline_runs)}")
    print(f"  Longest downtime         : {flatline_runs['duration_h'].max():.1f} hours")
    print(f"\n  Top 10 longest downtime periods:")
    print(flatline_runs.head(10)[["min", "max", "duration_h"]].to_string())

    # Plot a month of data that contains a flatline for visual confirmation
    worst_start = flatline_runs.iloc[0]["min"] - pd.Timedelta(days=3)
    worst_end   = flatline_runs.iloc[0]["max"] + pd.Timedelta(days=3)
    zoom = raw[(raw["timestamp"] >= worst_start) & (raw["timestamp"] <= worst_end)]

    fig, ax = plt.subplots(figsize=(14, 3))
    ok  = zoom[~zoom["is_flatline"]]
    bad = zoom[ zoom["is_flatline"]]
    ax.plot(ok["timestamp"],  ok["Temperatuur"],  color="#1D9E75", lw=0.8, label="Valid")
    ax.scatter(bad["timestamp"], bad["Temperatuur"], color="#D85A30", s=4, label="Flatline (downtime)", zorder=3)
    ax.set_title("Longest flatline period (zoomed)")
    ax.set_ylabel("°C")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig("exploration_plots/02_flatline_zoom.png", dpi=150)
    plt.close()
    print("\n  Saved: exploration_plots/02_flatline_zoom.png")

except Exception as e:
    print(f"  Could not run raw Vlinder check: {e}")


# ── 5. NWP model bias: t2m_mean vs observed ───────────────────────────────────

print("\n── 5. NWP model bias ──────────────────────────────────────")

# Convert NWP t2m (already in K) to °C for comparison
full = full.copy()
full["nwp_t2m_C"] = full["t2m_mean"] - 273.15
bias = full["nwp_t2m_C"] - full["obs_temp_C"]

print(f"  Mean bias (NWP - obs) : {bias.mean():.3f} °C")
print(f"  Std of bias           : {bias.std():.3f} °C")
print(f"  RMSE                  : {np.sqrt((bias**2).mean()):.3f} °C")
print(f"  Max overestimate      : {bias.max():.2f} °C")
print(f"  Max underestimate     : {bias.min():.2f} °C")

# Bias by hour of day (should show urban heat island pattern)
bias_by_hour = full.groupby("hour_of_day").apply(
    lambda g: pd.Series({"mean_bias": (g["nwp_t2m_C"] - g["obs_temp_C"]).mean()})
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].scatter(full["obs_temp_C"], full["nwp_t2m_C"], alpha=0.1, s=3, color="#185FA5")
lim = [full["obs_temp_C"].min() - 1, full["obs_temp_C"].max() + 1]
axes[0].plot(lim, lim, "r--", lw=1, label="Perfect forecast")
axes[0].set_xlabel("Observed (°C)")
axes[0].set_ylabel("NWP t2m forecast (°C)")
axes[0].set_title("NWP vs observed temperature")
axes[0].legend()

axes[1].bar(bias_by_hour["hour_of_day"], bias_by_hour["mean_bias"], color="#185FA5", alpha=0.8)
axes[1].axhline(0, color="black", lw=0.8)
axes[1].set_xlabel("Hour of day (UTC)")
axes[1].set_ylabel("Mean bias (°C)")
axes[1].set_title("NWP bias by hour of day")
axes[1].set_xticks(range(0, 24, 3))

plt.tight_layout()
plt.savefig("exploration_plots/03_nwp_bias.png", dpi=150)
plt.close()
print("\n  Saved: exploration_plots/03_nwp_bias.png")


# ── 6. Feature correlation with target ───────────────────────────────────────

print("\n── 6. Feature correlations with target ────────────────────")

# Focus on the _mean aggregated features + temporal features
mean_cols = [c for c in full.columns if c.endswith("_mean")]
temporal   = ["leadtime_hours", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
candidate_features = mean_cols + temporal

corr = full[candidate_features + ["obs_temp_K"]].corr()["obs_temp_K"].drop("obs_temp_K")
corr_sorted = corr.abs().sort_values(ascending=False)

print("  Top 15 features by absolute correlation with obs_temp_K:")
print(corr_sorted.head(15).to_string())

fig, ax = plt.subplots(figsize=(10, 6))
top20 = corr.reindex(corr_sorted.head(20).index)
colors = ["#1D9E75" if v > 0 else "#D85A30" for v in top20]
ax.barh(range(len(top20)), top20.values, color=colors, alpha=0.85)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20.index, fontsize=9)
ax.set_xlabel("Pearson correlation with observed temperature")
ax.set_title("Top 20 features correlated with target")
ax.axvline(0, color="black", lw=0.8)
plt.tight_layout()
plt.savefig("exploration_plots/04_feature_correlations.png", dpi=150)
plt.close()
print("\n  Saved: exploration_plots/04_feature_correlations.png")


# ── 7. Train / val / test split sanity check ──────────────────────────────────

print("\n── 7. Split sanity check ──────────────────────────────────")

# Check for date overlap (there must be none)
train_max = train["valid_time"].max()
val_min   = val["valid_time"].min()
val_max   = val["valid_time"].max()
test_min  = test["valid_time"].min()

overlap_1 = train_max >= val_min
overlap_2 = val_max   >= test_min

print(f"  Train ends  : {train_max.date()}")
print(f"  Val starts  : {val_min.date()}")
print(f"  Val ends    : {val_max.date()}")
print(f"  Test starts : {test_min.date()}")
print(f"  Train/Val overlap : {'❌ OVERLAP DETECTED' if overlap_1 else '✓ None'}")
print(f"  Val/Test overlap  : {'❌ OVERLAP DETECTED' if overlap_2 else '✓ None'}")

# Visualise the split
fig, ax = plt.subplots(figsize=(14, 2))
for df, label, color in [(train, "Train", "#1D9E75"), (val, "Val", "#EF9F27"), (test, "Test", "#D85A30")]:
    ax.barh(0, (df["valid_time"].max() - df["valid_time"].min()).days,
            left=mdates.date2num(df["valid_time"].min()),
            height=0.5, color=color, label=f"{label} ({len(df):,} rows)", alpha=0.85)
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.set_yticks([])
ax.legend(loc="upper left", fontsize=9)
ax.set_title("Train / Val / Test split over time")
plt.tight_layout()
plt.savefig("exploration_plots/05_split_timeline.png", dpi=150)
plt.close()
print("\n  Saved: exploration_plots/05_split_timeline.png")


# ── 8. Outlier detection on target ───────────────────────────────────────────

print("\n── 8. Outlier detection ───────────────────────────────────")

q1, q3 = target_C.quantile(0.25), target_C.quantile(0.75)
iqr = q3 - q1
lower, upper = q1 - 3 * iqr, q3 + 3 * iqr

outliers = full[(target_C < lower) | (target_C > upper)]
print(f"  IQR bounds (3×): [{lower:.1f} °C, {upper:.1f} °C]")
print(f"  Outliers found : {len(outliers)}")
if len(outliers) > 0:
    print(outliers[["valid_time", "obs_temp_C"]].to_string())

print("\n" + "=" * 60)
print("Exploration complete. Check exploration_plots/ for all figures.")
print("=" * 60)
