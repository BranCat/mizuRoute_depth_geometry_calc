"""
back-calculate-depth-compare.py
─────────────────────────────────────────────────────────────────────────────
Compares mizuRoute-simulated river depth (back-calculated via Manning's
equation from KWTroutedRunoff) against HYDAT-derived open-water depth
(from rating curve: true_depth = water_level − b).

Observation source : 07BJ001_openwater_depth_wasp_calibration.csv
                     (open-water season only, May–Oct, true_depth column)

Calibration        : wscale is calibrated so that mean simulated depth
                     matches the HYDAT mean depth over the 2016–2024
                     open-water period (TARGET_DEPTH).
─────────────────────────────────────────────────────────────────────────────
"""

import xarray as xr
import numpy as np
import netCDF4 as nc
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import brentq
from scipy import stats
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MIZU_NC  = Path("mizu3.nc")
TOPO_NC  = Path("topology.nc")
OBS_CSV  = Path("07BJ001_openwater_depth_wasp_calibration.csv")   # HYDAT curve-fit depths
OUT_PNG  = Path("compare_depth_seg356.png")

TARGET_SEG   = 356
MANN_N       = 0.025
WIDTH_EXP    = 0.20

# Mean HYDAT open-water depth for 2016–2024 (computed from OBS_CSV)
# Mean   = 1.5127 m  ← use this; pulled up slightly by high-flow events
# Median = 1.2224 m  ← use this if you want to down-weight flood peaks
TARGET_DEPTH = 1.5127

# Comparison window — must overlap with both HYDAT data and mizuRoute sim
TIME_START = "2016-01-01"
TIME_END   = "2024-12-31"   # HYDAT ends 2024; mizu goes to 2025

# ─────────────────────────────────────────────────────────────────────────────
# LOAD OBSERVED DATA  (HYDAT rating-curve depths, open-water only)
# ─────────────────────────────────────────────────────────────────────────────
print("Loading observed data ...")

df_obs_raw = pd.read_csv(OBS_CSV)
print(f"  Columns : {list(df_obs_raw.columns)}")

# Parse date and depth columns
df_obs_raw["date"]       = pd.to_datetime(df_obs_raw["date"], errors="coerce")
df_obs_raw["true_depth"] = pd.to_numeric(df_obs_raw["true_depth"], errors="coerce")
df_obs_raw = df_obs_raw.dropna(subset=["date", "true_depth"])

# Set index and sort
df_obs = df_obs_raw.set_index("date").sort_index()

# Filter to comparison window
df_obs = df_obs.loc[TIME_START:TIME_END]

# Keep only non-negative depths (already open-water only from CSV)
df_obs = df_obs[df_obs["true_depth"] >= 0]

if df_obs.empty:
    raise RuntimeError("No observed data after filtering — check TIME_START/TIME_END.")

# Rename to standard name for downstream code
df_obs = df_obs.rename(columns={"true_depth": "obs_depth"})

print(f"  Records    : {len(df_obs)}")
print(f"  Time range : {df_obs.index[0].date()} → {df_obs.index[-1].date()}")
print(f"  Depth range: {df_obs['obs_depth'].min():.3f} – "
      f"{df_obs['obs_depth'].max():.3f} m")
print(f"  Mean depth : {df_obs['obs_depth'].mean():.3f} m  "
      f"(median {df_obs['obs_depth'].median():.3f} m)")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MIZUROUTE OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading mizuRoute output ...")

ds   = xr.open_dataset(MIZU_NC)

# Verify variable exists before indexing
if "KWTroutedRunoff" not in ds:
    raise KeyError(
        f"'KWTroutedRunoff' not found. Available variables: {list(ds.data_vars)}"
    )

Q    = ds["KWTroutedRunoff"].values.astype(np.float64)   # (time, seg)
time = pd.DatetimeIndex(ds["time"].values)
print(f"  Timesteps  : {len(time):,}  ({time[0].date()} → {time[-1].date()})")
print(f"  Segments   : {Q.shape[1]}")

with nc.Dataset(TOPO_NC) as topo:
    slope   = topo["slope"][:].astype(np.float64)
    seg_ids = topo["segId"][:].astype(np.int32)

seg_idx = int(np.where(seg_ids == TARGET_SEG)[0][0])
Q_seg   = Q[:, seg_idx]
S_seg   = max(slope[seg_idx], 1e-8)   # guard against zero slope
print(f"  Target seg : {TARGET_SEG}  (index {seg_idx})")
print(f"  Slope      : {slope[seg_idx]:.6f}")
print(f"  Q range    : {Q_seg.min():.2f} – {Q_seg.max():.2f} m³/s")

# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATE wscale — match mean simulated depth to TARGET_DEPTH
# Uses only the 2016–2024 open-water period for consistency with obs
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nCalibrating wscale to TARGET_DEPTH = {TARGET_DEPTH:.4f} m ...")

# Restrict calibration to comparison window
cal_mask   = (time >= pd.Timestamp(TIME_START)) & (time <= pd.Timestamp(TIME_END))
Q_cal      = np.maximum(Q_seg[cal_mask], 0.0)

# Only keep open-water months (May–Oct) to match obs
time_cal   = time[cal_mask]
ow_mask    = pd.DatetimeIndex(time_cal).month.isin([5, 6, 7, 8, 9, 10])
Q_cal_ow   = Q_cal[ow_mask]

def mean_depth_error(wscale):
    w = np.maximum(wscale * (Q_cal_ow ** WIDTH_EXP), 1e-10)
    d = np.clip((Q_cal_ow * MANN_N / (w * np.sqrt(S_seg))) ** 0.6, 0.0, 50.0)
    return np.mean(d) - TARGET_DEPTH

wscale_opt = brentq(mean_depth_error, 1e-6, 1000.0, xtol=1e-8)
print(f"  Calibrated wscale : {wscale_opt:.6f}")

# ─────────────────────────────────────────────────────────────────────────────
# CALCULATE SIMULATED DEPTH — full time series
# ─────────────────────────────────────────────────────────────────────────────
Q_safe = np.maximum(Q_seg, 0.0)
width  = np.maximum(wscale_opt * (Q_safe ** WIDTH_EXP), 1e-10)
depth  = np.clip((Q_safe * MANN_N / (width * np.sqrt(S_seg))) ** 0.6, 0.0, 50.0)

# Resample to daily mean and filter to comparison window
df_sim = (pd.DataFrame({"sim_depth": depth}, index=time)
            .resample("D").mean()
            .loc[TIME_START:TIME_END])

# Keep open-water months only (to match obs which is May–Oct only)
df_sim = df_sim[df_sim.index.month.isin([5, 6, 7, 8, 9, 10])]

# ─────────────────────────────────────────────────────────────────────────────
# ALIGN OBSERVED AND SIMULATED
# ─────────────────────────────────────────────────────────────────────────────
df_obs_daily = df_obs[["obs_depth"]].resample("D").mean()

df = df_sim.join(df_obs_daily, how="inner").dropna()
print(f"\nPaired open-water records : {len(df)}")

if len(df) == 0:
    raise RuntimeError(
        "No overlapping open-water timesteps found. "
        "Check TIME_START/TIME_END and that both files cover the same period."
    )

obs = df["obs_depth"].values
sim = df["sim_depth"].values

# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────
def calc_metrics(obs, sim):
    slope_lr, intercept, r, _, _ = stats.linregress(obs, sim)
    r2    = r ** 2
    nse   = 1 - np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)
    r_kge = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim)  / np.std(obs)
    beta  = np.mean(sim) / np.mean(obs)
    kge   = 1 - np.sqrt((r_kge - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    rmse  = np.sqrt(np.mean((sim - obs) ** 2))
    mae   = np.mean(np.abs(sim - obs))
    # PBIAS: positive = model underestimates, negative = overestimates
    pbias = 100 * np.sum(obs - sim) / np.sum(obs)
    return dict(R2=r2, NSE=nse, KGE=kge, RMSE=rmse,
                MAE=mae, PBIAS=pbias,
                slope_lr=slope_lr, intercept=intercept)

m = calc_metrics(obs, sim)

print("\nPerformance Metrics  (open-water season, 2016–2024)")
print("=" * 45)
for k, v in m.items():
    if k not in ("slope_lr", "intercept"):
        print(f"  {k:<8}: {v:+.4f}")
print("=" * 45)
print(f"  Obs mean  : {obs.mean():.3f} m  (std {obs.std():.3f})")
print(f"  Sim mean  : {sim.mean():.3f} m  (std {sim.std():.3f})")

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS  (2 rows × 2 cols)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nPlotting → {OUT_PNG}")

fig = plt.figure(figsize=(15, 10))
gs  = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28)

ax_ts  = fig.add_subplot(gs[0, :])   # full-width time series
ax_sc  = fig.add_subplot(gs[1, 0])   # scatter
ax_res = fig.add_subplot(gs[1, 1])   # residual histogram

fig.suptitle(
    f"River Depth Comparison — Station {TARGET_SEG}  "
    f"(Open Water, May–Oct, {TIME_START[:4]}–{TIME_END[:4]})\n"
    f"Manning's equation  |  n={MANN_N}  |  w = {wscale_opt:.4f} × Q^{WIDTH_EXP}",
    fontsize=11, fontweight="bold"
)

# ── Time series ───────────────────────────────────────────────────────────────
ax_ts.plot(df.index, df["obs_depth"], lw=0.9, color="steelblue",
           label="Observed (HYDAT rating curve depth)", zorder=3)
ax_ts.plot(df.index, df["sim_depth"], lw=0.9, color="tomato", alpha=0.85,
           label="Simulated (Manning's / KWTroutedRunoff)", zorder=2)
ax_ts.set_ylabel("River Depth (m)")
ax_ts.set_title("Daily River Depth — Observed vs Simulated")
ax_ts.legend(fontsize=9)
ax_ts.grid(True, linestyle="--", alpha=0.4)
ax_ts.xaxis.set_major_locator(mdates.YearLocator())
ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(ax_ts.xaxis.get_majorticklabels(), rotation=30, ha="right")

# ── Scatter ───────────────────────────────────────────────────────────────────
ax_sc.scatter(obs, sim, s=4, alpha=0.3, color="steelblue")
lim = max(obs.max(), sim.max()) * 1.05
ax_sc.plot([0, lim], [0, lim], "k--", lw=1.2, label="1:1")

# Regression line
x_rng  = np.linspace(0, lim, 200)
y_reg  = m["slope_lr"] * x_rng + m["intercept"]
ax_sc.plot(x_rng, y_reg, "r-", lw=1.2,
           label=f"Fit: y={m['slope_lr']:.2f}x+{m['intercept']:.2f}")

ax_sc.set_xlim(0, lim); ax_sc.set_ylim(0, lim)
ax_sc.set_xlabel("Observed Depth (m)"); ax_sc.set_ylabel("Simulated Depth (m)")
ax_sc.set_title("Scatter: Observed vs Simulated")
ax_sc.legend(fontsize=8)
ax_sc.grid(True, linestyle="--", alpha=0.4)

# Metrics text box
metrics_text = (
    f"R²    = {m['R2']:.3f}\n"
    f"NSE   = {m['NSE']:.3f}\n"
    f"KGE   = {m['KGE']:.3f}\n"
    f"RMSE  = {m['RMSE']:.3f} m\n"
    f"PBIAS = {m['PBIAS']:+.1f}%"
)
ax_sc.text(0.04, 0.96, metrics_text, transform=ax_sc.transAxes,
           fontsize=8, verticalalignment="top",
           bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# ── Residual histogram ────────────────────────────────────────────────────────
residuals = sim - obs
ax_res.hist(residuals, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
ax_res.axvline(0,               color="black",  lw=1.5, linestyle="--",
               label="Zero residual")
ax_res.axvline(residuals.mean(), color="tomato", lw=1.5, linestyle="--",
               label=f"Mean = {residuals.mean():+.3f} m")
ax_res.set_xlabel("Residual  (sim − obs, m)")
ax_res.set_ylabel("Count")
ax_res.set_title("Depth Residuals Distribution")
ax_res.legend(fontsize=8)
ax_res.grid(True, linestyle="--", alpha=0.4)

plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.show()
print(f"Plot saved → {OUT_PNG}")
print("\nDone.")
