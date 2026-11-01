"""
export_all_segments.py
=======================
Calculate and export channel width, depth, and unit flow for ALL segments
using parameters calibrated against HYDAT rating-curve depth at segment 356.

Parameters
----------
  mann_n    = 0.025   (from param.nml.default &KWT)
  WIDTH_EXP = 0.20    (discharge-based hydraulic geometry exponent)
  wscale    = solved by Brent's method so mean open-water depth at seg 356
              matches HYDAT rating-curve mean depth (2016–2024): 1.5127 m

Equations
---------
  width     [m]    =  wscale * Q ^ WIDTH_EXP
  depth     [m]    =  ( Q * mann_n / (width * sqrt(slope)) ) ^ 0.6
  unit_flow [m2/s] =  Q / width

Observation reference
---------------------
  07BJ001_openwater_depth_wasp_calibration.csv
  Column: true_depth = water_level - b  (rating-curve estimated depth)
  Season: open-water only (May–Oct)

Outputs
-------
  channel_geometry_all.nc                 — NetCDF (time, seg): width, depth, unit_flow
  channel_geometry_all.csv               — per-segment summary statistics
  channel_geometry_all.png               — depth + width + unit flow (all segments)
  channel_geometry_seg356_validation.png — obs vs sim at seg 356 with stats
"""

import xarray as xr
import numpy as np
import netCDF4 as nc
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import brentq
from scipy.stats import pearsonr
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MIZU_NC  = Path("mizu3.nc")
TOPO_NC  = Path("topology.nc")
OBS_CSV  = Path("07BJ001_openwater_depth_wasp_calibration.csv")  # HYDAT curve-fit depths
OUT_NC   = Path("channel_geometry_all.nc")
OUT_CSV  = Path("channel_geometry_all.csv")
OUT_PNG  = Path("channel_geometry_all.png")
OUT_VAL  = Path("channel_geometry_seg356_validation.png")

# Manning's parameters (from param.nml.default &KWT)
MANN_N    = 0.025
WIDTH_EXP = 0.20

# Calibration reference
TARGET_SEG   = 356      # reference segment for wscale calibration
TARGET_DEPTH = 1.5127   # HYDAT mean open-water depth at seg 356, 2016–2024 (m)
MIN_DEPTH    = 0.01     # minimum depth floor — avoids hiding real low-flow conditions

# Comparison window — must overlap with both HYDAT (ends 2024) and mizuRoute sim
TIME_START = "2016-01-01"
TIME_END   = "2024-12-31"

# Open-water months for calibration and comparison (matches HYDAT CSV)
OW_MONTHS = [5, 6, 7, 8, 9, 10]
# ─────────────────────────────────────────────────────────────────────────────

# ── Load simulated data ───────────────────────────────────────────────────────
print("Loading simulated data ...")
ds   = xr.open_dataset(MIZU_NC)

if "KWTroutedRunoff" not in ds:
    raise KeyError(
        f"'KWTroutedRunoff' not found. Available variables: {list(ds.data_vars)}"
    )

Q    = ds["KWTroutedRunoff"].values.astype(np.float64)   # (time, seg)
time = pd.DatetimeIndex(ds["time"].values)

with nc.Dataset(TOPO_NC) as topo:
    slope   = topo["slope"][:].astype(np.float64)    # (seg,)
    seg_ids = topo["segId"][:].astype(np.int32)

n_time, n_seg = Q.shape
print(f"  Q shape   : {Q.shape}")
print(f"  Time range: {time[0].date()} → {time[-1].date()}")
print(f"  Segments  : {n_seg}  →  {list(seg_ids)}")

# ── Load observed data (HYDAT rating-curve depths, open-water only) ───────────
print("\nLoading observed data ...")
df_obs_raw = pd.read_csv(OBS_CSV)
print(f"  Columns : {list(df_obs_raw.columns)}")

df_obs_raw["date"]       = pd.to_datetime(df_obs_raw["date"], errors="coerce")
df_obs_raw["true_depth"] = pd.to_numeric(df_obs_raw["true_depth"], errors="coerce")
df_obs_raw = df_obs_raw.dropna(subset=["date", "true_depth"])

df_obs = df_obs_raw.set_index("date").sort_index()
df_obs = df_obs.loc[TIME_START:TIME_END]
df_obs = df_obs[df_obs["true_depth"] >= 0]

if df_obs.empty:
    raise RuntimeError("No observed data after filtering — check TIME_START/TIME_END.")

df_obs = df_obs.rename(columns={"true_depth": "obs_depth"})
print(f"  Records    : {len(df_obs)}")
print(f"  Time range : {df_obs.index[0].date()} → {df_obs.index[-1].date()}")
print(f"  Depth range: {df_obs['obs_depth'].min():.3f} – "
      f"{df_obs['obs_depth'].max():.3f} m")
print(f"  Mean depth : {df_obs['obs_depth'].mean():.3f} m  "
      f"(median {df_obs['obs_depth'].median():.3f} m)")

# ── Calibrate wscale using segment 356, open-water months, 2016–2024 ─────────
print(f"\nCalibrating wscale against seg {TARGET_SEG} "
      f"(TARGET_DEPTH = {TARGET_DEPTH} m) ...")

if TARGET_SEG not in seg_ids:
    raise ValueError(f"Segment {TARGET_SEG} not found in topology.")

ref_idx = int(np.where(seg_ids == TARGET_SEG)[0][0])
S_ref   = max(slope[ref_idx], 1e-8)

# Filter to calibration window: 2016–2024, open-water months only
cal_mask = (
    (time >= pd.Timestamp(TIME_START)) &
    (time <= pd.Timestamp(TIME_END)) &
    pd.DatetimeIndex(time).month.isin(OW_MONTHS)
)
Q_cal = np.maximum(Q[cal_mask, ref_idx], 0.0)
print(f"  Calibration timesteps : {cal_mask.sum():,} (open-water, {TIME_START[:4]}–{TIME_END[:4]})")

def mean_depth_error(wscale):
    w = np.maximum(wscale * (Q_cal ** WIDTH_EXP), 1e-10)
    d = np.clip((Q_cal * MANN_N / (w * np.sqrt(S_ref))) ** 0.6, 0.0, 50.0)
    return np.mean(d) - TARGET_DEPTH

wscale_opt = brentq(mean_depth_error, 1e-6, 1000.0, xtol=1e-8)
print(f"  Calibrated wscale : {wscale_opt:.6f}")
print(f"  mann_n = {MANN_N}   WIDTH_EXP = {WIDTH_EXP}   min_depth = {MIN_DEPTH}")

# ── Calculate width, depth & unit flow for all segments ───────────────────────
print("\nCalculating width, depth, and unit flow for all segments ...")

Q_safe   = np.maximum(Q, 0.0)
slope_2d = np.where(slope > 1e-8, slope, 1e-8)[np.newaxis, :]   # (1, seg)

width     = wscale_opt * (Q_safe ** WIDTH_EXP)
width     = np.where(width > 1e-10, width, 1e-10)

depth     = (Q_safe * MANN_N / (width * np.sqrt(slope_2d))) ** 0.6
depth     = np.clip(depth, MIN_DEPTH, 50.0)

unit_flow = Q_safe / width

print(f"  width     range: {width.min():.4f} – {width.max():.4f} m")
print(f"  depth     range: {depth.min():.4f} – {depth.max():.4f} m")
print(f"  unit_flow range: {unit_flow.min():.6f} – {unit_flow.max():.4f} m²/s")

# ── Save NetCDF ───────────────────────────────────────────────────────────────
print(f"\nSaving NetCDF → {OUT_NC}")
ds_out = xr.Dataset(
    {
        "channel_width": (
            ["time", "seg"],
            width.astype(np.float32),
            {"units": "m",
             "long_name": "Channel width",
             "equation": f"wscale * Q^{WIDTH_EXP}",
             "wscale": wscale_opt,
             "width_exp": WIDTH_EXP}
        ),
        "channel_depth": (
            ["time", "seg"],
            depth.astype(np.float32),
            {"units": "m",
             "long_name": "Channel flow depth",
             "equation": "(Q*mann_n / (width*sqrt(slope)))^0.6",
             "mann_n": MANN_N,
             "min_depth": MIN_DEPTH}
        ),
        "unit_flow": (
            ["time", "seg"],
            unit_flow.astype(np.float32),
            {"units": "m2 s-1",
             "long_name": "Unit flow (discharge per unit width)",
             "equation": "Q / width"}
        ),
        "discharge": (
            ["time", "seg"],
            Q.astype(np.float32),
            {"units": "m3 s-1",
             "long_name": "KWTroutedRunoff — routed discharge"}
        ),
        "slope": (["seg"], slope.astype(np.float32),
                  {"units": "m/m", "long_name": "Channel bed slope"}),
        "segId": (["seg"], seg_ids,
                  {"long_name": "Segment ID"}),
    },
    coords={"time": ds["time"].values},
    attrs={
        "description"                    : "Channel geometry for all segments — mizuRoute KWT",
        "obs_reference"                  : str(OBS_CSV),
        "obs_depth_method"               : "HYDAT rating curve: true_depth = water_level - b",
        "mann_n"                         : MANN_N,
        "wscale"                         : wscale_opt,
        "width_exp"                      : WIDTH_EXP,
        "min_depth_m"                    : MIN_DEPTH,
        "calibration_segment"            : int(TARGET_SEG),
        "calibration_target_mean_depth_m": TARGET_DEPTH,
        "calibration_period"             : f"{TIME_START} to {TIME_END}, open-water (May-Oct)",
    }
)
encoding = {v: {"zlib": True, "complevel": 4}
            for v in ["channel_width", "channel_depth", "unit_flow", "discharge"]}
ds_out.to_netcdf(OUT_NC, encoding=encoding)
ds.close()
print(f"  Saved: {OUT_NC}")

# ── Save per-segment summary CSV ──────────────────────────────────────────────
print(f"\nBuilding summary statistics → {OUT_CSV}")
rows = []
for i, sid in enumerate(seg_ids):
    d  = depth[:, i]
    w  = width[:, i]
    q  = Q[:, i]
    uf = unit_flow[:, i]
    rows.append({
        "seg_id"           : int(sid),
        "slope"            : round(float(slope[i]), 8),
        "mean_Q_m3s"       : round(float(q.mean()),  4),
        "max_Q_m3s"        : round(float(q.max()),   4),
        "mean_depth_m"     : round(float(d.mean()),  4),
        "min_depth_m"      : round(float(d.min()),   4),
        "max_depth_m"      : round(float(d.max()),   4),
        "p5_depth_m"       : round(float(np.percentile(d,  5)), 4),
        "p95_depth_m"      : round(float(np.percentile(d, 95)), 4),
        "mean_width_m"     : round(float(w.mean()),  4),
        "min_width_m"      : round(float(w.min()),   4),
        "max_width_m"      : round(float(w.max()),   4),
        "mean_unitflow_m2s": round(float(uf.mean()), 6),
        "min_unitflow_m2s" : round(float(uf.min()),  6),
        "max_unitflow_m2s" : round(float(uf.max()),  6),
        "p5_unitflow_m2s"  : round(float(np.percentile(uf,  5)), 6),
        "p95_unitflow_m2s" : round(float(np.percentile(uf, 95)), 6),
    })

df_stats = pd.DataFrame(rows).set_index("seg_id")
df_stats.to_csv(OUT_CSV)
print(f"  Saved: {OUT_CSV}")
print("\n" + df_stats[["mean_depth_m", "max_depth_m",
                        "mean_width_m", "max_width_m",
                        "mean_unitflow_m2s"]].to_string())

# ── Align obs & sim at segment 356 (open-water, daily) ───────────────────────
print(f"\nAligning obs and sim depth at seg {TARGET_SEG} ...")

sim_356 = (pd.Series(depth[:, ref_idx], index=time, name="sim_depth")
             .resample("D").mean()
             .loc[TIME_START:TIME_END])
# Keep open-water months only
sim_356 = sim_356[sim_356.index.month.isin(OW_MONTHS)]

obs_daily = df_obs["obs_depth"].resample("D").mean()

df_paired = pd.concat([sim_356, obs_daily], axis=1, join="inner").dropna()
df_paired.columns = ["sim_depth", "obs_depth"]
n_paired = len(df_paired)
print(f"  Paired open-water records : {n_paired}")

if n_paired < 3:
    raise RuntimeError("Too few paired obs/sim points to compute statistics.")

obs = df_paired["obs_depth"].values
sim = df_paired["sim_depth"].values

# ── Performance metrics ───────────────────────────────────────────────────────
def nse(obs, sim):
    return 1.0 - np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)

def kge(obs, sim):
    r     = pearsonr(obs, sim)[0]
    alpha = np.std(sim)  / np.std(obs)
    beta  = np.mean(sim) / np.mean(obs)
    return 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

def pbias(obs, sim):
    return 100.0 * np.sum(obs - sim) / np.sum(obs)

r2_   = pearsonr(obs, sim)[0] ** 2
nse_  = nse(obs, sim)
kge_  = kge(obs, sim)
rmse  = np.sqrt(np.mean((obs - sim) ** 2))
mae   = np.mean(np.abs(obs - sim))
pb    = pbias(obs, sim)
bias  = np.mean(sim - obs)

stats_lines = [
    f"n         = {n_paired}",
    f"R²        = {r2_:.3f}",
    f"NSE       = {nse_:.3f}",
    f"KGE       = {kge_:.3f}",
    f"RMSE      = {rmse:.3f} m",
    f"MAE       = {mae:.3f} m",
    f"PBIAS     = {pb:+.1f} %",
    f"Mean bias = {bias:+.3f} m",
]
stats_text = "\n".join(stats_lines)

print(f"\n{'─'*40}")
print(f"  Performance — Seg {TARGET_SEG}  (open-water, {TIME_START[:4]}–{TIME_END[:4]})")
print(f"{'─'*40}")
for line in stats_lines:
    print(f"  {line}")
print(f"{'─'*40}")

# ── Plot 1: all-segment overview (depth / width / unit flow) ──────────────────
print(f"\nPlotting all-segment overview → {OUT_PNG}")

df_depth = (pd.DataFrame(depth,     index=time, columns=seg_ids)
              .resample("D").mean().loc[TIME_START:TIME_END])
df_width = (pd.DataFrame(width,     index=time, columns=seg_ids)
              .resample("D").mean().loc[TIME_START:TIME_END])
df_uf    = (pd.DataFrame(unit_flow, index=time, columns=seg_ids)
              .resample("D").mean().loc[TIME_START:TIME_END])

fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)
fig.suptitle(
    f"Channel Geometry — All {n_seg} Segments\n"
    f"mann_n={MANN_N}   wscale={wscale_opt:.4f}   WIDTH_EXP={WIDTH_EXP}   "
    f"min_depth={MIN_DEPTH} m\n"
    f"wscale calibrated at seg {TARGET_SEG}: mean open-water depth = {TARGET_DEPTH} m "
    f"({TIME_START[:4]}–{TIME_END[:4]})",
    fontsize=11, fontweight="bold"
)

cmap = plt.cm.viridis(np.linspace(0, 1, n_seg))

# depth panel
ax = axes[0]
for i, sid in enumerate(seg_ids):
    ax.plot(df_depth.index, df_depth[sid], lw=0.7, color=cmap[i], alpha=0.7,
            label=str(sid))
ax.axhline(df_depth.values.mean(), color="red", ls="--", lw=1.2,
           label=f"Overall mean = {df_depth.values.mean():.2f} m")
ax.set_ylabel("Depth (m)", fontsize=11)
ax.set_title("Daily Mean Depth per Segment", fontsize=10)
ax.legend(fontsize=6, ncol=6, loc="upper right", framealpha=0.7,
          title="seg ID", title_fontsize=6)
ax.grid(True, alpha=0.25)
ax.set_ylim(bottom=0)

# width panel
ax = axes[1]
for i, sid in enumerate(seg_ids):
    ax.plot(df_width.index, df_width[sid], lw=0.7, color=cmap[i], alpha=0.7)
ax.axhline(df_width.values.mean(), color="red", ls="--", lw=1.2,
           label=f"Overall mean = {df_width.values.mean():.2f} m")
ax.set_ylabel("Width (m)", fontsize=11)
ax.set_title("Daily Mean Width per Segment", fontsize=10)
ax.legend(fontsize=9, loc="upper right", framealpha=0.7)
ax.grid(True, alpha=0.25)
ax.set_ylim(bottom=0)

# unit flow panel
ax = axes[2]
for i, sid in enumerate(seg_ids):
    ax.plot(df_uf.index, df_uf[sid], lw=0.7, color=cmap[i], alpha=0.7)
ax.axhline(df_uf.values.mean(), color="red", ls="--", lw=1.2,
           label=f"Overall mean = {df_uf.values.mean():.4f} m²/s")
ax.set_ylabel("Unit Flow (m²/s)", fontsize=11)
ax.set_xlabel("Date", fontsize=11)
ax.set_title("Daily Mean Unit Flow per Segment  (Q / width)", fontsize=10)
ax.legend(fontsize=9, loc="upper right", framealpha=0.7)
ax.grid(True, alpha=0.25)
ax.set_ylim(bottom=0)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved: {OUT_PNG}")

# ── Plot 2: seg 356 validation — obs vs sim ───────────────────────────────────
print(f"\nPlotting validation → {OUT_VAL}")

fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                         gridspec_kw={"height_ratios": [2, 1]})
fig.suptitle(
    f"Observed vs Simulated River Depth — Segment {TARGET_SEG}  "
    f"(Open Water, May–Oct)\n"
    f"Obs: HYDAT rating curve (true_depth = h − b)   |   "
    f"Sim: Manning's eq  n={MANN_N}  wscale={wscale_opt:.4f}  WIDTH_EXP={WIDTH_EXP}",
    fontsize=11, fontweight="bold"
)

# time series panel
ax = axes[0]
ax.plot(df_paired.index, df_paired["sim_depth"],
        color="steelblue", lw=0.9, alpha=0.85, label="Simulated")
ax.plot(df_paired.index, df_paired["obs_depth"],
        color="tomato", lw=0.9, alpha=0.85, label="Observed (HYDAT)")
ax.set_ylabel("Depth (m)", fontsize=11)
ax.set_title(f"Time Series — Segment {TARGET_SEG}  "
             f"({TIME_START[:4]}–{TIME_END[:4]}, open-water months only)",
             fontsize=10)
ax.legend(fontsize=9, loc="upper right", framealpha=0.8)
ax.grid(True, alpha=0.25)
ax.set_ylim(bottom=0)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax.text(0.01, 0.97, stats_text,
        transform=ax.transAxes, fontsize=8.5, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="grey", alpha=0.85))

# scatter panel
ax = axes[1]
lim_min = min(obs.min(), sim.min()) * 0.95
lim_max = max(obs.max(), sim.max()) * 1.05
ax.scatter(obs, sim, s=8, alpha=0.4, color="steelblue", label="Paired obs/sim")
ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1.2, label="1:1 line")
# regression line
m_lr, b_lr = np.polyfit(obs, sim, 1)
x_rng = np.linspace(lim_min, lim_max, 200)
ax.plot(x_rng, m_lr * x_rng + b_lr, "r-", lw=1.2,
        label=f"Fit: y = {m_lr:.2f}x + {b_lr:.2f}")
ax.set_xlabel("Observed Depth (m)", fontsize=11)
ax.set_ylabel("Simulated Depth (m)", fontsize=11)
ax.set_title(f"Scatter — n = {n_paired} paired daily open-water values", fontsize=10)
ax.set_xlim(lim_min, lim_max)
ax.set_ylim(lim_min, lim_max)
ax.set_aspect("equal", adjustable="box")
ax.legend(fontsize=9, loc="upper left", framealpha=0.8)
ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig(OUT_VAL, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved: {OUT_VAL}")
print("\nDone.")
