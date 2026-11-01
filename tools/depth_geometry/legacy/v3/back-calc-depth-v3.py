"""
back-calculate-depth-compare-v3.py
─────────────────────────────────────────────────────────────────────────────
v3 fixes two real problems found in v2 results:

  (a) v2's Nelder-Mead polish step did NOT enforce bounds, so the optimiser
      drifted to WIDTH_EXP = -0.046 (physically backwards — channel
      narrowing as flow rises).  Replaced with L-BFGS-B (bounded).

  (b) The mizuRoute KWroutedRunoff for seg 187 contains clear numerical
      outliers (p99 = 1,586 m³/s but p99.9 = 108,896 m³/s and max =
      402,524 m³/s).  These ~0.1% of timesteps are dominating the
      calibration.  v3 calibrates after excluding paired days above
      a percentile threshold (Q_CAL_CAP_PCT, default 99) so the fit
      reflects the realistic 99% of flows.  Metrics are reported on
      BOTH the full paired set AND the filtered set so you can see
      the honest picture.

Note: the *real* fix is upstream — mizuRoute should not be producing
402,524 m³/s at this gauge.  Investigate forcings, time-step stability,
or spin-up artefacts.  This script just makes the depth-side calibration
robust to those outliers.
─────────────────────────────────────────────────────────────────────────────
"""

import xarray as xr
import numpy as np
import netCDF4 as nc
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import differential_evolution, minimize
from scipy import stats
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MIZU_NC  = Path("yukon-whitehorse.nc")
TOPO_NC  = Path("topology.nc")
OBS_CSV  = Path("09AB001_daily_depth_2007_2017.csv")

TARGET_SEG = 189
MANN_N     = 0.025
OUT_PNG    = Path(f"compare_depth_seg{TARGET_SEG}_v3.png")

# Calibration bounds
WSCALE_BOUNDS    = (0.1, 500.0)
WIDTH_EXP_BOUNDS = (0.05, 0.50)   # physically plausible hydraulic-geometry range

# Outlier handling for calibration: exclude paired days where simulated Q
# exceeds this percentile of the paired Q distribution. Set to 100 to disable.
Q_CAL_CAP_PCT = 99.0

# Comparison window
TIME_START = "2008-01-01"
TIME_END   = "2017-12-31"

# Optimisation
DE_SEED    = 42
DE_MAXITER = 200
DE_POPSIZE = 20

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def sim_depth(wscale, wexp, Q_arr, n, S):
    """Manning + width law: w = wscale * Q^wexp, h = (Q n / (w √S))^0.6"""
    w = np.maximum(wscale * (Q_arr ** wexp), 1e-10)
    return np.clip((Q_arr * n / (w * np.sqrt(S))) ** 0.6, 0.0, 50.0)

def kge(obs, sim):
    r     = np.corrcoef(obs, sim)[0, 1]
    #alpha = np.std(sim)  / np.std(obs)
    alpha = np.std(sim, ddof=1) / np.std(obs, ddof=1)
    beta  = np.mean(sim) / np.mean(obs)
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

def calc_metrics(obs, sim):
    slope_lr, intercept, r, _, _ = stats.linregress(obs, sim)
    r2  = r ** 2
    nse = 1 - np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)
    return dict(
        R2        = r2,
        NSE       = nse,
        KGE       = kge(obs, sim),
        RMSE      = np.sqrt(np.mean((sim - obs) ** 2)),
        MAE       = np.mean(np.abs(sim - obs)),
        PBIAS     = 100 * np.sum(obs - sim) / np.sum(obs),
        slope_lr  = slope_lr,
        intercept = intercept,
    )

def at_bound(v, lo, hi, tol=1e-3):
    return abs(v - lo) < tol or abs(v - hi) < tol

# ─────────────────────────────────────────────────────────────────────────────
# LOAD OBSERVED
# ─────────────────────────────────────────────────────────────────────────────
print("Loading observed data ...")

df_obs_raw = pd.read_csv(OBS_CSV)
df_obs_raw["date"]       = pd.to_datetime(df_obs_raw["date"], errors="coerce")
df_obs_raw["true_depth"] = pd.to_numeric(df_obs_raw["true_depth"], errors="coerce")
df_obs_raw = df_obs_raw.dropna(subset=["date", "true_depth"])

df_obs = (df_obs_raw.set_index("date").sort_index()
                    .loc[TIME_START:TIME_END])
df_obs = df_obs[df_obs["true_depth"] >= 0]
df_obs = df_obs.rename(columns={"true_depth": "obs_depth"})

if df_obs.empty:
    raise RuntimeError("No observed data after filtering.")

print(f"  Records    : {len(df_obs)}")
print(f"  Time range : {df_obs.index[0].date()} → {df_obs.index[-1].date()}")
print(f"  Depth      : mean={df_obs['obs_depth'].mean():.3f} m, "
      f"std={df_obs['obs_depth'].std():.3f} m, "
      f"max={df_obs['obs_depth'].max():.2f} m")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MIZUROUTE
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading mizuRoute output ...")

ds = xr.open_dataset(MIZU_NC)
if "KWroutedRunoff" not in ds:
    raise KeyError(f"'KWroutedRunoff' not found in {MIZU_NC}.")

Q    = ds["KWroutedRunoff"].values.astype(np.float64)
time = pd.DatetimeIndex(ds["time"].values)

with nc.Dataset(TOPO_NC) as topo:
    slope   = topo["slope"][:].astype(np.float64)
    seg_ids = topo["segId"][:].astype(np.int64)

seg_idx = int(np.where(seg_ids == TARGET_SEG)[0][0])
Q_seg   = np.maximum(Q[:, seg_idx], 0.0)
S_seg   = max(slope[seg_idx], 1e-8)
print(f"  Target seg : {TARGET_SEG}  (index {seg_idx})")
print(f"  Slope      : {slope[seg_idx]:.6f}")

qpct = np.percentile(Q_seg, [50, 90, 95, 99, 99.9])
print(f"  Q (m³/s)   : min={Q_seg.min():.2f}  median={qpct[0]:.2f}  "
      f"p95={qpct[2]:.2f}  p99={qpct[3]:.2f}  "
      f"p99.9={qpct[4]:.2f}  max={Q_seg.max():.2f}")
if Q_seg.max() > 10 * qpct[4]:
    print("  ⚠ Q maximum is >10× the 99.9th-percentile value — likely "
          "numerical artefact in mizuRoute output.")

# ─────────────────────────────────────────────────────────────────────────────
# BUILD PAIRED DATA AND CALIBRATION SUBSET
# ─────────────────────────────────────────────────────────────────────────────
df_sim_Q = (pd.DataFrame({"Q": Q_seg}, index=time)
              .resample("D").mean()
              .loc[TIME_START:TIME_END])

df_pair    = df_sim_Q.join(df_obs[["obs_depth"]], how="inner").dropna()
df_pair_ow = df_pair[df_pair.index.month.isin([5,6,7,8,9,10])]
print(f"\nPaired open-water records : {len(df_pair_ow)}")

# Build the calibration subset by excluding extreme-Q days
if Q_CAL_CAP_PCT < 100:
    q_cap = np.percentile(df_pair_ow["Q"].values, Q_CAL_CAP_PCT)
    df_cal = df_pair_ow[df_pair_ow["Q"] <= q_cap]
    print(f"  Q-cap for calibration   : Q ≤ p{Q_CAL_CAP_PCT:g} = {q_cap:.2f} m³/s")
    print(f"  Calibration subset size : {len(df_cal)}  "
          f"(excluded {len(df_pair_ow)-len(df_cal)} extreme days)")
else:
    df_cal = df_pair_ow
    print(f"  No Q-cap applied — calibrating on all {len(df_cal)} paired days")

Q_cal   = df_cal["Q"].values
obs_cal = df_cal["obs_depth"].values

# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION  (maximise KGE)
# ─────────────────────────────────────────────────────────────────────────────
print("\nCalibrating ...")
print(f"  Bounds → wscale ∈ {WSCALE_BOUNDS}, "
      f"WIDTH_EXP ∈ {WIDTH_EXP_BOUNDS}")

def neg_kge(params):
    wscale, wexp = params
    d = sim_depth(wscale, wexp, Q_cal, MANN_N, S_seg)
    if not np.all(np.isfinite(d)) or np.std(d) == 0:
        return 1.0
    return -kge(obs_cal, d)

de_result = differential_evolution(
    neg_kge,
    bounds   = [WSCALE_BOUNDS, WIDTH_EXP_BOUNDS],
    seed     = DE_SEED,
    maxiter  = DE_MAXITER,
    popsize  = DE_POPSIZE,
    tol      = 1e-7,
    polish   = False,
    updating = "deferred",
)

# Bounded local polish — L-BFGS-B respects the bounds (Nelder-Mead does not).
polish = minimize(
    neg_kge,
    x0      = de_result.x,
    method  = "L-BFGS-B",
    bounds  = [WSCALE_BOUNDS, WIDTH_EXP_BOUNDS],
    options = {"ftol": 1e-10, "gtol": 1e-8, "maxiter": 1000},
)

wscale_opt, WIDTH_EXP_opt = polish.x
kge_train = -polish.fun

print(f"  Calibrated wscale    : {wscale_opt:.4f}")
print(f"  Calibrated WIDTH_EXP : {WIDTH_EXP_opt:.4f}")
print(f"  Calibration KGE      : {kge_train:.4f}")

if at_bound(WIDTH_EXP_opt, *WIDTH_EXP_BOUNDS):
    print("  ⚠ WIDTH_EXP at bound — try a different Q_CAL_CAP_PCT, or "
          "widen WIDTH_EXP_BOUNDS, or fix the upstream Q outliers.")
if at_bound(wscale_opt, *WSCALE_BOUNDS):
    print("  ⚠ wscale at bound — widen WSCALE_BOUNDS.")

# ─────────────────────────────────────────────────────────────────────────────
# APPLY TO FULL OPEN-WATER PERIOD
# ─────────────────────────────────────────────────────────────────────────────
# depth_full = sim_depth(wscale_opt, WIDTH_EXP_opt, Q_seg, MANN_N, S_seg)

q_global_cap = np.percentile(Q_seg, 99.9)
Q_seg_capped = np.minimum(Q_seg, q_global_cap)
depth_full = sim_depth(wscale_opt, WIDTH_EXP_opt, Q_seg_capped, MANN_N, S_seg)



df_sim = (pd.DataFrame({"sim_depth": depth_full}, index=time)
            .resample("D").mean()
            .loc[TIME_START:TIME_END])
df_sim = df_sim[df_sim.index.month.isin([5,6,7,8,9,10])]

df_obs_daily = df_obs[["obs_depth"]].resample("D").mean()
df = df_sim.join(df_obs_daily, how="inner").dropna()

# Metrics on full paired set
obs_all = df["obs_depth"].values
sim_all = df["sim_depth"].values
m_all   = calc_metrics(obs_all, sim_all)

# Metrics on filtered set (Q ≤ q_cap) — apples-to-apples with calibration
if Q_CAL_CAP_PCT < 100:
    Q_on_eval_dates = df_sim_Q.reindex(df.index)["Q"].values
    mask_in = Q_on_eval_dates <= q_cap
    obs_f = obs_all[mask_in]
    sim_f = sim_all[mask_in]
    m_f   = calc_metrics(obs_f, sim_f) if len(obs_f) > 1 else None
else:
    m_f   = None

# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────
def print_metric_block(title, m, n):
    print(f"\n{title}  (n={n})")
    print("=" * 50)
    for k, v in m.items():
        if k not in ("slope_lr", "intercept"):
            print(f"  {k:<8}: {v:+.4f}")
    print("=" * 50)

print_metric_block("Metrics on FULL paired open-water set",
                   m_all, len(obs_all))
print(f"  Obs mean  : {obs_all.mean():.3f} m  (std {obs_all.std():.3f})")
print(f"  Sim mean  : {sim_all.mean():.3f} m  (std {sim_all.std():.3f})")

if m_f is not None:
    print_metric_block(f"Metrics on FILTERED set (Q ≤ p{Q_CAL_CAP_PCT:g})",
                       m_f, len(obs_f))

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nPlotting → {OUT_PNG}")

fig = plt.figure(figsize=(15, 10))
gs  = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28)

ax_ts  = fig.add_subplot(gs[0, :])
ax_sc  = fig.add_subplot(gs[1, 0])
ax_res = fig.add_subplot(gs[1, 1])

fig.suptitle(
    f"River Depth Comparison — Segment {TARGET_SEG} (v3, KGE-calibrated, "
    f"Q≤p{Q_CAL_CAP_PCT:g} for calibration)\n"
    f"Open Water May–Oct {TIME_START[:4]}–{TIME_END[:4]}  |  "
    f"n={MANN_N}  |  w = {wscale_opt:.3f} × Q^{WIDTH_EXP_opt:.3f}",
    fontsize=11, fontweight="bold"
)

ax_ts.plot(df.index, df["obs_depth"], lw=0.9, color="steelblue",
           label="Observed (HYDAT rating-curve)", zorder=3)
ax_ts.plot(df.index, df["sim_depth"], lw=0.9, color="tomato", alpha=0.85,
           label="Simulated (Manning / KWroutedRunoff)", zorder=2)
ax_ts.set_ylabel("River Depth (m)")
ax_ts.set_title("Daily River Depth — Observed vs Simulated")
ax_ts.legend(fontsize=9)
ax_ts.grid(True, linestyle="--", alpha=0.4)
ax_ts.xaxis.set_major_locator(mdates.YearLocator())
ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(ax_ts.xaxis.get_majorticklabels(), rotation=30, ha="right")

ax_sc.scatter(obs_all, sim_all, s=4, alpha=0.3, color="steelblue")
lim = max(obs_all.max(), sim_all.max()) * 1.05
ax_sc.plot([0, lim], [0, lim], "k--", lw=1.2, label="1:1")
x_rng = np.linspace(0, lim, 200)
y_reg = m_all["slope_lr"] * x_rng + m_all["intercept"]
ax_sc.plot(x_rng, y_reg, "r-", lw=1.2,
           label=f"Fit: y={m_all['slope_lr']:.2f}x+{m_all['intercept']:.2f}")
ax_sc.set_xlim(0, lim); ax_sc.set_ylim(0, lim)
ax_sc.set_xlabel("Observed Depth (m)"); ax_sc.set_ylabel("Simulated Depth (m)")
ax_sc.set_title("Scatter: Observed vs Simulated (full set)")
ax_sc.legend(fontsize=8)
ax_sc.grid(True, linestyle="--", alpha=0.4)
ax_sc.text(0.04, 0.96,
           f"R²    = {m_all['R2']:.3f}\nNSE   = {m_all['NSE']:.3f}\n"
           f"KGE   = {m_all['KGE']:.3f}\nRMSE  = {m_all['RMSE']:.3f} m\n"
           f"PBIAS = {m_all['PBIAS']:+.1f}%",
           transform=ax_sc.transAxes, fontsize=8, verticalalignment="top",
           bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

residuals = sim_all - obs_all
ax_res.hist(residuals, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
ax_res.axvline(0, color="black", lw=1.5, linestyle="--", label="Zero")
ax_res.axvline(residuals.mean(), color="tomato", lw=1.5, linestyle="--",
               label=f"Mean = {residuals.mean():+.3f} m")
ax_res.set_xlabel("Residual  (sim − obs, m)")
ax_res.set_ylabel("Count")
ax_res.set_title("Depth Residuals Distribution")
ax_res.legend(fontsize=8)
ax_res.grid(True, linestyle="--", alpha=0.4)

plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.show()
print(f"Plot saved → {OUT_PNG}\nDone.")