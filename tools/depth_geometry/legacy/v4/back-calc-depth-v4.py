"""
back-calculate-depth-compare-v4.py
─────────────────────────────────────────────────────────────────────────────
v4 makes two changes over v3 to improve physical soundness:

  (a) WIDTH_EXP lower bound raised from 0.05 → 0.25.
      Leopold & Maddock (1953) empirical hydraulic geometry gives width
      exponents of 0.3–0.5 for natural channels.  Allowing 0.05 let the
      optimiser collapse width to a near-constant, which is unphysical.
      The new bound enforces a realistic minimum width-discharge sensitivity.
      Accept that raw KGE on depth may be slightly lower — the parameter is
      now physically meaningful.

  (b) Objective switched from raw-KGE to log-KGE (KGE computed on
      log-transformed depths).
      Raw KGE is dominated by high-flow events; log-KGE weights low and
      moderate flows more equally, giving the optimiser information across
      the full flow range.  This naturally pushes WIDTH_EXP away from its
      bound because width scaling matters more at lower flows.
      A small epsilon (LOG_EPS = 0.01 m) is added before log to avoid
      log(0) on near-zero depths.

      Both raw-KGE and log-KGE are reported in the output so you can
      compare with v3.

All other logic (data loading, Q-cap, DE + L-BFGS-B, metrics, plots)
is unchanged from v3.
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
MIZU_NC  = Path("yukon-upper-kw.nc")
TOPO_NC  = Path("topology.nc")
OBS_CSV  = Path("09CD001_openwater_depth_wasp_calibration.csv")

TARGET_SEG = 239
MANN_N     = 0.025
OUT_PNG    = Path(f"compare_depth_seg{TARGET_SEG}_v4.png")

# Calibration bounds
# v3: WIDTH_EXP_BOUNDS = (0.05, 0.50)  ← allowed unphysical near-constant width
# v4: raised lower bound to 0.25 to enforce Leopold & Maddock physical range
WSCALE_BOUNDS    = (0.1, 500.0)
WIDTH_EXP_BOUNDS = (0.25, 0.50)

# Log-KGE epsilon: added to depth before log to avoid log(0)
LOG_EPS = 0.01   # metres

# Outlier cap for calibration
Q_CAL_CAP_PCT = 99.0

TIME_START = "2015-01-01"
TIME_END   = "2024-12-31"

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
    """KGE on raw values."""
    if np.std(obs, ddof=1) == 0 or np.std(sim, ddof=1) == 0:
        return np.nan
    r     = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim, ddof=1) / np.std(obs, ddof=1)
    beta  = np.mean(sim) / np.mean(obs)
    return 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)


def log_kge(obs, sim, eps=LOG_EPS):
    """KGE computed on log-transformed depths.

    Weights low-to-moderate flows more equally than raw KGE, giving the
    optimiser sensitivity to width scaling across the full flow range.
    """
    lo = np.log(obs + eps)
    ls = np.log(sim + eps)
    if np.std(lo, ddof=1) == 0 or np.std(ls, ddof=1) == 0:
        return np.nan
    r     = np.corrcoef(lo, ls)[0, 1]
    alpha = np.std(ls, ddof=1) / np.std(lo, ddof=1)
    beta  = np.mean(ls) / np.mean(lo)
    return 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)


def calc_metrics(obs, sim):
    slope_lr, intercept, r, _, _ = stats.linregress(obs, sim)
    nse = 1.0 - np.sum((obs - sim)**2) / np.sum((obs - np.mean(obs))**2)
    return dict(
        R2        = r**2,
        NSE       = nse,
        KGE       = kge(obs, sim),
        logKGE    = log_kge(obs, sim),
        RMSE      = np.sqrt(np.mean((sim - obs)**2)),
        MAE       = np.mean(np.abs(sim - obs)),
        PBIAS     = 100.0 * np.sum(obs - sim) / np.sum(obs),
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
df_obs = df_obs[df_obs["true_depth"] >= 0].rename(columns={"true_depth": "obs_depth"})

if df_obs.empty:
    raise RuntimeError("No observed data after filtering.")

print(f"  Records    : {len(df_obs)}")
print(f"  Time range : {df_obs.index[0].date()} → {df_obs.index[-1].date()}")
print(f"  Depth      : mean={df_obs['obs_depth'].mean():.3f} m, "
      f"std={df_obs['obs_depth'].std(ddof=1):.3f} m, "
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
    print("  ⚠ Q maximum is >10× the 99.9th-percentile — likely numerical "
          "artefact in mizuRoute output.")

# ─────────────────────────────────────────────────────────────────────────────
# BUILD PAIRED DATA AND CALIBRATION SUBSET
# ─────────────────────────────────────────────────────────────────────────────
df_sim_Q = (pd.DataFrame({"Q": Q_seg}, index=time)
              .resample("D").mean()
              .loc[TIME_START:TIME_END])

df_obs_daily = df_obs[["obs_depth"]].resample("D").mean()
df_pair      = df_sim_Q.join(df_obs_daily, how="inner").dropna()
df_pair_ow   = df_pair[df_pair.index.month.isin([5, 6, 7, 8, 9, 10])]
print(f"\nPaired open-water records : {len(df_pair_ow)}")

if Q_CAL_CAP_PCT < 100:
    q_cap  = np.percentile(df_pair_ow["Q"].values, Q_CAL_CAP_PCT)
    df_cal = df_pair_ow[df_pair_ow["Q"] <= q_cap]
    print(f"  Q-cap for calibration   : Q ≤ p{Q_CAL_CAP_PCT:g} = {q_cap:.2f} m³/s")
    print(f"  Calibration subset size : {len(df_cal)}  "
          f"(excluded {len(df_pair_ow) - len(df_cal)} extreme days)")
else:
    q_cap  = np.inf
    df_cal = df_pair_ow
    print(f"  No Q-cap — calibrating on all {len(df_cal)} paired days")

if len(df_cal) < 30:
    raise RuntimeError(f"Too few calibration points ({len(df_cal)}) — check Q_CAL_CAP_PCT.")

Q_cal   = df_cal["Q"].values
obs_cal = df_cal["obs_depth"].values

# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION  (maximise log-KGE)
# ─────────────────────────────────────────────────────────────────────────────
print("\nCalibrating (objective: log-KGE) ...")
print(f"  Bounds → wscale ∈ {WSCALE_BOUNDS}, WIDTH_EXP ∈ {WIDTH_EXP_BOUNDS}")
print(f"  (v3 had WIDTH_EXP_BOUNDS lower = 0.05; v4 raises it to "
      f"{WIDTH_EXP_BOUNDS[0]} for physical plausibility)")


def neg_log_kge(params):
    wscale, wexp = params
    d = sim_depth(wscale, wexp, Q_cal, MANN_N, S_seg)
    if not np.all(np.isfinite(d)) or np.std(d, ddof=1) == 0:
        return 1.0
    lk = log_kge(obs_cal, d)
    return np.nan_to_num(-lk, nan=1.0)


de_result = differential_evolution(
    neg_log_kge,
    bounds   = [WSCALE_BOUNDS, WIDTH_EXP_BOUNDS],
    seed     = DE_SEED,
    maxiter  = DE_MAXITER,
    popsize  = DE_POPSIZE,
    tol      = 1e-7,
    polish   = False,
    updating = "deferred",
)

polish = minimize(
    neg_log_kge,
    x0      = de_result.x,
    method  = "L-BFGS-B",
    bounds  = [WSCALE_BOUNDS, WIDTH_EXP_BOUNDS],
    options = {"ftol": 1e-10, "gtol": 1e-8, "maxiter": 1000},
)

wscale_opt, wexp_opt = polish.x
log_kge_train = -polish.fun

# Also report raw KGE at calibrated parameters for comparison with v3
d_cal_opt    = sim_depth(wscale_opt, wexp_opt, Q_cal, MANN_N, S_seg)
raw_kge_train = kge(obs_cal, d_cal_opt)

print(f"  Calibrated wscale    : {wscale_opt:.4f}")
print(f"  Calibrated WIDTH_EXP : {wexp_opt:.4f}")
print(f"  Calibration log-KGE  : {log_kge_train:.4f}  ← v4 objective")
print(f"  Calibration raw-KGE  : {raw_kge_train:.4f}  ← v3 objective (for reference)")

for val, lo, hi, name in [
    (wscale_opt, *WSCALE_BOUNDS,    "wscale"),
    (wexp_opt,   *WIDTH_EXP_BOUNDS, "WIDTH_EXP"),
]:
    if at_bound(val, lo, hi):
        print(f"  ⚠ {name} is at its bound — consider widening bounds.")

# ─────────────────────────────────────────────────────────────────────────────
# APPLY TO FULL OPEN-WATER PERIOD
# ─────────────────────────────────────────────────────────────────────────────
if Q_CAL_CAP_PCT < 100:
    q_global_cap = np.percentile(Q_seg, 99.9)
    Q_seg_used   = np.minimum(Q_seg, q_global_cap)
else:
    Q_seg_used = Q_seg

depth_full = sim_depth(wscale_opt, wexp_opt, Q_seg_used, MANN_N, S_seg)

df_sim = (pd.DataFrame({"sim_depth": depth_full}, index=time)
            .resample("D").mean()
            .loc[TIME_START:TIME_END])
df_sim = df_sim[df_sim.index.month.isin([5, 6, 7, 8, 9, 10])]

df_eval = df_sim.join(df_obs_daily, how="inner").dropna()
df_eval["Q"] = df_sim_Q.reindex(df_eval.index)["Q"]

obs_all = df_eval["obs_depth"].values
sim_all = df_eval["sim_depth"].values
m_all   = calc_metrics(obs_all, sim_all)

if Q_CAL_CAP_PCT < 100:
    mask_in = df_eval["Q"].values <= q_cap
    obs_f, sim_f = obs_all[mask_in], sim_all[mask_in]
    m_f = calc_metrics(obs_f, sim_f) if len(obs_f) > 1 else None
else:
    m_f = None

# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────
DISPLAY_KEYS = ("R2", "NSE", "KGE", "logKGE", "RMSE", "MAE", "PBIAS")

def print_metric_block(title, m, n):
    print(f"\n{title}  (n={n})")
    print("=" * 50)
    for k in DISPLAY_KEYS:
        print(f"  {k:<8}: {m[k]:+.4f}")
    print("=" * 50)

print_metric_block("Metrics on FULL paired open-water set", m_all, len(obs_all))
print(f"  Obs mean  : {obs_all.mean():.3f} m  (std {obs_all.std(ddof=1):.3f})")
print(f"  Sim mean  : {sim_all.mean():.3f} m  (std {sim_all.std(ddof=1):.3f})")

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
    f"River Depth Comparison — Segment {TARGET_SEG} (v4, log-KGE calibrated, "
    f"WIDTH_EXP ∈ {WIDTH_EXP_BOUNDS})\n"
    f"Open Water May–Oct {TIME_START[:4]}–{TIME_END[:4]}  |  "
    f"n={MANN_N}  |  w = {wscale_opt:.3f} × Q^{wexp_opt:.3f}  |  "
    f"log-KGE={log_kge_train:.3f}  raw-KGE={raw_kge_train:.3f}",
    fontsize=11, fontweight="bold"
)

ax_ts.plot(df_eval.index, df_eval["obs_depth"], lw=0.9, color="steelblue",
           label="Observed (HYDAT rating-curve)", zorder=3)
ax_ts.plot(df_eval.index, df_eval["sim_depth"], lw=0.9, color="tomato", alpha=0.85,
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
           f"R²     = {m_all['R2']:.3f}\nNSE    = {m_all['NSE']:.3f}\n"
           f"KGE    = {m_all['KGE']:.3f}\nlogKGE = {m_all['logKGE']:.3f}\n"
           f"RMSE   = {m_all['RMSE']:.3f} m\nPBIAS  = {m_all['PBIAS']:+.1f}%",
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