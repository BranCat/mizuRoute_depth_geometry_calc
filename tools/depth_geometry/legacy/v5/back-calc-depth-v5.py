"""
back-calculate-depth-compare-v5.py
─────────────────────────────────────────────────────────────────────────────
v5 adds a second depth model and compares both in one run:

  MODEL A (unchanged from v4): Manning + power-law width
      w = wscale · Q^WIDTH_EXP,   h = (Q·n / (w·√S))^0.6
      Structural limit: depth exponent ≤ 0.6·(1−WIDTH_EXP).
      NOTE: with WIDTH_EXP_BOUNDS up to 0.70, that ceiling is ~0.18 —
      still below the HYDAT-derived exponent for 09CD001 (~0.43). If the
      calibrated WIDTH_EXP still pins at its upper bound, Model A is
      structurally incapable of reaching the target exponent and the fix
      is a different functional form, not a wider bound.

  MODEL B (new): direct at-a-station rating
      h = α · Q^β,  with β FREE (bounds 0.2–1.5).
      Motivation: at-a-station hydraulic geometry lets the data set the
      exponent directly, rather than deriving it structurally from a
      width-law assumption the way Model A does.

  Both are calibrated with log-KGE (v4 objective) on the same Q-capped
  open-water subset, evaluated on the same paired sets, and plotted
  side by side.

  CHANGES IN THIS REVISION:
  - Fixed HYDAT reference curve: was hardcoded to a DIFFERENT station
    (09AB001, h=0.00195*Q^1.086) even though this script calibrates
    against 09CD001. Now derived from HYDAT_ALPHA/HYDAT_BETA constants
    below, which should be set from the actual curve-fitting-depth-
    estimate.py output for whichever station OBS_CSV points to.
  - Added a p99 -> p99.9 jump check, since the original ">10x max vs
    p99.9" check misses a sharp jump concentrated in the last ~0.1%
    of the distribution (which is what showed up in practice: p99.9
    was ~24x p99, but max was only ~2.6x p99.9, so the original check
    didn't fire).
  - Widened WIDTH_EXP upper bound for a cleaner diagnostic of whether
    Model A's structural ceiling, not just the bound, is the limiter.
  - Added a comment clarifying the calibration-cap vs evaluation-cap
    difference (p99 vs p99.9) is intentional, not an oversight.
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
OUT_PNG    = Path(f"compare_depth_seg{TARGET_SEG}_v5.png")

# HYDAT-derived depth curve for THIS station (from curve-fitting-depth-
# estimate.py's log-space fit output). Used only as a printed reference
# point to sanity-check the calibrated models below — update these two
# numbers whenever OBS_CSV points to a different station, or pull them
# directly from the OBS_CSV columns (a_coeff / c_exponent -> alpha, 1/c)
# at load time instead of hardcoding, to avoid this going stale again.
HYDAT_ALPHA = 0.1280   # a^(-1/c) from the log-space fit
HYDAT_BETA  = 0.4261   # 1/c from the log-space fit

# Model A bounds (Manning width law) — same as v4
WSCALE_BOUNDS    = (0.1, 500.0)
WIDTH_EXP_BOUNDS = (0.25, 0.70)   # widened from (0.25, 0.50) to test
                                  # whether 0.5 was a real optimum or just
                                  # the old bound — see note above.

# Model B bounds (direct rating h = alpha·Q^beta)
# alpha is searched in LOG10 space: the valid alpha spans orders of magnitude
# (gauge rating suggests ~0.1) and linear-space search gets lost in the
# clipped-depth plateau where the objective is flat.
LOG10_ALPHA_BOUNDS = (-5.0, 1.0)
BETA_BOUNDS        = (0.2, 1.5)

LOG_EPS       = 0.01     # m, epsilon for log-KGE
Q_CAL_CAP_PCT = 99.0     # calibration is trained on Q <= p99 only (excludes
                          # the most extreme days). Full-period evaluation
                          # below instead *clips* (not excludes) at p99.9 --
                          # a deliberately wider range than the model was
                          # trained on, to see how it behaves just outside
                          # the calibration envelope rather than only
                          # inside it. This is intentional, not a mismatch.

TIME_START = "2015-01-01"
TIME_END   = "2024-12-31"
# Everything before TIME_START (2013-01-01 through 2014-12-31) is the model's
# warm-up/spin-up period: storages initialize from empty at sim_start, and
# mizuRoute output during this stretch (including the cold-start spike
# right at 2013-01-01/01-02 identified during debugging) is not physically
# meaningful and is never used for calibration or evaluation. This is a
# deliberate, confirmed exclusion, not just a convenient date range.
WARMUP_END = TIME_START

DE_SEED    = 42
DE_MAXITER = 200
DE_POPSIZE = 20

# ─────────────────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────────────────
def sim_depth_manning(params, Q_arr, n, S):
    wscale, wexp = params
    w = np.maximum(wscale * (Q_arr ** wexp), 1e-10)
    return np.clip((Q_arr * n / (w * np.sqrt(S))) ** 0.6, 0.0, 50.0)


def sim_depth_rating(params, Q_arr, n=None, S=None):
    log10_alpha, beta = params
    alpha = 10.0 ** log10_alpha
    return np.clip(alpha * np.power(np.maximum(Q_arr, 1e-10), beta), 0.0, 50.0)


MODELS = {
    "A_manning": dict(
        fn     = sim_depth_manning,
        bounds = [WSCALE_BOUNDS, WIDTH_EXP_BOUNDS],
        names  = ["wscale", "WIDTH_EXP"],
        label  = "Manning width-law (v4)",
    ),
    "B_rating": dict(
        fn     = sim_depth_rating,
        bounds = [LOG10_ALPHA_BOUNDS, BETA_BOUNDS],
        names  = ["log10_alpha", "beta"],
        label  = "Direct rating h=α·Q^β (v5)",
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def kge(obs, sim):
    if np.std(obs, ddof=1) == 0 or np.std(sim, ddof=1) == 0:
        return np.nan
    r     = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim, ddof=1) / np.std(obs, ddof=1)
    beta  = np.mean(sim) / np.mean(obs)
    return 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)


def log_kge(obs, sim, eps=LOG_EPS):
    return kge(np.log(obs + eps), np.log(sim + eps))


def calc_metrics(obs, sim):
    slope_lr, intercept, r, _, _ = stats.linregress(obs, sim)
    nse = 1.0 - np.sum((obs - sim)**2) / np.sum((obs - np.mean(obs))**2)
    return dict(
        R2=r**2, NSE=nse, KGE=kge(obs, sim), logKGE=log_kge(obs, sim),
        RMSE=np.sqrt(np.mean((sim - obs)**2)),
        MAE=np.mean(np.abs(sim - obs)),
        PBIAS=100.0 * np.sum(obs - sim) / np.sum(obs),
        slope_lr=slope_lr, intercept=intercept,
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

df_obs = (df_obs_raw.set_index("date").sort_index().loc[TIME_START:TIME_END])
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

# NOTE: percentiles/diagnostics below are computed on the TIME_START:TIME_END
# analysis window only, not the full raw Q_seg series. The full series can
# include a cold-start spin-up spike right at the model's sim_start (channel/
# basin storage initializing from empty), which is a known, expected model
# artifact rather than a routing bug -- but it inflates percentile stats if
# left in, even when it falls entirely outside the analysis window and never
# reaches calibration (as turned out to be the case here). Slicing first
# avoids that misleading signal.
window_mask = (time >= pd.Timestamp(TIME_START)) & (time <= pd.Timestamp(TIME_END))
Q_seg_window = Q_seg[window_mask]
n_excluded_spinup = len(Q_seg) - len(Q_seg_window)
if n_excluded_spinup > 0:
    print(f"  (Note: {n_excluded_spinup} timesteps before {TIME_START} or "
          f"after {TIME_END} excluded from diagnostics below, to avoid "
          f"spin-up artifacts outside the analysis window skewing the stats.)")

qpct = np.percentile(Q_seg_window, [50, 90, 95, 99, 99.9])
print(f"  Q (m³/s)   : min={Q_seg_window.min():.2f}  median={qpct[0]:.2f}  "
      f"p95={qpct[2]:.2f}  p99={qpct[3]:.2f}  "
      f"p99.9={qpct[4]:.2f}  max={Q_seg_window.max():.2f}")
if Q_seg_window.max() > 10 * qpct[4]:
    print("  ⚠ Q maximum is >10× the 99.9th percentile — likely numerical "
          "artefact in mizuRoute output.")
# NEW CHECK: catches a sharp jump concentrated between p99 and p99.9, which
# the max-vs-p99.9 check above can miss entirely (as it did in practice —
# max was only ~2.6x p99.9, but p99.9 itself was ~24x p99).
if qpct[4] > 10 * qpct[3]:
    print(f"  ⚠ p99.9 ({qpct[4]:.2f}) is >10× p99 ({qpct[3]:.2f}) — sharp "
          "jump concentrated in the top ~0.1% of timesteps. Worth inspecting "
          "those specific days directly (e.g. df_sim_Q above q_cap, sorted) "
          "before trusting the calibration cap to fully isolate this.")

# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC — inspect the extreme-Q days directly
#
# Yukon River context: even accounting for this being a genuinely large
# system, realistic peak flows for the full Yukon River (at its mouth) run
# roughly 25,000-30,000 m3/s during major floods. This is the upper Yukon
# (a sub-basin, segment 239), where realistic peaks should be well below
# that. Values far above that within the analysis window would still be
# worth investigating -- this diagnostic pulls the actual dates/values
# (restricted to the analysis window, same spin-up caveat as above) so you
# can see whether extreme values are concentrated in a handful of specific
# days (routing instability / numerical artefact, or a real high-flow event)
# or spread more broadly (a different kind of bug, e.g. a unit/scaling error).
# ─────────────────────────────────────────────────────────────────────────────
df_qdiag = pd.DataFrame({"Q_m3s": Q_seg_window}, index=time[window_mask]).sort_values(
    "Q_m3s", ascending=False)
n_extreme = int(np.sum(Q_seg_window > qpct[3]))   # count above p99
print(f"\n  Extreme-Q diagnostic (within {TIME_START} to {TIME_END}): "
      f"{n_extreme} timesteps exceed p99 ({qpct[3]:.2f} m3/s) "
      f"out of {len(Q_seg_window)} total.")
print(f"  Top 20 highest simulated Q at segment {TARGET_SEG}:")
print(df_qdiag.head(20).to_string())

qdiag_path = Path(f"extreme_Q_seg{TARGET_SEG}_diagnostic.csv")
df_qdiag[df_qdiag["Q_m3s"] > qpct[3]].to_csv(qdiag_path)
print(f"  Full list of {n_extreme} above-p99 timesteps saved → {qdiag_path}")

# Quick clustering check: are the extreme days consecutive (a single event/
# instability) or scattered across many separate dates (systemic)?
extreme_dates = df_qdiag[df_qdiag["Q_m3s"] > qpct[3]].index.sort_values()
if len(extreme_dates) > 1:
    gaps_days = extreme_dates.to_series().diff().dt.days.dropna()
    n_clusters = int((gaps_days > 1).sum()) + 1
    print(f"  These {len(extreme_dates)} extreme timesteps fall into "
          f"~{n_clusters} separate date cluster(s) "
          f"(consecutive-day runs counted as one cluster).")
    if n_clusters <= 5:
        print("  NOTE: concentrated into very few clusters — consistent with")
        print("        isolated routing-instability events (e.g. cold-start")
        print("        spin-up right at sim_start, or a real high-flow event)")
        print("        rather than a constant/systemic bias. Check the")
        print("        cluster's start date against the model's sim_start and")
        print("        against SUMMA/mizuRoute logs for those timesteps.")
    else:
        print("  NOTE: spread across many separate dates — less consistent")
        print("        with a one-off instability; worth checking for a unit")
        print("        or scaling issue instead (e.g. mm/s vs m3/s mismatch,")
        print("        or an unintended double-accumulation in routing).")

# ─────────────────────────────────────────────────────────────────────────────
# PAIRED DATA / CALIBRATION SUBSET
# ─────────────────────────────────────────────────────────────────────────────
df_sim_Q = (pd.DataFrame({"Q": Q_seg}, index=time)
              .resample("D").mean().loc[TIME_START:TIME_END])
df_obs_daily = df_obs[["obs_depth"]].resample("D").mean()
df_pair    = df_sim_Q.join(df_obs_daily, how="inner").dropna()
df_pair_ow = df_pair[df_pair.index.month.isin([5, 6, 7, 8, 9, 10])]
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

if len(df_cal) < 30:
    raise RuntimeError(f"Too few calibration points ({len(df_cal)}).")

Q_cal   = df_cal["Q"].values
obs_cal = df_cal["obs_depth"].values

# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATE BOTH MODELS (objective: log-KGE)
# ─────────────────────────────────────────────────────────────────────────────
# Analytic OLS seed for Model B:  log(h) = log(alpha) + beta·log(Q)
beta_ols, ln_alpha_ols = np.polyfit(np.log(Q_cal), np.log(obs_cal + LOG_EPS), 1)
seed_B = np.array([np.log10(np.exp(ln_alpha_ols)), beta_ols])
seed_B = np.clip(seed_B,
                 [LOG10_ALPHA_BOUNDS[0], BETA_BOUNDS[0]],
                 [LOG10_ALPHA_BOUNDS[1], BETA_BOUNDS[1]])
print(f"\nModel B OLS seed (log-log regression): "
      f"alpha={10**seed_B[0]:.6f}, beta={seed_B[1]:.4f}")

results = {}
for key, mdl in MODELS.items():
    print(f"\nCalibrating MODEL {key} — {mdl['label']} ...")
    print(f"  Bounds → {mdl['names'][0]} ∈ {mdl['bounds'][0]}, "
          f"{mdl['names'][1]} ∈ {mdl['bounds'][1]}")

    def neg_log_kge(params, fn=mdl["fn"]):
        d = fn(params, Q_cal, MANN_N, S_seg)
        if not np.all(np.isfinite(d)) or np.std(d, ddof=1) == 0:
            return 2.0   # worse than any real log-KGE deficit in valid region
        return np.nan_to_num(-log_kge(obs_cal, d), nan=2.0)

    # Build DE init population; for Model B inject the OLS seed so DE starts
    # inside the (narrow) valid region instead of the clipped-depth plateau.
    rng = np.random.default_rng(DE_SEED)
    lo  = np.array([b[0] for b in mdl["bounds"]])
    hi  = np.array([b[1] for b in mdl["bounds"]])
    init_pop = lo + rng.random((DE_POPSIZE * len(lo), len(lo))) * (hi - lo)
    if key == "B_rating":
        init_pop[0] = seed_B

    de = differential_evolution(
        neg_log_kge, bounds=mdl["bounds"], seed=DE_SEED,
        maxiter=DE_MAXITER, popsize=DE_POPSIZE, tol=1e-7,
        polish=False, updating="deferred", init=init_pop)
    pol = minimize(neg_log_kge, x0=de.x, method="L-BFGS-B",
                   bounds=mdl["bounds"],
                   options={"ftol": 1e-10, "gtol": 1e-8, "maxiter": 1000})

    # keep the better of DE and polished result (polish can stall on plateaus)
    p_opt = pol.x if pol.fun <= de.fun else de.x

    d_cal = mdl["fn"](p_opt, Q_cal, MANN_N, S_seg)
    if key == "B_rating":
        print(f"  Calibrated alpha     : {10**p_opt[0]:.6f}  "
              f"(log10_alpha={p_opt[0]:.4f})")
        print(f"  Calibrated beta      : {p_opt[1]:.6f}")
    else:
        print(f"  Calibrated {mdl['names'][0]:9s}: {p_opt[0]:.6f}")
        print(f"  Calibrated {mdl['names'][1]:9s}: {p_opt[1]:.6f}")
    print(f"  Calibration log-KGE  : {-neg_log_kge(p_opt):.4f}")
    print(f"  Calibration raw-KGE  : {kge(obs_cal, d_cal):.4f}")
    for v, (lo_b, hi_b), nm in zip(p_opt, mdl["bounds"], mdl["names"]):
        if at_bound(v, lo_b, hi_b):
            print(f"  ⚠ {nm} is at its bound.")
    if key == "B_rating":
        print(f"  (HYDAT-derived depth curve for {OBS_CSV.stem}: "
              f"h = {HYDAT_ALPHA:.4f}·Q^{HYDAT_BETA:.4f})")

    results[key] = dict(params=p_opt, **mdl)

# ─────────────────────────────────────────────────────────────────────────────
# APPLY BOTH MODELS TO FULL OPEN-WATER PERIOD
# ─────────────────────────────────────────────────────────────────────────────
q_global_cap = np.percentile(Q_seg_window, 99.9) if Q_CAL_CAP_PCT < 100 else np.inf
Q_seg_used   = np.minimum(Q_seg, q_global_cap)

df_eval = None
for key, r in results.items():
    depth_full = r["fn"](r["params"], Q_seg_used, MANN_N, S_seg)
    ser = (pd.DataFrame({f"sim_{key}": depth_full}, index=time)
             .resample("D").mean().loc[TIME_START:TIME_END])
    ser = ser[ser.index.month.isin([5, 6, 7, 8, 9, 10])]
    df_eval = ser if df_eval is None else df_eval.join(ser, how="inner")

df_eval = df_eval.join(df_obs_daily, how="inner").dropna()
df_eval["Q"] = df_sim_Q.reindex(df_eval.index)["Q"]
obs_all = df_eval["obs_depth"].values
mask_in = df_eval["Q"].values <= q_cap

# ─────────────────────────────────────────────────────────────────────────────
# REPORT — side-by-side
# ─────────────────────────────────────────────────────────────────────────────
DISPLAY_KEYS = ("R2", "NSE", "KGE", "logKGE", "RMSE", "MAE", "PBIAS")
metrics = {}
for key in results:
    sim = df_eval[f"sim_{key}"].values
    metrics[key] = dict(
        full=calc_metrics(obs_all, sim),
        filt=calc_metrics(obs_all[mask_in], sim[mask_in]),
        sim=sim)

print(f"\n{'='*64}")
print(f"  SIDE-BY-SIDE — FULL paired open-water set (n={len(obs_all)})")
print(f"{'='*64}")
hdr = f"  {'metric':<8}" + "".join(f"{results[k]['label'][:26]:>28}" for k in results)
print(hdr)
for mk in DISPLAY_KEYS:
    print(f"  {mk:<8}" + "".join(f"{metrics[k]['full'][mk]:>+28.4f}" for k in results))
print(f"  {'sim mean':<8}" +
      "".join(f"{metrics[k]['sim'].mean():>28.3f}" for k in results))
print(f"  {'sim std':<8}" +
      "".join(f"{metrics[k]['sim'].std(ddof=1):>28.3f}" for k in results))
print(f"  {'obs mean':<8}{obs_all.mean():>28.3f}")
print(f"  {'obs std':<8}{obs_all.std(ddof=1):>28.3f}")

print(f"\n{'='*64}")
print(f"  SIDE-BY-SIDE — FILTERED set Q ≤ p{Q_CAL_CAP_PCT:g} (n={mask_in.sum()})")
print(f"{'='*64}")
for mk in DISPLAY_KEYS:
    print(f"  {mk:<8}" + "".join(f"{metrics[k]['filt'][mk]:>+28.4f}" for k in results))

# ─────────────────────────────────────────────────────────────────────────────
# LOW / MID / HIGH FLOW TERCILE BREAKDOWN
#
# Checks whether the moderate R2/NSE seen above is spread evenly across the
# whole flow range (pointing to a whole-curve issue, e.g. a wrong exponent)
# or concentrated in one flow regime (pointing to something specific to
# low, mid, or high flow, e.g. ice-transition contamination at the low end
# or routing behavior at the high end). Terciles are computed on Q itself
# (not depth), split into three equal-count groups.
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*64}")
print(f"  LOW / MID / HIGH FLOW TERCILE BREAKDOWN (full paired set)")
print(f"{'='*64}")

Q_all = df_eval["Q"].values
q_terc = np.percentile(Q_all, [100/3, 200/3])
tercile_masks = {
    "low  (Q < p33)":            Q_all <  q_terc[0],
    "mid  (p33 <= Q < p67)":      (Q_all >= q_terc[0]) & (Q_all < q_terc[1]),
    "high (Q >= p67)":            Q_all >= q_terc[1],
}
print(f"  Tercile breakpoints: p33={q_terc[0]:.1f} m3/s, p67={q_terc[1]:.1f} m3/s\n")

for band_name, band_mask in tercile_masks.items():
    n_band = int(band_mask.sum())
    if n_band < 10:
        print(f"  {band_name:<24} n={n_band:<6} (too few points, skipped)")
        continue
    obs_band = obs_all[band_mask]
    line = f"  {band_name:<24} n={n_band:<6}"
    for key in results:
        sim_band = metrics[key]["sim"][band_mask]
        m_band = calc_metrics(obs_band, sim_band)
        line += f"  {results[key]['label'][:18]:<18} R2={m_band['R2']:.3f} NSE={m_band['NSE']:.3f}"
    print(line)

print("\n  If R2/NSE are similarly moderate across all three bands, the fit")
print("  quality is a whole-curve issue (consistent with the calibrated")
print("  exponent, ~0.27, being systematically lower than the HYDAT-derived")
print("  exponent, ~0.43 -- see printed reference above). If one band is")
print("  clearly worse than the others, that points to something regime-")
print("  specific (e.g. ice-transition days leaking into 'open water', or")
print("  routing behavior that only breaks down at high flow) instead.")

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS — time series + scatter per model
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nPlotting → {OUT_PNG}")
fig = plt.figure(figsize=(15, 10))
gs  = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28)
ax_ts = fig.add_subplot(gs[0, :])

pA, pB = results["A_manning"]["params"], results["B_rating"]["params"]
fig.suptitle(
    f"Depth Comparison — Segment {TARGET_SEG} (v5: Manning vs direct rating)\n"
    f"A: w={pA[0]:.2f}·Q^{pA[1]:.3f}  |  "
    f"B: h={10**pB[0]:.5f}·Q^{pB[1]:.3f}  |  "
    f"Open water May–Oct {TIME_START[:4]}–{TIME_END[:4]}",
    fontsize=11, fontweight="bold")

ax_ts.plot(df_eval.index, df_eval["obs_depth"], lw=1.0, color="black",
           label="Observed (HYDAT rating)", zorder=3)
ax_ts.plot(df_eval.index, df_eval["sim_A_manning"], lw=0.8, color="tomato",
           alpha=0.8, label="A: Manning width-law", zorder=2)
ax_ts.plot(df_eval.index, df_eval["sim_B_rating"], lw=0.8, color="seagreen",
           alpha=0.8, label="B: direct rating", zorder=2)
ax_ts.set_ylabel("River Depth (m)")
ax_ts.set_title("Daily River Depth — Observed vs Both Models")
ax_ts.legend(fontsize=9); ax_ts.grid(True, ls="--", alpha=0.4)
ax_ts.xaxis.set_major_locator(mdates.YearLocator())
ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(ax_ts.xaxis.get_majorticklabels(), rotation=30, ha="right")

lim = max(obs_all.max(),
          *[metrics[k]["sim"].max() for k in results]) * 1.05
for ax, key, color in [(fig.add_subplot(gs[1, 0]), "A_manning", "tomato"),
                       (fig.add_subplot(gs[1, 1]), "B_rating", "seagreen")]:
    sim = metrics[key]["sim"]
    m   = metrics[key]["full"]
    ax.scatter(obs_all, sim, s=4, alpha=0.3, color=color)
    ax.plot([0, lim], [0, lim], "k--", lw=1.2, label="1:1")
    x_rng = np.linspace(0, lim, 200)
    ax.plot(x_rng, m["slope_lr"] * x_rng + m["intercept"], "r-", lw=1.2,
            label=f"y={m['slope_lr']:.2f}x+{m['intercept']:.2f}")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Observed Depth (m)"); ax.set_ylabel("Simulated Depth (m)")
    ax.set_title(results[key]["label"])
    ax.legend(fontsize=8); ax.grid(True, ls="--", alpha=0.4)
    ax.text(0.04, 0.96,
            f"R²     = {m['R2']:.3f}\nNSE    = {m['NSE']:.3f}\n"
            f"KGE    = {m['KGE']:.3f}\nlogKGE = {m['logKGE']:.3f}\n"
            f"RMSE   = {m['RMSE']:.3f} m\nPBIAS  = {m['PBIAS']:+.1f}%",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.show()
print(f"Plot saved → {OUT_PNG}\nDone.")