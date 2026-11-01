"""
export_all_segments_v5.py
=========================
Calculate and export channel width, depth, and unit flow for ALL segments
of the Upper Yukon River network, using parameters calibrated against
HYDAT rating-curve-derived depth at the reference segment
(segment 239, gauge 09CD001).

Changes from v4
---------------
  • FIXED provenance mismatch: v4's docstring/metadata and its calibrated
    Model B parameters (ALPHA_B=0.001954, BETA_B=1.104167) described a
    DIFFERENT calibration effort — segment 189 / gauge 09AB001, level-
    observed depth, 2014-2017, KGE=0.758 — while TARGET_SEG was actually
    set to 239 and OBS_CSV actually pointed at the 09CD001 file. Those
    09AB001 numbers were themselves traced back to a stale HYDAT reference
    that had already been flagged and corrected in back-calc-depth-v5.py's
    docstring earlier in this project; they were never real calibrated
    output for segment 239. All parameters/metadata below are now the
    actual back-calc-depth-v5.py output for seg 239 / gauge 09CD001.
  • HYBRID depth model (structure kept from v4, values corrected):
      - seg 239 (gauged, calibrated): direct rating  h = ALPHA_B · Q^BETA_B
        (Model B from back-calc-depth-v5.py)
      - all other segments: Manning + width law (Model A) — the only
        transferable option at ungauged segments.
      - NOTE: unlike the v4 case (where Model A structurally could not
        match Model B: KGE 0.384 vs 0.758), at seg 239 Model A and Model B
        converged to the SAME effective power-law curve during calibration
        (verified: (MANN_N/(WSCALE_OPT*sqrt(slope)))^0.6 == ALPHA_B almost
        exactly, and their exponents match: 0.6*(1-WIDTH_EXP_OPT) ==
        BETA_B). The hybrid structure is kept for clarity/provenance
        (which segment's depth is the calibrated relationship vs the
        transferred one), but at seg 239 itself it makes no numerical
        difference which branch is used.
    A per-segment "depth_method" flag is stored in the NetCDF, as in v4.
  • Q capped at p99 per segment, computed only within the post-warm-up
    analysis window (TIME_START onward). v4 computed the per-segment
    outlier diagnostic and Q-cap over the FULL raw series, which would
    reintroduce the cold-start spin-up spike (identified and excluded in
    back-calc-depth-v5.py) into every segment's cap calculation here.
  • TIME_START/TIME_END and VAL_START/VAL_END updated to match this
    project's actual data availability (simulation run 2013 onward with
    2013-2014 as model warm-up; HYDAT-derived observed depth available
    2015 onward for validation) instead of the v4 script's 2008-2017 /
    2014-2017 window, which belonged to the other gauge's project.
  • obs_depth_method / datum metadata corrected: this station has no
    independent absolute elevation datum set (GAUGE_TO_CGVD2013 = None in
    curve-fitting-depth-estimate.py), so depth here is relative to the
    fitted zero-flow gauge stage only, not an absolute CGVD-referenced bed
    elevation. v4's specific "628.961 m CGVD2013" value belonged to the
    other gauge and has been removed rather than corrected, since no
    equivalent absolute reference currently exists for this station.
  • Optional WIDTH_ANCHOR_M rescales exported width/unit_flow to a known
    real width at median Q, same as v4 (kept, since it's independent of
    which gauge's parameters are used).
"""

import xarray as xr
import numpy as np
import netCDF4 as nc
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import pearsonr
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MIZU_NC = Path("yukon-upper-kw.nc")
TOPO_NC = Path("topology.nc")
OBS_CSV = Path("09CD001_openwater_depth_wasp_calibration.csv")
OUT_NC  = Path("channel_geometry_all_v5.nc")
OUT_CSV = Path("channel_geometry_all_v5.csv")
OUT_PNG = Path("channel_geometry_all_v5.png")

# --- Calibrated parameters (back-calc-depth-v5.py, seg 239, gauge 09CD001) --
# Model B — direct rating, used for depth at the gauged segment.
# Source: back-calc-depth-v5.py calibration run, log-KGE objective,
# Q<=p99 cap, full paired open-water set (n=1732 after cap), 2015-2024.
ALPHA_B = 0.421667
BETA_B  = 0.270671
# Model B calibration performance (log-KGE=0.5461, raw KGE=0.5904); full
# paired-set metrics (n=1750): R2=0.3945, NSE=0.3435, KGE=0.6000. See the
# low/mid/high tercile breakdown from back-calc-depth-v5.py for the known
# low-flow performance gap (R2=0.014 in the lowest Q tercile) before relying
# on this for low-flow days specifically.

# Model A — Manning + width law, used for depth at ungauged segments.
# Same source/run as Model B above.
MANN_N        = 0.025
WSCALE_OPT    = 4.286006
WIDTH_EXP_OPT = 0.548888       # NOT at bound (bounds were 0.25-0.70)

# --- Optional width anchor ----------------------------------------------------
# If you know the real width at median flow at the reference segment (e.g.
# from satellite imagery), set it here to rescale EXPORTED width & unit_flow
# across all segments. Depth is unaffected.
WIDTH_ANCHOR_M = None          # e.g. 130.0 ; None = no rescaling

# --- Reference segment & validation settings ---------------------------------
TARGET_SEG = 239              # gauge 09CD001, Upper Yukon River domain
OUT_VAL    = Path(f"channel_geometry_seg{TARGET_SEG}_validation_v5.png")
MIN_DEPTH  = 0.01

# Model warm-up period: 2013-01-01 through 2014-12-31 is spin-up (channel/
# basin storage initializing from empty; includes the cold-start Q spike at
# 2013-01-01/01-02 identified during earlier debugging). Excluded from
# export, plotting, and all diagnostics below — not just calibration.
TIME_START = "2015-01-01"      # plotting/export window
TIME_END   = "2024-12-31"
VAL_START  = "2015-01-01"      # HYDAT-derived observed depth, post-warm-up
VAL_END    = "2024-12-31"
OW_MONTHS  = [5, 6, 7, 8, 9, 10]

# --- Q-cap percentile applied per segment before geometry calculation ---------
# Computed only within TIME_START:TIME_END (see warm-up note above) —
# computing this over the full raw series would let the pre-2015 spin-up
# spike inflate the cap for every segment, not just segment 239.
Q_CAP_PCT = 99.0

# --- Outlier diagnostic threshold ---------------------------------------------
Q_OUTLIER_RATIO = 10.0   # flag segments where max(Q) > ratio × p99

# ─────────────────────────────────────────────────────────────────────────────
# LOAD SIMULATED DATA
# ─────────────────────────────────────────────────────────────────────────────
print("Loading simulated data ...")
ds = xr.open_dataset(MIZU_NC)
if "KWroutedRunoff" not in ds:
    raise KeyError(f"'KWroutedRunoff' not found in {MIZU_NC}.")

Q    = ds["KWroutedRunoff"].values.astype(np.float64)
time = pd.DatetimeIndex(ds["time"].values)

with nc.Dataset(TOPO_NC) as topo:
    slope   = topo["slope"][:].astype(np.float64)
    seg_ids = topo["segId"][:].astype(np.int64)

n_time, n_seg = Q.shape
print(f"  Q shape    : {Q.shape}")
print(f"  Time range : {time[0].date()} → {time[-1].date()}")
print(f"  Segments   : {n_seg}")

if TARGET_SEG not in seg_ids:
    raise ValueError(f"Segment {TARGET_SEG} not found in topology.")
ref_idx = int(np.where(seg_ids == TARGET_SEG)[0][0])

# Warm-up exclusion mask, applied before any diagnostic or cap computation.
window_mask = (time >= pd.Timestamp(TIME_START)) & (time <= pd.Timestamp(TIME_END))
n_excluded_spinup = int((~window_mask).sum())
if n_excluded_spinup > 0:
    print(f"  (Note: {n_excluded_spinup} timesteps before {TIME_START} or "
          f"after {TIME_END} excluded as model warm-up before any "
          f"diagnostic, cap, or export below.)")

# ─────────────────────────────────────────────────────────────────────────────
# REPORT APPLIED PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
print("\nApplying calibrated parameters (no recalibration in this script):")
print(f"  Depth, seg {TARGET_SEG} (Model B): h = {ALPHA_B} · Q^{BETA_B}")
print(f"  Depth, other segs (Model A)  : Manning n={MANN_N}, "
      f"wscale={WSCALE_OPT}, WIDTH_EXP={WIDTH_EXP_OPT}")
print(f"  Q-cap = p{Q_CAP_PCT:g} per segment (within {TIME_START} to {TIME_END})")
print(f"  Width anchor: {WIDTH_ANCHOR_M if WIDTH_ANCHOR_M else 'none'}")
print("  (Provenance: back-calc-depth-v5.py, gauge 09CD001, seg 239, "
      f"open-water 2015-2024)")

# ─────────────────────────────────────────────────────────────────────────────
# Q-OUTLIER DIAGNOSTIC PER SEGMENT (vectorised, post-warm-up window only)
# ─────────────────────────────────────────────────────────────────────────────
print("\nQ-outlier diagnostic per segment (post-warm-up window):")
Q_safe        = np.maximum(Q, 0.0)
Q_safe_window = Q_safe[window_mask, :]
p99_per_seg   = np.percentile(Q_safe_window, 99, axis=0)
qmax_per_seg  = Q_safe_window.max(axis=0)
ratio_per_seg = qmax_per_seg / np.maximum(p99_per_seg, 1e-6)
outlier_mask  = ratio_per_seg > Q_OUTLIER_RATIO
n_outlier     = int(outlier_mask.sum())

if n_outlier > 0:
    print(f"  ⚠ {n_outlier}/{n_seg} segments show suspicious Q peaks "
          f"(max > {Q_OUTLIER_RATIO:.0f}× p99, within analysis window):")
    for i in np.where(outlier_mask)[0]:
        print(f"      seg {seg_ids[i]:>4}:  p99={p99_per_seg[i]:>10.1f}  "
              f"max={qmax_per_seg[i]:>10.1f}  ratio={ratio_per_seg[i]:>6.1f}×")
else:
    print(f"  ✓ No segment has max(Q) > {Q_OUTLIER_RATIO:.0f}× its p99 "
          "within the analysis window.")

# ─────────────────────────────────────────────────────────────────────────────
# APPLY Q-CAP PER SEGMENT (p99, post-warm-up window)
# ─────────────────────────────────────────────────────────────────────────────
q_cap_per_seg = np.percentile(Q_safe_window, Q_CAP_PCT, axis=0)
Q_capped      = np.minimum(Q_safe, q_cap_per_seg[np.newaxis, :])
n_capped_days = int((Q_safe_window > q_cap_per_seg[np.newaxis, :]).sum())
print(f"\nQ capped at p{Q_CAP_PCT:g} per segment (cap computed within "
      f"{TIME_START} to {TIME_END}; {n_capped_days} segment-timesteps "
      "affected in that window):")
print(f"  Q_capped range (post-warm-up) : "
      f"{Q_capped[window_mask, :].min():.2f} – "
      f"{Q_capped[window_mask, :].max():.2f} m³/s")

# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE WIDTH, DEPTH, UNIT FLOW
# ─────────────────────────────────────────────────────────────────────────────
print("\nCalculating geometry for all segments ...")

slope_2d = np.where(slope > 1e-8, slope, 1e-8)[np.newaxis, :]

# Manning width (always needed: Manning depth at ungauged segments uses it)
width_manning = np.maximum(WSCALE_OPT * (Q_capped ** WIDTH_EXP_OPT), 1e-10)

# Depth — Model A (Manning) everywhere, then overwrite ref segment w/ Model B.
# (As noted in the header, these two are numerically equivalent at seg 239
# for this calibration — this line is kept mainly for explicit provenance.)
depth = np.clip(
    (Q_capped * MANN_N / (width_manning * np.sqrt(slope_2d))) ** 0.6,
    MIN_DEPTH, 50.0)
depth[:, ref_idx] = np.clip(
    ALPHA_B * (Q_capped[:, ref_idx] ** BETA_B), MIN_DEPTH, 50.0)

depth_method = np.array(
    ["manning_A"] * n_seg, dtype=object)
depth_method[ref_idx] = "rating_B"

# Exported width — optionally anchored to a known real width at median Q
if WIDTH_ANCHOR_M is not None:
    q_med_ref   = np.median(Q_capped[window_mask, ref_idx])
    w_med_model = WSCALE_OPT * (q_med_ref ** WIDTH_EXP_OPT)
    width_scale = WIDTH_ANCHOR_M / w_med_model
    print(f"  Width anchor: model width at median Q (seg {TARGET_SEG}) = "
          f"{w_med_model:.1f} m → scaled by {width_scale:.3f} "
          f"to match {WIDTH_ANCHOR_M} m")
else:
    width_scale = 1.0

width     = width_manning * width_scale
unit_flow = Q_capped / width

print(f"  width     range (post-warm-up) : "
      f"{width[window_mask, :].min():.2f} – {width[window_mask, :].max():.2f} m")
print(f"  depth     range (post-warm-up) : "
      f"{depth[window_mask, :].min():.3f} – {depth[window_mask, :].max():.3f} m")
print(f"  unit_flow range (post-warm-up) : "
      f"{unit_flow[window_mask, :].min():.6f} – "
      f"{unit_flow[window_mask, :].max():.4f} m²/s")
if depth[window_mask, :].max() > 10:
    print("  ⚠ depth.max() > 10 m within analysis window — check for "
          "residual Q spikes.")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE NETCDF
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nSaving NetCDF → {OUT_NC}")
ds_out = xr.Dataset(
    {
        "channel_width": (
            ["time", "seg"], width.astype(np.float32),
            {"units": "m", "long_name": "Channel width",
             "equation": "wscale * Q_capped^WIDTH_EXP  (x width_scale)",
             "wscale": WSCALE_OPT, "width_exp": WIDTH_EXP_OPT,
             "width_scale_applied": float(width_scale)}),
        "channel_depth": (
            ["time", "seg"], depth.astype(np.float32),
            {"units": "m", "long_name": "Channel flow depth",
             "equation": (f"seg {TARGET_SEG}: ALPHA_B*Q_capped^BETA_B "
                          "(rating, Model B); others: "
                          "(Q_capped*n/(width_manning*sqrt(slope)))^0.6"),
             "alpha_B": ALPHA_B, "beta_B": BETA_B,
             "mann_n": MANN_N, "min_depth": MIN_DEPTH,
             "depth_datum_note": (
                 "Depth is water_level minus the fitted zero-flow gauge "
                 "stage (b), i.e. depth above the rating-curve datum. No "
                 "independent absolute elevation reference (e.g. CGVD2013) "
                 "has been established for this station, so this is a "
                 "gauge-datum-relative depth, not an absolute bed "
                 "elevation-referenced depth.")}),
        "unit_flow": (
            ["time", "seg"], unit_flow.astype(np.float32),
            {"units": "m2 s-1",
             "long_name": "Unit flow (discharge per unit width)",
             "equation": "Q_capped / width"}),
        "discharge": (
            ["time", "seg"], Q.astype(np.float32),
            {"units": "m3 s-1",
             "long_name": "KWroutedRunoff — routed discharge (uncapped)"}),
        "discharge_capped": (
            ["time", "seg"], Q_capped.astype(np.float32),
            {"units": "m3 s-1",
             "long_name": f"Routed discharge capped at p{Q_CAP_PCT:g}/segment "
                          f"(cap computed within {TIME_START} to {TIME_END})"}),
        "slope": (["seg"], slope.astype(np.float32),
                  {"units": "m/m", "long_name": "Channel bed slope"}),
        "segId": (["seg"], seg_ids, {"long_name": "Segment ID"}),
        "depth_method": (["seg"], depth_method.astype(str),
                         {"long_name": ("Depth model per segment: rating_B "
                                        "(calibrated) or manning_A "
                                        "(transferred)")}),
    },
    coords={"time": ds["time"].values},
    attrs={
        "description"         : ("Channel geometry, Upper Yukon River "
                                 "network — mizuRoute KWT"),
        "reference_gauge"     : "09CD001 Upper Yukon River",
        "calibration_segment" : int(TARGET_SEG),
        "obs_reference"       : str(OBS_CSV),
        "obs_depth_method"    : ("HYDAT rating-curve-derived depth: "
                                 "true_depth = water_level - b, b fitted "
                                 "via log-space power-law rating curve "
                                 "(curve-fitting-depth-estimate.py). No "
                                 "independent absolute datum established."),
        "calibration_source"  : ("back-calc-depth-v5.py: log-KGE objective, "
                                 "DE + L-BFGS-B, Q<=p99 cap, open-water "
                                 "May-Oct, 2015-2024 (post-warm-up)."),
        "calibration_period"  : f"{VAL_START} to {VAL_END}, open-water (May-Oct)",
        "model_B_performance" : ("seg 239, full paired open-water set "
                                 "n=1750: R2=0.3945, NSE=0.3435, KGE=0.6000, "
                                 "logKGE=0.5643. Known limitation: low-flow "
                                 "tercile (Q<p33) R2=0.014, NSE=-1.235 -- "
                                 "much weaker at low flow than high flow "
                                 "(high tercile R2=0.711); see back-calc-"
                                 "depth-v5.py tercile breakdown."),
        "model_A_performance" : ("seg 239: numerically equivalent to Model "
                                 "B at this segment for this calibration "
                                 "(same effective exponent and constant); "
                                 "kept as the transferable form for "
                                 "ungauged segments, where Model B's fitted "
                                 "constant/exponent do not apply."),
        "note_low_flow_limitation": (
                                 "Depth estimates at low flow (below "
                                 "~p33 of Q) are considerably less reliable "
                                 "than at high flow for this gauge/segment. "
                                 "Root cause not yet isolated between "
                                 "shoulder-season contamination, rating-"
                                 "curve b-uncertainty, or model-structure "
                                 "limitations at low flow -- treat low-flow "
                                 "depth output here as provisional."),
        "note_width_bias"     : ("Manning-calibrated width law has not been "
                                 "independently validated against real "
                                 "channel width; set WIDTH_ANCHOR_M to "
                                 "rescale exported width/unit_flow if a "
                                 "known reference width becomes available."),
        "note_q_outliers"     : (f"{n_outlier} of {n_seg} segments show "
                                 f"max(Q) > {Q_OUTLIER_RATIO:.0f}x p99(Q) "
                                 f"within {TIME_START} to {TIME_END} — "
                                 "possible numerical artefacts or genuine "
                                 "high-flow events; Q capped at "
                                 f"p{Q_CAP_PCT:g} per segment "
                                 f"({n_capped_days} segment-timesteps "
                                 "affected). Data before "
                                 f"{TIME_START} is model warm-up and is "
                                 "excluded from all diagnostics and export."),
        "q_cap_percentile"    : Q_CAP_PCT,
        "min_depth_m"         : MIN_DEPTH,
        "analysis_window"     : f"{TIME_START} to {TIME_END} (post-warm-up)",
    }
)
encoding = {v: {"zlib": True, "complevel": 4}
            for v in ["channel_width", "channel_depth", "unit_flow",
                      "discharge", "discharge_capped"]}
ds_out.to_netcdf(OUT_NC, encoding=encoding)
ds.close()
print(f"  Saved: {OUT_NC}")

# ─────────────────────────────────────────────────────────────────────────────
# PER-SEGMENT SUMMARY CSV (post-warm-up window only)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nBuilding summary statistics → {OUT_CSV}")
rows = []
for i, sid in enumerate(seg_ids):
    d  = depth[window_mask, i]
    w  = width[window_mask, i]
    uf = unit_flow[window_mask, i]
    q  = Q[window_mask, i]
    rows.append({
        "seg_id"           : int(sid),
        "depth_method"     : depth_method[i],
        "slope"            : round(float(slope[i]), 8),
        "mean_Q_m3s"       : round(float(q.mean()), 4),
        "p99_Q_m3s"        : round(float(np.percentile(q, 99)), 4),
        "max_Q_m3s"        : round(float(q.max()), 4),
        "q_cap_m3s"        : round(float(q_cap_per_seg[i]), 4),
        "mean_depth_m"     : round(float(d.mean()), 4),
        "min_depth_m"      : round(float(d.min()), 4),
        "max_depth_m"      : round(float(d.max()), 4),
        "p5_depth_m"       : round(float(np.percentile(d, 5)), 4),
        "p95_depth_m"      : round(float(np.percentile(d, 95)), 4),
        "mean_width_m"     : round(float(w.mean()), 4),
        "min_width_m"      : round(float(w.min()), 4),
        "max_width_m"      : round(float(w.max()), 4),
        "mean_unitflow_m2s": round(float(uf.mean()), 6),
        "p5_unitflow_m2s"  : round(float(np.percentile(uf, 5)), 6),
        "p95_unitflow_m2s" : round(float(np.percentile(uf, 95)), 6),
    })

df_stats = pd.DataFrame(rows).set_index("seg_id")
df_stats.to_csv(OUT_CSV)
print(f"  Saved: {OUT_CSV}")
print("\n" + df_stats[["depth_method", "mean_depth_m", "max_depth_m",
                        "mean_width_m", "mean_unitflow_m2s"]].to_string())

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION — OBSERVED vs SIMULATED (Model B) AT REFERENCE SEGMENT
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nLoading observed data for validation at seg {TARGET_SEG} ...")
df_obs_raw = pd.read_csv(OBS_CSV)
df_obs_raw["date"]       = pd.to_datetime(df_obs_raw["date"], errors="coerce")
df_obs_raw["true_depth"] = pd.to_numeric(df_obs_raw["true_depth"], errors="coerce")
df_obs_raw = df_obs_raw.dropna(subset=["date", "true_depth"])

df_obs = (df_obs_raw.set_index("date").sort_index().loc[VAL_START:VAL_END])
df_obs = df_obs[df_obs["true_depth"] >= 0].rename(
    columns={"true_depth": "obs_depth"})

sim_ref = (pd.Series(depth[:, ref_idx], index=time, name="sim_depth")
             .resample("D").mean().loc[VAL_START:VAL_END])
sim_ref = sim_ref[sim_ref.index.month.isin(OW_MONTHS)]

obs_daily = df_obs["obs_depth"].resample("D").mean()
df_paired = pd.concat([sim_ref, obs_daily], axis=1, join="inner").dropna()
df_paired.columns = ["sim_depth", "obs_depth"]
n_paired = len(df_paired)
print(f"  Paired open-water records : {n_paired} "
      f"({VAL_START} to {VAL_END})")

if n_paired < 3:
    raise RuntimeError("Too few paired obs/sim points.")

obs = df_paired["obs_depth"].values
sim = df_paired["sim_depth"].values

def nse(o, s):
    return 1.0 - np.sum((o - s)**2) / np.sum((o - np.mean(o))**2)

def kge(o, s):
    r     = pearsonr(o, s)[0]
    alpha = np.std(s, ddof=1) / np.std(o, ddof=1)
    beta  = np.mean(s) / np.mean(o)
    return 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

stats_lines = [
    f"n         = {n_paired}",
    f"R²        = {pearsonr(obs, sim)[0]**2:.3f}",
    f"NSE       = {nse(obs, sim):.3f}",
    f"KGE       = {kge(obs, sim):.3f}",
    f"RMSE      = {np.sqrt(np.mean((obs - sim)**2)):.3f} m",
    f"MAE       = {np.mean(np.abs(obs - sim)):.3f} m",
    f"PBIAS     = {100.0 * np.sum(obs - sim) / np.sum(obs):+.1f} %",
    f"Mean bias = {np.mean(sim - obs):+.3f} m",
]
stats_text = "\n".join(stats_lines)

print(f"\n{'─'*44}")
print(f"  Performance — Seg {TARGET_SEG} (Model B rating depth)")
print(f"  Open-water, {VAL_START} to {VAL_END}")
print(f"{'─'*44}")
for line in stats_lines:
    print(f"  {line}")
print(f"{'─'*44}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — ALL-SEGMENT OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nPlotting all-segment overview → {OUT_PNG}")

df_depth = (pd.DataFrame(depth, index=time, columns=seg_ids)
              .resample("D").mean().loc[TIME_START:TIME_END])
df_width = (pd.DataFrame(width, index=time, columns=seg_ids)
              .resample("D").mean().loc[TIME_START:TIME_END])
df_uf    = (pd.DataFrame(unit_flow, index=time, columns=seg_ids)
              .resample("D").mean().loc[TIME_START:TIME_END])

fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)
fig.suptitle(
    f"Channel Geometry — All {n_seg} Segments (v5, hybrid depth)\n"
    f"seg {TARGET_SEG}: h={ALPHA_B}·Q^{BETA_B} (rating)   |   "
    f"others: Manning n={MANN_N}, wscale={WSCALE_OPT:.2f}, "
    f"WIDTH_EXP={WIDTH_EXP_OPT}   |   Q capped at p{Q_CAP_PCT:g}\n"
    f"Calibrated at gauge 09CD001, seg {TARGET_SEG}, {TIME_START}–{TIME_END}",
    fontsize=11, fontweight="bold")

cmap = plt.cm.viridis(np.linspace(0, 1, n_seg))

for ax, df_v, ylab, title in [
    (axes[0], df_depth, "Depth (m)", "Daily Mean Depth per Segment "
     f"(seg {TARGET_SEG} = rating model, highlighted)"),
    (axes[1], df_width, "Width (m)", "Daily Mean Width per Segment"),
    (axes[2], df_uf, "Unit Flow (m²/s)",
     "Daily Mean Unit Flow per Segment (Q_capped / width)"),
]:
    for i, sid in enumerate(seg_ids):
        if sid == TARGET_SEG and df_v is df_depth:
            continue
        ax.plot(df_v.index, df_v[sid], lw=0.7, color=cmap[i], alpha=0.6)
    if df_v is df_depth:
        ax.plot(df_v.index, df_v[TARGET_SEG], lw=1.4, color="crimson",
                label=f"seg {TARGET_SEG} (rating model)")
        ax.legend(fontsize=8, loc="upper right", framealpha=0.7)
    ax.axhline(df_v.values.mean(), color="red", ls="--", lw=1.0)
    ax.set_ylabel(ylab, fontsize=11)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(bottom=0)

axes[2].set_xlabel("Date", fontsize=11)
axes[2].xaxis.set_major_locator(mdates.YearLocator())
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=30, ha="right")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved: {OUT_PNG}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — VALIDATION AT REFERENCE SEGMENT
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nPlotting validation → {OUT_VAL}")

fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                         gridspec_kw={"height_ratios": [2, 1]})
fig.suptitle(
    f"Observed vs Simulated River Depth — Segment {TARGET_SEG} "
    f"(Open Water, May–Oct)\n"
    f"Obs: HYDAT rating-curve-derived depth (h − b, gauge 09CD001)   |   "
    f"Sim: rating model h = {ALPHA_B}·Q^{BETA_B}, Q capped at p{Q_CAP_PCT:g}",
    fontsize=11, fontweight="bold")

ax = axes[0]
ax.plot(df_paired.index, df_paired["sim_depth"], color="steelblue",
        lw=0.9, alpha=0.85, label="Simulated (rating model)")
ax.plot(df_paired.index, df_paired["obs_depth"], color="tomato",
        lw=0.9, alpha=0.85, label="Observed (HYDAT rating-based)")
ax.set_ylabel("Depth (m)", fontsize=11)
ax.set_title(f"Time Series — Segment {TARGET_SEG} "
             f"({VAL_START[:4]}–{VAL_END[:4]}, open-water months)",
             fontsize=10)
ax.legend(fontsize=9, loc="upper right", framealpha=0.8)
ax.grid(True, alpha=0.25)
ax.set_ylim(bottom=0)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax.text(0.01, 0.97, stats_text, transform=ax.transAxes, fontsize=8.5,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="grey", alpha=0.85))

ax = axes[1]
lim_min = min(obs.min(), sim.min()) * 0.95
lim_max = max(obs.max(), sim.max()) * 1.05
ax.scatter(obs, sim, s=8, alpha=0.4, color="steelblue", label="Paired obs/sim")
ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1.2, label="1:1 line")
m_lr, b_lr = np.polyfit(obs, sim, 1)
x_rng = np.linspace(lim_min, lim_max, 200)
ax.plot(x_rng, m_lr * x_rng + b_lr, "r-", lw=1.2,
        label=f"Fit: y = {m_lr:.2f}x + {b_lr:.2f}")
ax.set_xlabel("Observed Depth (m)", fontsize=11)
ax.set_ylabel("Simulated Depth (m)", fontsize=11)
ax.set_title(f"Scatter — n = {n_paired} paired daily open-water values",
             fontsize=10)
ax.set_xlim(lim_min, lim_max); ax.set_ylim(lim_min, lim_max)
ax.set_aspect("equal", adjustable="box")
ax.legend(fontsize=9, loc="upper left", framealpha=0.8)
ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig(OUT_VAL, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved: {OUT_VAL}")
print("\nDone.")