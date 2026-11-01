""" 
    This python script is updated on 07-22-2026 with the help of Claude AI.
    
    This script is updated from the original curve-fitting-depth-estimate.py 
    to include a log-space fit for the rating curve, which is more appropriate 
    for power-law relationships and gives better estimates of the zero-flow stage (b). 
    It also includes additional diagnostics and plots to assess the quality of the fit 
    and the sensitivity of depth estimates to uncertainty in b.

"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =============================================================================
# CONFIGURATION — update these per station
# =============================================================================
HYDAT_PATH = "<WORK_ROOT>/mizu-depth-output-codes/Hydat.sqlite3"
STATION    = "09CD001"
OUT_PATH   = "<WORK_ROOT>/mizu-depth-output-codes"

# Geodetic offsets: gauge datum → absolute elevation (m above sea level)
# Set to None if unknown; script will skip absolute bed elevation output
GAUGE_TO_CGVD2013 = None
GAUGE_TO_CGVD28   = None

# =============================================================================
# DATABASE CONNECTION
# =============================================================================
conn = sqlite3.connect(HYDAT_PATH)

# =============================================================================
# LOAD & UNPIVOT DLY_LEVELS (LEVEL1..LEVEL31 -> one row per day)
# =============================================================================
levels_raw = pd.read_sql(
    f"SELECT * FROM DLY_LEVELS WHERE STATION_NUMBER = '{STATION}'", conn)

if levels_raw.empty:
    raise RuntimeError(f"No level data found for station {STATION} in DLY_LEVELS.")

level_rows = []
for _, row in levels_raw.iterrows():
    for day in range(1, 32):
        val = row.get(f"LEVEL{day}")
        if pd.notna(val):
            try:
                date = pd.Timestamp(year=int(row["YEAR"]),
                                    month=int(row["MONTH"]), day=day)
                level_rows.append({"date": date, "water_level": float(val)})
            except ValueError:
                pass

levels = pd.DataFrame(level_rows).sort_values("date").reset_index(drop=True)
print(f"Daily levels loaded : {len(levels)} records "
      f"({levels['date'].min().date()} to {levels['date'].max().date()})")
print(f"  Level range       : {levels['water_level'].min():.4f} – "
      f"{levels['water_level'].max():.4f} m")

# =============================================================================
# LOAD & UNPIVOT DLY_FLOWS (FLOW1..FLOW31 -> one row per day)
# =============================================================================
flows_raw = pd.read_sql(
    f"SELECT * FROM DLY_FLOWS WHERE STATION_NUMBER = '{STATION}'", conn)

if flows_raw.empty:
    raise RuntimeError(f"No flow data found for station {STATION} in DLY_FLOWS.")

flow_rows = []
for _, row in flows_raw.iterrows():
    for day in range(1, 32):
        val = row.get(f"FLOW{day}")
        if pd.notna(val):
            try:
                date = pd.Timestamp(year=int(row["YEAR"]),
                                    month=int(row["MONTH"]), day=day)
                flow_rows.append({"date": date, "flow": float(val)})
            except ValueError:
                pass

if not flow_rows:
    raise RuntimeError(
        f"flow_rows is empty for station {STATION}.\n"
        "Check: (1) all FLOW columns are NULL, "
        "(2) YEAR/MONTH not integers, "
        "(3) wrong station ID."
    )

flows = pd.DataFrame(flow_rows).sort_values("date").reset_index(drop=True)
print(f"Daily flows loaded  : {len(flows)} records "
      f"({flows['date'].min().date()} to {flows['date'].max().date()})")

conn.close()

# =============================================================================
# MERGE & FILTER
# =============================================================================
df = pd.merge(levels, flows, on="date", how="inner")
df = df[(df["water_level"] > 0) & (df["flow"] > 0)].reset_index(drop=True)
print(f"Matched records     : {len(df)}")

df["month"]  = df["date"].dt.month
df["season"] = df["month"].isin([5, 6, 7, 8, 9, 10])
df_open = df[df["season"]].copy()
df_ice  = df[~df["season"]].copy()

print(f"\nOpen water records  : {len(df_open)} (May–Oct)")
print(f"Ice-affected records: {len(df_ice)} (Nov–Apr)")

# =============================================================================
# RATING CURVE FIT  Q = a * (h - b)^c   [open water only]
#
# CHANGE (log-space fit): b is the parameter your depth values depend on
# directly (depth = h - b), so it needs to be well constrained by the LOW
# stage / low flow days, where the curve's shape near h -> b pins b down.
# A linear least-squares fit on Q minimizes squared error in discharge
# units, which are dominated by high-flow days (hundreds to low thousands
# of m3/s) -- low-flow days barely influence the loss, even though they are
# the days that matter most for locating b accurately.
#
# Fitting in log space (log Q vs log(h-b)) gives every flow magnitude
# roughly equal weight, which is the standard approach for power-law
# rating curves and should produce a more reliable b for depth estimation,
# not just a better-looking R2 on discharge.
# =============================================================================
def rating_curve(h, a, b, c):
    return a * np.power(np.maximum(h - b, 1e-6), c)

def rating_curve_log(h, a, b, c):
    return np.log(a) + c * np.log(np.maximum(h - b, 1e-6))

h_open  = df_open["water_level"].values
q_open  = df_open["flow"].values
h_min   = h_open.min()

b_upper = h_min * 0.9999
b_init  = h_min * 0.80

p0     = [200.0, b_init, 1.8]
bounds = ([1e-3, 0.0,    0.5],
          [1e7,  b_upper, 5.0])

# --- Linear-space fit (kept for comparison / backward compatibility) -------
popt_lin, pcov_lin = curve_fit(rating_curve, h_open, q_open,
                                p0=p0, bounds=bounds, maxfev=50000)

# --- Log-space fit (now the primary fit used downstream) -------------------
popt, pcov = curve_fit(rating_curve_log, h_open, np.log(q_open),
                        p0=p0, bounds=bounds, maxfev=50000)
a, b, c = popt
perr    = np.sqrt(np.diag(pcov))

a_lin, b_lin, c_lin = popt_lin
perr_lin = np.sqrt(np.diag(pcov_lin))

print(f"\n  Linear-space fit (previous method): a={a_lin:.4f}  b={b_lin:.4f} ± {perr_lin[1]:.4f}  c={c_lin:.4f}")
print(f"  Log-space fit    (this run's method): a={a:.4f}  b={b:.4f} ± {perr[1]:.4f}  c={c:.4f}")
print(f"  Shift in b from switching to log-space fit: {b - b_lin:+.4f} m")

# --- Parameter correlation check --------------------------------------------
# High correlation between b and c (or a and b) means many different
# combinations fit the data almost equally well -- i.e. b is not tightly
# pinned down by the data even if R2 looks good. This is the more direct
# check on "how much can I trust b" than R2 on discharge.
corr = pcov / np.outer(perr, perr)
corr_df = pd.DataFrame(corr, index=["a", "b", "c"], columns=["a", "b", "c"])
print("\n  Parameter correlation matrix (log-space fit):")
print(corr_df.round(3).to_string())
if abs(corr_df.loc["b", "c"]) > 0.9:
    print("  NOTE: |corr(b,c)| > 0.9 -- b and c are strongly correlated.")
    print("        The printed +/- uncertainty on b likely understates how")
    print("        much b could shift under a different (still-plausible) fit.")

# ── Detect b collapse ──────────────────────────────────────────────────────
b_collapsed = (b < 0.001 * h_min) if h_min > 0 else (b < 0.001)
if b_collapsed:
    print("\n  WARNING: b collapsed to near-zero.")
    print("  This station's gauge datum is likely already referenced near the bed.")
    print("  True depth ≈ water_level directly. Verify in WSC station metadata.")

# =============================================================================
# R² FOR OPEN WATER, ICE, AND ALL DATA
# =============================================================================
def r2_score(obs, pred):
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    return 1.0 - ss_res / ss_tot

q_pred_open = rating_curve(df_open["water_level"], a, b, c)
q_pred_ice  = rating_curve(df_ice["water_level"],  a, b, c)
q_pred_all  = rating_curve(df["water_level"],       a, b, c)

r2_open = r2_score(df_open["flow"].values, q_pred_open)
r2_ice  = r2_score(df_ice["flow"].values,  q_pred_ice)
r2_all  = r2_score(df["flow"].values,      q_pred_all)

# R2 computed in log space too, since that's the space we actually fit in --
# this is the more honest "goodness of fit" number for the log-space method.
r2_open_log = r2_score(np.log(df_open["flow"].values), np.log(np.maximum(q_pred_open, 1e-9)))

df_ice = df_ice.copy()
df_ice["q_residual"] = df_ice["flow"].values - q_pred_ice

# =============================================================================
# PRINT RESULTS
# =============================================================================
print(f"\n{'='*56}")
print(f"  RATING CURVE PARAMETERS  (open water, May–Oct, log-space fit)")
print(f"{'='*56}")
print(f"  Q = ({a:.4f} ± {perr[0]:.4f})")
print(f"      × (h - ({b:.4f} ± {perr[1]:.4f}))")
print(f"      ^ ({c:.4f} ± {perr[2]:.4f})")
print(f"  R²  open water  (linear Q space)      : {r2_open:.4f}")
print(f"  R²  open water  (log Q space, the      ")
print(f"      space actually being fit)          : {r2_open_log:.4f}")
print(f"  R²  ice-affected : {r2_ice:.4f}  "
      f"(drop = {r2_open - r2_ice:.4f}  — backwater signal)")
print(f"  R²  all data     : {r2_all:.4f}")
print(f"{'='*56}")

print(f"\n  Gauge datum diagnostics:")
print(f"  h_min (lowest recorded)    : {h_min:.4f} m")
print(f"  b (fitted zero-flow stage) : {b:.4f} m")
print(f"  b / h_min ratio            : {b/h_min:.4f}  "
      f"({'near-bed datum' if b_collapsed else 'meaningful offset'})")

if GAUGE_TO_CGVD2013 is not None:
    print(f"\n  Bed elevation CGVD2013 : {b + GAUGE_TO_CGVD2013:.4f} m a.s.l.")
if GAUGE_TO_CGVD28 is not None:
    print(f"  Bed elevation CGVD28   : {b + GAUGE_TO_CGVD28:.4f} m a.s.l.")
if GAUGE_TO_CGVD2013 is None:
    print(f"\n  NOTE: Set GAUGE_TO_CGVD2013 / GAUGE_TO_CGVD28 at top of script")
    print(f"        for station {STATION} to get absolute bed elevation.")

# =============================================================================
# DEPTH SENSITIVITY TO b's UNCERTAINTY
#
# Since depth = water_level - b, any error in b shifts every depth value
# by the same amount. This shows how much the reported depth range would
# move if the true b were at the edges of its +/- uncertainty band, which
# is a more direct "is this good enough" check than R2 for your actual
# use case (absolute depth, not discharge).
# =============================================================================
b_lo = b - perr[1]
b_hi = b + perr[1]
depth_open_center = df_open["water_level"] - b
depth_open_lo      = df_open["water_level"] - b_hi   # larger b -> smaller depth
depth_open_hi      = df_open["water_level"] - b_lo   # smaller b -> larger depth

print(f"\n  Depth sensitivity to b uncertainty (open water season):")
print(f"    b - err = {b_lo:.4f} m  -> depth range {depth_open_hi.min():.3f} – {depth_open_hi.max():.3f} m")
print(f"    b       = {b:.4f} m  -> depth range {depth_open_center.min():.3f} – {depth_open_center.max():.3f} m")
print(f"    b + err = {b_hi:.4f} m  -> depth range {depth_open_lo.min():.3f} – {depth_open_lo.max():.3f} m")
print(f"    Mean depth shifts by ~{perr[1]:.3f} m across this band")
print(f"    (i.e. every depth value has at least this much systematic")
print(f"     uncertainty on top of any curve-fit scatter)")

# =============================================================================
# HYSTERESIS / LOOPING-CURVE CHECK
#
# If the same stage produces different discharge on rising vs. falling
# limbs (common in unsteady natural-channel flow), a single power-law
# curve can't represent both well no matter how it's fit. This flags
# whether that's likely happening here.
# =============================================================================
df_open_sorted = df_open.sort_values("date").copy()
df_open_sorted["stage_diff"] = df_open_sorted["water_level"].diff()
rising  = df_open_sorted[df_open_sorted["stage_diff"] > 0]
falling = df_open_sorted[df_open_sorted["stage_diff"] < 0]

if len(rising) > 20 and len(falling) > 20:
    q_pred_rising  = rating_curve(rising["water_level"],  a, b, c)
    q_pred_falling = rating_curve(falling["water_level"], a, b, c)
    bias_rising  = (rising["flow"].values  - q_pred_rising).mean()
    bias_falling = (falling["flow"].values - q_pred_falling).mean()
    print(f"\n  Hysteresis check:")
    print(f"    Mean residual on rising limb  : {bias_rising:+.2f} m3/s")
    print(f"    Mean residual on falling limb : {bias_falling:+.2f} m3/s")
    if abs(bias_rising - bias_falling) > 0.1 * np.mean(np.abs(df_open["flow"])):
        print("    NOTE: rising/falling residuals differ notably -- possible")
        print("          looping rating curve (unsteady-flow hysteresis).")
        print("          A single static power-law curve may be systematically")
        print("          biased depending on flood stage/timing.")

# =============================================================================
# TRUE DEPTH
# =============================================================================
for frame in [df, df_open, df_ice]:
    frame["true_depth"] = frame["water_level"] - b

print(f"\n--- True Depth Statistics (open water season) ---")
print(f"  Mean   : {df_open['true_depth'].mean():.3f} m")
print(f"  Median : {df_open['true_depth'].median():.3f} m")
print(f"  Min    : {df_open['true_depth'].min():.3f} m")
print(f"  Max    : {df_open['true_depth'].max():.3f} m")
if b_collapsed:
    print("  NOTE: depth ≈ water level because b ≈ 0 (near-bed gauge datum).")

# =============================================================================
# EXPORT 1 — Full dataset (all seasons)
# =============================================================================
df.to_csv(OUT_PATH + "/hydat_levels_flows_depth.csv", index=False)
print(f"\nFull data saved → {OUT_PATH}/hydat_levels_flows_depth.csv")

# =============================================================================
# EXPORT 2 — Open-water season only, formatted for WASP calibration
# =============================================================================
# Add WASP-relevant hydraulic geometry columns
df_open = df_open.copy()
df_open["year"]          = df_open["date"].dt.year
df_open["month"]         = df_open["date"].dt.month
df_open["day"]           = df_open["date"].dt.day
df_open["station"]       = STATION
df_open["b_zero_flow_m"] = b          # bed elevation reference
df_open["b_uncertainty_m"] = perr[1]  # +/- uncertainty on b (see sensitivity note above)
df_open["a_coeff"]       = a          # rating curve coefficient
df_open["c_exponent"]    = c          # rating curve exponent
df_open["q_predicted"]   = q_pred_open          # curve-predicted discharge
df_open["q_residual"]    = df_open["flow"] - q_pred_open  # observed - predicted
df_open["r2_open_water"] = r2_open    # fit quality reference (linear Q space)
df_open["r2_open_water_log"] = r2_open_log  # fit quality reference (log Q space, actual fit space)

# Select and order columns clearly for WASP calibration use
wasp_cols = [
    "station",        # station ID
    "date",           # calendar date
    "year",           # year
    "month",          # month (5–10 only)
    "day",            # day of month
    "water_level",    # raw gauge reading (m)
    "true_depth",     # estimated hydraulic mean depth (m) = water_level - b
    "flow",           # observed discharge (m³/s)
    "q_predicted",    # rating curve predicted discharge (m³/s)
    "q_residual",     # flow residual — how well curve fits each day (m³/s)
    "b_zero_flow_m",  # zero-flow stage b (m) — bed elevation reference
    "b_uncertainty_m",# +/- uncertainty on b (m) -- propagates directly to true_depth
    "a_coeff",        # rating curve a parameter
    "c_exponent",     # rating curve c parameter
    "r2_open_water",  # open-water R² for reference (linear Q space)
    "r2_open_water_log", # open-water R² for reference (log Q space, actual fit space)
]

df_wasp = df_open[wasp_cols].reset_index(drop=True)

wasp_path = OUT_PATH + f"/{STATION}_openwater_depth_wasp_calibration.csv"
df_wasp.to_csv(wasp_path, index=False)

print(f"WASP calibration file saved → {wasp_path}")
print(f"\n--- WASP Calibration File Summary ---")
print(f"  Station          : {STATION}")
print(f"  Records          : {len(df_wasp)} daily open-water records")
print(f"  Date range       : {df_wasp['date'].min().date()} "
      f"to {df_wasp['date'].max().date()}")
print(f"  Months included  : May–October only (months 5–10)")
print(f"  Depth range      : {df_wasp['true_depth'].min():.3f} – "
      f"{df_wasp['true_depth'].max():.3f} m")
print(f"  Flow range       : {df_wasp['flow'].min():.1f} – "
      f"{df_wasp['flow'].max():.1f} m³/s")
print(f"  Rating curve     : Q = {a:.4f} × (h − {b:.4f})^{c:.4f}")
print(f"  R² (open water, log-space, actual fit) : {r2_open_log:.4f}")
print(f"  R² (open water, linear Q space)        : {r2_open:.4f}")
print(f"  b (zero-flow)    : {b:.4f} ± {perr[1]:.4f} m")
print(f"\n  WASP hydraulic geometry parameters:")
print(f"  Depth exponent β = 1/c = {1/c:.4f}  (use in WASP HydroGeom table)")
print(f"  Depth coeff    α = a^(-1/c) = {a**(-1/c):.4f}")
print(f"  → WASP depth formula: depth = {a**(-1/c):.4f} × Q^{1/c:.4f}")

# =============================================================================
# PLOTS  (3 rows × 2 cols)
# =============================================================================
fig, axes = plt.subplots(3, 2, figsize=(14, 15))
fig.suptitle(f"Rating Curve Analysis — Station {STATION}",
             fontsize=13, fontweight="bold")

h_range    = np.linspace(df["water_level"].min(), df["water_level"].max(), 400)
h_eff_open = np.maximum(df_open["water_level"] - b, 1e-6)
h_eff_ice  = np.maximum(df_ice["water_level"]  - b, 1e-6)
h_eff_rng  = np.maximum(h_range - b, 1e-6)

# ── Plot 1: Rating curve linear scale ────────────────────────────────────────
ax = axes[0, 0]
ax.scatter(df_ice["water_level"],  df_ice["flow"],  s=5, alpha=0.25,
           color="royalblue", label=f"Ice-affected (Nov–Apr)  R²={r2_ice:.3f}")
ax.scatter(df_open["water_level"], df_open["flow"], s=5, alpha=0.25,
           color="tomato",    label=f"Open water (May–Oct)  R²={r2_open:.3f}")
ax.plot(h_range, rating_curve(h_range, a, b, c),
        color="black", linewidth=2,
        label=f"Q={a:.2f}·(h-{b:.3f})^{c:.3f}  (log-space fit)")
ax.axvline(b, color="sienna", linestyle="--", linewidth=1.2,
           label=f"b = {b:.4f} m")
ax.set_xlabel("Water Level (m, gauge datum)")
ax.set_ylabel("Discharge (m³/s)")
ax.set_title("Rating Curve — Linear Scale")
ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.4)

# ── Plot 2: Rating curve log–log scale ───────────────────────────────────────
ax = axes[0, 1]
ax.scatter(h_eff_ice,  df_ice["flow"],  s=5, alpha=0.25, color="royalblue",
           label="Ice-affected")
ax.scatter(h_eff_open, df_open["flow"], s=5, alpha=0.25, color="tomato",
           label="Open water")
ax.plot(h_eff_rng, rating_curve(h_range, a, b, c),
        color="black", linewidth=2, label=f"slope c = {c:.3f}")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("(h − b)  effective depth (m, log)")
ax.set_ylabel("Discharge (m³/s, log)")
ax.set_title("Rating Curve — Log–Log Scale\n"
             "(straight line = valid power law; a visible kink suggests\n"
             "a compound channel that a single curve can't capture)")
ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.4, which="both")

# ── Plot 3: True depth time series ───────────────────────────────────────────
ax = axes[1, 0]
ax.plot(df_ice["date"],  df_ice["true_depth"],  color="royalblue",
        linewidth=0.7, alpha=0.7, label="Ice-affected (Nov–Apr)")
ax.plot(df_open["date"], df_open["true_depth"], color="tomato",
        linewidth=0.7, alpha=0.7, label="Open water (May–Oct)")
ax.axhline(df_open["true_depth"].mean(), color="black", linestyle="--",
           linewidth=1.2,
           label=f"Open water mean = {df_open['true_depth'].mean():.2f} m")
ax.set_xlabel("Date"); ax.set_ylabel("Estimated True Depth (m)")
ax.set_title(f"River Depth Time Series  (b = {b:.4f} ± {perr[1]:.4f} m)")
ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.4)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

# ── Plot 4: Ice-season flow residuals (backwater signal) ─────────────────────
ax = axes[1, 1]
ax.scatter(df_ice["date"], df_ice["q_residual"],
           s=5, alpha=0.4, color="royalblue")
ax.axhline(0, color="black", linewidth=1.2)
ax.axhline(df_ice["q_residual"].mean(), color="darkblue", linestyle="--",
           linewidth=1.5,
           label=f"Mean residual = {df_ice['q_residual'].mean():.1f} m³/s")
ax.fill_between(df_ice["date"], df_ice["q_residual"], 0,
                where=df_ice["q_residual"] < 0,
                color="royalblue", alpha=0.15, label="Flow suppressed by ice")
ax.fill_between(df_ice["date"], df_ice["q_residual"], 0,
                where=df_ice["q_residual"] > 0,
                color="tomato",    alpha=0.15, label="Flow above rating curve")
ax.set_xlabel("Date")
ax.set_ylabel("Q observed − Q predicted (m³/s)")
ax.set_title("Ice-Season Flow Residuals\n"
             "(negative = backwater suppresses discharge)")
ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.4)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

# ── Plot 5: Depth distribution histogram ─────────────────────────────────────
ax = axes[2, 0]
ax.hist(df_open["true_depth"], bins=40, color="tomato",    alpha=0.6,
        edgecolor="white", linewidth=0.5, label="Open water (May–Oct)")
ax.hist(df_ice["true_depth"],  bins=40, color="royalblue", alpha=0.6,
        edgecolor="white", linewidth=0.5, label="Ice-affected (Nov–Apr)")
ax.axvline(df_open["true_depth"].mean(), color="darkred",  linestyle="--",
           linewidth=1.5,
           label=f"Open mean  = {df_open['true_depth'].mean():.2f} m")
ax.axvline(df_ice["true_depth"].mean(),  color="darkblue", linestyle="--",
           linewidth=1.5,
           label=f"Ice mean   = {df_ice['true_depth'].mean():.2f} m")
ax.set_xlabel("Estimated True Depth (m)"); ax.set_ylabel("Frequency")
ax.set_title("Depth Distribution: Open Water vs Ice-Affected")
ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.4)

# ── Plot 6: Monthly depth boxplot ────────────────────────────────────────────
ax = axes[2, 1]
month_names  = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
monthly_data = [df[df["month"] == m]["true_depth"].values for m in range(1, 13)]
bp = ax.boxplot(monthly_data, patch_artist=True,
                medianprops={"color": "black", "linewidth": 1.5},
                flierprops={"marker": "o", "markersize": 2, "alpha": 0.3})
for i, patch in enumerate(bp["boxes"]):
    patch.set_facecolor("tomato" if 5 <= i + 1 <= 10 else "royalblue")
    patch.set_alpha(0.6)
ax.set_xticklabels(month_names)
ax.set_xlabel("Month"); ax.set_ylabel("Estimated True Depth (m)")
ax.set_title("Monthly Depth Distribution\n"
             "(red = open water, blue = ice-affected)")
ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(OUT_PATH + "/rating_curve_fit.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved.")