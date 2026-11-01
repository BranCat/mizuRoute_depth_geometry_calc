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
# =============================================================================
def rating_curve(h, a, b, c):
    return a * np.power(np.maximum(h - b, 1e-6), c)

h_open  = df_open["water_level"].values
q_open  = df_open["flow"].values
h_min   = h_open.min()

b_upper = h_min * 0.9999
b_init  = h_min * 0.80

p0     = [200.0, b_init, 1.8]
bounds = ([1e-3, 0.0,    0.5],
          [1e7,  b_upper, 5.0])

popt, pcov = curve_fit(rating_curve, h_open, q_open,
                       p0=p0, bounds=bounds, maxfev=50000)
a, b, c = popt
perr    = np.sqrt(np.diag(pcov))

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

df_ice = df_ice.copy()
df_ice["q_residual"] = df_ice["flow"].values - q_pred_ice

# =============================================================================
# PRINT RESULTS
# =============================================================================
print(f"\n{'='*56}")
print(f"  RATING CURVE PARAMETERS  (open water, May–Oct)")
print(f"{'='*56}")
print(f"  Q = ({a:.4f} ± {perr[0]:.4f})")
print(f"      × (h - ({b:.4f} ± {perr[1]:.4f}))")
print(f"      ^ ({c:.4f} ± {perr[2]:.4f})")
print(f"  R²  open water   : {r2_open:.4f}")
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
df_open["a_coeff"]       = a          # rating curve coefficient
df_open["c_exponent"]    = c          # rating curve exponent
df_open["q_predicted"]   = q_pred_open          # curve-predicted discharge
df_open["q_residual"]    = df_open["flow"] - q_pred_open  # observed - predicted
df_open["r2_open_water"] = r2_open    # fit quality reference

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
    "a_coeff",        # rating curve a parameter
    "c_exponent",     # rating curve c parameter
    "r2_open_water",  # open-water R² for reference
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
print(f"  R² (open water)  : {r2_open:.4f}")
print(f"  b (zero-flow)    : {b:.4f} m")
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
        label=f"Q={a:.2f}·(h-{b:.3f})^{c:.3f}")
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
             "(straight line = valid power law)")
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
ax.set_title(f"River Depth Time Series  (b = {b:.4f} m)")
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
