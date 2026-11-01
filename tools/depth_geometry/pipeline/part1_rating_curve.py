"""
part1_rating_curve.py

Fits a power-law rating curve  Q = a * (h - b)^c  to WSC/HYDAT daily level
and flow records for one station, using a log-space least-squares fit
(more appropriate for power laws than fitting in linear Q space, and gives
a better-constrained zero-flow stage b -- see notes inline).

This is a refactor of the original standalone curve-fitting-depth-estimate.py
/ curve-fit-river-depth-assess.py script into a single callable function,
fit_rating_curve(cfg), so it can be driven by a control file and chained
into part2_depth_compare.py automatically. All original console output,
diagnostics, exports, and plots are preserved -- only the entry point
changed (function call instead of top-level script execution).

Required cfg keys:
    station     WSC/HYDAT station number, e.g. "09CD001"
    hydat_path  path to local HYDAT sqlite3 file
    out_path    directory to write outputs into

Optional cfg keys:
    gauge_to_cgvd2013   float or None (default None)
    gauge_to_cgvd28     float or None (default None)
    make_plot           bool (default True)
    show_plot           bool (default False) -- set True only for interactive use

Returns a dict (see bottom of fit_rating_curve) with the fitted parameters,
uncertainties, R^2 values, depth statistics, and output file paths. This
dict is what part2_depth_compare.py consumes to avoid re-hardcoding
HYDAT_ALPHA / HYDAT_BETA.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# =============================================================================
# MODEL FUNCTIONS
# =============================================================================
def rating_curve(h, a, b, c):
    return a * np.power(np.maximum(h - b, 1e-6), c)


def rating_curve_log(h, a, b, c):
    return np.log(a) + c * np.log(np.maximum(h - b, 1e-6))


def r2_score(obs, pred):
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def fit_rating_curve(cfg: dict) -> dict:
    station = cfg["station"]
    hydat_path = cfg["hydat_path"]
    out_path = Path(cfg["out_path"])
    out_path.mkdir(parents=True, exist_ok=True)

    gauge_to_cgvd2013 = cfg.get("gauge_to_cgvd2013")
    gauge_to_cgvd28 = cfg.get("gauge_to_cgvd28")
    make_plot = cfg.get("make_plot", True)
    show_plot = cfg.get("show_plot", False)

    if make_plot and not show_plot:
        matplotlib.use("Agg")  # headless-safe; avoids blocking in batch runs

    print(f"\n{'#' * 70}")
    print(f"# PART 1: RATING CURVE / OBSERVED DEPTH  --  STATION {station}")
    print(f"{'#' * 70}")

    # -------------------------------------------------------------------
    # DATABASE CONNECTION
    # -------------------------------------------------------------------
    conn = sqlite3.connect(hydat_path)

    # -------------------------------------------------------------------
    # LOAD & UNPIVOT DLY_LEVELS (LEVEL1..LEVEL31 -> one row per day)
    # -------------------------------------------------------------------
    levels_raw = pd.read_sql(
        f"SELECT * FROM DLY_LEVELS WHERE STATION_NUMBER = '{station}'", conn)

    if levels_raw.empty:
        raise RuntimeError(f"No level data found for station {station} in DLY_LEVELS.")

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
    print(f"  Level range       : {levels['water_level'].min():.4f} - "
          f"{levels['water_level'].max():.4f} m")

    # -------------------------------------------------------------------
    # LOAD & UNPIVOT DLY_FLOWS (FLOW1..FLOW31 -> one row per day)
    # -------------------------------------------------------------------
    flows_raw = pd.read_sql(
        f"SELECT * FROM DLY_FLOWS WHERE STATION_NUMBER = '{station}'", conn)

    if flows_raw.empty:
        raise RuntimeError(f"No flow data found for station {station} in DLY_FLOWS.")

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
            f"flow_rows is empty for station {station}.\n"
            "Check: (1) all FLOW columns are NULL, "
            "(2) YEAR/MONTH not integers, "
            "(3) wrong station ID."
        )

    flows = pd.DataFrame(flow_rows).sort_values("date").reset_index(drop=True)
    print(f"Daily flows loaded  : {len(flows)} records "
          f"({flows['date'].min().date()} to {flows['date'].max().date()})")

    conn.close()

    # -------------------------------------------------------------------
    # MERGE & FILTER
    # -------------------------------------------------------------------
    df = pd.merge(levels, flows, on="date", how="inner")
    df = df[(df["water_level"] > 0) & (df["flow"] > 0)].reset_index(drop=True)
    print(f"Matched records     : {len(df)}")

    df["month"] = df["date"].dt.month
    df["season"] = df["month"].isin([5, 6, 7, 8, 9, 10])
    df_open = df[df["season"]].copy()
    df_ice = df[~df["season"]].copy()

    print(f"\nOpen water records  : {len(df_open)} (May-Oct)")
    print(f"Ice-affected records: {len(df_ice)} (Nov-Apr)")

    # -------------------------------------------------------------------
    # RATING CURVE FIT   Q = a * (h - b)^c    [open water only, log-space]
    # -------------------------------------------------------------------
    h_open = df_open["water_level"].values
    q_open = df_open["flow"].values
    h_min = h_open.min()

    b_upper = h_min * 0.9999
    b_init = h_min * 0.80

    p0 = [200.0, b_init, 1.8]
    bounds = ([1e-3, 0.0, 0.5],
              [1e7, b_upper, 5.0])

    # Linear-space fit (kept for comparison / backward compatibility)
    popt_lin, pcov_lin = curve_fit(rating_curve, h_open, q_open,
                                    p0=p0, bounds=bounds, maxfev=50000)

    # Log-space fit (primary fit used downstream)
    popt, pcov = curve_fit(rating_curve_log, h_open, np.log(q_open),
                            p0=p0, bounds=bounds, maxfev=50000)
    a, b, c = popt
    perr = np.sqrt(np.diag(pcov))

    a_lin, b_lin, c_lin = popt_lin
    perr_lin = np.sqrt(np.diag(pcov_lin))

    print(f"\n  Linear-space fit (previous method): a={a_lin:.4f}  b={b_lin:.4f} +/- {perr_lin[1]:.4f}  c={c_lin:.4f}")
    print(f"  Log-space fit    (this run's method): a={a:.4f}  b={b:.4f} +/- {perr[1]:.4f}  c={c:.4f}")
    print(f"  Shift in b from switching to log-space fit: {b - b_lin:+.4f} m")

    # Parameter correlation check
    corr = pcov / np.outer(perr, perr)
    corr_df = pd.DataFrame(corr, index=["a", "b", "c"], columns=["a", "b", "c"])
    print("\n  Parameter correlation matrix (log-space fit):")
    print(corr_df.round(3).to_string())
    if abs(corr_df.loc["b", "c"]) > 0.9:
        print("  NOTE: |corr(b,c)| > 0.9 -- b and c are strongly correlated.")
        print("        The printed +/- uncertainty on b likely understates how")
        print("        much b could shift under a different (still-plausible) fit.")

    # Detect b collapse
    b_collapsed = (b < 0.001 * h_min) if h_min > 0 else (b < 0.001)
    if b_collapsed:
        print("\n  WARNING: b collapsed to near-zero.")
        print("  This station's gauge datum is likely already referenced near the bed.")
        print("  True depth ~ water_level directly. Verify in WSC station metadata.")

    # -------------------------------------------------------------------
    # R^2 FOR OPEN WATER, ICE, AND ALL DATA
    # -------------------------------------------------------------------
    q_pred_open = rating_curve(df_open["water_level"], a, b, c)
    q_pred_ice = rating_curve(df_ice["water_level"], a, b, c)
    q_pred_all = rating_curve(df["water_level"], a, b, c)

    r2_open = r2_score(df_open["flow"].values, q_pred_open)
    r2_ice = r2_score(df_ice["flow"].values, q_pred_ice)
    r2_all = r2_score(df["flow"].values, q_pred_all)

    r2_open_log = r2_score(np.log(df_open["flow"].values), np.log(np.maximum(q_pred_open, 1e-9)))

    df_ice = df_ice.copy()
    df_ice["q_residual"] = df_ice["flow"].values - q_pred_ice

    # -------------------------------------------------------------------
    # PRINT RESULTS
    # -------------------------------------------------------------------
    print(f"\n{'=' * 56}")
    print(f"  RATING CURVE PARAMETERS  (open water, May-Oct, log-space fit)")
    print(f"{'=' * 56}")
    print(f"  Q = ({a:.4f} +/- {perr[0]:.4f})")
    print(f"      x (h - ({b:.4f} +/- {perr[1]:.4f}))")
    print(f"      ^ ({c:.4f} +/- {perr[2]:.4f})")
    print(f"  R2  open water (linear Q space)        : {r2_open:.4f}")
    print(f"  R2  open water (log Q space, actual fit): {r2_open_log:.4f}")
    print(f"  R2  ice-affected : {r2_ice:.4f}  (drop = {r2_open - r2_ice:.4f}  - backwater signal)")
    print(f"  R2  all data     : {r2_all:.4f}")
    print(f"{'=' * 56}")

    print(f"\n  Gauge datum diagnostics:")
    print(f"  h_min (lowest recorded)    : {h_min:.4f} m")
    print(f"  b (fitted zero-flow stage) : {b:.4f} m")
    print(f"  b / h_min ratio            : {b / h_min:.4f}  "
          f"({'near-bed datum' if b_collapsed else 'meaningful offset'})")

    bed_cgvd2013 = None
    bed_cgvd28 = None
    if gauge_to_cgvd2013 is not None:
        bed_cgvd2013 = b + gauge_to_cgvd2013
        print(f"\n  Bed elevation CGVD2013 : {bed_cgvd2013:.4f} m a.s.l.")
    if gauge_to_cgvd28 is not None:
        bed_cgvd28 = b + gauge_to_cgvd28
        print(f"  Bed elevation CGVD28   : {bed_cgvd28:.4f} m a.s.l.")
    if gauge_to_cgvd2013 is None:
        print(f"\n  NOTE: Set gauge_to_cgvd2013 / gauge_to_cgvd28 in the control file")
        print(f"        for station {station} to get absolute bed elevation.")

    # -------------------------------------------------------------------
    # DEPTH SENSITIVITY TO b's UNCERTAINTY
    # -------------------------------------------------------------------
    b_lo = b - perr[1]
    b_hi = b + perr[1]
    depth_open_center = df_open["water_level"] - b
    depth_open_lo = df_open["water_level"] - b_hi
    depth_open_hi = df_open["water_level"] - b_lo

    print(f"\n  Depth sensitivity to b uncertainty (open water season):")
    print(f"    b - err = {b_lo:.4f} m  -> depth range {depth_open_hi.min():.3f} - {depth_open_hi.max():.3f} m")
    print(f"    b       = {b:.4f} m  -> depth range {depth_open_center.min():.3f} - {depth_open_center.max():.3f} m")
    print(f"    b + err = {b_hi:.4f} m  -> depth range {depth_open_lo.min():.3f} - {depth_open_lo.max():.3f} m")
    print(f"    Mean depth shifts by ~{perr[1]:.3f} m across this band")

    # -------------------------------------------------------------------
    # HYSTERESIS / LOOPING-CURVE CHECK
    # -------------------------------------------------------------------
    df_open_sorted = df_open.sort_values("date").copy()
    df_open_sorted["stage_diff"] = df_open_sorted["water_level"].diff()
    rising = df_open_sorted[df_open_sorted["stage_diff"] > 0]
    falling = df_open_sorted[df_open_sorted["stage_diff"] < 0]

    if len(rising) > 20 and len(falling) > 20:
        q_pred_rising = rating_curve(rising["water_level"], a, b, c)
        q_pred_falling = rating_curve(falling["water_level"], a, b, c)
        bias_rising = (rising["flow"].values - q_pred_rising).mean()
        bias_falling = (falling["flow"].values - q_pred_falling).mean()
        print(f"\n  Hysteresis check:")
        print(f"    Mean residual on rising limb  : {bias_rising:+.2f} m3/s")
        print(f"    Mean residual on falling limb : {bias_falling:+.2f} m3/s")
        if abs(bias_rising - bias_falling) > 0.1 * np.mean(np.abs(df_open["flow"])):
            print("    NOTE: rising/falling residuals differ notably -- possible")
            print("          looping rating curve (unsteady-flow hysteresis).")

    # -------------------------------------------------------------------
    # TRUE DEPTH
    # -------------------------------------------------------------------
    for frame in [df, df_open, df_ice]:
        frame["true_depth"] = frame["water_level"] - b

    print(f"\n--- True Depth Statistics (open water season) ---")
    print(f"  Mean   : {df_open['true_depth'].mean():.3f} m")
    print(f"  Median : {df_open['true_depth'].median():.3f} m")
    print(f"  Min    : {df_open['true_depth'].min():.3f} m")
    print(f"  Max    : {df_open['true_depth'].max():.3f} m")
    if b_collapsed:
        print("  NOTE: depth ~ water level because b ~ 0 (near-bed gauge datum).")

    # -------------------------------------------------------------------
    # EXPORT 1 - Full dataset (all seasons)
    # -------------------------------------------------------------------
    full_csv_path = out_path / "hydat_levels_flows_depth.csv"
    df.to_csv(full_csv_path, index=False)
    print(f"\nFull data saved -> {full_csv_path}")

    # -------------------------------------------------------------------
    # EXPORT 2 - Open-water season only, formatted for WASP calibration
    # -------------------------------------------------------------------
    df_open = df_open.copy()
    df_open["year"] = df_open["date"].dt.year
    df_open["month"] = df_open["date"].dt.month
    df_open["day"] = df_open["date"].dt.day
    df_open["station"] = station
    df_open["b_zero_flow_m"] = b
    df_open["b_uncertainty_m"] = perr[1]
    df_open["a_coeff"] = a
    df_open["c_exponent"] = c
    df_open["q_predicted"] = q_pred_open
    df_open["q_residual"] = df_open["flow"] - q_pred_open
    df_open["r2_open_water"] = r2_open
    df_open["r2_open_water_log"] = r2_open_log

    wasp_cols = [
        "station", "date", "year", "month", "day",
        "water_level", "true_depth", "flow", "q_predicted", "q_residual",
        "b_zero_flow_m", "b_uncertainty_m", "a_coeff", "c_exponent",
        "r2_open_water", "r2_open_water_log",
    ]

    df_wasp = df_open[wasp_cols].reset_index(drop=True)

    wasp_csv_path = out_path / f"{station}_openwater_depth_wasp_calibration.csv"
    df_wasp.to_csv(wasp_csv_path, index=False)

    print(f"WASP calibration file saved -> {wasp_csv_path}")
    print(f"\n--- WASP Calibration File Summary ---")
    print(f"  Station         : {station}")
    print(f"  Records         : {len(df_wasp)} daily open-water records")
    print(f"  Date range      : {df_wasp['date'].min().date()} to {df_wasp['date'].max().date()}")
    print(f"  Months included : May-October only (months 5-10)")
    print(f"  Depth range     : {df_wasp['true_depth'].min():.3f} - {df_wasp['true_depth'].max():.3f} m")
    print(f"  Flow range      : {df_wasp['flow'].min():.1f} - {df_wasp['flow'].max():.1f} m3/s")
    print(f"  Rating curve    : Q = {a:.4f} x (h - {b:.4f})^{c:.4f}")
    print(f"  R2 (open water, log-space, actual fit) : {r2_open_log:.4f}")
    print(f"  R2 (open water, linear Q space)        : {r2_open:.4f}")
    print(f"  b (zero-flow)   : {b:.4f} +/- {perr[1]:.4f} m")
    print(f"\n  WASP hydraulic geometry parameters:")
    print(f"  Depth exponent beta = 1/c = {1 / c:.4f}  (use in WASP HydroGeom table)")
    print(f"  Depth coeff    alpha = a^(-1/c) = {a ** (-1 / c):.4f}")
    print(f"  -> WASP depth formula: depth = {a ** (-1 / c):.4f} x Q^{1 / c:.4f}")

    # -------------------------------------------------------------------
    # PLOTS (3 rows x 2 cols)
    # -------------------------------------------------------------------
    plot_path = None
    if make_plot:
        fig, axes = plt.subplots(3, 2, figsize=(14, 15))
        fig.suptitle(f"Rating Curve Analysis - Station {station}",
                     fontsize=13, fontweight="bold")

        h_range = np.linspace(df["water_level"].min(), df["water_level"].max(), 400)
        h_eff_open = np.maximum(df_open["water_level"] - b, 1e-6)
        h_eff_ice = np.maximum(df_ice["water_level"] - b, 1e-6)
        h_eff_rng = np.maximum(h_range - b, 1e-6)

        ax = axes[0, 0]
        ax.scatter(df_ice["water_level"], df_ice["flow"], s=5, alpha=0.25,
                   color="royalblue", label=f"Ice-affected (Nov-Apr)  R2={r2_ice:.3f}")
        ax.scatter(df_open["water_level"], df_open["flow"], s=5, alpha=0.25,
                   color="tomato", label=f"Open water (May-Oct)  R2={r2_open:.3f}")
        ax.plot(h_range, rating_curve(h_range, a, b, c),
                color="black", linewidth=2,
                label=f"Q={a:.2f}x(h-{b:.3f})^{c:.3f}  (log-space fit)")
        ax.axvline(b, color="sienna", linestyle="--", linewidth=1.2, label=f"b = {b:.4f} m")
        ax.set_xlabel("Water Level (m, gauge datum)")
        ax.set_ylabel("Discharge (m3/s)")
        ax.set_title("Rating Curve - Linear Scale")
        ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.4)

        ax = axes[0, 1]
        ax.scatter(h_eff_ice, df_ice["flow"], s=5, alpha=0.25, color="royalblue", label="Ice-affected")
        ax.scatter(h_eff_open, df_open["flow"], s=5, alpha=0.25, color="tomato", label="Open water")
        ax.plot(h_eff_rng, rating_curve(h_range, a, b, c),
                color="black", linewidth=2, label=f"slope c = {c:.3f}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("(h - b)  effective depth (m, log)")
        ax.set_ylabel("Discharge (m3/s, log)")
        ax.set_title("Rating Curve - Log-Log Scale")
        ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.4, which="both")

        ax = axes[1, 0]
        ax.plot(df_ice["date"], df_ice["true_depth"], color="royalblue",
                linewidth=0.7, alpha=0.7, label="Ice-affected (Nov-Apr)")
        ax.plot(df_open["date"], df_open["true_depth"], color="tomato",
                linewidth=0.7, alpha=0.7, label="Open water (May-Oct)")
        ax.axhline(df_open["true_depth"].mean(), color="black", linestyle="--", linewidth=1.2,
                   label=f"Open water mean = {df_open['true_depth'].mean():.2f} m")
        ax.set_xlabel("Date"); ax.set_ylabel("Estimated True Depth (m)")
        ax.set_title(f"River Depth Time Series  (b = {b:.4f} +/- {perr[1]:.4f} m)")
        ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.4)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

        ax = axes[1, 1]
        ax.scatter(df_ice["date"], df_ice["q_residual"], s=5, alpha=0.4, color="royalblue")
        ax.axhline(0, color="black", linewidth=1.2)
        ax.axhline(df_ice["q_residual"].mean(), color="darkblue", linestyle="--", linewidth=1.5,
                   label=f"Mean residual = {df_ice['q_residual'].mean():.1f} m3/s")
        ax.fill_between(df_ice["date"], df_ice["q_residual"], 0,
                        where=df_ice["q_residual"] < 0, color="royalblue", alpha=0.15,
                        label="Flow suppressed by ice")
        ax.fill_between(df_ice["date"], df_ice["q_residual"], 0,
                        where=df_ice["q_residual"] > 0, color="tomato", alpha=0.15,
                        label="Flow above rating curve")
        ax.set_xlabel("Date"); ax.set_ylabel("Q observed - Q predicted (m3/s)")
        ax.set_title("Ice-Season Flow Residuals")
        ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.4)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

        ax = axes[2, 0]
        ax.hist(df_open["true_depth"], bins=40, color="tomato", alpha=0.6,
               edgecolor="white", linewidth=0.5, label="Open water (May-Oct)")
        ax.hist(df_ice["true_depth"], bins=40, color="royalblue", alpha=0.6,
               edgecolor="white", linewidth=0.5, label="Ice-affected (Nov-Apr)")
        ax.axvline(df_open["true_depth"].mean(), color="darkred", linestyle="--", linewidth=1.5,
                   label=f"Open mean = {df_open['true_depth'].mean():.2f} m")
        ax.axvline(df_ice["true_depth"].mean(), color="darkblue", linestyle="--", linewidth=1.5,
                   label=f"Ice mean = {df_ice['true_depth'].mean():.2f} m")
        ax.set_xlabel("Estimated True Depth (m)"); ax.set_ylabel("Frequency")
        ax.set_title("Depth Distribution: Open Water vs Ice-Affected")
        ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.4)

        ax = axes[2, 1]
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly_data = [df[df["month"] == m]["true_depth"].values for m in range(1, 13)]
        bp = ax.boxplot(monthly_data, patch_artist=True,
                        medianprops={"color": "black", "linewidth": 1.5},
                        flierprops={"marker": "o", "markersize": 2, "alpha": 0.3})
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor("tomato" if 5 <= i + 1 <= 10 else "royalblue")
            patch.set_alpha(0.6)
        ax.set_xticklabels(month_names)
        ax.set_xlabel("Month"); ax.set_ylabel("Estimated True Depth (m)")
        ax.set_title("Monthly Depth Distribution\n(red = open water, blue = ice-affected)")
        ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        plot_path = out_path / "rating_curve_fit.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        if show_plot:
            plt.show()
        plt.close(fig)
        print(f"Plot saved -> {plot_path}")

    # -------------------------------------------------------------------
    # RESULT DICT -- consumed by part2_depth_compare.py
    # -------------------------------------------------------------------
    return {
        "station": station,
        "a": a, "b": b, "c": c,
        "a_err": perr[0], "b_err": perr[1], "c_err": perr[2],
        "b_collapsed": bool(b_collapsed),
        "r2_open": r2_open, "r2_open_log": r2_open_log,
        "r2_ice": r2_ice, "r2_all": r2_all,
        "n_open": len(df_open), "n_ice": len(df_ice),
        "depth_mean": float(df_open["true_depth"].mean()),
        "depth_median": float(df_open["true_depth"].median()),
        "depth_min": float(df_open["true_depth"].min()),
        "depth_max": float(df_open["true_depth"].max()),
        "bed_cgvd2013": bed_cgvd2013,
        "bed_cgvd28": bed_cgvd28,
        # WASP hydraulic geometry, derived here so part2 never has to
        # hardcode HYDAT_ALPHA / HYDAT_BETA again:
        #   h = alpha * Q^beta  <=>  Q = a*(h-b)^c  solved for h
        "hydat_alpha": a ** (-1 / c),
        "hydat_beta": 1 / c,
        "full_csv_path": str(full_csv_path),
        "wasp_csv_path": str(wasp_csv_path),
        "plot_path": str(plot_path) if plot_path else None,
    }


if __name__ == "__main__":
    # Standalone smoke-test entry point -- edit these to run this file alone
    # without going through the control file / orchestrator.
    demo_cfg = {
        "station": "09CD001",
        "hydat_path": "<WORK_ROOT>/mizu-depth-output-codes/Hydat.sqlite3",
        "out_path": "<WORK_ROOT>/mizu-depth-output-codes",
        "gauge_to_cgvd2013": None,
        "gauge_to_cgvd28": None,
    }
    result = fit_rating_curve(demo_cfg)
    print("\nReturned result dict keys:", list(result.keys()))
