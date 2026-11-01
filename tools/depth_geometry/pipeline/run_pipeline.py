"""
run_pipeline.py

Orchestrator for the river-depth pipeline. Reads control_file.py, and for
each station:
    1. Runs Part 1 (fit_rating_curve) -- observed depth from HYDAT level/flow
    2. Runs Part 2 (compare_depth_models) -- depth from simulated discharge,
       automatically using Part 1's fitted rating curve (a, b, c) instead of
       requiring you to hand-copy HYDAT_ALPHA / HYDAT_BETA between scripts.
    3. Collects both parts' results into one row of a combined summary table.

After all stations finish, writes a single combined_summary.csv (one row
per station, both models' key metrics side by side) plus a plain-text
report, and prints a final table to the console.

Run with:
    python run_pipeline.py
"""

import sys
import traceback
from pathlib import Path
from datetime import datetime

import pandas as pd

from control_file import DEFAULTS, STATIONS
from part1_rating_curve import fit_rating_curve
from part2_depth_compare import compare_depth_models
from part3_channel_geometry import export_channel_geometry


def merged_cfg(station_entry: dict) -> dict:
    """Station-specific keys override DEFAULTS."""
    return {**DEFAULTS, **station_entry}


def flatten_result(rc: dict, cmp: dict, geo: dict) -> dict:
    """Flatten Part 1 + Part 2 + Part 3 results into one summary row per station."""
    row = {
        "station": rc["station"],
        "target_seg": cmp["target_seg"],
        # --- Part 1: rating curve / observed depth ---
        "rating_a": rc["a"], "rating_b": rc["b"], "rating_c": rc["c"],
        "rating_b_err": rc["b_err"],
        "r2_open_water": rc["r2_open"],
        "r2_open_water_log": rc["r2_open_log"],
        "r2_ice_affected": rc["r2_ice"],
        "n_open_water_days": rc["n_open"],
        "obs_depth_mean_m": rc["depth_mean"],
        "obs_depth_median_m": rc["depth_median"],
        "obs_depth_min_m": rc["depth_min"],
        "obs_depth_max_m": rc["depth_max"],
        "b_collapsed": rc["b_collapsed"],
        # --- Part 2: depth-from-simulation comparison ---
        "n_paired_open_water": cmp["n_paired_open_water"],
        "n_calibration": cmp["n_calibration"],
        "extreme_q_cluster_count": cmp["extreme_q_cluster_count"],
    }

    # Flatten both models' full-period metrics side by side
    for model_key, model_result in cmp["models"].items():
        prefix = model_key  # "A_manning" / "B_rating"
        for pname, pval in model_result["params"].items():
            row[f"{prefix}_{pname}"] = pval
        for mname, mval in model_result["metrics_full"].items():
            row[f"{prefix}_full_{mname}"] = mval
        for mname, mval in model_result["metrics_filtered"].items():
            row[f"{prefix}_filt_{mname}"] = mval

    # --- Part 3: channel geometry export (all segments) ---
    row["n_segments_exported"] = geo["n_segments"]
    row["n_outlier_segments"] = geo["n_outlier_segments"]
    row["geometry_nc_path"] = geo["geometry_nc_path"]
    row["geometry_csv_path"] = geo["geometry_csv_path"]
    if geo["validation"]:
        for mname, mval in geo["validation"].items():
            row[f"geometry_validation_{mname}"] = mval

    return row


def run_one_station(station_entry: dict) -> dict:
    cfg = merged_cfg(station_entry)

    for required in ("station", "mizu_nc", "topo_nc", "target_seg"):
        if required not in cfg:
            raise KeyError(
                f"Station entry is missing required key '{required}': {station_entry}")

    # --- Per-station output folder -------------------------------------
    # All of this station's outputs (Part 1/2/3 files) go into their own
    # subfolder, e.g. <out_path>/09CD001/, instead of being mixed together
    # in the top-level out_path. Only this station's cfg dict is touched --
    # merged_cfg() returns a fresh dict each call, so DEFAULTS["out_path"]
    # (used later for the combined summary across all stations) is
    # untouched.
    station_out_path = Path(cfg["out_path"]) / str(cfg["station"])
    station_out_path.mkdir(parents=True, exist_ok=True)
    cfg["out_path"] = str(station_out_path)
    print(f"  Output folder: {station_out_path}")

    # --- Part 1 ---
    rc = fit_rating_curve(cfg)

    # --- Part 2, chained automatically off Part 1's result ---
    # obs_csv and hydat_alpha/beta come from rc, not from cfg, so there is
    # nothing to keep in sync by hand between the two parts.
    cmp = compare_depth_models(cfg, rating_curve_result=rc)

    # --- Part 3, chained automatically off Part 1 + Part 2's results ---
    # alpha_b/beta_b/wscale_opt/width_exp_opt come from cmp (Part 2's
    # calibrated models), obs_csv (for validation) comes from rc -- neither
    # needs to be hand-copied between scripts.
    geo = export_channel_geometry(cfg, rating_curve_result=rc, depth_compare_result=cmp)

    return flatten_result(rc, cmp, geo)


def main():
    if not STATIONS:
        print("No stations configured in control_file.py -- nothing to do.")
        return

    print(f"River depth pipeline starting: {len(STATIONS)} station(s) to process.")
    print(f"Run started: {datetime.now().isoformat(timespec='seconds')}")

    summary_rows = []
    failures = []

    for i, station_entry in enumerate(STATIONS, start=1):
        station_id = station_entry.get("station", f"<station #{i}, no id given>")
        print(f"\n{'=' * 70}")
        print(f"STATION {i}/{len(STATIONS)}: {station_id}")
        print(f"{'=' * 70}")
        try:
            row = run_one_station(station_entry)
            summary_rows.append(row)
        except Exception as exc:  # noqa: BLE001 - keep batch running past one bad station
            print(f"\n[ERROR] Station {station_id} failed: {exc}")
            traceback.print_exc(file=sys.stdout)
            failures.append({"station": station_id, "error": str(exc)})

    # -------------------------------------------------------------------
    # COMBINED SUMMARY
    # -------------------------------------------------------------------
    out_path = Path(DEFAULTS["out_path"])
    out_path.mkdir(parents=True, exist_ok=True)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_csv_path = out_path / "combined_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)

        print(f"\n{'#' * 70}")
        print(f"# PIPELINE COMPLETE: {len(summary_rows)}/{len(STATIONS)} station(s) succeeded")
        print(f"{'#' * 70}")
        print(f"\nCombined summary saved -> {summary_csv_path}")

        # Compact console table: the metrics you'd actually scan first
        display_cols = [
            "station", "target_seg",
            "r2_open_water_log", "obs_depth_mean_m",
            "A_manning_full_R2", "A_manning_full_KGE", "A_manning_full_NSE",
            "B_rating_full_R2", "B_rating_full_KGE", "B_rating_full_NSE",
            "n_segments_exported", "geometry_validation_KGE",
        ]
        display_cols = [c for c in display_cols if c in summary_df.columns]
        print("\n--- Quick comparison across stations ---")
        print(summary_df[display_cols].to_string(index=False))

        # Plain-text report alongside the CSV, for quick reading without
        # opening a spreadsheet
        report_path = out_path / "combined_summary_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"River Depth Pipeline -- Combined Summary\n")
            f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"Stations succeeded: {len(summary_rows)}/{len(STATIONS)}\n")
            f.write("=" * 70 + "\n\n")
            for row in summary_rows:
                f.write(f"Station {row['station']}  (segment {row['target_seg']})\n")
                f.write(f"  Rating curve : Q = {row['rating_a']:.4f} x "
                        f"(h - {row['rating_b']:.4f})^{row['rating_c']:.4f}\n")
                f.write(f"  R2 open water (log-space fit) : {row['r2_open_water_log']:.4f}\n")
                f.write(f"  Observed depth mean/median    : "
                        f"{row['obs_depth_mean_m']:.3f} / {row['obs_depth_median_m']:.3f} m\n")
                f.write(f"  Model A (Manning)  R2={row.get('A_manning_full_R2', float('nan')):.3f}  "
                        f"KGE={row.get('A_manning_full_KGE', float('nan')):.3f}  "
                        f"NSE={row.get('A_manning_full_NSE', float('nan')):.3f}\n")
                f.write(f"  Model B (direct rating) R2={row.get('B_rating_full_R2', float('nan')):.3f}  "
                        f"KGE={row.get('B_rating_full_KGE', float('nan')):.3f}  "
                        f"NSE={row.get('B_rating_full_NSE', float('nan')):.3f}\n")
                f.write(f"  Channel geometry exported for {row.get('n_segments_exported', '?')} segments "
                        f"-> {row.get('geometry_nc_path', '?')}\n")
                if "geometry_validation_KGE" in row:
                    f.write(f"  Geometry validation at seg {row['target_seg']}: "
                            f"R2={row.get('geometry_validation_R2', float('nan')):.3f}  "
                            f"KGE={row.get('geometry_validation_KGE', float('nan')):.3f}  "
                            f"NSE={row.get('geometry_validation_NSE', float('nan')):.3f}\n")
                f.write("\n")
            if failures:
                f.write("-" * 70 + "\n")
                f.write(f"FAILED STATIONS ({len(failures)}):\n")
                for fail in failures:
                    f.write(f"  {fail['station']}: {fail['error']}\n")
        print(f"Text report saved -> {report_path}")

    if failures:
        print(f"\n[!] {len(failures)} station(s) failed -- see traceback(s) above.")
        for fail in failures:
            print(f"    {fail['station']}: {fail['error']}")

    if not summary_rows:
        print("\nNo stations completed successfully -- nothing to summarize.")
        sys.exit(1)


if __name__ == "__main__":
    main()