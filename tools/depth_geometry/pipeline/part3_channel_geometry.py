"""
part3_channel_geometry.py

Applies the Part 2 calibrated depth models across EVERY segment of the
river network (not just the gauged/calibration segment) to back-calculate
channel width, depth, and unit flow for the whole domain:

    - Reference segment (TARGET_SEG, gauged): Model B direct rating,
        h = ALPHA_B * Q^BETA_B  (the calibrated "B_rating" model from Part 2)
    - All other (ungauged) segments: Model A Manning + width law,
        w = wscale * Q^WIDTH_EXP,  h = (Q*n / (w*sqrt(S)))^0.6
        (the calibrated "A_manning" model from Part 2 -- the only
        transferable option at segments with no observed depth to
        calibrate against directly)

A per-segment "depth_method" flag records which branch was used.

This is a refactor of the original standalone export_all_segments_v5.py
into a single callable function, export_channel_geometry(cfg,
rating_curve_result, depth_compare_result), so it can be driven by the
control file and chained directly after part1_rating_curve.py and
part2_depth_compare.py.

KEY CHANGE vs the original script: ALPHA_B / BETA_B / WSCALE_OPT /
WIDTH_EXP_OPT are no longer hardcoded constants copied by hand from a
calibration run (the exact mismatch v5's own docstring describes fixing,
where stale numbers from a different gauge/segment were left in place).
When depth_compare_result (Part 2's return dict) is passed in, all four
are derived directly from it. Passing them directly in cfg is still
supported for standalone use without running Part 1/2 first.

Required cfg keys:
    mizu_nc, topo_nc, target_seg, mann_n, out_path   (same as Part 2)

Optional cfg keys (defaults match the original script):
    time_start, time_end        analysis window, default "2015-01-01"/"2024-12-31"
    alpha_b, beta_b              only used if depth_compare_result is None
    wscale_opt, width_exp_opt    only used if depth_compare_result is None
    obs_csv                      only used if rating_curve_result is None
                                 (needed for the validation step at target_seg)
    width_anchor_m               rescale exported width/unit_flow to a known
                                 real width at median Q; default None (no rescale)
    q_cap_pct                    per-segment Q cap percentile, default 99.0
    q_outlier_ratio              flag threshold for max(Q) > ratio*p99, default 10.0
    min_depth                    floor on exported depth (m), default 0.01
    ow_months                    open-water months for validation, default [5..10]
    make_plot, show_plot         default True / False

Returns a dict with export paths, per-segment summary stats, and the
observed-vs-simulated validation metrics at the reference segment -- this
feeds into the orchestrator's combined summary alongside Part 1/2's results.
"""

from pathlib import Path

import json
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import pearsonr


def nse(o, s):
    return 1.0 - np.sum((o - s) ** 2) / np.sum((o - np.mean(o)) ** 2)


def kge(o, s):
    r = pearsonr(o, s)[0]
    alpha = np.std(s, ddof=1) / np.std(o, ddof=1)
    beta = np.mean(s) / np.mean(o)
    return 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def export_channel_geometry(
    cfg: dict,
    rating_curve_result: dict = None,
    depth_compare_result: dict = None,
) -> dict:
    mizu_nc = Path(cfg["mizu_nc"])
    topo_nc = Path(cfg["topo_nc"])
    target_seg = cfg["target_seg"]
    mann_n = cfg["mann_n"]
    out_path = Path(cfg["out_path"])
    out_path.mkdir(parents=True, exist_ok=True)

    time_start = cfg.get("time_start", "2015-01-01")
    time_end = cfg.get("time_end", "2024-12-31")
    val_start, val_end = time_start, time_end

    width_anchor_m = cfg.get("width_anchor_m")
    q_cap_pct = cfg.get("q_cap_pct", cfg.get("q_cal_cap_pct", 99.0))
    q_outlier_ratio = cfg.get("q_outlier_ratio", 10.0)
    min_depth = cfg.get("min_depth", 0.01)
    ow_months = cfg.get("ow_months", [5, 6, 7, 8, 9, 10])
    make_plot = cfg.get("make_plot", True)
    show_plot = cfg.get("show_plot", False)

    if make_plot and not show_plot:
        matplotlib.use("Agg")

    # -------------------------------------------------------------------
    # Resolve calibrated parameters -- from Part 2's result when given,
    # otherwise from cfg (standalone use), otherwise error out clearly.
    # -------------------------------------------------------------------
    if depth_compare_result is not None:
        b_params = depth_compare_result["models"]["B_rating"]["params"]
        a_params = depth_compare_result["models"]["A_manning"]["params"]
        alpha_b = 10.0 ** b_params["log10_alpha"]
        beta_b = b_params["beta"]
        wscale_opt = a_params["wscale"]
        width_exp_opt = a_params["WIDTH_EXP"]
        station_label = depth_compare_result["station"]
    else:
        # Standalone use (no depth_compare_result passed in, e.g. running
        # this file directly rather than through run_pipeline.py). Rather
        # than requiring alpha_b/beta_b/wscale_opt/width_exp_opt to be
        # hardcoded in cfg (the exact stale-provenance bug this project's
        # v5 script docstring described fixing once already), first try to
        # auto-load them from the JSON file Part 2 saves after every run
        # ({station}_calibrated_models.json in out_path). Explicit cfg
        # values still win if you deliberately want to override; a JSON
        # path can also be given directly via cfg["model_params_json"].
        alpha_b = cfg.get("alpha_b")
        beta_b = cfg.get("beta_b")
        wscale_opt = cfg.get("wscale_opt")
        width_exp_opt = cfg.get("width_exp_opt")
        station_label = cfg.get("station", Path(cfg.get("obs_csv", "unknown")).stem)

        need_lookup = None in (alpha_b, beta_b, wscale_opt, width_exp_opt)
        if need_lookup:
            model_params_json = cfg.get("model_params_json")
            if model_params_json is None:
                guess = Path(cfg["out_path"]) / f"{station_label}_calibrated_models.json"
                if guess.exists():
                    model_params_json = guess
            if model_params_json and Path(model_params_json).exists():
                with open(model_params_json, "r", encoding="utf-8") as f:
                    mp = json.load(f)
                alpha_b = alpha_b if alpha_b is not None else mp["alpha_b"]
                beta_b = beta_b if beta_b is not None else mp["beta_b"]
                wscale_opt = wscale_opt if wscale_opt is not None else mp["wscale_opt"]
                width_exp_opt = width_exp_opt if width_exp_opt is not None else mp["width_exp_opt"]
                print(f"  (Auto-loaded calibrated parameters from {model_params_json} "
                      f"-- no depth_compare_result or hardcoded values needed.)")

        if None in (alpha_b, beta_b, wscale_opt, width_exp_opt):
            raise ValueError(
                "export_channel_geometry needs calibrated parameters, from one of:\n"
                "  (a) depth_compare_result (Part 2's return value, when chained "
                "via run_pipeline.py),\n"
                "  (b) a '{station}_calibrated_models.json' file in out_path "
                "(saved automatically by part2_depth_compare.py), or\n"
                "  (c) alpha_b/beta_b/wscale_opt/width_exp_opt set directly in cfg.\n"
                "None of these were available."
            )

    if rating_curve_result is not None:
        obs_csv = Path(rating_curve_result["wasp_csv_path"])
    else:
        obs_csv = Path(cfg["obs_csv"])

    out_nc = out_path / f"channel_geometry_{station_label}.nc"
    out_csv = out_path / f"channel_geometry_{station_label}.csv"
    out_png = out_path / f"channel_geometry_{station_label}_overview.png"
    out_val = out_path / f"channel_geometry_seg{target_seg}_{station_label}_validation.png"

    print(f"\n{'#' * 70}")
    print(f"# PART 3: CHANNEL GEOMETRY - ALL SEGMENTS  --  "
          f"STATION {station_label}, REFERENCE SEG {target_seg}")
    print(f"{'#' * 70}")

    # -------------------------------------------------------------------
    # LOAD SIMULATED DATA
    # -------------------------------------------------------------------
    print("Loading simulated data ...")
    ds = xr.open_dataset(mizu_nc)
    # mizuRoute names the routed-flow variable differently by version/scheme
    q_var = next((v for v in ("KWroutedRunoff", "KWTroutedRunoff", "IRFroutedRunoff")
                  if v in ds), None)
    if q_var is None:
        raise KeyError(f"No routed-runoff variable (KWroutedRunoff / KWTroutedRunoff / "
                       f"IRFroutedRunoff) found in {mizu_nc}.")
    print(f"  Using routed-runoff variable: {q_var}")

    Q = ds[q_var].values.astype(np.float64)
    time = pd.DatetimeIndex(ds["time"].values)

    with nc.Dataset(topo_nc) as topo:
        slope = topo["slope"][:].astype(np.float64)
        seg_ids = topo["segId"][:].astype(np.int64)

    n_time, n_seg = Q.shape
    print(f"  Q shape    : {Q.shape}")
    print(f"  Time range : {time[0].date()} - {time[-1].date()}")
    print(f"  Segments   : {n_seg}")

    if target_seg not in seg_ids:
        raise ValueError(f"Segment {target_seg} not found in topology.")
    ref_idx = int(np.where(seg_ids == target_seg)[0][0])

    window_mask = (time >= pd.Timestamp(time_start)) & (time <= pd.Timestamp(time_end))
    n_excluded_spinup = int((~window_mask).sum())
    if n_excluded_spinup > 0:
        print(f"  (Note: {n_excluded_spinup} timesteps before {time_start} or "
              f"after {time_end} excluded as model warm-up before any "
              f"diagnostic, cap, or export below.)")

    # -------------------------------------------------------------------
    # REPORT APPLIED PARAMETERS
    # -------------------------------------------------------------------
    print("\nApplying calibrated parameters (no recalibration in this script):")
    print(f"  Depth, seg {target_seg} (Model B): h = {alpha_b:.6f} * Q^{beta_b:.6f}")
    print(f"  Depth, other segs (Model A)   : Manning n={mann_n}, "
          f"wscale={wscale_opt:.6f}, WIDTH_EXP={width_exp_opt:.6f}")
    print(f"  Q-cap = p{q_cap_pct:g} per segment (within {time_start} to {time_end})")
    print(f"  Width anchor: {width_anchor_m if width_anchor_m else 'none'}")
    print(f"  (Provenance: part2_depth_compare.py, station {station_label}, "
          f"segment {target_seg})")

    # -------------------------------------------------------------------
    # Q-OUTLIER DIAGNOSTIC PER SEGMENT (post-warm-up window only)
    # -------------------------------------------------------------------
    print("\nQ-outlier diagnostic per segment (post-warm-up window):")
    Q_safe = np.maximum(Q, 0.0)
    Q_safe_window = Q_safe[window_mask, :]
    p99_per_seg = np.percentile(Q_safe_window, 99, axis=0)
    qmax_per_seg = Q_safe_window.max(axis=0)
    ratio_per_seg = qmax_per_seg / np.maximum(p99_per_seg, 1e-6)
    outlier_mask = ratio_per_seg > q_outlier_ratio
    n_outlier = int(outlier_mask.sum())

    if n_outlier > 0:
        print(f"  [!] {n_outlier}/{n_seg} segments show suspicious Q peaks "
              f"(max > {q_outlier_ratio:.0f}x p99, within analysis window):")
        for i in np.where(outlier_mask)[0]:
            print(f"      seg {seg_ids[i]:>4}: p99={p99_per_seg[i]:>10.1f}  "
                  f"max={qmax_per_seg[i]:>10.1f}  ratio={ratio_per_seg[i]:>6.1f}x")
    else:
        print(f"  No segment has max(Q) > {q_outlier_ratio:.0f}x its p99 "
              "within the analysis window.")

    # -------------------------------------------------------------------
    # APPLY Q-CAP PER SEGMENT
    # -------------------------------------------------------------------
    q_cap_per_seg = np.percentile(Q_safe_window, q_cap_pct, axis=0)
    Q_capped = np.minimum(Q_safe, q_cap_per_seg[np.newaxis, :])
    n_capped_days = int((Q_safe_window > q_cap_per_seg[np.newaxis, :]).sum())
    print(f"\nQ capped at p{q_cap_pct:g} per segment (cap computed within "
          f"{time_start} to {time_end}; {n_capped_days} segment-timesteps "
          "affected in that window):")
    print(f"  Q_capped range (post-warm-up) : "
          f"{Q_capped[window_mask, :].min():.2f} - "
          f"{Q_capped[window_mask, :].max():.2f} m3/s")

    # -------------------------------------------------------------------
    # COMPUTE WIDTH, DEPTH, UNIT FLOW
    # -------------------------------------------------------------------
    print("\nCalculating geometry for all segments ...")

    slope_2d = np.where(slope > 1e-8, slope, 1e-8)[np.newaxis, :]

    width_manning = np.maximum(wscale_opt * (Q_capped ** width_exp_opt), 1e-10)

    depth = np.clip(
        (Q_capped * mann_n / (width_manning * np.sqrt(slope_2d))) ** 0.6,
        min_depth, 50.0)
    depth[:, ref_idx] = np.clip(
        alpha_b * (Q_capped[:, ref_idx] ** beta_b), min_depth, 50.0)

    depth_method = np.array(["manning_A"] * n_seg, dtype=object)
    depth_method[ref_idx] = "rating_B"

    if width_anchor_m is not None:
        q_med_ref = np.median(Q_capped[window_mask, ref_idx])
        w_med_model = wscale_opt * (q_med_ref ** width_exp_opt)
        width_scale = width_anchor_m / w_med_model
        print(f"  Width anchor: model width at median Q (seg {target_seg}) = "
              f"{w_med_model:.1f} m -> scaled by {width_scale:.3f} "
              f"to match {width_anchor_m} m")
    else:
        width_scale = 1.0

    width = width_manning * width_scale
    unit_flow = Q_capped / width

    print(f"  width     range (post-warm-up) : "
          f"{width[window_mask, :].min():.2f} - {width[window_mask, :].max():.2f} m")
    print(f"  depth     range (post-warm-up) : "
          f"{depth[window_mask, :].min():.3f} - {depth[window_mask, :].max():.3f} m")
    print(f"  unit_flow range (post-warm-up) : "
          f"{unit_flow[window_mask, :].min():.6f} - "
          f"{unit_flow[window_mask, :].max():.4f} m2/s")
    if depth[window_mask, :].max() > 10:
        print("  [!] depth.max() > 10 m within analysis window - check for "
              "residual Q spikes.")

    # -------------------------------------------------------------------
    # SAVE NETCDF
    # -------------------------------------------------------------------
    print(f"\nSaving NetCDF -> {out_nc}")
    ds_out = xr.Dataset(
        {
            "channel_width": (
                ["time", "seg"], width.astype(np.float32),
                {"units": "m", "long_name": "Channel width",
                 "equation": "wscale * Q_capped^WIDTH_EXP  (x width_scale)",
                 "wscale": wscale_opt, "width_exp": width_exp_opt,
                 "width_scale_applied": float(width_scale)}),
            "channel_depth": (
                ["time", "seg"], depth.astype(np.float32),
                {"units": "m", "long_name": "Channel flow depth",
                 "equation": (f"seg {target_seg}: alpha_b*Q_capped^beta_b "
                              "(rating, Model B); others: "
                              "(Q_capped*n/(width_manning*sqrt(slope)))^0.6"),
                 "alpha_b": alpha_b, "beta_b": beta_b,
                 "mann_n": mann_n, "min_depth": min_depth}),
            "unit_flow": (
                ["time", "seg"], unit_flow.astype(np.float32),
                {"units": "m2 s-1",
                 "long_name": "Unit flow (discharge per unit width)",
                 "equation": "Q_capped / width"}),
            "discharge": (
                ["time", "seg"], Q.astype(np.float32),
                {"units": "m3 s-1",
                 "long_name": "KWroutedRunoff - routed discharge (uncapped)"}),
            "discharge_capped": (
                ["time", "seg"], Q_capped.astype(np.float32),
                {"units": "m3 s-1",
                 "long_name": f"Routed discharge capped at p{q_cap_pct:g}/segment "
                              f"(cap computed within {time_start} to {time_end})"}),
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
            "description": ("Channel geometry, all segments -- mizuRoute KWT"),
            "reference_gauge": station_label,
            "calibration_segment": int(target_seg),
            "obs_reference": str(obs_csv),
            "calibration_source": ("part2_depth_compare.py: log-KGE objective, "
                                    f"DE + L-BFGS-B, Q<=p{q_cap_pct:g} cap, open-water."),
            "calibration_period": f"{val_start} to {val_end}, open-water",
            "note_q_outliers": (f"{n_outlier} of {n_seg} segments show "
                                 f"max(Q) > {q_outlier_ratio:.0f}x p99(Q) "
                                 f"within {time_start} to {time_end}. Q capped at "
                                 f"p{q_cap_pct:g} per segment "
                                 f"({n_capped_days} segment-timesteps affected)."),
            "q_cap_percentile": q_cap_pct,
            "min_depth_m": min_depth,
            "analysis_window": f"{time_start} to {time_end} (post-warm-up)",
        }
    )
    encoding = {v: {"zlib": True, "complevel": 4}
                for v in ["channel_width", "channel_depth", "unit_flow",
                          "discharge", "discharge_capped"]}
    ds_out.to_netcdf(out_nc, encoding=encoding)
    ds.close()
    print(f"  Saved: {out_nc}")

    # -------------------------------------------------------------------
    # PER-SEGMENT SUMMARY CSV
    # -------------------------------------------------------------------
    print(f"\nBuilding summary statistics -> {out_csv}")
    rows = []
    for i, sid in enumerate(seg_ids):
        d = depth[window_mask, i]
        w = width[window_mask, i]
        uf = unit_flow[window_mask, i]
        q = Q[window_mask, i]
        rows.append({
            "seg_id": int(sid),
            "depth_method": depth_method[i],
            "slope": round(float(slope[i]), 8),
            "mean_Q_m3s": round(float(q.mean()), 4),
            "p99_Q_m3s": round(float(np.percentile(q, 99)), 4),
            "max_Q_m3s": round(float(q.max()), 4),
            "q_cap_m3s": round(float(q_cap_per_seg[i]), 4),
            "mean_depth_m": round(float(d.mean()), 4),
            "min_depth_m": round(float(d.min()), 4),
            "max_depth_m": round(float(d.max()), 4),
            "p5_depth_m": round(float(np.percentile(d, 5)), 4),
            "p95_depth_m": round(float(np.percentile(d, 95)), 4),
            "mean_width_m": round(float(w.mean()), 4),
            "min_width_m": round(float(w.min()), 4),
            "max_width_m": round(float(w.max()), 4),
            "mean_unitflow_m2s": round(float(uf.mean()), 6),
            "p5_unitflow_m2s": round(float(np.percentile(uf, 5)), 6),
            "p95_unitflow_m2s": round(float(np.percentile(uf, 95)), 6),
        })

    df_stats = pd.DataFrame(rows).set_index("seg_id")
    df_stats.to_csv(out_csv)
    print(f"  Saved: {out_csv}")
    print("\n" + df_stats[["depth_method", "mean_depth_m", "max_depth_m",
                            "mean_width_m", "mean_unitflow_m2s"]].to_string())

    # -------------------------------------------------------------------
    # VALIDATION - OBSERVED vs SIMULATED (Model B) AT REFERENCE SEGMENT
    # -------------------------------------------------------------------
    print(f"\nLoading observed data for validation at seg {target_seg} ...")
    df_obs_raw = pd.read_csv(obs_csv)
    df_obs_raw["date"] = pd.to_datetime(df_obs_raw["date"], errors="coerce")
    df_obs_raw["true_depth"] = pd.to_numeric(df_obs_raw["true_depth"], errors="coerce")
    df_obs_raw = df_obs_raw.dropna(subset=["date", "true_depth"])

    df_obs = (df_obs_raw.set_index("date").sort_index().loc[val_start:val_end])
    df_obs = df_obs[df_obs["true_depth"] >= 0].rename(columns={"true_depth": "obs_depth"})

    sim_ref = (pd.Series(depth[:, ref_idx], index=time, name="sim_depth")
               .resample("D").mean().loc[val_start:val_end])
    sim_ref = sim_ref[sim_ref.index.month.isin(ow_months)]

    obs_daily = df_obs["obs_depth"].resample("D").mean()
    df_paired = pd.concat([sim_ref, obs_daily], axis=1, join="inner").dropna()
    df_paired.columns = ["sim_depth", "obs_depth"]
    n_paired = len(df_paired)
    print(f"  Paired open-water records : {n_paired} ({val_start} to {val_end})")

    validation = None
    if n_paired < 3:
        print("  [!] Too few paired obs/sim points -- skipping validation stats/plot.")
    else:
        obs = df_paired["obs_depth"].values
        sim = df_paired["sim_depth"].values

        validation = dict(
            n=n_paired,
            R2=float(pearsonr(obs, sim)[0] ** 2),
            NSE=float(nse(obs, sim)),
            KGE=float(kge(obs, sim)),
            RMSE=float(np.sqrt(np.mean((obs - sim) ** 2))),
            MAE=float(np.mean(np.abs(obs - sim))),
            PBIAS=float(100.0 * np.sum(obs - sim) / np.sum(obs)),
            mean_bias=float(np.mean(sim - obs)),
        )

        stats_lines = [
            f"n         = {validation['n']}",
            f"R2       = {validation['R2']:.3f}",
            f"NSE      = {validation['NSE']:.3f}",
            f"KGE      = {validation['KGE']:.3f}",
            f"RMSE     = {validation['RMSE']:.3f} m",
            f"MAE      = {validation['MAE']:.3f} m",
            f"PBIAS    = {validation['PBIAS']:+.1f} %",
            f"Mean bias = {validation['mean_bias']:+.3f} m",
        ]
        stats_text = "\n".join(stats_lines)

        print(f"\n{'-' * 44}")
        print(f"  Performance - Seg {target_seg} (Model B rating depth)")
        print(f"  Open-water, {val_start} to {val_end}")
        print(f"{'-' * 44}")
        for line in stats_lines:
            print(f"  {line}")
        print(f"{'-' * 44}")

        # ---------------------------------------------------------------
        # PLOT 1 - ALL-SEGMENT OVERVIEW
        # ---------------------------------------------------------------
        if make_plot:
            print(f"\nPlotting all-segment overview -> {out_png}")

            df_depth = (pd.DataFrame(depth, index=time, columns=seg_ids)
                        .resample("D").mean().loc[time_start:time_end])
            df_width = (pd.DataFrame(width, index=time, columns=seg_ids)
                        .resample("D").mean().loc[time_start:time_end])
            df_uf = (pd.DataFrame(unit_flow, index=time, columns=seg_ids)
                     .resample("D").mean().loc[time_start:time_end])

            fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)
            fig.suptitle(
                f"Channel Geometry - All {n_seg} Segments (hybrid depth)\n"
                f"seg {target_seg}: h={alpha_b:.4f}*Q^{beta_b:.4f} (rating)  |  "
                f"others: Manning n={mann_n}, wscale={wscale_opt:.2f}, "
                f"WIDTH_EXP={width_exp_opt:.3f}  |  Q capped at p{q_cap_pct:g}\n"
                f"Calibrated at station {station_label}, seg {target_seg}, "
                f"{time_start}-{time_end}",
                fontsize=11, fontweight="bold")

            cmap = plt.cm.viridis(np.linspace(0, 1, n_seg))

            for ax, df_v, ylab, title in [
                (axes[0], df_depth, "Depth (m)", "Daily Mean Depth per Segment "
                 f"(seg {target_seg} = rating model, highlighted)"),
                (axes[1], df_width, "Width (m)", "Daily Mean Width per Segment"),
                (axes[2], df_uf, "Unit Flow (m2/s)",
                 "Daily Mean Unit Flow per Segment (Q_capped / width)"),
            ]:
                for i, sid in enumerate(seg_ids):
                    if sid == target_seg and df_v is df_depth:
                        continue
                    ax.plot(df_v.index, df_v[sid], lw=0.7, color=cmap[i], alpha=0.6)
                if df_v is df_depth:
                    ax.plot(df_v.index, df_v[target_seg], lw=1.4, color="crimson",
                            label=f"seg {target_seg} (rating model)")
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
            plt.savefig(out_png, dpi=150, bbox_inches="tight")
            if show_plot:
                plt.show()
            plt.close(fig)
            print(f"  Saved: {out_png}")

            # -----------------------------------------------------------
            # PLOT 2 - VALIDATION AT REFERENCE SEGMENT
            # -----------------------------------------------------------
            print(f"\nPlotting validation -> {out_val}")

            fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                                      gridspec_kw={"height_ratios": [2, 1]})
            fig.suptitle(
                f"Observed vs Simulated River Depth - Segment {target_seg} "
                f"(Open Water)\n"
                f"Obs: HYDAT rating-curve-derived depth (station {station_label})  |  "
                f"Sim: rating model h = {alpha_b:.4f}*Q^{beta_b:.4f}, "
                f"Q capped at p{q_cap_pct:g}",
                fontsize=11, fontweight="bold")

            ax = axes[0]
            ax.plot(df_paired.index, df_paired["sim_depth"], color="steelblue",
                    lw=0.9, alpha=0.85, label="Simulated (rating model)")
            ax.plot(df_paired.index, df_paired["obs_depth"], color="tomato",
                    lw=0.9, alpha=0.85, label="Observed (HYDAT rating-based)")
            ax.set_ylabel("Depth (m)", fontsize=11)
            ax.set_title(f"Time Series - Segment {target_seg} "
                         f"({val_start[:4]}-{val_end[:4]}, open-water months)",
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
            ax.set_title(f"Scatter - n = {n_paired} paired daily open-water values",
                         fontsize=10)
            ax.set_xlim(lim_min, lim_max); ax.set_ylim(lim_min, lim_max)
            ax.set_aspect("equal", adjustable="box")
            ax.legend(fontsize=9, loc="upper left", framealpha=0.8)
            ax.grid(True, alpha=0.25)

            plt.tight_layout()
            plt.savefig(out_val, dpi=150, bbox_inches="tight")
            if show_plot:
                plt.show()
            plt.close(fig)
            print(f"  Saved: {out_val}")

    print("\nDone.")

    # -------------------------------------------------------------------
    # RESULT DICT -- consumed by the orchestrator's combined summary
    # -------------------------------------------------------------------
    return {
        "station": station_label,
        "target_seg": target_seg,
        "n_segments": n_seg,
        "n_outlier_segments": n_outlier,
        "alpha_b": alpha_b, "beta_b": beta_b,
        "wscale_opt": wscale_opt, "width_exp_opt": width_exp_opt,
        "geometry_nc_path": str(out_nc),
        "geometry_csv_path": str(out_csv),
        "overview_png_path": str(out_png) if make_plot else None,
        "validation_png_path": str(out_val) if (make_plot and validation) else None,
        "validation": validation,
    }


if __name__ == "__main__":
    # Standalone smoke-test entry point -- edit these to run this file alone
    # (without Part 1/2 / the orchestrator). alpha_b/beta_b/wscale_opt/
    # width_exp_opt are left unset here deliberately: export_channel_geometry()
    # will auto-load them from "{station}_calibrated_models.json" in out_path
    # (saved automatically by part2_depth_compare.py's last run for this
    # station), rather than requiring hardcoded numbers that can go stale.
    # Set them explicitly in this dict only if you want to override, or if
    # that JSON file doesn't exist yet (e.g. Part 2 has never been run).
    demo_cfg = {
        "station": "09CD001",
        "mizu_nc": "yukon-upper-kw.nc",
        "topo_nc": "topology.nc",
        "target_seg": 239,
        "mann_n": 0.025,
        "out_path": ".",
        "obs_csv": "09CD001_openwater_depth_wasp_calibration.csv",
    }
    result = export_channel_geometry(demo_cfg)
    print("\nReturned result dict keys:", list(result.keys()))
