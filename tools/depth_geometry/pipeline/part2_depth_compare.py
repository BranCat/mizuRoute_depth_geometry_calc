"""
part2_depth_compare.py

Calibrates two structural depth models against observed (HYDAT-derived)
depth, driven by simulated mizuRoute discharge at a target river segment:

    MODEL A: Manning + power-law width
        w = wscale * Q^WIDTH_EXP,  h = (Q*n / (w*sqrt(S)))^0.6

    MODEL B: direct at-a-station rating
        h = alpha * Q^beta,  with alpha, beta free (bounds configurable)

Both are calibrated with log-KGE on a Q-capped open-water subset, evaluated
on the same paired sets, and plotted side by side.

This is a refactor of the original standalone back-calculate-depth-
compare-v5.py into a single callable function, compare_depth_models(cfg,
rating_curve_result), so it can be driven by a control file and chained
directly after part1_rating_curve.py.

KEY CHANGE vs the original script: HYDAT_ALPHA / HYDAT_BETA are no longer
hardcoded constants that go stale when OBS_CSV points to a different
station. When rating_curve_result (Part 1's return dict) is passed in,
alpha/beta are derived directly from it (hydat_alpha / hydat_beta keys,
i.e. a^(-1/c) and 1/c from the log-space rating-curve fit). Passing
hydat_alpha / hydat_beta directly in cfg is still supported for standalone
use without running Part 1 first.

Required cfg keys:
    mizu_nc      path to mizuRoute output netCDF
    topo_nc      path to topology netCDF (must contain 'slope', 'segId')
    target_seg   river segment ID to extract and compare against obs
    mann_n       Manning's roughness coefficient for this segment
    out_path     directory to write outputs into
    obs_csv      path to Part 1's WASP calibration CSV (auto-filled by the
                 orchestrator from rating_curve_result if not given)

Optional cfg keys (all have defaults matching the original script):
    time_start, time_end            analysis window, default "2015-01-01"/"2024-12-31"
    hydat_alpha, hydat_beta         only used if rating_curve_result is None
    wscale_bounds, width_exp_bounds Model A bounds
    log10_alpha_bounds, beta_bounds Model B bounds
    log_eps                         epsilon for log-KGE, default 0.01
    q_cal_cap_pct                   calibration Q percentile cap, default 99.0
    de_seed, de_maxiter, de_popsize differential_evolution settings
    make_plot, show_plot            default True / False

Returns a dict with calibrated params and metrics for both models -- this
is what the orchestrator flattens into the combined multi-station summary.
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
from scipy.optimize import differential_evolution, minimize
from scipy import stats


# =============================================================================
# MODELS
# =============================================================================
def sim_depth_manning(params, Q_arr, n, S):
    wscale, wexp = params
    w = np.maximum(wscale * (Q_arr ** wexp), 1e-10)
    return np.clip((Q_arr * n / (w * np.sqrt(S))) ** 0.6, 0.0, 50.0)


def sim_depth_rating(params, Q_arr, n=None, S=None):
    log10_alpha, beta = params
    alpha = 10.0 ** log10_alpha
    return np.clip(alpha * np.power(np.maximum(Q_arr, 1e-10), beta), 0.0, 50.0)


# =============================================================================
# METRICS
# =============================================================================
def kge(obs, sim):
    if np.std(obs, ddof=1) == 0 or np.std(sim, ddof=1) == 0:
        return np.nan
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim, ddof=1) / np.std(obs, ddof=1)
    beta = np.mean(sim) / np.mean(obs)
    return 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def log_kge(obs, sim, eps):
    return kge(np.log(obs + eps), np.log(sim + eps))


def calc_metrics(obs, sim, log_eps):
    slope_lr, intercept, r, _, _ = stats.linregress(obs, sim)
    nse = 1.0 - np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)
    return dict(
        R2=r ** 2, NSE=nse, KGE=kge(obs, sim), logKGE=log_kge(obs, sim, log_eps),
        RMSE=np.sqrt(np.mean((sim - obs) ** 2)),
        MAE=np.mean(np.abs(sim - obs)),
        PBIAS=100.0 * np.sum(obs - sim) / np.sum(obs),
        slope_lr=slope_lr, intercept=intercept,
    )


def at_bound(v, lo, hi, tol=1e-3):
    return abs(v - lo) < tol or abs(v - hi) < tol


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def compare_depth_models(cfg: dict, rating_curve_result: dict = None) -> dict:
    mizu_nc = Path(cfg["mizu_nc"])
    topo_nc = Path(cfg["topo_nc"])
    target_seg = cfg["target_seg"]
    mann_n = cfg["mann_n"]
    out_path = Path(cfg["out_path"])
    out_path.mkdir(parents=True, exist_ok=True)

    time_start = cfg.get("time_start", "2015-01-01")
    time_end = cfg.get("time_end", "2024-12-31")

    wscale_bounds = cfg.get("wscale_bounds", (0.1, 500.0))
    width_exp_bounds = cfg.get("width_exp_bounds", (0.25, 0.70))
    log10_alpha_bounds = cfg.get("log10_alpha_bounds", (-5.0, 1.0))
    beta_bounds = cfg.get("beta_bounds", (0.2, 1.5))
    log_eps = cfg.get("log_eps", 0.01)
    q_cal_cap_pct = cfg.get("q_cal_cap_pct", 99.0)
    de_seed = cfg.get("de_seed", 42)
    de_maxiter = cfg.get("de_maxiter", 200)
    de_popsize = cfg.get("de_popsize", 20)
    make_plot = cfg.get("make_plot", True)
    show_plot = cfg.get("show_plot", False)

    if make_plot and not show_plot:
        matplotlib.use("Agg")

    # Reference depth curve for sanity-check printout only. Derived from
    # Part 1's fit when available -- this is what keeps it from going
    # stale when the pipeline is re-run for a different station.
    if rating_curve_result is not None:
        hydat_alpha = rating_curve_result["hydat_alpha"]
        hydat_beta = rating_curve_result["hydat_beta"]
        obs_csv = Path(rating_curve_result["wasp_csv_path"])
        station_label = rating_curve_result["station"]
    else:
        # Standalone use (no rating_curve_result passed in, e.g. running
        # this file directly rather than through run_pipeline.py). Rather
        # than requiring hydat_alpha/hydat_beta to be hardcoded in cfg
        # (the exact stale-value trap this project hit once already),
        # auto-derive them from Part 1's own WASP CSV columns -- that CSV
        # already carries a_coeff/c_exponent from whichever fit actually
        # produced it, so this stays correct even if Part 1 is re-run with
        # different data. Explicit cfg["hydat_alpha"/"hydat_beta"] still
        # wins if you deliberately want to override.
        obs_csv = Path(cfg["obs_csv"])
        station_label = cfg.get("station", obs_csv.stem)
        hydat_alpha = cfg.get("hydat_alpha")
        hydat_beta = cfg.get("hydat_beta")
        if hydat_alpha is None or hydat_beta is None:
            try:
                _peek = pd.read_csv(obs_csv, nrows=1)
                if "a_coeff" in _peek.columns and "c_exponent" in _peek.columns:
                    a0 = float(_peek["a_coeff"].iloc[0])
                    c0 = float(_peek["c_exponent"].iloc[0])
                    hydat_alpha = a0 ** (-1 / c0)
                    hydat_beta = 1 / c0
                    print(f"  (Auto-derived hydat_alpha/beta from {obs_csv.name}'s "
                          f"a_coeff/c_exponent columns -- no rating_curve_result "
                          f"or hardcoded values needed.)")
            except (FileNotFoundError, KeyError, ValueError, ZeroDivisionError):
                pass  # fall through; hydat_alpha/beta stay None if truly unavailable

    out_png = out_path / f"compare_depth_seg{target_seg}_{station_label}.png"

    models = {
        "A_manning": dict(
            fn=sim_depth_manning,
            bounds=[wscale_bounds, width_exp_bounds],
            names=["wscale", "WIDTH_EXP"],
            label="Manning width-law",
        ),
        "B_rating": dict(
            fn=sim_depth_rating,
            bounds=[log10_alpha_bounds, beta_bounds],
            names=["log10_alpha", "beta"],
            label="Direct rating h=a*Q^b",
        ),
    }

    print(f"\n{'#' * 70}")
    print(f"# PART 2: DEPTH-FROM-SIMULATION COMPARISON  --  "
          f"STATION {station_label}, SEGMENT {target_seg}")
    print(f"{'#' * 70}")

    # -------------------------------------------------------------------
    # LOAD OBSERVED
    # -------------------------------------------------------------------
    print("Loading observed data ...")
    df_obs_raw = pd.read_csv(obs_csv)
    df_obs_raw["date"] = pd.to_datetime(df_obs_raw["date"], errors="coerce")
    df_obs_raw["true_depth"] = pd.to_numeric(df_obs_raw["true_depth"], errors="coerce")
    df_obs_raw = df_obs_raw.dropna(subset=["date", "true_depth"])

    df_obs = (df_obs_raw.set_index("date").sort_index().loc[time_start:time_end])
    df_obs = df_obs[df_obs["true_depth"] >= 0].rename(columns={"true_depth": "obs_depth"})
    if df_obs.empty:
        raise RuntimeError("No observed data after filtering.")

    print(f"  Records    : {len(df_obs)}")
    print(f"  Time range : {df_obs.index[0].date()} - {df_obs.index[-1].date()}")
    print(f"  Depth      : mean={df_obs['obs_depth'].mean():.3f} m, "
          f"std={df_obs['obs_depth'].std(ddof=1):.3f} m, "
          f"max={df_obs['obs_depth'].max():.2f} m")

    # -------------------------------------------------------------------
    # LOAD MIZUROUTE
    # -------------------------------------------------------------------
    print("\nLoading mizuRoute output ...")
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

    seg_idx = int(np.where(seg_ids == target_seg)[0][0])
    Q_seg = np.maximum(Q[:, seg_idx], 0.0)
    S_seg = max(slope[seg_idx], 1e-8)

    print(f"  Target seg : {target_seg}  (index {seg_idx})")
    print(f"  Slope      : {slope[seg_idx]:.6f}")

    window_mask = (time >= pd.Timestamp(time_start)) & (time <= pd.Timestamp(time_end))
    Q_seg_window = Q_seg[window_mask]
    n_excluded_spinup = len(Q_seg) - len(Q_seg_window)
    if n_excluded_spinup > 0:
        print(f"  (Note: {n_excluded_spinup} timesteps before {time_start} or "
              f"after {time_end} excluded from diagnostics below.)")

    qpct = np.percentile(Q_seg_window, [50, 90, 95, 99, 99.9])
    print(f"  Q (m3/s)   : min={Q_seg_window.min():.2f}  median={qpct[0]:.2f}  "
          f"p95={qpct[2]:.2f}  p99={qpct[3]:.2f}  "
          f"p99.9={qpct[4]:.2f}  max={Q_seg_window.max():.2f}")
    if Q_seg_window.max() > 10 * qpct[4]:
        print("  [!] Q maximum is >10x the 99.9th percentile - likely numerical "
              "artefact in mizuRoute output.")
    if qpct[4] > 10 * qpct[3]:
        print(f"  [!] p99.9 ({qpct[4]:.2f}) is >10x p99 ({qpct[3]:.2f}) - sharp "
              "jump concentrated in the top ~0.1% of timesteps.")

    # -------------------------------------------------------------------
    # DIAGNOSTIC - inspect the extreme-Q days directly
    # -------------------------------------------------------------------
    df_qdiag = pd.DataFrame({"Q_m3s": Q_seg_window}, index=time[window_mask]).sort_values(
        "Q_m3s", ascending=False)
    n_extreme = int(np.sum(Q_seg_window > qpct[3]))
    print(f"\n  Extreme-Q diagnostic (within {time_start} to {time_end}): "
          f"{n_extreme} timesteps exceed p99 ({qpct[3]:.2f} m3/s) "
          f"out of {len(Q_seg_window)} total.")

    qdiag_path = out_path / f"extreme_Q_seg{target_seg}_{station_label}_diagnostic.csv"
    df_qdiag[df_qdiag["Q_m3s"] > qpct[3]].to_csv(qdiag_path)
    print(f"  Full list of {n_extreme} above-p99 timesteps saved -> {qdiag_path}")

    extreme_dates = df_qdiag[df_qdiag["Q_m3s"] > qpct[3]].index.sort_values()
    n_clusters = None
    if len(extreme_dates) > 1:
        gaps_days = extreme_dates.to_series().diff().dt.days.dropna()
        n_clusters = int((gaps_days > 1).sum()) + 1
        print(f"  These {len(extreme_dates)} extreme timesteps fall into "
              f"~{n_clusters} separate date cluster(s).")

    # -------------------------------------------------------------------
    # PAIRED DATA / CALIBRATION SUBSET
    # -------------------------------------------------------------------
    df_sim_Q = (pd.DataFrame({"Q": Q_seg}, index=time)
                .resample("D").mean().loc[time_start:time_end])
    df_obs_daily = df_obs[["obs_depth"]].resample("D").mean()
    df_pair = df_sim_Q.join(df_obs_daily, how="inner").dropna()
    df_pair_ow = df_pair[df_pair.index.month.isin([5, 6, 7, 8, 9, 10])]
    print(f"\nPaired open-water records : {len(df_pair_ow)}")

    if q_cal_cap_pct < 100:
        q_cap = np.percentile(df_pair_ow["Q"].values, q_cal_cap_pct)
        df_cal = df_pair_ow[df_pair_ow["Q"] <= q_cap]
        print(f"  Q-cap for calibration      : Q <= p{q_cal_cap_pct:g} = {q_cap:.2f} m3/s")
        print(f"  Calibration subset size    : {len(df_cal)}  "
              f"(excluded {len(df_pair_ow) - len(df_cal)} extreme days)")
    else:
        q_cap = np.inf
        df_cal = df_pair_ow

    if len(df_cal) < 30:
        raise RuntimeError(f"Too few calibration points ({len(df_cal)}).")

    Q_cal = df_cal["Q"].values
    obs_cal = df_cal["obs_depth"].values

    # -------------------------------------------------------------------
    # CALIBRATE BOTH MODELS (objective: log-KGE)
    # -------------------------------------------------------------------
    beta_ols, ln_alpha_ols = np.polyfit(np.log(Q_cal), np.log(obs_cal + log_eps), 1)
    seed_B = np.array([np.log10(np.exp(ln_alpha_ols)), beta_ols])
    seed_B = np.clip(seed_B,
                      [log10_alpha_bounds[0], beta_bounds[0]],
                      [log10_alpha_bounds[1], beta_bounds[1]])
    print(f"\nModel B OLS seed (log-log regression): "
          f"alpha={10 ** seed_B[0]:.6f}, beta={seed_B[1]:.4f}")

    results = {}
    for key, mdl in models.items():
        print(f"\nCalibrating MODEL {key} - {mdl['label']} ...")
        print(f"  Bounds -> {mdl['names'][0]}: {mdl['bounds'][0]}, "
              f"{mdl['names'][1]}: {mdl['bounds'][1]}")

        def neg_log_kge(params, fn=mdl["fn"]):
            d = fn(params, Q_cal, mann_n, S_seg)
            if not np.all(np.isfinite(d)) or np.std(d, ddof=1) == 0:
                return 2.0
            return np.nan_to_num(-log_kge(obs_cal, d, log_eps), nan=2.0)

        rng = np.random.default_rng(de_seed)
        lo = np.array([b_[0] for b_ in mdl["bounds"]])
        hi = np.array([b_[1] for b_ in mdl["bounds"]])
        init_pop = lo + rng.random((de_popsize * len(lo), len(lo))) * (hi - lo)
        if key == "B_rating":
            init_pop[0] = seed_B

        de = differential_evolution(
            neg_log_kge, bounds=mdl["bounds"], seed=de_seed,
            maxiter=de_maxiter, popsize=de_popsize, tol=1e-7,
            polish=False, updating="deferred", init=init_pop)
        pol = minimize(neg_log_kge, x0=de.x, method="L-BFGS-B",
                       bounds=mdl["bounds"],
                       options={"ftol": 1e-10, "gtol": 1e-8, "maxiter": 1000})

        p_opt = pol.x if pol.fun <= de.fun else de.x

        d_cal = mdl["fn"](p_opt, Q_cal, mann_n, S_seg)
        if key == "B_rating":
            print(f"  Calibrated alpha : {10 ** p_opt[0]:.6f}  (log10_alpha={p_opt[0]:.4f})")
            print(f"  Calibrated beta  : {p_opt[1]:.6f}")
        else:
            print(f"  Calibrated {mdl['names'][0]:9s}: {p_opt[0]:.6f}")
            print(f"  Calibrated {mdl['names'][1]:9s}: {p_opt[1]:.6f}")
        print(f"  Calibration log-KGE : {-neg_log_kge(p_opt):.4f}")
        print(f"  Calibration raw-KGE : {kge(obs_cal, d_cal):.4f}")
        for v, (lo_b, hi_b), nm in zip(p_opt, mdl["bounds"], mdl["names"]):
            if at_bound(v, lo_b, hi_b):
                print(f"  [!] {nm} is at its bound.")
        if key == "B_rating" and hydat_alpha is not None:
            print(f"  (HYDAT-derived depth curve for {station_label}: "
                  f"h = {hydat_alpha:.4f}*Q^{hydat_beta:.4f})")

        results[key] = dict(params=p_opt, **mdl)

    # -------------------------------------------------------------------
    # APPLY BOTH MODELS TO FULL OPEN-WATER PERIOD
    # -------------------------------------------------------------------
    q_global_cap = np.percentile(Q_seg_window, 99.9) if q_cal_cap_pct < 100 else np.inf
    Q_seg_used = np.minimum(Q_seg, q_global_cap)

    df_eval = None
    for key, r in results.items():
        depth_full = r["fn"](r["params"], Q_seg_used, mann_n, S_seg)
        ser = (pd.DataFrame({f"sim_{key}": depth_full}, index=time)
               .resample("D").mean().loc[time_start:time_end])
        ser = ser[ser.index.month.isin([5, 6, 7, 8, 9, 10])]
        df_eval = ser if df_eval is None else df_eval.join(ser, how="inner")

    df_eval = df_eval.join(df_obs_daily, how="inner").dropna()
    df_eval["Q"] = df_sim_Q.reindex(df_eval.index)["Q"]
    obs_all = df_eval["obs_depth"].values
    mask_in = df_eval["Q"].values <= q_cap

    # -------------------------------------------------------------------
    # REPORT - side-by-side
    # -------------------------------------------------------------------
    display_keys = ("R2", "NSE", "KGE", "logKGE", "RMSE", "MAE", "PBIAS")
    metrics = {}
    for key in results:
        sim = df_eval[f"sim_{key}"].values
        metrics[key] = dict(
            full=calc_metrics(obs_all, sim, log_eps),
            filt=calc_metrics(obs_all[mask_in], sim[mask_in], log_eps),
            sim=sim)

    print(f"\n{'=' * 64}")
    print(f"  SIDE-BY-SIDE - FULL paired open-water set (n={len(obs_all)})")
    print(f"{'=' * 64}")
    hdr = f"  {'metric':<8}" + "".join(f"{results[k]['label'][:26]:>28}" for k in results)
    print(hdr)
    for mk in display_keys:
        print(f"  {mk:<8}" + "".join(f"{metrics[k]['full'][mk]:>+28.4f}" for k in results))
    print(f"  {'sim mean':<8}" + "".join(f"{metrics[k]['sim'].mean():>28.3f}" for k in results))
    print(f"  {'sim std':<8}" + "".join(f"{metrics[k]['sim'].std(ddof=1):>28.3f}" for k in results))
    print(f"  {'obs mean':<8}{obs_all.mean():>28.3f}")
    print(f"  {'obs std':<8}{obs_all.std(ddof=1):>28.3f}")

    print(f"\n{'=' * 64}")
    print(f"  SIDE-BY-SIDE - FILTERED set Q <= p{q_cal_cap_pct:g} (n={mask_in.sum()})")
    print(f"{'=' * 64}")
    for mk in display_keys:
        print(f"  {mk:<8}" + "".join(f"{metrics[k]['filt'][mk]:>+28.4f}" for k in results))

    # -------------------------------------------------------------------
    # LOW / MID / HIGH FLOW TERCILE BREAKDOWN
    # -------------------------------------------------------------------
    print(f"\n{'=' * 64}")
    print(f"  LOW / MID / HIGH FLOW TERCILE BREAKDOWN (full paired set)")
    print(f"{'=' * 64}")

    Q_all = df_eval["Q"].values
    q_terc = np.percentile(Q_all, [100 / 3, 200 / 3])
    tercile_masks = {
        "low  (Q < p33)": Q_all < q_terc[0],
        "mid  (p33<=Q<p67)": (Q_all >= q_terc[0]) & (Q_all < q_terc[1]),
        "high (Q >= p67)": Q_all >= q_terc[1],
    }
    print(f"  Tercile breakpoints: p33={q_terc[0]:.1f} m3/s, p67={q_terc[1]:.1f} m3/s\n")

    tercile_results = {}
    for band_name, band_mask in tercile_masks.items():
        n_band = int(band_mask.sum())
        if n_band < 10:
            print(f"  {band_name:<20} n={n_band:<6} (too few points, skipped)")
            continue
        obs_band = obs_all[band_mask]
        line = f"  {band_name:<20} n={n_band:<6}"
        band_metrics = {}
        for key in results:
            sim_band = metrics[key]["sim"][band_mask]
            m_band = calc_metrics(obs_band, sim_band, log_eps)
            band_metrics[key] = m_band
            line += f"  {results[key]['label'][:18]:<18} R2={m_band['R2']:.3f} NSE={m_band['NSE']:.3f}"
        print(line)
        tercile_results[band_name] = band_metrics

    # -------------------------------------------------------------------
    # PLOTS
    # -------------------------------------------------------------------
    plot_path = None
    if make_plot:
        print(f"\nPlotting -> {out_png}")
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28)
        ax_ts = fig.add_subplot(gs[0, :])

        pA, pB = results["A_manning"]["params"], results["B_rating"]["params"]
        fig.suptitle(
            f"Depth Comparison - Segment {target_seg} - Station {station_label}\n"
            f"A: w={pA[0]:.2f}*Q^{pA[1]:.3f}  |  "
            f"B: h={10 ** pB[0]:.5f}*Q^{pB[1]:.3f}  |  "
            f"Open water May-Oct {time_start[:4]}-{time_end[:4]}",
            fontsize=11, fontweight="bold")

        ax_ts.plot(df_eval.index, df_eval["obs_depth"], lw=1.0, color="black",
                  label="Observed (HYDAT rating)", zorder=3)
        ax_ts.plot(df_eval.index, df_eval["sim_A_manning"], lw=0.8, color="tomato",
                  alpha=0.8, label="A: Manning width-law", zorder=2)
        ax_ts.plot(df_eval.index, df_eval["sim_B_rating"], lw=0.8, color="seagreen",
                  alpha=0.8, label="B: direct rating", zorder=2)
        ax_ts.set_ylabel("River Depth (m)")
        ax_ts.set_title("Daily River Depth - Observed vs Both Models")
        ax_ts.legend(fontsize=9); ax_ts.grid(True, ls="--", alpha=0.4)
        ax_ts.xaxis.set_major_locator(mdates.YearLocator())
        ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax_ts.xaxis.get_majorticklabels(), rotation=30, ha="right")

        lim = max(obs_all.max(), *[metrics[k]["sim"].max() for k in results]) * 1.05
        for ax, key, color in [(fig.add_subplot(gs[1, 0]), "A_manning", "tomato"),
                               (fig.add_subplot(gs[1, 1]), "B_rating", "seagreen")]:
            sim = metrics[key]["sim"]
            m = metrics[key]["full"]
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
                   f"R2       = {m['R2']:.3f}\nNSE      = {m['NSE']:.3f}\n"
                   f"KGE      = {m['KGE']:.3f}\nlogKGE = {m['logKGE']:.3f}\n"
                   f"RMSE   = {m['RMSE']:.3f} m\nPBIAS  = {m['PBIAS']:+.1f}%",
                   transform=ax.transAxes, fontsize=8, va="top",
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        if show_plot:
            plt.show()
        plt.close(fig)
        plot_path = out_png
        print(f"Plot saved -> {out_png}\nDone.")

    # -------------------------------------------------------------------
    # SAVE CALIBRATED MODEL PARAMETERS -- so Part 3 (or any other later
    # step) can pick these up automatically even when run standalone,
    # without hardcoding numbers copied by hand from a console log. This
    # is the same class of bug the original v5 export script's docstring
    # described fixing (stale ALPHA_B/BETA_B from a different gauge/
    # segment left in place); saving to disk here removes the manual
    # copy step entirely, regardless of how Part 3 is invoked later.
    # -------------------------------------------------------------------
    model_params_path = out_path / f"{station_label}_calibrated_models.json"
    model_params_out = {
        "station": station_label,
        "target_seg": target_seg,
        "mann_n": mann_n,
        "alpha_b": float(10 ** results["B_rating"]["params"][0]),
        "beta_b": float(results["B_rating"]["params"][1]),
        "wscale_opt": float(results["A_manning"]["params"][0]),
        "width_exp_opt": float(results["A_manning"]["params"][1]),
    }
    with open(model_params_path, "w", encoding="utf-8") as f:
        json.dump(model_params_out, f, indent=2)
    print(f"\nCalibrated model parameters saved -> {model_params_path}")
    print("  (Part 3 will auto-load this file if run standalone for this station.)")

    # -------------------------------------------------------------------
    # RESULT DICT -- consumed by the orchestrator's combined summary
    # -------------------------------------------------------------------
    return {
        "station": station_label,
        "target_seg": target_seg,
        "n_paired_open_water": len(obs_all),
        "n_calibration": len(df_cal),
        "models": {
            key: dict(
                label=results[key]["label"],
                params=dict(zip(results[key]["names"], results[key]["params"].tolist())),
                metrics_full=metrics[key]["full"],
                metrics_filtered=metrics[key]["filt"],
            )
            for key in results
        },
        "tercile_metrics": tercile_results,
        "extreme_q_cluster_count": n_clusters,
        "extreme_q_diagnostic_csv": str(qdiag_path),
        "plot_path": str(plot_path) if plot_path else None,
        "model_params_json_path": str(model_params_path),
    }


if __name__ == "__main__":
    # Standalone smoke-test entry point -- edit these to run this file alone
    # (without Part 1 / the orchestrator). hydat_alpha/hydat_beta are left
    # unset here deliberately: compare_depth_models() will auto-derive them
    # from OBS_CSV's own a_coeff/c_exponent columns (Part 1's actual fit for
    # this station), rather than requiring hardcoded numbers that can go
    # stale. Set them explicitly in this dict only if you want to override.
    demo_cfg = {
        "mizu_nc": "yukon-upper-kw.nc",
        "topo_nc": "topology.nc",
        "target_seg": 239,
        "mann_n": 0.025,
        "out_path": ".",
        "obs_csv": "09CD001_openwater_depth_wasp_calibration.csv",
    }
    result = compare_depth_models(demo_cfg, rating_curve_result=None)
    print("\nReturned result dict keys:", list(result.keys()))
