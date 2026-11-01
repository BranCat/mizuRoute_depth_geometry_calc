"""
control_file.py

Single place to configure the river-depth pipeline: WSC gauge number,
mizuRoute/topology file paths, target river segment, Manning's n, and any
model-fitting hyperparameters you want to override.

Edit STATIONS to add/remove stations. Each entry is merged on top of
DEFAULTS, so you only need to specify what's different for that station.

Run the pipeline with:
    python run_pipeline.py
"""

# =============================================================================
# GLOBAL DEFAULTS
# Applied to every station in STATIONS below, unless a station-specific
# entry overrides a given key.
# =============================================================================
DEFAULTS = {
    # --- Part 1: HYDAT / rating curve ---------------------------------
    "hydat_path": "<WORK_ROOT>/river-depth-geometry-automation/Hydat.sqlite3",
    "out_path": "<WORK_ROOT>/river-depth-geometry-automation",
    "gauge_to_cgvd2013": None,
    "gauge_to_cgvd28": None,

    # --- Part 2: simulated discharge / channel geometry ----------------
    "mann_n": 0.025,
    "time_start": "2015-01-01",
    "time_end": "2024-12-31",

    # Model A (Manning width-law) calibration bounds
    "wscale_bounds": (0.1, 500.0),
    "width_exp_bounds": (0.25, 0.70),

    # Model B (direct at-a-station rating h = alpha * Q^beta) bounds
    "log10_alpha_bounds": (-5.0, 1.0),
    "beta_bounds": (0.2, 1.5),

    # Fitting hyperparameters
    "log_eps": 0.01,
    "q_cal_cap_pct": 99.0,
    "de_seed": 42,
    "de_maxiter": 200,
    "de_popsize": 20,

    # --- Part 3: channel geometry export (all segments) ----------------
    "width_anchor_m": None,   # e.g. 130.0 to rescale width/unit_flow to a
                               # known real width at median Q; None = no rescale
    "q_outlier_ratio": 10.0,  # flag segments where max(Q) > ratio * p99
    "min_depth": 0.01,        # floor on exported depth, meters
    "ow_months": [5, 6, 7, 8, 9, 10],

    # Plotting
    "make_plot": True,
    "show_plot": False,   # set True only if running interactively
}

# =============================================================================
# STATIONS
# One entry per WSC gauge / basin you want to run through the full
# pipeline. Only include keys that differ from DEFAULTS above -- anything
# omitted here falls back to the default.
#
# Required per station: station, mizu_nc, topo_nc, target_seg
# =============================================================================
STATIONS = [
    {
        "station": "09AB001",
        "mizu_nc": "<WORK_ROOT>/summa-mizu-sim-assess-automation/sim-data/yukon-whitehorse.nc",
        "topo_nc": "<WORK_ROOT>/river-depth-geometry-automation/topology-whitehorse-yukon.nc",
        "target_seg": 239,  # gauge 07BJ001 (was 365, which does not exist in swan_river.nc)
        # Example override: this station's channel is rougher than default
        # "mann_n": 0.030,
    },

    # Add more stations here to batch-process, e.g.:
    # {
    #     "station": "09EB001",
    #     "mizu_nc": "liard-lower-kw.nc",
    #     "topo_nc": "liard_topology.nc",
    #     "target_seg": 512,
    #     "mann_n": 0.028,
    #     "time_start": "2011-01-01",
    #     "time_end": "2023-12-31",
    # },
]
