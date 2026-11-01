# Channel depth and geometry from mizuRoute discharge

Post-processing tools that estimate **channel depth, width and unit flow** for every reach of a mizuRoute river network, calibrated against depth derived from Water Survey of Canada (WSC/HYDAT) gauges.

The outputs (`channel_geometry_<station>.nc`) give SedHydro the four variables it reads from a geometry file: `channel_width`, `channel_depth`, `unit_flow` and `discharge`.

Author: Frank Han, University of Calgary. Developed March–October 2026.

---

## Method

Depth comes from Manning's equation for a wide rectangular channel, with width tied to discharge by a power law:

```
width  w = wscale · Q^b
depth  h = ( Q · n / (w · √S) )^0.6        n = 0.025, S = reach slope from the topology file
unit flow  q = Q / w
```

At the **gauged reach**, depth can instead come from a direct at-a-station rating, `h = α · Q^β` (Model B). Every other reach uses the Manning and width law (Model A), because that is the only form that transfers to reaches without observations.

The observed depth used for calibration comes from a HYDAT rating curve, `Q = a · (h_stage − b)^c`. The observed depth is then `h_stage − b`, using open-water days only (May–October).

---

## Version history

The method developed river by river, from Swan (March 2026) to the Upper Yukon (July 2026), and was then combined into one pipeline (October 2026). Each version fixed a problem found with the previous one. See [CHANGELOG.md](CHANGELOG.md) for details.

| Version | Script(s) | Width | Depth | Calibration | First used for |
|---|---|---|---|---|---|
| Initial | `legacy/00_initial/compute_river_depth.py` | 0.001·√(upstream area), the mizuRoute default | Manning | none | — |
| v1 | `legacy/v1/back-calculate-depth-compare.py`, `export_all_segments.py` | wscale·Q^0.20 (fixed exponent) | Manning | wscale matched to observed **mean** depth (Brent) | Swan (07BJ001) |
| v2 | *script not preserved* (see `legacy/v2/README.md`) | wscale·Q^b | Manning | Nelder-Mead, **unbounded** | Athabasca (07DD001) |
| v3 | `legacy/v3/back-calc-depth-v3.py` | b ≥ 0.05 | Manning | KGE, bounded L-BFGS-B; Q > p99 excluded | Hay, Notikewin, Smoky |
| v4 | `legacy/v4/back-calc-depth-v4.py` | b ≥ 0.25 | Manning | log-KGE | Upper Yukon |
| v5 | `legacy/v5/back-calc-depth-v5.py`, `channel-geometry-calc.py` | b ≥ 0.25 | **rating at gauge (B)**, Manning elsewhere (A) | log-KGE; Q capped at p99 after warm-up | Upper Yukon |
| **Pipeline** | `pipeline/` | as v5 | as v5 | as v5; one control file for all stations | all stations |

Rating-curve fitting from HYDAT, used for the observed depth, is in `legacy/rating_curve/`. The `-update` version fits in log space, which better constrains the zero-flow stage *b*. `rtest.py` is an experimental segmented fit using AIC.

---

## Running the pipeline

Requirements: Python 3 with `numpy`, `pandas`, `scipy`, `xarray`, `netCDF4` and `matplotlib`, plus a local copy of the HYDAT SQLite database (`Hydat.sqlite3`, downloadable from Environment and Climate Change Canada).

1. Edit `pipeline/control_file.py`:
   - replace the `<WORK_ROOT>` placeholder in `hydat_path`, `out_path` and the station paths with your own folder
   - add one entry per station to `STATIONS`, with `station`, `mizu_nc` (mizuRoute output), `topo_nc` (topology with `segId` and `slope`) and `target_seg` (the gauged reach)
2. Run:
   ```bash
   cd pipeline
   python run_pipeline.py
   ```
3. For each station, a folder `<out_path>/<station>/` gets:
   - Part 1: `rating_curve_fit.png` and `<station>_openwater_depth_wasp_calibration.csv` (observed depth)
   - Part 2: `compare_depth_seg<N>_<station>.png` and `<station>_calibrated_models.json` (Model A and Model B parameters)
   - Part 3: `channel_geometry_<station>.nc` (time × seg: width, depth, unit flow, discharge), a per-segment summary CSV, an overview plot and a validation plot at the gauge
   - in `out_path`: `combined_summary.csv` and `combined_summary_report.txt`

The routed-flow variable is detected automatically (`KWroutedRunoff`, `KWTroutedRunoff` or `IRFroutedRunoff`).

---

## Results (pipeline, 2015–2024)

Calibrated parameters from `results/<station>/<station>_calibrated_models.json`:

| River | Gauge | Gauge reach | Model A width law | Model B rating at gauge |
|---|---|---|---|---|
| Swan | 07BJ001 | 356 | 6.96·Q^0.25 | 0.314·Q^0.459 |
| Athabasca | 07DD001 | 222 | 60.21·Q^0.25 | 0.117·Q^0.476 |
| Smoky | 07GJ001 | 114 | 4.15·Q^0.479 | 0.411·Q^0.313 |
| Notikewin | 07HC001 | 203 | 11.41·Q^0.25 | 0.151·Q^0.531 |
| Hay | 07OB001 | 187 | 14.45·Q^0.25 | 0.142·Q^0.527 |
| Upper Yukon | 09CD001 | 239 | *see note* | *see note* |

Notes:
- The width exponent sits at its lower bound (0.25) for four of the five Alberta rivers, so the bound rather than the data sets it there. Smoky is the exception.
- **Upper Yukon (09CD001):** the saved `calibrated_models.json` refers to segment 187 and has both parameters at their bounds. It is stale and should be regenerated for segment 239 before it is used. The validation plots in `results/09CD001/` are for segment 239.
- The pipeline caps discharge at the 99th percentile per reach. This removes mizuRoute spin-up spikes, but also flattens the largest floods.
- `results/` holds only small summary files. NetCDF outputs, HYDAT-derived time series and the HYDAT database are not included in the repository (see `.gitignore`).

A fuller write-up, including old-versus-new geometry comparisons and validation against observed depth for each river, is in `docs/River_depth_geometry_method_summary_1030_2026.docx`.

---

## Use in SedHydro (Paper 2)

The sediment runs in SedHydro Paper 2 used geometry files from different versions: Swan v1, Athabasca v2, Hay, Notikewin and Smoky v3, and Yukon v4. A sensitivity test that replaces the Athabasca v2 geometry (a constant width of about 438 m) with the pipeline geometry is under way (October 2026).

---

## Known issues

- Six small helper scripts from the original working folder were lost to file corruption and are not included: `data-retrieve-demo.py`, `netcdf-check.py`, `netcdf-compare.py`, `netcdf-mizu-plot.py`, `plot_KWTroutedRunoff_seg356.py` and `wsc_station_query.py`.
- Scripts in `legacy/` are kept as they were used, including station settings; local paths were replaced by the placeholder `<WORK_ROOT>`. They are a record of the method's development, not maintained code.
- `pipeline/control_file.py` contains `<WORK_ROOT>` placeholders and the most recent station entry; adapt it before running.
