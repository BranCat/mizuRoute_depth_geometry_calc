# Changelog

Development of the channel depth and geometry method, in order. Dates are approximate (month of first use).

## Pipeline: October 2026
- Combined rating-curve fitting (Part 1), depth-model calibration (Part 2) and network export (Part 3) into `pipeline/`, driven by `control_file.py` and `run_pipeline.py`.
- Part 1's fitted rating curve is passed straight to Part 2, and Part 2's calibrated parameters to Part 3. Before, these values were copied between scripts by hand, which is how stale numbers from another gauge had ended up in v4.
- Each station's results go to their own folder, `<out_path>/<station>/`, and all stations are collected in a combined summary table.
- Fixes, 30 October 2026: Parts 2 and 3 accept `KWroutedRunoff`, `KWTroutedRunoff` or `IRFroutedRunoff` (mizuRoute v3.0.1 writes `KWTroutedRunoff`); the Swan target segment in the control file was corrected from 365 (which does not exist) to 356.

## v5: July 2026 (Upper Yukon, 09CD001)
- Added **Model B**, a direct at-a-station rating `h = α·Q^β` with β free, used at the gauged reach. Model A (Manning and width law) is kept for ungauged reaches, and the export records which one each reach uses (`depth_method`).
- Fixed the HYDAT reference curve, which had been hard-coded from a different station (09AB001).
- The discharge cap (p99) is computed only within the post-warm-up window, and a p99 → p99.9 jump check was added to detect spin-up spikes.
- `channel-geometry-calc.py` fixed a provenance mismatch: the v4 export carried Model B parameters from a different gauge and segment.

## v4: July 2026 (Upper Yukon)
- Raised the width-exponent lower bound from 0.05 to 0.25, following Leopold & Maddock (1953), who report 0.3–0.5 for natural channels. A bound of 0.05 let width collapse to almost constant.
- Changed the objective from KGE to **log-KGE**, so that low and moderate flows carry weight in the calibration.

## v3: May–June 2026 (Hay, Notikewin, Smoky)
- Replaced unbounded Nelder-Mead with bounded **L-BFGS-B** (with differential evolution). In v2 the exponent had drifted to −0.046, meaning channels narrowed as flow rose.
- Calibration excludes days above the 99th-percentile discharge. mizuRoute produced numerical outliers of up to about 400,000 m³/s in the first weeks of a simulation. Metrics are reported on both the full and the filtered sets.

## v2: April 2026 (Athabasca, 07DD001)
- Let the width exponent be calibrated as well as the scale (Nelder-Mead, unbounded).
- Problem: the exponent collapsed to about 0, giving a **constant width of about 438 m at every reach and every flow**. This is what led to v3. The script was not preserved; see `legacy/v2/README.md`.

## v1: March 2026 (Swan, 07BJ001)
- First calibration against observed depth: `w = wscale·Q^0.20`, with wscale solved (Brent's method) so the simulated mean open-water depth at the gauge (segment 356) matches the HYDAT rating-curve mean, 1.513 m for 2016–2024.
- Limitation: it matches only the mean, not the variability.

## Initial
- `compute_river_depth.py`: Manning depth with the mizuRoute default width, 0.001·√(upstream area). Not calibrated.

## Rating curve (observed depth)
- `curve-fitting-depth-estimate.py`: fits `Q = a·(h − b)^c` to HYDAT daily levels and flows, using open water only (May–October); depth = h − b.
- `curve-fitting-depth-estimate-update.py` (July 2026): log-space fit, which better constrains b, plus extra diagnostics and a sensitivity check on b.
- `rtest.py`: experimental segmented fit with AIC model selection.
