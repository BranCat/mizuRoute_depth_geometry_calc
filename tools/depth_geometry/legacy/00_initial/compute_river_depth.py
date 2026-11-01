"""
compute_river_depth.py
─────────────────────────────────────────────────────────────────────────────
Post-processes mizuRoute output to estimate river depth at every reach and
every timestep using Manning's equation.

Inputs:
  - mizuRoute output NetCDF  (contains KWTroutedRunoff)
  - topology NetCDF          (contains slope, area, segId, downSegId, hruToSegId)

Output:
  - NetCDF with river_depth (m), river_width (m), river_velocity (m/s)
    for every reach and timestep

Manning's equation (wide rectangular channel assumption):
  Q  = (1/n) * w * d^(5/3) * S^(1/2)
  d  = ( Q * n / (w * S^0.5) )^(3/5)
  v  = Q / (w * d)

─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import xarray as xr
import pandas as pd
import os

# ── Configuration ─────────────────────────────────────────────────────────────
MIZU_OUTPUT  = "/anvil/projects/x-ees250064/symfluence-FH/SYMFLUENCE_data/domain_Swan_river_t1/summa_sim_local/simulations/run_1/mizuRoute/proc_29_run_1.nc"
TOPOLOGY     = "/anvil/projects/x-ees250064/symfluence-FH/SYMFLUENCE_data/domain_Swan_river_t1/summa_sim_local/settings/mizuRoute/topology.nc"
OUTPUT_FILE  = "/anvil/projects/x-ees250064/symfluence-FH/SYMFLUENCE_data/domain_Swan_river_t1/summa_sim_local/simulations/run_1/mizuRoute/proc_29_run_1_depth.nc"

# Manning's n and width scale from param.nml.default
MANN_N  = 0.025   # Manning's roughness coefficient
WSCALE  = 0.001   # width = WSCALE * sqrt(upstream_area_m2)

# Minimum values to avoid numerical issues
MIN_SLOPE = 1e-5  # minimum slope (m/m) — avoids division issues on flat reaches
MIN_DEPTH = 0.001 # minimum depth (m)  — clamp unrealistically small values
# ──────────────────────────────────────────────────────────────────────────────


def build_upstream_area(topo):
    """
    Accumulate upstream drainage area for every reach by walking the
    network from headwaters to outlet.
    Returns a dict: segId -> cumulative upstream area (m²)
    """
    seg_ids    = topo["segId"].values.astype(int)
    down_ids   = topo["downSegId"].values.astype(int)
    hru_ids    = topo["hruId"].values.astype(int)
    hru_seg    = topo["hruToSegId"].values.astype(int)  # which seg each HRU drains to
    hru_area   = topo["area"].values                    # m²

    # Map each segId to its local (own-HRU) area
    seg_local_area = {sid: 0.0 for sid in seg_ids}
    for hid, sid, a in zip(hru_ids, hru_seg, hru_area):
        if sid in seg_local_area:
            seg_local_area[sid] += a

    # Map segId -> downSegId
    down_map = {sid: did for sid, did in zip(seg_ids, down_ids)}

    # Topological sort (simple iterative accumulation)
    cum_area = {sid: seg_local_area[sid] for sid in seg_ids}
    max_iter = len(seg_ids) * 2
    for _ in range(max_iter):
        updated = False
        for sid in seg_ids:
            did = down_map[sid]
            if did in cum_area and did != -9999:
                new_val = cum_area[sid] + seg_local_area.get(did, 0.0)
                # Pass upstream area downstream
                contribution = cum_area[sid]
                old_down     = cum_area[did]
                # Recompute downstream from all tributaries each pass
                # (simple approach: done via repeated passes)
                pass
        break  # exit — we use the proper approach below

    # Proper approach: sort reaches from headwater to outlet
    # Build in-degree (number of upstream reaches)
    indegree = {sid: 0 for sid in seg_ids}
    children = {sid: [] for sid in seg_ids}   # sid -> list of upstream segs
    for sid in seg_ids:
        did = down_map[sid]
        if did in indegree:
            indegree[did] += 1
            children[did].append(sid)

    # Kahn's topological sort (headwaters first)
    from collections import deque
    queue = deque([sid for sid in seg_ids if indegree[sid] == 0])
    order = []
    temp_indegree = indegree.copy()
    while queue:
        node = queue.popleft()
        order.append(node)
        did = down_map[node]
        if did in temp_indegree and did != -9999:
            temp_indegree[did] -= 1
            if temp_indegree[did] == 0:
                queue.append(did)

    # Accumulate area in topological order
    cum_area = {sid: seg_local_area[sid] for sid in seg_ids}
    for sid in order:
        did = down_map[sid]
        if did in cum_area and did != -9999:
            cum_area[did] += cum_area[sid]

    return cum_area


def compute_depth(Q, slope, width, mann_n, min_slope, min_depth):
    """
    Vectorised Manning's equation for wide rectangular channel.
    Q     : discharge (m³/s), any shape
    slope : channel slope (m/m), broadcast-compatible with Q
    width : channel width (m),   broadcast-compatible with Q
    Returns depth (m), velocity (m/s)
    """
    slope_safe = np.maximum(slope, min_slope)
    width_safe = np.maximum(width, 0.1)   # at least 10 cm wide
    Q_safe     = np.maximum(Q, 0.0)

    # d = (Q * n / (w * S^0.5)) ^ (3/5)
    depth = (Q_safe * mann_n / (width_safe * np.sqrt(slope_safe))) ** (3.0 / 5.0)
    depth = np.maximum(depth, min_depth)

    # v = Q / (w * d)
    velocity = Q_safe / (width_safe * depth)

    return depth.astype(np.float32), velocity.astype(np.float32)


def main():
    print("=" * 65)
    print("  mizuRoute River Depth Post-Processor")
    print("=" * 65)

    # ── Load topology ──────────────────────────────────────────────────────────
    print(f"\n[1] Loading topology: {os.path.basename(TOPOLOGY)}")
    topo   = xr.open_dataset(TOPOLOGY)
    seg_ids_topo = topo["segId"].values.astype(int)
    slopes_topo  = topo["slope"].values

    # Build upstream area lookup
    print("    Computing upstream drainage area per reach ...")
    cum_area_map = build_upstream_area(topo)
    topo.close()

    # Arrays aligned to topology seg order
    seg_slopes = np.array([slopes_topo[i]
                           for i in range(len(seg_ids_topo))])
    seg_widths = np.array([WSCALE * np.sqrt(cum_area_map[sid])
                           for sid in seg_ids_topo])

    print(f"    Reach widths  — min: {seg_widths.min():.1f} m  "
          f"max: {seg_widths.max():.1f} m  "
          f"mean: {seg_widths.mean():.1f} m")
    print(f"    Reach slopes  — min: {seg_slopes.min():.2e}  "
          f"max: {seg_slopes.max():.4f}  "
          f"mean: {seg_slopes.mean():.4f}")

    # ── Load mizuRoute output ──────────────────────────────────────────────────
    print(f"\n[2] Loading mizuRoute output: {os.path.basename(MIZU_OUTPUT)}")
    ds = xr.open_dataset(MIZU_OUTPUT)

    if "KWTroutedRunoff" not in ds:
        raise KeyError("KWTroutedRunoff not found in output. "
                       "Check route_opt setting.")

    Q_all   = ds["KWTroutedRunoff"].values  # (time, seg)  m³/s
    mizu_reach_ids = ds["reachID"].values.astype(int)
    times   = ds["time"]
    n_time, n_seg = Q_all.shape
    print(f"    Timesteps : {n_time:,}  |  Reaches : {n_seg}")

    # ── Align topology seg order with mizuRoute output seg order ───────────────
    # mizuRoute output seg order may differ from topology seg order
    topo_seg_index = {sid: i for i, sid in enumerate(seg_ids_topo)}
    mizu_to_topo   = np.array([topo_seg_index[sid]
                                for sid in mizu_reach_ids])

    slopes_aligned = seg_slopes[mizu_to_topo]   # (seg,)
    widths_aligned = seg_widths[mizu_to_topo]   # (seg,)

    # Broadcast to (time, seg) for vectorised compute
    slopes_2d = slopes_aligned[np.newaxis, :]   # (1, seg) -> broadcasts
    widths_2d = widths_aligned[np.newaxis, :]   # (1, seg) -> broadcasts

    # ── Compute depth and velocity ─────────────────────────────────────────────
    print(f"\n[3] Computing depth and velocity for {n_time:,} timesteps ...")
    depth_all, vel_all = compute_depth(
        Q_all, slopes_2d, widths_2d, MANN_N, MIN_SLOPE, MIN_DEPTH
    )

    # ── Summary statistics ─────────────────────────────────────────────────────
    print(f"\n  Depth   — min: {depth_all.min():.3f} m  "
          f"max: {depth_all.max():.3f} m  "
          f"mean: {depth_all.mean():.3f} m")
    print(f"  Velocity— min: {vel_all.min():.4f} m/s  "
          f"max: {vel_all.max():.3f} m/s  "
          f"mean: {vel_all.mean():.4f} m/s")

    # ── Build output Dataset ───────────────────────────────────────────────────
    print(f"\n[4] Writing output: {os.path.basename(OUTPUT_FILE)}")

    # Width array (constant in time — just one value per reach)
    width_arr = widths_aligned.astype(np.float32)

    ds_out = xr.Dataset(
        {
            "KWTroutedRunoff": (["time", "seg"], Q_all,
                {
                    "long_name": "routed runoff in reach - lagrangian kinematic wave",
                    "units"    : "m3/s",
                }),
            "river_depth": (["time", "seg"], depth_all,
                {
                    "long_name"  : "estimated river depth from Manning's equation",
                    "units"      : "m",
                    "method"     : f"d=(Q*n/(w*S^0.5))^(3/5), n={MANN_N}, wscale={WSCALE}",
                    "note"       : "wide rectangular channel assumption",
                }),
            "river_width": (["seg"], width_arr,
                {
                    "long_name" : "estimated bankfull channel width",
                    "units"     : "m",
                    "formula"   : f"width = {WSCALE} * sqrt(upstream_area_m2)",
                }),
            "river_velocity": (["time", "seg"], vel_all,
                {
                    "long_name" : "estimated depth-averaged flow velocity",
                    "units"     : "m/s",
                    "formula"   : "v = Q / (w * d)",
                }),
            "reachID": (["seg"], mizu_reach_ids,
                {
                    "long_name": "reach ID",
                    "units"    : "-",
                }),
        },
        coords={"time": times, "seg": np.arange(n_seg)},
        attrs={
            "description" : "River depth computed from mizuRoute KWTroutedRunoff via Manning's equation",
            "source_mizu" : os.path.basename(MIZU_OUTPUT),
            "source_topo" : os.path.basename(TOPOLOGY),
            "mann_n"      : MANN_N,
            "wscale"      : WSCALE,
            "min_slope"   : MIN_SLOPE,
        }
    )

    encoding = {
        "river_depth"   : {"zlib": True, "complevel": 4, "dtype": "float32"},
        "river_width"   : {"zlib": True, "complevel": 4, "dtype": "float32"},
        "river_velocity": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "KWTroutedRunoff": {"zlib": True, "complevel": 4, "dtype": "float32"},
    }
    ds_out.to_netcdf(OUTPUT_FILE, encoding=encoding)
    ds.close()

    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 ** 2)
    print(f"  Done. File written ({size_mb:.1f} MB): {OUTPUT_FILE}")

    # ── Per-reach summary at outlet ───────────────────────────────────────────
    outlet_idx = 0  # reachID 356 is first in output (downSegId = -9999)
    print(f"\n  Outlet reach (reachID={mizu_reach_ids[outlet_idx]}) stats:")
    print(f"    Width    : {widths_aligned[outlet_idx]:.1f} m")
    print(f"    Q mean   : {Q_all[:, outlet_idx].mean():.2f} m³/s")
    print(f"    Depth mean: {depth_all[:, outlet_idx].mean():.3f} m")
    print(f"    Depth max : {depth_all[:, outlet_idx].max():.3f} m")

    print(f"\n{'='*65}")
    print("  Complete.")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
