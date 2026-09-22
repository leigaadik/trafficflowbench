"""Diagnose why the conservation term does or does not carry signal.

Prints the absolute magnitudes behind E_LWR:
    sum|N|      the accumulation the identity is about
    sum|dN|     observed change in accumulation
    sum|rhs|    what inflow/outflow/ramps say the change should be
    sum|resid|  the gap

The upstream docstring says the conservation signal is ~0.75% of the
accumulated total, and that a topology-derived flux misses by ~0.22%. If the
observation noise on the released layer is larger than 0.75%, no boundary flux
derived from that layer can ever make S_LWR discriminative.

    python mywork/diagnose_t3.py --boundary-flux reports/submit/boundary_flux_self.parquet \
        --submission reports/calibrate/D12_I5_N_truth.csv --panel D12_I5_N
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
for _p in (_SRC, Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from holdout import holdout_scope
from task3.score_task3 import (
    REGIMES, derive_physics, link_order, load_boundary_flux,
    load_ramp_flows, network_parameters, read_csv_checked,
)

STATE_COLUMNS = {"panel", "timestamp", "station_id", "link_id", "mask_regime",
                 "speed_kmh", "flow_vph"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=Path, required=True)
    ap.add_argument("--boundary-flux", type=Path, default=None)
    ap.add_argument("--release-root", type=Path, default=Path("data/kaggle_public"))
    ap.add_argument("--panel", action="append", default=None)
    args = ap.parse_args()
    release = args.release_root.resolve()

    from holdout import panels as known_panels
    panels = args.panel or known_panels(release)
    modes = json.loads((_SRC.parent / "config" / "task3_lwr_modes.json").read_text(encoding="utf-8"))
    state = read_csv_checked(args.submission.resolve(), STATE_COLUMNS, "state")

    for panel in panels:
        panel_dir = release / "corridors" / panel
        params = network_parameters(panel_dir).set_index("link_id")
        order = link_order(panel_dir, "train")
        ramps = load_ramp_flows(panel_dir, "train")
        flux = load_boundary_flux(args.boundary_flux.resolve(), panel) if args.boundary_flux else None
        sp = state[state.panel == panel]

        with holdout_scope(restrict_fit=False):
            frames = []
            for regime in REGIMES:
                f, _ = derive_physics(panel, release, "train", sp, regime,
                                      order, params, ramps, flux)
                frames.append(f)
        d = pd.concat(frames, ignore_index=True)

        p = network_parameters(panel_dir).set_index("link_id")
        x = d.join(p, on="link_id", rsuffix="_net")
        num = ["speed_kmh", "flow_vph", "inflow_vph", "outflow_vph", "on_ramp_flow_vph",
               "off_ramp_flow_vph", "accumulation_N"]
        for c in num:
            x[c] = pd.to_numeric(x[c], errors="coerce")
        x = x.dropna(subset=num + ["length_km"])
        x = x.sort_values(["link_id", "timestamp"])
        x["dN"] = x.groupby("link_id").accumulation_N.shift(-1) - x.accumulation_N
        x["dt"] = (x.groupby("link_id").timestamp.shift(-1) - x.timestamp).dt.total_seconds() / 3600.0
        x = x[(x.dt > 0) & (x.dt <= 5.1 / 60.0)].copy()
        rhs = x.dt * (x.inflow_vph + x.on_ramp_flow_vph - x.outflow_vph - x.off_ramp_flow_vph)
        resid = x.dN - rhs

        print(f"\n=== {panel} ===")
        print(f"  transitions            {len(x):,}")
        print(f"  mean accumulation N    {x.accumulation_N.mean():,.1f} veh")
        print(f"  sum|dN|                {x.dN.abs().sum():,.0f}")
        print(f"  sum|rhs|               {rhs.abs().sum():,.0f}")
        print(f"  sum|resid|             {resid.abs().sum():,.0f}")
        print(f"  E_LWR = sum|resid|/sum|rhs|   {resid.abs().sum() / (rhs.abs().sum() + 1e-9):.4f}")
        print(f"  sum|resid| as share of sum|N|: "
              f"{100 * resid.abs().sum() / (x.accumulation_N.abs().sum() + 1e-9):.2f}%")
        print(f"  sum|rhs|   as share of sum|N|: "
              f"{100 * rhs.abs().sum() / (x.accumulation_N.abs().sum() + 1e-9):.2f}%")


if __name__ == "__main__":
    main()
