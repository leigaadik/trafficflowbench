"""Calibrate Task 3 locally by supplying true boundary fluxes.

Task 3 is scored on the Task 1 submission, so the same three inputs are reused:
zero, baseline, and truth. With a real boundary flux file the conservation term
stops flooring at zero and S_LWR becomes discriminative.

Leaderboard reference (docs/BASELINES.md, ten-panel validation mean):
    baseline 0.3549 | perfect 0.9636

The perfect answer should land near 0.96 rather than 1.0: S_FD is a validity
check against a triangular diagram fitted to real detectors, so even the true
state does not sit perfectly on it. S_physics = (S_FD + 2*S_LWR)/3.

    python mywork/calibrate_t3.py --submission truth reports/calibrate/D12_I5_N_truth.csv \
        --submission baseline reports/calibrate/D12_I5_N_baseline.csv \
        --boundary-flux reports/submit/boundary_flux.parquet --panel D12_I5_N
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
for _p in (_SRC, Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from holdout import holdout_scope
from task3.score_task3 import (
    REGIMES,
    derive_physics,
    link_order,
    load_boundary_flux,
    load_ramp_flows,
    network_parameters,
    read_csv_checked,
    score_panel,
)

STATE_COLUMNS = {"panel", "timestamp", "station_id", "link_id", "mask_regime",
                 "speed_kmh", "flow_vph"}
ONLINE = {"zero": 0.0, "baseline": 0.3549, "truth": 0.9636}


def score_one(release: Path, panel: str, submission: Path, flux: Path | None,
              split: str = "train") -> dict:
    modes = json.loads((_SRC.parent / "config" / "task3_lwr_modes.json").read_text(encoding="utf-8"))
    state = read_csv_checked(Path(submission).resolve(), STATE_COLUMNS, "state submission")
    panel_dir = release / "corridors" / panel
    params = network_parameters(panel_dir).set_index("link_id")
    order = link_order(panel_dir, split)
    ramps = load_ramp_flows(panel_dir, split)
    boundary = load_boundary_flux(Path(flux).resolve(), panel) if flux else None

    sp = state[state.panel == panel]
    frames = []
    # derive_physics walks the split through score_task3.files, patched to the
    # eval month by the scope.
    with holdout_scope(restrict_fit=False):
        for regime in REGIMES:
            frame, _ = derive_physics(panel, release, split, sp, regime,
                                      order, params, ramps, boundary)
            frames.append(frame)
    physics = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    rows = [score_panel(panel, physics, release, modes, r) for r in REGIMES]
    return {
        "S_physics": float(sum(r["S_physics"] for r in rows) / len(rows)),
        "S_FD": float(sum(r["S_FD"] for r in rows) / len(rows)),
        "S_LWR": float(sum(r["S_LWR"] for r in rows) / len(rows)),
        # E_LWR = sum|residual| / sum|rhs|. At 1.0 the residual is as large as
        # the term it is compared against, which means the frame carries no
        # conservation signal at all rather than a slightly wrong flux.
        "E_LWR": float(sum(r["E_LWR"] for r in rows) / len(rows)),
        "n_rows": int(sum(r["n_rows"] for r in rows)),
        "mode": rows[0].get("mode"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", action="append", nargs=2,
                    metavar=("NAME", "PATH"), required=True,
                    help="repeatable; NAME is your label, e.g. baseline")
    ap.add_argument("--boundary-flux", type=Path, default=None,
                    help="parquet from build_boundary_flux.py; omit to reproduce the "
                         "topology fallback (S_LWR should then floor at 0)")
    ap.add_argument("--release-root", type=Path, default=Path("data/kaggle_public"))
    ap.add_argument("--panel", action="append", default=None)
    args = ap.parse_args()
    release = args.release_root.resolve()

    from holdout import panels as known_panels
    panels = args.panel or known_panels(release)

    print(f"boundary flux: {args.boundary_flux or 'NONE (topology fallback)'}")
    for name, path in args.submission:
        scores = []
        for panel in panels:
            s = score_one(release, panel, Path(path), args.boundary_flux)
            scores.append(s)
            print(f"  {name:9s} {panel:11s} S_physics={s['S_physics']:.4f} "
                  f"(S_FD={s['S_FD']:.4f} S_LWR={s['S_LWR']:.4f} "
                  f"E_LWR={s['E_LWR']:.3f} rows={s['n_rows']:,})", flush=True)
        mean = sum(s["S_physics"] for s in scores) / len(scores)
        online = ONLINE.get(name)
        extra = f"   online={online:.4f}   delta={mean - online:+.4f}" if online is not None else ""
        print(f"  => {name:9s} mean S_physics={mean:.4f}{extra}\n")


if __name__ == "__main__":
    main()
