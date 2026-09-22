"""Build true boundary fluxes for Task 3, from the unmasked train layer.

Why this exists
---------------
``score_task3.py --boundary-flux`` expects an organizer-held file of
``(timestamp, link_id, inflow_vph, outflow_vph)``. It is never published, and
without it the evaluator falls back to applying the released topology to the
participant's *own* submitted flows. That fallback is too coarse - the
conservation signal is ~0.75% of the accumulated total while the fallback misses
by ~0.22% - so S_LWR floors at 0 for everyone and the public evaluator cannot
tell a good reconstruction from a bad one.

On ``train`` the unmasked flow field *is* available, and it is exactly what the
organizer's boundary fluxes are derived from. So running the released topology
over truth instead of over the submission reproduces them, and S_LWR becomes
discriminative locally. No upstream code is modified: ``topology_boundary_flows``
is called as-is, just with a different input frame.

    python mywork/build_boundary_flux.py --release-root data/kaggle_public \
        --output reports/submit/boundary_flux.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
for _p in (_SRC, Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from holdout import is_eval_file


def boundary_from_topology(state: pd.DataFrame, panel_dir: Path,
                           outflow_mode: str = "self") -> pd.DataFrame:
    """Per-link inflow/outflow implied by the released topology and a flow field.

    ``outflow_mode`` decides what leaves a link:

    ``self``        a link's outflow is its own measured flow. This is what the
                    cell conservation identity means: what a link discharges is
                    what the detector on it counts.
    ``downstream``  the upstream's outflow is the *next* link's flow, which is
                    what score_task3.topology_boundary_flows does. It only holds
                    in steady state, and that gap is the ~0.22% error the
                    upstream docstring blames for S_LWR flooring at zero.
    """
    topo_path = panel_dir / "network" / "lwr_mainline_topology.csv"
    if not topo_path.exists():
        return pd.DataFrame(columns=["timestamp", "link_id", "inflow_vph", "outflow_vph"])
    topo = pd.read_csv(topo_path, dtype=str)
    topo["link_id"] = topo.link_id.astype(str)
    topo = topo.drop_duplicates("link_id").set_index("link_id")

    pivot = state.pivot_table(index="timestamp", columns="link_id",
                              values="flow_vph", aggfunc="mean")
    known = set(pivot.columns.astype(str))
    rows = []
    for link in state.link_id.astype(str).drop_duplicates():
        if link not in pivot.columns:
            continue
        incoming = []
        if link in topo.index:
            incoming = [x for x in str(topo.loc[link].get("incoming_link_ids", "")).split(";")
                        if x in known]
        inflow = pivot[incoming].sum(axis=1) if incoming else pivot[link]
        if outflow_mode == "self":
            outflow = pivot[link]
        else:
            outgoing = []
            if link in topo.index:
                outgoing = [x for x in str(topo.loc[link].get("outgoing_link_ids", "")).split(";")
                            if x in known]
            outflow = pivot[outgoing].sum(axis=1) if outgoing else pivot[link]
        rows.append(pd.DataFrame({"timestamp": pivot.index, "link_id": link,
                                  "inflow_vph": inflow.to_numpy(),
                                  "outflow_vph": outflow.to_numpy()}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["timestamp", "link_id", "inflow_vph", "outflow_vph"])


def truth_state(release: Path, panel: str, split: str = "train") -> pd.DataFrame:
    """The real per-link flow field, one row per (timestamp, link)."""
    root = release / "corridors" / panel / split / "mainline_states"
    paths = [p for p in sorted(root.glob("**/*.parquet")) if is_eval_file(p)]
    if not paths:
        raise FileNotFoundError(f"{panel}: no unmasked partitions in the eval month")
    pieces = [
        pd.read_parquet(p, columns=["timestamp", "link_id", "flow_vph", "is_score_eligible"])
        for p in paths
    ]
    frame = pd.concat(pieces, ignore_index=True)
    frame["link_id"] = frame.link_id.astype(str)
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    frame = frame[frame.is_score_eligible.astype(bool)]
    frame["flow_vph"] = pd.to_numeric(frame.flow_vph, errors="coerce")
    frame = frame.dropna(subset=["flow_vph"])
    # Several detector stations can sit on one link; the evaluator also averages.
    return frame.groupby(["timestamp", "link_id"], as_index=False).flow_vph.mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", type=Path, required=True)
    ap.add_argument("--panel", action="append", default=None)
    ap.add_argument("--split", default="train", choices=["train", "validation", "private"])
    ap.add_argument("--outflow-mode", default="self", choices=["self", "downstream"],
                    help="'self' (default) treats a link's outflow as its own measured "
                         "flow; 'downstream' reproduces the upstream fallback")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    release = args.release_root.resolve()

    from holdout import panels as known_panels
    panels = args.panel or known_panels(release)

    out = []
    for panel in panels:
        state = truth_state(release, panel, args.split)
        flux = boundary_from_topology(state, release / "corridors" / panel,
                                      args.outflow_mode)
        flux["panel"] = panel
        out.append(flux)
        print(f"  {panel}: {len(flux):,} flux rows", flush=True)

    frame = pd.concat(out, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    print(f"wrote {args.output}  ({len(frame):,} rows)")


if __name__ == "__main__":
    main()
