"""Three-point calibration of the local Task 1 evaluator.

Builds three submissions on the eval month and scores them with the upstream
evaluator restricted to that month:

    zero       every target cell filled with 0
    baseline   historical mean, profile fitted on the first eight months only
    truth      the unmasked observation layer copied straight in

The point is not to get a pretty number. It is to check that the evaluator spans
the right range and is anchored the same way the leaderboard is:

    input        leaderboard     local should be
    empty        0.0000          ~0
    baseline     0.55343         L0   <- the gap is holdout-vs-validation difficulty
    perfect      0.9945          ~0.99

If the two ends do not land, the evaluator is wrong and every offline decision
made with it is worthless. Fix it before tuning anything.

    python mywork/calibrate_t1.py --panel D12_I5_N
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
for _p in (_SRC, Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from holdout import EVAL_END, EVAL_START, holdout_scope, is_eval_file
from task1.baseline_task1_historical_mean import DEFAULT_RELEASE

OUTPUT_COLUMNS = ["panel", "timestamp", "station_id", "link_id", "mask_regime",
                  "speed_kmh", "flow_vph"]

# Leaderboard reference points for S_state specifically, from docs/BASELINES.md
# (ten-panel validation mean). Do not compare against 0.55343 / 0.9945 - those
# are S_total across all four tasks, not Task 1 alone.
ONLINE = {"zero": 0.0000, "baseline": 0.6929, "truth": 1.0000}
# The uploaded baseline scored 0.55343 overall; kept only as a reminder that a
# Task 1 gain has to be multiplied by 0.35 to reach the total.
S_STATE_WEIGHT = 0.35


def read_template(release: Path, panel: str) -> pd.DataFrame:
    path = release / "task1" / panel / "train" / "sample_submission_state.csv"
    if not path.exists():
        raise FileNotFoundError(f"no train template for {panel}: {path}")
    return pd.read_csv(path)


def eval_months(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows whose timestamp falls in the eval month."""
    day = frame.timestamp.astype(str).str[:10]
    return frame[(day >= EVAL_START) & (day <= EVAL_END)].copy()


def truth_lookup(release: Path, panel: str) -> pd.DataFrame:
    """True speed/flow for the eval month, keyed the way the template writes them."""
    from task1.score_task1 import release_files

    paths = [p for p in release_files(release, panel, "train") if is_eval_file(p)]
    if not paths:
        raise FileNotFoundError(f"{panel}: no unmasked partitions inside the eval month")
    frames = [
        pd.read_parquet(p, columns=["timestamp", "station_id", "link_id",
                                    "speed_kmh", "flow_vph", "is_score_eligible"])
        for p in paths
    ]
    frame = pd.concat(frames, ignore_index=True)
    frame["timestamp"] = frame.timestamp.astype(str)
    frame["station_id"] = frame.station_id.astype(str)
    frame["link_id"] = frame.link_id.astype(str)
    frame = frame[frame.is_score_eligible.astype(bool)]
    return frame.drop_duplicates(["timestamp", "station_id", "link_id"])


def build_zero(release: Path, panel: str) -> pd.DataFrame:
    frame = eval_months(read_template(release, panel))
    frame["speed_kmh"] = 0.0
    frame["flow_vph"] = 0.0
    return frame[OUTPUT_COLUMNS]


def build_truth(release: Path, panel: str) -> pd.DataFrame:
    frame = eval_months(read_template(release, panel))
    truth = truth_lookup(release, panel)
    merged = frame[["panel", "timestamp", "station_id", "link_id", "mask_regime"]].merge(
        truth[["timestamp", "station_id", "link_id", "speed_kmh", "flow_vph"]],
        on=["timestamp", "station_id", "link_id"], how="left",
    )
    return merged[OUTPUT_COLUMNS]


def build_baseline(release: Path, panel: str, out_dir: Path) -> pd.DataFrame:
    """Historical mean with the profile fitted on the first eight months only."""
    from task1.build_task1_baseline_submission import build_panel_submission

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{panel}_holdout_baseline.csv"
    if path.exists():
        path.unlink()
    # restrict_fit keeps build_profile inside the fit period; the rows it writes
    # still cover all of train, but only eval-month rows are scored later.
    with holdout_scope(restrict_fit=True):
        build_panel_submission(panel, release, "train", path, True)
    return pd.read_csv(path)


def score(release: Path, panel: str, frame: pd.DataFrame, out_dir: Path, name: str):
    from task1.score_task1 import read_submission, score_panel

    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f"{panel}_{name}.csv"
    frame[OUTPUT_COLUMNS].to_csv(csv, index=False)
    submission, errors = read_submission(csv)
    if errors:
        print(f"  warnings: {errors}")
    with holdout_scope(restrict_fit=False):
        rows = score_panel(panel, release, "train", submission)
    panel_row = next(r for r in rows if r["level"] == "panel")
    regimes = [r for r in rows if r["level"] == "panel_regime"]
    return {
        "S_state": panel_row["S_state"],
        "n_target": panel_row["n_target_cells"],
        "n_missing": panel_row["n_missing_predictions"],
        "by_regime": {r["regime"]: round(r["S_state"], 4) for r in regimes},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE)
    ap.add_argument("--panel", action="append", default=None,
                    help="score only selected panel(s); repeatable")
    ap.add_argument("--output-root", type=Path,
                    default=Path(__file__).resolve().parents[1] / "reports" / "calibrate")
    args = ap.parse_args()
    release = args.release_root.resolve()
    from holdout import panels as known_panels
    panels = args.panel or known_panels(release)
    out_dir = args.output_root.resolve()

    results = {}
    for panel in panels:
        print(f"[{panel}] building three submissions ...", flush=True)
        results[panel] = {}
        for name, builder in (("zero", build_zero), ("truth", build_truth),
                              ("baseline", None)):
            frame = build_baseline(release, panel, out_dir) if builder is None else builder(release, panel)
            results[panel][name] = score(release, panel, frame, out_dir, name)
            r = results[panel][name]
            print(f"  {name:9s} S_state={r['S_state']:.4f}  n={r['n_target']:,}  "
                  f"missing={r['n_missing']:,}  {r['by_regime']}", flush=True)

    print("\n=== Task 1 holdout calibration (fit<=8 months, eval=2031-02) ===")
    for name, online in ONLINE.items():
        scores = [results[p][name]["S_state"] for p in panels]
        local = float(np.mean(scores))
        flag = ""
        if name == "truth" and local < 0.90:
            flag = "  <-- truth should be near 1.0: evaluator is wrong"
        if name == "zero" and local > 0.05:
            flag = "  <-- zeros should score ~0: evaluator is wrong"
        print(f"  {name:9s} local={local:.4f}   online={online:.5f}   "
              f"delta={local - online:+.4f}{flag}")
    print(f"  panels: {', '.join(panels)}")


if __name__ == "__main__":
    main()
