"""Score any Task 1 submission on the holdout month.

This is the entry point you use while iterating: whatever produced the CSV -
the upstream baseline, a model of your own, a quick experiment - gets one
number that is comparable to the recorded anchors.

    python mywork/eval_holdout.py --submission path/to/state_submission.csv
    python mywork/eval_holdout.py --submission x.csv --panel D12_I5_N --panel D12_I5_S

The CSV needs the columns the leaderboard template uses:
    panel, timestamp, station_id, link_id, mask_regime, speed_kmh, flow_vph

Only rows inside 2031-02 are scored, and the profile this file was built with
must have been fitted on <= 2031-01-31, otherwise the number is contaminated.

Anchors (ten-panel mean, see EXPERIMENTS.md):
    zero 0.0000 | baseline 0.7306 | truth 1.0000
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

from holdout import EVAL_END, EVAL_START, holdout_scope
from holdout import panels as known_panels
from task1.baseline_task1_historical_mean import DEFAULT_RELEASE
from task1.score_task1 import read_submission, score_panel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=Path, required=True)
    ap.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE)
    ap.add_argument("--panel", action="append", default=None)
    ap.add_argument("--output", type=Path, default=None,
                    help="optional CSV to write the per-panel report to")
    args = ap.parse_args()
    release = args.release_root.resolve()

    submission, errors = read_submission(args.submission.resolve())
    for error in errors:
        print(f"WARNING: {error}")

    panels = args.panel or known_panels(release)
    rows = []
    for panel in panels:
        with holdout_scope(restrict_fit=False):
            scored = score_panel(panel, release, "train", submission)
        panel_row = next(r for r in scored if r["level"] == "panel")
        regimes = {r["regime"]: r["S_state"] for r in scored if r["level"] == "panel_regime"}
        rows.append({
            "panel": panel,
            "S_state": round(panel_row["S_state"], 4),
            "n_target": int(panel_row["n_target_cells"]),
            "n_missing": int(panel_row["n_missing_predictions"]),
            "R1": round(regimes.get("R1", float("nan")), 4),
            "R2": round(regimes.get("R2", float("nan")), 4),
            "R3": round(regimes.get("R3", float("nan")), 4),
        })

    report = pd.DataFrame(rows)
    print(report.to_string(index=False))
    print(f"\nmean S_state = {report.S_state.mean():.4f}   "
          f"(baseline anchor 0.7306, delta {report.S_state.mean() - 0.7306:+.4f})")
    print(f"eval window: {EVAL_START} .. {EVAL_END}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.output, index=False)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
