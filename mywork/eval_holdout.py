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

# Historical-mean baseline per panel on the holdout month (fit <= 2031-01-31).
# Compare against the mean of the panels you actually scored, not the ten-panel
# number: panels differ hugely (0.514 to 0.899), so scoring one corridor and
# reading a ten-panel delta tells you nothing.
PANEL_BASELINE = {
    "D7_I10_E": 0.7207, "D7_I10_W": 0.8108, "D7_I210_E": 0.8988,
    "D7_I210_W": 0.7890, "D7_I405_N": 0.5140, "D7_I405_S": 0.6535,
    "D12_I5_N": 0.6572, "D12_I5_S": 0.6359, "D12_I405_N": 0.8341,
    "D12_I405_S": 0.7919,
}


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
    report["baseline"] = report.panel.map(PANEL_BASELINE)
    report["delta"] = (report.S_state - report.baseline).round(4)
    print(report.to_string(index=False))

    anchor = float(report.baseline.mean())
    current = float(report.S_state.mean())
    print(f"\nmean S_state = {current:.4f}   baseline({len(report)} panel(s)) = {anchor:.4f}"
          f"   delta = {current - anchor:+.4f}")
    print(f"eval window: {EVAL_START} .. {EVAL_END}")
    worst = report.nsmallest(1, "delta")
    if not worst.empty and worst.delta.iloc[0] < -0.02:
        print(f"NOTE: {worst.panel.iloc[0]} dropped {abs(worst.delta.iloc[0]):.4f} - "
              "check per-panel rows, the mean hides local regressions.")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.output, index=False)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
