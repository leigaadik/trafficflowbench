"""One offline evaluation pass across all four tasks.

Everything here runs against the local holdout: Task 1 on 2031-02, Task 2 on the
train windows with reconstructed labels, Task 3 with conservation-solved
boundary fluxes. Task 4 can only report S_link.

    python mywork/evaluate.py --release-root data/kaggle_public \
        --state  reports/submit/state_submission.csv \
        --queue  reports/submit/queue_submission.csv \
        --odme   reports/submit/odme/baseline_submission.csv

Omit a task and it is skipped (and reported as skipped, never as 0). Use
--panel for a fast subset while iterating.

Read the numbers as relative, not absolute: the holdout month is easier than
validation (Task 1 reads ~0.038 high) and Task 3 has a dead zone, so watch
E_LWR rather than S_LWR.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
_ROOT = Path(__file__).resolve().parents[1]
for _p in (_SRC, Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from holdout import panels as known_panels

ANCHORS_PATH = Path(__file__).resolve().parent / "anchors.json"
WEIGHTS = {"state": 0.35, "queue": 0.30, "physics": 0.15, "odme": 0.20}
# S_ODME cannot be computed locally (no path-flow truth), so the total is
# reported with the leaderboard baseline value held constant and flagged.
ODME_PLACEHOLDER = 0.8359


def load_anchors() -> dict:
    if ANCHORS_PATH.exists():
        return json.loads(ANCHORS_PATH.read_text(encoding="utf-8"))
    return {}


def table(rows: list[dict], anchor_key: str | None, anchors: dict,
          value_col: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if anchor_key and anchors.get(anchor_key):
        base = anchors[anchor_key]
        frame["baseline"] = frame.panel.map(base)
        frame["delta"] = (frame[value_col] - frame.baseline).round(4)
    return frame


def eval_task1(release: Path, panels: list[str], submission: Path) -> pd.DataFrame:
    from task1.score_task1 import read_submission, score_panel
    from holdout import holdout_scope

    sub, errors = read_submission(Path(submission).resolve())
    for e in errors:
        print(f"  WARNING: {e}")
    rows = []
    for panel in panels:
        with holdout_scope(restrict_fit=False):
            scored = score_panel(panel, release, "train", sub)
        r = next(x for x in scored if x["level"] == "panel")
        rows.append({"panel": panel, "S_state": round(r["S_state"], 4),
                     "n_target": int(r["n_target_cells"])})
    return pd.DataFrame(rows)


def eval_task2(release: Path, panels: list[str], submission: Path,
               truth: Path, out_csv: Path) -> pd.DataFrame:
    cmd = [sys.executable, str(_SRC / "task2" / "score_task2.py"),
           "--submission", str(Path(submission).resolve()),
           "--release-root", str(release),
           "--split", "train",
           "--truth-file", str(Path(truth).resolve()),
           "--output", str(out_csv)]
    for p in panels:
        cmd += ["--panel", p]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    detail = pd.read_csv(out_csv)
    rows = detail[detail.level == "panel"][["panel", "IoU_ST"]].copy()
    rows["IoU_ST"] = rows.IoU_ST.round(4)
    return rows


def eval_task3(release: Path, panels: list[str], submission: Path,
               flux: Path) -> pd.DataFrame:
    from calibrate_t3 import score_one

    rows = []
    for panel in panels:
        s = score_one(release, panel, submission, flux)
        rows.append({"panel": panel, "S_physics": round(s["S_physics"], 4),
                     "S_FD": round(s["S_FD"], 4), "S_LWR": round(s["S_LWR"], 4),
                     "E_LWR": round(s["E_LWR"], 4)})
    return pd.DataFrame(rows)


def eval_task4(release: Path, panels: list[str], submission: Path,
               out_csv: Path) -> pd.DataFrame:
    cmd = [sys.executable, str(_SRC / "task4" / "score_task4.py"),
           "--submission", str(Path(submission).resolve()),
           "--release-root", str(release),
           "--split", "train",
           "--output", str(out_csv)]
    for p in panels:
        cmd += ["--panel", p]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    detail = pd.read_csv(out_csv)
    rows = detail[detail.level == "panel"][["panel", "S_link"]].copy()
    rows["S_link"] = rows.S_link.round(4)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", type=Path, default=Path("data/kaggle_public"))
    ap.add_argument("--state", type=Path, help="Task 1 submission CSV")
    ap.add_argument("--queue", type=Path, help="Task 2 submission CSV")
    ap.add_argument("--odme", type=Path, help="Task 4 submission CSV")
    ap.add_argument("--queue-truth", type=Path,
                    default=Path("reports/submit/queue_targets_train.parquet"))
    ap.add_argument("--boundary-flux", type=Path,
                    default=Path("reports/submit/flux_conservation_all.parquet"))
    ap.add_argument("--panel", action="append", default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    release = args.release_root.resolve()
    panels = args.panel or known_panels(release)
    anchors = load_anchors()
    out_dir = _ROOT / "reports" / "evaluate"
    out_dir.mkdir(parents=True, exist_ok=True)

    scores: dict[str, float] = {}
    frames: dict[str, pd.DataFrame] = {}

    if args.state:
        frames["state"] = eval_task1(release, panels, args.state)
        scores["state"] = float(frames["state"].S_state.mean())
        if not args.odme:
            pass
        # Task 3 is scored on the same file, so it comes along for free.
        if args.boundary_flux and args.boundary_flux.exists():
            frames["physics"] = eval_task3(release, panels, args.state,
                                           args.boundary_flux.resolve())
            scores["physics"] = float(frames["physics"].S_physics.mean())
        else:
            print(f"  (Task 3 skipped: boundary flux not found at {args.boundary_flux})")

    if args.queue:
        frames["queue"] = eval_task2(release, panels, args.queue,
                                     args.queue_truth.resolve(),
                                     out_dir / "task2_detail.csv")
        scores["queue"] = float(frames["queue"].IoU_ST.mean())

    if args.odme:
        frames["odme"] = eval_task4(release, panels, args.odme,
                                    out_dir / "task4_detail.csv")
        print("  (Task 4 reports S_link only; S_od/S_dev/S_attr need withheld "
              "path flows and cannot be scored locally)")

    print()
    for key in ("state", "queue", "physics", "odme"):
        if key not in frames:
            print(f"--- Task {key}: not evaluated")
            continue
        frame = frames[key]
        col = {"state": "S_state", "queue": "IoU_ST",
               "physics": "S_physics", "odme": "S_link"}[key]
        frame = table(frame.to_dict("records"), key, anchors, col)
        print(f"--- {key}")
        print(frame.to_string(index=False))
        if key in scores:
            base = anchors.get(key, {})
            if base:
                anchor_mean = sum(base.get(p, float("nan")) for p in frame.panel) / len(frame)
                if anchor_mean == anchor_mean:
                    print(f"    mean={scores[key]:.4f}  "
                          f"baseline={anchor_mean:.4f}  delta={scores[key] - anchor_mean:+.4f}")
            else:
                print(f"    mean={scores[key]:.4f}")
        print()

    if {"state", "queue", "physics"} <= set(scores):
        total = (WEIGHTS["state"] * scores["state"]
                 + WEIGHTS["queue"] * scores["queue"]
                 + WEIGHTS["physics"] * scores["physics"]
                 + WEIGHTS["odme"] * ODME_PLACEHOLDER)
        print("=== weighted total ===")
        for key in ("state", "queue", "physics"):
            print(f"  {WEIGHTS[key]:.2f} x {scores[key]:.4f} = "
                  f"{WEIGHTS[key] * scores[key]:.4f}   ({key})")
        print(f"  {WEIGHTS['odme']:.2f} x {ODME_PLACEHOLDER:.4f} = "
              f"{WEIGHTS['odme'] * ODME_PLACEHOLDER:.4f}   (odme, PLACEHOLDER - "
              f"online baseline, not measured)")
        print(f"  total (T4 held at baseline) = {total:.4f}")
        print("  Note: not comparable to the leaderboard. The holdout month is "
              "easier than validation, so this reads high.")

    if args.output:
        payload = {k: v.to_dict("records") for k, v in frames.items()}
        payload["scores"] = scores
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
