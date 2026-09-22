"""Build Task 2 queue truth for the train split, from the unmasked layer.

``score_task2.py`` needs a truth parquet with ``queue_true``, and the public
package ships none for any split. On train the unmasked observation layer does
carry the horizon - it is only the *masked* view that blanks the 30-minute
forecast span and the hour after it - so the labels can be reconstructed.

The frame is built on top of ``sample_submission_queue.csv`` so the key set is
identical to what the evaluator merges against; that merge uses
``validate="one_to_one"`` and aborts on any mismatch.

Caveat that limits how far this number can be trusted: the contract says labels
come from the *underlying state*, not the noisy observation. Only the noisy
observation is published, so labels near the cutoff can flip. The cutoff is
0.60 * free speed (~63-70 km/h) while typical noise is a few km/h, so the bias
is small but real. Treat the result as a relative signal, not an absolute one.

    python mywork/build_queue_truth.py --release-root data/kaggle_public \
        --output reports/submit/queue_targets_train.parquet
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

from task2.build_task2_persistence_submission import read_window_index
from task2.queue_utils import thresholds

DATE_IN_NAME = lambda name: name.replace("synthetic_mainline_", "").replace(".parquet", "")


def observation_field(release: Path, panel: str, split: str, dates: set[str]) -> pd.DataFrame:
    root = release / "corridors" / panel / split / "mainline_states"
    wanted = {d.replace("-", "_") for d in dates}
    pieces = []
    for path in sorted(root.glob("**/*.parquet")):
        stem = path.stem
        date = stem.replace("synthetic_mainline_", "")
        if date not in wanted:
            continue
        pieces.append(pd.read_parquet(
            path, columns=["timestamp", "link_id", "speed_kmh", "is_score_eligible"]))
    if not pieces:
        return pd.DataFrame(columns=["timestamp", "link_id", "speed_kmh", "is_score_eligible"])
    frame = pd.concat(pieces, ignore_index=True)
    frame["link_id"] = frame.link_id.astype(str)
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    frame["speed_kmh"] = pd.to_numeric(frame.speed_kmh, errors="coerce")
    return frame.groupby(["timestamp", "link_id"], as_index=False).agg(
        speed_kmh=("speed_kmh", "mean"),
        is_score_eligible=("is_score_eligible", "max"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", type=Path, required=True)
    ap.add_argument("--split", default="train", choices=["train", "validation", "private"])
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    release = args.release_root.resolve()
    windows = read_window_index(release, [args.split])
    if windows.empty:
        raise SystemExit(f"no Task 2 windows for split {args.split}")

    out = []
    for panel in sorted(windows.panel.astype(str).unique()):
        panel_dir = release / "corridors" / panel
        links = pd.read_csv(panel_dir / "network" / "links.csv")
        links["link_id"] = links.link_id.astype(str)
        cut, source = thresholds(panel_dir, links)
        w = windows[windows.panel.astype(str) == panel]
        obs = observation_field(release, panel, args.split, set(w.date.astype(str)))
        template = pd.read_csv(
            release / "task2" / panel / args.split / "sample_submission_queue.csv",
            usecols=["window_id", "timestamp", "link_id"])
        template["window_id"] = template.window_id.astype(str)
        template["link_id"] = template.link_id.astype(str)
        template["timestamp"] = pd.to_datetime(template.timestamp, utc=True)
        merged = template.merge(obs, on=["timestamp", "link_id"], how="left")
        merged["queue_true"] = merged.speed_kmh <= merged.link_id.map(cut).fillna(63.0)
        merged["panel"] = panel
        unmapped = int(merged.speed_kmh.isna().sum())
        print(f"  {panel}: {len(merged):,} cells, {unmapped:,} without truth "
              f"(threshold source: {source})", flush=True)
        out.append(merged[["panel", "window_id", "timestamp", "link_id",
                           "is_score_eligible", "queue_true"]])

    frame = pd.concat(out, ignore_index=True)
    frame["is_score_eligible"] = frame.is_score_eligible.fillna(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    print(f"wrote {args.output}  ({len(frame):,} rows, "
          f"queue_true rate {frame.queue_true.mean():.3f})")


if __name__ == "__main__":
    main()
