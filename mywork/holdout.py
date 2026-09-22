"""Local holdout split for offline evaluation.

    fit   train 前 8 个月   2030-06-01 .. 2031-01-31   用于学习 profile / 拟合模型
    eval  train 第 9 个月   2031-02-01 .. 2031-02-28   只用于评估，绝不参与拟合

Why the split exists at all
---------------------------
The historical-mean baseline learns its profile from the *unmasked* train layer
(``baseline_task1_historical_mean.files`` returns ``mainline_states`` first), and
that layer contains the true value of every masked target cell. Scoring on train
without a split therefore leaks: each target's own truth sits in the profile with
roughly a 1/39 weight. A holdout removes that, so offline numbers mean something.

The split never affects what gets uploaded
------------------------------------------
``submission_key.csv`` covers only ``validation`` and ``private`` - not a single
train row. Holding out a month changes how the model is fitted, not which cells
the leaderboard asks for. For the final submission, fit on all nine months.

How the split is applied
------------------------
Rather than copy the upstream evaluators, this module patches the two file
listers the upstream code calls, so the official scoring logic is reused verbatim
and stays correct when upstream is merged:

    baseline_task1_historical_mean.files  -> fit period only   (learning)
    score_task1.masked_files              -> eval period only  (scoring)
"""
from __future__ import annotations

import re
import sys
from contextlib import contextmanager
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

FIT_END = "2031-01-31"
EVAL_START = "2031-02-01"
EVAL_END = "2031-02-28"

# Daily partitions are named synthetic_mainline_YYYY_MM_DD.parquet.
_FILE_DATE = re.compile(r"_(\d{4})_(\d{2})_(\d{2})\.parquet$")


def file_date(path) -> str | None:
    """The calendar day a daily partition holds, or None if it is not one."""
    m = _FILE_DATE.search(Path(path).name)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"


def is_fit_file(path) -> bool:
    d = file_date(path)
    return d is not None and d <= FIT_END


def is_eval_file(path) -> bool:
    d = file_date(path)
    return d is not None and EVAL_START <= d <= EVAL_END


@contextmanager
def holdout_scope(restrict_fit: bool = True):
    """Restrict upstream file access to the holdout window.

    ``restrict_fit`` additionally keeps profile learning inside the fit period.
    Turn it off when you want the full nine months, e.g. for a final submission.
    """
    from task1 import baseline_task1_historical_mean as bhm
    from task1 import score_task1 as s1

    original_files = bhm.files
    original_masked = s1.masked_files

    if restrict_fit:
        bhm.files = lambda panel_dir, split: [
            p for p in original_files(panel_dir, split) if is_fit_file(p)
        ]
    # Scoring always runs on the eval month, whether or not fitting is restricted.
    s1.masked_files = lambda release, panel, split, regime: [
        p for p in original_masked(release, panel, split, regime) if is_eval_file(p)
    ]

    # Task 3 walks the whole split through its own imported `files`, so it needs
    # the same treatment or it would score nine months instead of the holdout.
    try:
        from task3 import score_task3 as s3
    except Exception:
        s3 = None
    original_s3_files = None
    if s3 is not None:
        original_s3_files = s3.files
        s3.files = lambda panel_dir, split: [
            p for p in original_s3_files(panel_dir, split) if is_eval_file(p)
        ]

    try:
        yield
    finally:
        bhm.files = original_files
        s1.masked_files = original_masked
        if s3 is not None:
            s3.files = original_s3_files


def panels(release: Path = None) -> list[str]:
    import json

    root = Path(release) if release else _SRC.parent
    manifest = json.loads((root / "config" / "corridors.json").read_text(encoding="utf-8"))
    return [p["corridor_id"] for p in manifest["panels"]]
