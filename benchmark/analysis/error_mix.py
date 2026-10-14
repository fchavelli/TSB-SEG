"""SMS error types: the breakdown of each Default SD run and its colours.

``results/sms_error_breakdown.parquet`` splits the SMS penalty of every Default run of the
state-detection methods into its four error types (Delay, Transition,
Isolation, Missing); Figures 5c and 6c show their average shares.
"""

from __future__ import annotations

import pandas as pd

from analysis.helpers import ALGO_TASK, METRIC_SD, SMS_ERRORS
from analysis.residual_dist import (
    ZONES,
)

PENALTY_COLS = ["penalty_delay", "penalty_transition", "penalty_isolation", "penalty_missing"]
PENALTY_LABELS = ["Delay", "Transition", "Isolation", "Missing"]
# The four error types in stacking order: muted tones that darken from the
# benign type to the severe ones, apart from the CPD/SD family colours (blue,
# orange) of the other figures. Worst pairwise CIELAB dE (Machado 2009
# simulation): deuteranopia 22, protanopia 24, tritanopia 35, normal vision 40.
PENALTY_COLORS = {
    "penalty_delay": "#9fc3c0",
    "penalty_transition": "#e6cf8f",
    "penalty_isolation": "#c96f7e",
    "penalty_missing": "#5e4a6e",
}
# Ink of a % label drawn on each segment colour.
PENALTY_TEXT = {
    "penalty_delay": "#0b0b0b",
    "penalty_transition": "#0b0b0b",
    "penalty_isolation": "#0b0b0b",
    "penalty_missing": "#ffffff",
}


DEFAULT_FRAMES = (("df_default_ng", "unsupervised"), ("df_default_g", "semi_supervised"))


def default_sd_runs(d) -> pd.DataFrame:
    """Default runs of the SD methods, both modes, with a ``supervision`` column.

    The runs whose SMS penalty Figures 5c and 6c break down; listed for
    ``scripts/sms_error_breakdown.py manifest``.
    """
    sd_algos = {a for a, task in ALGO_TASK.items() if task == "SD"}
    frames = []
    for name, supervision in DEFAULT_FRAMES:
        df = getattr(d, name)
        df = df[df["algorithm"].isin(sd_algos)].dropna(subset=[METRIC_SD]).copy()
        df["supervision"] = supervision
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def read_sms_breakdown(d) -> pd.DataFrame:
    """``SMS_ERRORS``, refused unless it covers exactly ``default_sd_runs(d)``."""
    sms = pd.read_parquet(SMS_ERRORS)
    want = set(default_sd_runs(d)["run_id"])
    have = set(sms["run_uuid"])
    if want != have:
        raise ValueError(
            f"{SMS_ERRORS} is stale: {len(want - have)} Default SD runs "
            f"missing, {len(have - want)} runs no longer read. Rerun "
            "scripts/sms_error_breakdown.py."
        )
    return sms


def penalty_mix_by_zone(df: pd.DataFrame, d) -> pd.DataFrame:
    """Average per-run penalty shares per residual zone (rows sum to 1).

    ``df`` carries ``run_id`` and ``zone`` (``assign_zone``). Runs with no
    SMS penalty (perfect score) have nothing to decompose and are dropped.
    """
    sms = read_sms_breakdown(d).rename(columns={"run_uuid": "run_id"})
    # A guided copy (data.GUIDED_FROM_NON_GUIDED) has the run_id of its
    # non-guided run: one row per run.
    sms = sms[["run_id"] + PENALTY_COLS].drop_duplicates("run_id")
    merged = df[["run_id", "zone"]].merge(sms, on="run_id", how="inner")
    totals = merged[PENALTY_COLS].sum(axis=1)
    merged = merged[totals > 0].copy()
    merged[PENALTY_COLS] = merged[PENALTY_COLS].div(totals[totals > 0], axis=0)
    return merged.groupby("zone")[PENALTY_COLS].mean().reindex(list(ZONES))
