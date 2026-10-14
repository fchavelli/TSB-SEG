"""Residual r = SMS - BiC of the SD runs, and its three zones (Figure 5).

Every run of the state-detection methods, Default and grid, Non-Guided and
Guided, is pooled; the envelope holds 95 % of the residuals (2.5 % in each
tail) and splits them into three zones around r = 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.helpers import ALGO_TASK, METRIC_CPD, METRIC_SD

# Zone codes, in order: below the envelope, inside it, above it. The paper
# figure (rq2_figure.py) draws both tails in dark ink and the inside in grey.
ZONES = ("under", "near", "over")
ZONE_PALETTE = ("#2b2a28", "#a8a7a1", "#2b2a28")


def zone_labels(
    delta_lo: float, delta_up: float, style: str = r"\scriptstyle "
) -> tuple[str, str, str]:
    """Labels of the three residual zones, carrying the envelope of the data."""
    lo, up = f"{delta_lo:.2f}", f"{delta_up:.2f}"
    return (rf"${style}r < -{lo}$", rf"${style}r \in [-{lo}, +{up}]$", rf"${style}r > +{up}$")


def pool_runs(d) -> pd.DataFrame:
    frames = []
    for src in (d.df_default_ng, d.df_grid_ng, d.df_default_g, d.df_grid_g):
        if src is None or len(src) == 0:
            continue
        cols = {"run_id", "algorithm", "dataset", METRIC_CPD, METRIC_SD}
        if not cols.issubset(src.columns):
            continue
        frames.append(src[list(cols)].copy())
    df = pd.concat(frames, ignore_index=True)
    sd_algos = {a for a, t in ALGO_TASK.items() if t == "SD"}
    df = df[df["algorithm"].isin(sd_algos)]
    df = df[df["dataset"].str.lower() != "pamap2"]
    df = df.dropna(subset=[METRIC_CPD, METRIC_SD, "run_id"])
    df = df[df[METRIC_CPD].between(0, 1) & df[METRIC_SD].between(0, 1)]
    return df.drop_duplicates(subset="run_id")


def envelope(df: pd.DataFrame) -> tuple[float, float]:
    delta_up = float(np.quantile(df[METRIC_SD] - df[METRIC_CPD], 0.975))
    delta_lo = float(np.quantile(df[METRIC_CPD] - df[METRIC_SD], 0.975))
    return delta_lo, delta_up


def assign_zone(df: pd.DataFrame, delta_lo: float, delta_up: float) -> pd.DataFrame:
    """Add ``residual`` and ``zone`` (one of ``ZONES``) columns."""
    under, near, over = ZONES

    def _z(r: float) -> str:
        if r < -delta_lo:
            return under
        if r > delta_up:
            return over
        return near

    df = df.copy()
    df["residual"] = df[METRIC_SD] - df[METRIC_CPD]
    df["zone"] = df["residual"].map(_z)
    return df
