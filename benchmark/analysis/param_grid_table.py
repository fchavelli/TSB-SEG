"""Hyperparameter grid of every TSB-SEG method, as actually evaluated.

Reads the grids from the runs behind the paper (the ``grid_*`` parameters of
the grid parents in ``df_grid_{ng,g}``, 8 datasets), not from
``configs/algorithm/*.yaml``, whose grid is not the one that ran when a method
was rerun in one mode only. The default of each parameter is read the same
way, from the parents of the default runs (``algo_*`` parameters of
``df_default_{ng,g}``).

One row per (method, parameter): the swept parameters, plus the parameter
that receives the number of events ``K`` in Guided mode (the aliases of
``tsbseg.utils.main_helpers``). In the Guided column, ``K`` marks that
parameter and "-" a Non-Guided stopping rule that ``K`` replaces (the
``unsupervised_only_parameters`` of the configs, e.g. the penalties).

Output: ``table_param_grids.tex`` (a ``longtable``).
"""

from __future__ import annotations

import pandas as pd
import yaml

from analysis.data import _ALGO_CONFIG_DIR
from analysis.helpers import ALGO_DISPLAY, MAIN_DATASETS, analyzer
from tsbseg.utils.main_helpers import (
    CHANGE_POINT_PARAM_ALIASES,
    SEGMENT_COUNT_PARAM_ALIASES,
    STATE_COUNT_PARAM_ALIASES,
)

# Rows of the table, in the order of the taxonomy (Section 3.1).
BLOCKS = [
    (
        "CPD, distribution-based",
        [
            "amoc",
            "binseg",
            "bottomup",
            "window",
            "changefinder",
            "pelt",
            "dynp",
            "prophet",
            "icid",
            "ggs",
        ],
    ),
    ("CPD, subsequence-based", ["igts", "fluss", "espresso", "tire", "clasp"]),
    ("SD", ["autoplait", "hdp-hsmm", "ticc", "time2state", "e2usd", "clap"]),
    ("Extended-scope pool", ["bocd", "eagglo", "tscp2", "tglad", "hidalgo"]),
]

# Parameters the pipeline fills with K in Guided mode, and the K they get.
_K_SYMBOL = {
    **{a: r"$n_{CP}$" for a in CHANGE_POINT_PARAM_ALIASES},
    **{a: r"$n_{CP}{+}1$" for a in SEGMENT_COUNT_PARAM_ALIASES},
    **{a: r"$n_S$" for a in STATE_COUNT_PARAM_ALIASES},
}

# Methods run in their default configuration only (no grid): the parameters
# that drive them, shown with their default value.
_DEFAULT_ONLY = {
    "bocd": ["hazard_lambda", "cp_prob_threshold", "min_distance"],
    "eagglo": ["alpha", "penalty"],
    "tscp2": ["window_size", "code_size", "epochs", "temperature"],
}

# Logged grid axes that are bookkeeping, not hyperparameters.
_IGNORED = {"execution_role"}


def _grid_values(df_grid: pd.DataFrame) -> tuple[dict, dict]:
    """``{algo: {param: sorted values}}`` and ``{algo: configs per dataset}``."""
    g = df_grid[df_grid["dataset"].isin(MAIN_DATASETS)]
    parents = g[["parent_run_id", "algorithm"]].drop_duplicates()
    p = analyzer.manager.fetch_run_params(list(parents["parent_run_id"].dropna()))
    p = p[p["key"].str.startswith("grid_")].copy()
    p["param"] = p["key"].str[len("grid_") :]
    p = p[~p["param"].isin(_IGNORED)]
    p = p.merge(parents, left_on="run_id", right_on="parent_run_id")
    values = {
        algo: {k: _sorted(v.unique()) for k, v in sub.groupby("param")["value"]}
        for algo, sub in p.groupby("algorithm")
    }
    n_cfg = g.groupby(["algorithm", "dataset"])["parent_run_id"].nunique()
    return values, n_cfg.groupby("algorithm").max().to_dict()


def _defaults(df_default: pd.DataFrame) -> dict:
    """``{algo: {param: value}}`` from the parents of the default runs."""
    g = df_default[df_default["dataset"].isin(MAIN_DATASETS)]
    parents = g[["parent_run_id", "algorithm"]].drop_duplicates()
    p = analyzer.manager.fetch_run_params(list(parents["parent_run_id"].dropna()))
    p = p[p["key"].str.startswith("algo_")].copy()
    p["param"] = p["key"].str[len("algo_") :]
    p = p.merge(parents, left_on="run_id", right_on="parent_run_id")
    out = {}
    for algo, sub in p.groupby("algorithm"):
        vals = sub.groupby("param")["value"].agg(lambda s: _sorted(s.unique()))
        out[algo] = {k: v[0] if len(v) == 1 else v for k, v in vals.items()}
    return out


def _sorted(values) -> list[str]:
    def key(v):
        try:
            return (0, float(v), "")
        except ValueError:
            return (1, 0.0, v)

    return sorted(values, key=key)


# Categorical values as the paper writes them, per parameter (the code names
# are terse: "l2" is a segment cost for the ruptures detectors but the L2-norm
# reduction for multivariate_strategy).
_DISPLAY = {
    "model": {"l1": r"$\ell_1$", "l2": r"$\ell_2$", "rbf": "RBF", "cosine": "cosine"},
    "multivariate_strategy": {"l2": "L2 norm", "ensembling": "ensembling"},
    "distance": {
        "euclidean_distance": "euclidean",
        "znormed_euclidean_distance": "z-norm.",
        "cinvariant_euclidean_distance": "CID",
    },
    "window_size": {"suss": "SuSS", "fft": "FFT", "acf": "ACF"},
    "classifier": {"rocket": "ROCKET", "mrhydra": "MR-Hydra"},
    "penalty": {"None": "none"},  # EAgglo: no penalty on the number of segments
}


def _fmt_value(v, param: str = "") -> str:
    if isinstance(v, list):  # default runs that differ across datasets
        return "/".join(_fmt_value(x, param) for x in v)
    if v in _DISPLAY.get(param, {}):
        return _DISPLAY[param][v]
    if v is None or v == "None":
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v).replace("_", r"\_")
    if f == int(f) and "e" not in str(v):
        return str(int(f))
    return f"{f:g}"


def _fmt_set(values: list[str], param: str = "") -> str:
    if len(values) == 1:
        return _fmt_value(values[0], param)
    order = list(_DISPLAY.get(param, {}))
    if order and all(v in order for v in values):
        values = sorted(values, key=order.index)
    return r"\{" + ", ".join(_fmt_value(v, param) for v in values) + r"\}"


def _k_param(algo: str, defaults: dict) -> str | None:
    """The parameter that receives K, if the method takes one."""
    names = [k for k in defaults.get(algo, {}) if k in _K_SYMBOL]
    if not names:
        return None
    # A parameter with a Non-Guided value (FLUSS n_segments = 5, ClaSP
    # n_segments = "learn") is the one a reader needs to see.
    set_ = [k for k in names if defaults[algo][k] not in (None, "None")]
    return sorted(set_ or names)[0]


def _rows(algo, ng, g, n_ng, n_g, dflt_ng, dflt_g) -> list[tuple[str, str, str, str]]:
    """(parameter, default, NG values, G values) for one method."""
    runs_ng = algo in n_ng or algo in dflt_ng
    runs_g = algo in n_g or algo in dflt_g
    d = {**dflt_g.get(algo, {}), **dflt_ng.get(algo, {})}
    k = _k_param(algo, {algo: d})
    swept = {
        p for grid in (ng.get(algo, {}), g.get(algo, {})) for p, v in grid.items() if len(v) > 1
    }
    params = (
        sorted(swept)
        + [p for p in _DEFAULT_ONLY.get(algo, []) if p not in swept]
        + ([k] if k and k not in swept else [])
    )
    na = r"\multicolumn{1}{c}{\emph{n/a}}"
    rows = []
    for p in params:
        # The K parameter of a Guided-only method has no Default-mode value.
        default = "-" if p == k and not runs_ng else _fmt_value(d.get(p), p)
        if not runs_ng:
            ng_cell = na
        else:
            v = ng.get(algo, {}).get(p)
            ng_cell = _fmt_set(v, p) if v else default
        if not runs_g:
            g_cell = na
        elif p == k:
            g_cell = _K_SYMBOL[p]
        else:
            v = g.get(algo, {}).get(p)
            fixed = _guided_instance(algo).get(p)
            # A swept Non-Guided stopping rule that K replaces: pinned to one value
            # when K is given (unsupervised_only_parameters), or not logged by
            # guided runs older than the parameter.
            replaced = (
                k is not None
                and fixed is None
                and (v is None or len(v) == 1)
                and len(ng.get(algo, {}).get(p, [])) > 1
            )
            if v is None and fixed is not None:
                # Guided runs older than the parameter: the value they used, as
                # recorded by the config's guided_instance (ChangeFinder: k = 2).
                g_cell = _fmt_value(str(fixed), p)
            else:
                g_cell = "-" if replaced else (_fmt_set(v, p) if v else default)
            if runs_ng and not replaced and g_cell == ng_cell:
                g_cell = "same"
        rows.append((r"\texttt{" + p.replace("_", r"\_") + "}", default, ng_cell, g_cell))
    return rows


def _guided_instance(algo: str) -> dict:
    """``guided_instance`` of the algorithm's config: values its guided runs used."""
    path = _ALGO_CONFIG_DIR / f"{algo}.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return dict(yaml.safe_load(f).get("guided_instance") or {})


def _n_cfg(n: dict, algo: str) -> str:
    return str(n[algo]) if algo in n else "-"


def build_table(d) -> str:
    ng, n_ng = _grid_values(d.df_grid_ng)
    g, n_g = _grid_values(d.df_grid_g)
    dflt_ng = _defaults(d.df_default_ng)
    dflt_g = _defaults(d.df_default_g)

    head = (
        r"\textbf{Method} & \textbf{\#cfg} & \textbf{Parameter} & \textbf{Default} "
        r"& \textbf{Non-Guided values} & \textbf{Guided values} \\"
    )
    lines = [
        "% Auto-generated by analysis/param_grid_table.py from the MLflow runs.",
        r"\begingroup",
        r"\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.05}\footnotesize",
        r"\begin{longtable}{@{}l c l l >{\raggedright\arraybackslash}p{3.2cm} "
        r">{\raggedright\arraybackslash}p{2.6cm}@{}}",
        r"\caption{\textbf{Hyperparameter grids, as evaluated.} For each method, the "
        r"parameters swept in Parameter-search mode, their value in Default mode and the "
        r"values tried in each mode; \#cfg is the number of configurations per dataset "
        r"(Non-Guided/Guided). In Guided mode the parameter marked with $n_{CP}$, "
        r"$n_{CP}{+}1$ (number of segments) or $n_S$ receives the ground-truth number of "
        r"events of each series, and ``-'' marks a Non-Guided stopping rule that it "
        r"replaces; ``same'': the Non-Guided values. The penalties of PELT, BinSeg, BottomUp and Window are multiples $X$ "
        r"of the BIC penalty $u \log n$ (Appendix~\ref{app:param_grids}). Time2State and "
        r"E2USD skip the window~64/step~100 combinations (step larger than the window). "
        r"PELT, AMOC, iCID, EAgglo and tGLAD take no $K$: their Guided grid is their "
        r"Non-Guided one (PELT and AMOC are not rerun). ClaSP distances: euclidean, "
        r"z-normalised euclidean and complexity-invariant (CID) euclidean; window-size "
        r"selection: SuSS, FFT or ACF; L2 norm and ensembling: the multivariate "
        r"adaptations of Appendix~\ref{app:mts_adapt}. tGLAD's default "
        r"runs used 100 epochs, 2000 on ActRecTut, MoCap and SKAB. \#cfg ``-'': "
        r"default configuration only (BOCD, EAgglo, TS-CP$^2$: main parameters shown); "
        r"\emph{n/a}: mode not evaluated.}"
        r"\label{tab:param_grids} \\",
        r"\toprule",
        head,
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{6}{l}{\emph{(continued from previous page)}} \\",
        r"\toprule",
        head,
        r"\midrule",
        r"\endhead",
        r"\midrule \multicolumn{6}{r}{\emph{(continued on next page)}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for b, (title, algos) in enumerate(BLOCKS):
        if b:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{6}}{{l}}{{\emph{{{title}}}}} \\*")
        lines.append(r"\midrule")
        for i, algo in enumerate(algos):
            name = ALGO_DISPLAY.get(algo, algo)
            cfg = f"{_n_cfg(n_ng, algo)}/{_n_cfg(n_g, algo)}"
            rows = _rows(algo, ng, g, n_ng, n_g, dflt_ng, dflt_g)
            if not rows:
                rows = [
                    (r"\multicolumn{4}{l}{\emph{default configuration only}}", None, None, None)
                ]
            for j, (p, dv, nv, gv) in enumerate(rows):
                first = (name, cfg) if j == 0 else ("", "")
                end = r"\\" if j == len(rows) - 1 else r"\\*"
                if dv is None:
                    lines.append(f"{first[0]} & {first[1]} & {p} {end}")
                else:
                    lines.append(f"{first[0]} & {first[1]} & {p} & {dv} & {nv} & {gv} {end}")
            if i < len(algos) - 1:
                lines.append(r"\addlinespace[2pt]")
    lines += [r"\end{longtable}", r"\endgroup"]
    return "\n".join(lines)
