"""Regenerate every figure and table of the TSB-SEG paper.

    python -m analysis.paper            # from benchmark/, or: make paper

Reads ``TSSEG_ANALYSIS_DB`` (default ``results/results.db``, built by
``scripts/build_results_db.py``) and writes to ``TSSEG_ANALYSIS_OUT`` (default
``figures/paper``). Each file the paper includes is listed in
``PAPER_FILES``; the run fails if one of them was not produced.
"""

from __future__ import annotations

from pathlib import Path

from analysis import (
    data,
    excluded_algos_analysis,
    feature_sensitivity,
    hyperparameters_figs,
    main_pool_grid_table,
    metric_correlations,
    param_grid_table,
    performance,
    rq1_extra,
    rq2_figure,
    rq3_figure,
    runtime,
    runtime_split,
    score_runtime_split,
)
from analysis.helpers import OUTPUT_DIR

PAPER_FILES = Path(__file__).with_name("paper_files.txt")


def main() -> None:
    d = data.load_data()
    performance.make_all_figures(d)  # CD diagrams, spiders, strips
    rq1_extra.make_all(d)  # Default-mode CDs, dataset boxplots, oracle table
    rq2_figure.make_figure(d)  # Figure 5
    rq3_figure.make_figure(d)  # Figure 6
    runtime.make_figure(d)  # SD runtime-accuracy panels
    runtime_split.make_figure(d)
    score_runtime_split.make_figure(d)  # PAMAP2 case study
    feature_sensitivity.make_figure(d)
    metric_correlations.make_figure(d)
    hyperparameters_figs.main(d)
    tex = param_grid_table.build_table(d)
    (OUTPUT_DIR / "table_param_grids.tex").write_text(tex + "\n")
    main_pool_grid_table.main(d)
    excluded_algos_analysis.main()

    expected = [
        line.strip()
        for line in PAPER_FILES.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    missing = [f for f in expected if not (OUTPUT_DIR / f).is_file()]
    if missing:
        raise SystemExit(f"not produced: {missing}")
    print(f"{len(expected)} paper files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
