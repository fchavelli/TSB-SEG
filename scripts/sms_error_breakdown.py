#!/usr/bin/env python
"""SMS error-type breakdown of the Default state-detection runs (Fig. 5c, 6c).

Figures 5c and 6c split the SMS penalty of every Default run of the
state-detection methods into its four error types (delay, transition,
isolation, missing). The breakdown needs the predicted labels, which the run
artifacts keep (they are not part of the published results), so it is
computed in two steps.

1. ``manifest``: the Default SD runs that the analysis reads
   (``analysis.error_mix.default_sd_runs``), with their logged SMS; the
   figures refuse a breakdown of any other set of runs::

       python scripts/sms_error_breakdown.py manifest -o sms_manifest.csv

2. ``compute``, where the artifacts are (``scripts/sms_error_breakdown.sbatch``
   on a SLURM cluster)::

       python scripts/sms_error_breakdown.py compute sms_manifest.csv \\
           -o sms_error_breakdown.parquet --artifacts <mlartifacts> ... --data data

   Ground truth comes from the data loader (``trial_index`` is 1-based, in the
   order of ``data_loading/series.csv``). Two checks per run, and nothing is
   written if either fails:

   - the change points of the loaded labels equal the run's
     ``ground_truth_change_points.npy`` (right series);
   - the recomputed SMS equals the logged one (same predictions, same metric).

The output goes to ``results/sms_error_breakdown.parquet``
(``analysis.helpers.SMS_ERRORS``).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SMS_WEIGHTS = {"delay": 0.1, "transition": 0.3, "isolation": 0.8, "missing": 0.5}
ERROR_TYPES = ("delay", "transition", "isolation", "missing")
MANIFEST_COLS = [
    "run_uuid",
    "experiment_id",
    "algorithm",
    "dataset",
    "trial_index",
    "supervision",
    "sms_logged",
]


# ── manifest ─────────────────────────────────────────────────────────────


def build_manifest(out: Path) -> None:
    sys.path.insert(0, str(ROOT / "benchmark"))
    from analysis.data import GUIDED_FROM_NON_GUIDED, load_data
    from analysis.error_mix import default_sd_runs
    from analysis.helpers import METRIC_SD

    man = default_sd_runs(load_data()).rename(
        columns={"run_id": "run_uuid", METRIC_SD: "sms_logged"}
    )
    man["trial_index"] = man["trial_index"].astype(int)
    if man.duplicated(["run_uuid", "supervision"]).any():
        raise SystemExit("a run appears twice among the Default SD runs of one mode")
    # The guided rows of GUIDED_FROM_NON_GUIDED (autoplait among the SD methods)
    # are copies of the non-guided runs: same run_uuid, one row per mode.
    shared = man[man["run_uuid"].duplicated(keep=False)]
    if not shared["algorithm"].isin(GUIDED_FROM_NON_GUIDED).all():
        raise SystemExit("a run is shared by both modes outside GUIDED_FROM_NON_GUIDED")
    man = man[MANIFEST_COLS].sort_values(["algorithm", "supervision", "dataset", "trial_index"])
    man.to_csv(out, index=False)
    counts = man.groupby(["algorithm", "supervision"]).size().unstack(fill_value=0)
    print(counts.to_string())
    print(f"{len(man)} runs -> {out}")


# ── compute ──────────────────────────────────────────────────────────────


def locate_artifacts(run_ids: list[str], roots: list[Path]) -> dict[str, Path]:
    """``{run_uuid: artifacts dir}``, first root first.

    A run may sit under another experiment id than in the results (a run
    copied from another tracking store): every experiment directory of
    every root is searched.
    """
    found: dict[str, Path] = {}
    for root in roots:
        for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for run in run_ids:
                if run in found:
                    continue
                art = exp_dir / run / "artifacts"
                if (art / "predicted_labels.npy").is_file():
                    found[run] = art
    return found


def load_ground_truth(datasets: list[str], data_root: Path) -> dict[str, list]:
    from tsbseg.data_loading.loaders import load_dataset

    gt = {}
    for ds in sorted(datasets):
        _, ys = load_dataset(ds, data_root=str(data_root), return_X_y=True)
        gt[ds] = [np.asarray(ys)] if isinstance(ys, np.ndarray) else [np.asarray(y) for y in ys]
        print(f"  loaded {ds}: {len(gt[ds])} series", flush=True)
    return gt


def breakdown(job: tuple) -> dict:
    from tsseg.metrics.state_detection import _state_matching_score

    row, art, y_true = job
    y_pred = np.load(art / "predicted_labels.npy", allow_pickle=False)
    cp_true = set((np.flatnonzero(np.diff(y_true)) + 1).tolist())
    cp_run = np.load(art / "ground_truth_change_points.npy", allow_pickle=False)
    cp_run = set(cp_run.astype(int).tolist()) - {0, len(y_true)}
    n = min(len(y_true), len(y_pred))
    score, errors = _state_matching_score(
        y_true[:n], y_pred[:n], SMS_WEIGHTS, return_mapped=False, return_errors=True
    )
    count, size, penalty = defaultdict(int), defaultdict(int), defaultdict(float)
    for e in errors:
        count[e["type"]] += 1
        size[e["type"]] += e["size"]
        penalty[e["type"]] += e["penalty"]
    return {
        "run_uuid": row["run_uuid"],
        "algorithm": row["algorithm"],
        "dataset": row["dataset"],
        "trial_index": row["trial_index"],
        "supervision": row["supervision"],
        "n_timepoints": n,
        "sms_score": score,
        "n_errors_total": len(errors),
        **{f"n_{t}": count[t] for t in ERROR_TYPES},
        **{f"size_{t}": size[t] for t in ERROR_TYPES},
        **{f"penalty_{t}": penalty[t] for t in ERROR_TYPES},
        "_gt_ok": cp_run == cp_true,
        "_sms_ok": bool(np.isclose(score, row["sms_logged"], rtol=0, atol=1e-9)),
        "_len_pred": len(y_pred),
        "_len_true": len(y_true),
    }


def compute(manifest: Path, out: Path, roots: list[Path], data_root: Path, workers: int) -> None:
    man = pd.read_csv(manifest)
    print(f"{len(man)} runs in {manifest}", flush=True)
    arts = locate_artifacts(man["run_uuid"].tolist(), roots)
    missing = sorted(set(man["run_uuid"]) - set(arts))
    if missing:
        raise SystemExit(f"{len(missing)} runs without artifacts, e.g. {missing[:5]}")

    gt = load_ground_truth(man["dataset"].unique().tolist(), data_root)
    jobs = []
    for row in man.to_dict("records"):
        ys = gt[row["dataset"]]
        k = int(row["trial_index"]) - 1
        if not 0 <= k < len(ys):
            raise SystemExit(f"{row['run_uuid']}: trial_index {k + 1} out of range")
        jobs.append((row, arts[row["run_uuid"]], ys[k]))

    with Pool(workers) as pool:
        rows = []
        for i, r in enumerate(pool.imap(breakdown, jobs, chunksize=16)):
            rows.append(r)
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    df = pd.DataFrame(rows)

    bad_gt = df[~df["_gt_ok"]]
    bad_sms = df[~df["_sms_ok"]]
    resized = df[df["_len_pred"] != df["_len_true"]]
    print(
        f"runs {len(df)}; ground-truth mismatches {len(bad_gt)}; "
        f"SMS mismatches {len(bad_sms)}; prediction length != series length "
        f"{len(resized)}"
    )
    if len(bad_gt) or len(bad_sms):
        cols = ["run_uuid", "algorithm", "dataset", "trial_index", "sms_score"]
        print(bad_gt[cols].head(10).to_string())
        print(bad_sms[cols].head(10).to_string())
        raise SystemExit("checks failed: nothing written")
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")])
    df.sort_values(["algorithm", "supervision", "dataset", "trial_index"]).to_parquet(
        out, index=False
    )
    print(f"written {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("manifest", help="list the Default SD runs read by the analysis")
    p.add_argument("-o", "--out", type=Path, required=True)

    p = sub.add_parser("compute", help="breakdown from the run artifacts")
    p.add_argument("manifest", type=Path)
    p.add_argument("-o", "--out", type=Path, required=True)
    p.add_argument("--artifacts", type=Path, nargs="+", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)

    args = parser.parse_args()
    if args.command == "manifest":
        build_manifest(args.out)
    else:
        compute(args.manifest, args.out, args.artifacts, args.data, args.workers)


if __name__ == "__main__":
    main()
