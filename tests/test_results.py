"""The published results (``results/``) are intact and rebuild the database the
analysis reads."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _manifest() -> dict:
    return json.loads((RESULTS / "manifest.json").read_text())


def test_checksums():
    for name, digest in _manifest()["sha256"].items():
        if name == "results.db":
            continue  # rebuilt locally, not published
        h = hashlib.sha256((RESULTS / name).read_bytes()).hexdigest()
        assert h == digest, name


def test_rebuilt_database_has_every_run(tmp_path):
    db = tmp_path / "results.db"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_results_db.py"),
            "--results",
            str(RESULTS),
            "--out",
            str(db),
        ],
        check=True,
    )
    with sqlite3.connect(db) as conn:
        counts = dict(
            conn.execute(
                "SELECT experiment_id, COUNT(*) FROM runs GROUP BY experiment_id"
            ).fetchall()
        )
    for exp in _manifest()["experiments"].values():
        assert counts[exp["id"]] == exp["runs"], exp["name"]
