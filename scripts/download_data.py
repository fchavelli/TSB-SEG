#!/usr/bin/env python
"""Download the TSB-SEG datasets.

    python scripts/download_data.py              # every dataset
    python scripts/download_data.py mocap pump   # some datasets

The redistributable datasets are hosted on the Hugging Face Hub, one
repository each (``data/registry.json`` pins the revision and the sha256 of
every file). A dataset lands as ``data/<name>.parquet``, which the loader
reads directly. USC-HAD and UTSA have
no explicit license and are not redistributed: they are downloaded from their
source (USC archive, ClaP repository at a fixed commit) into ``data/usc-had/``
and ``data/utsa/``, in the layout the loader reads. Every file is checked
against its sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "registry.json"
HUB = "https://huggingface.co/datasets"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path, sha256: str) -> None:
    """Download ``url`` to ``dest`` unless it is already there; check its sha256."""
    if dest.is_file() and digest(dest) == sha256:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url)
    if url.startswith(HUB) and os.environ.get("HF_TOKEN"):
        # Only needed while a repository is private.
        request.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
    with (
        tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp,
        urllib.request.urlopen(request, timeout=120) as response,
    ):
        shutil.copyfileobj(response, tmp)
    tmp_path = Path(tmp.name)
    if digest(tmp_path) != sha256:
        tmp_path.unlink()
        raise SystemExit(f"{url}: sha256 mismatch")
    tmp_path.chmod(0o644)  # temporary files are created private
    tmp_path.replace(dest)


def extract(archive: Path, prefix: str, target: Path) -> None:
    """Extract the members of ``archive`` under ``prefix`` into ``target``."""
    target = target.resolve()
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            if member.is_dir() or not member.filename.startswith(prefix):
                continue
            dest = (target / member.filename.removeprefix(prefix)).resolve()
            if target not in dest.parents:
                raise SystemExit(f"{archive.name}: unsafe path {member.filename}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)


def download(name: str, entry: dict, root: Path) -> None:
    if entry["kind"] == "hf":
        for path, f in entry["files"].items():
            url = f"{HUB}/{entry['repo']}/resolve/{entry['revision']}/{path}"
            fetch(url, root / f["dest"], f["sha256"])
    elif entry["kind"] == "zip":
        archive = root / "data" / "downloads" / Path(entry["url"]).name
        fetch(entry["url"], archive, entry["sha256"])
        extract(archive, entry["member_prefix"], root / entry["target"])
    elif entry["kind"] == "files":
        for file, sha256 in entry["files"].items():
            fetch(entry["base_url"] + file, root / entry["target"] / file, sha256)
    else:
        raise SystemExit(f"{name}: unknown kind {entry['kind']}")
    print(f"{name}: ok ({entry['license']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="datasets (default: all)")
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args()
    registry = json.loads(REGISTRY.read_text())
    unknown = sorted(set(args.names) - set(registry))
    if unknown:
        raise SystemExit(f"unknown: {unknown}; known: {sorted(registry)}")
    names = args.names or sorted(registry)
    for name in names:
        download(name, registry[name], args.root)


if __name__ == "__main__":
    main()
