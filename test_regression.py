"""The pipeline's output must not change unless a change is meant to change it.

* test_output_matches_snapshot: every benchmark drawing restores exactly as
  recorded in regression/snapshot/ (see regression_snapshot.py). After an
  intended change, record the new output with
      python regression_snapshot.py --rebless      (or REBLESS=1 python -m pytest test_regression.py)
  and list the differences in the PR.
* test_restoration_is_deterministic: the same image gives the same report
  twice in one process and in a fresh process.

Both need the benchmark images in test_images/ (not tracked by git) and are
skipped without them.
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest

import regression_snapshot as snap

ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGES = snap.benchmark_images() if os.path.isdir(os.path.join(ROOT, "test_images")) else []
needs_images = pytest.mark.skipif(not IMAGES, reason="benchmark images (test_images/) not present")


@needs_images
def test_output_matches_snapshot():
    from restoration.pipeline import default_workers, map_in_processes

    rebless = os.environ.get("REBLESS") == "1"
    failures = []
    for image, record, diffs in map_in_processes(snap._check_one, [(i,) for i in IMAGES], default_workers()):
        if rebless:
            snap.write_snapshot(record)
        elif diffs:
            failures.append(f"{image}: " + "; ".join(diffs[:3]))
    assert not failures, f"{len(failures)} image(s) differ from the snapshot:\n" + "\n".join(failures)


def _stable(report):
    """The report without anything that measures time."""
    if isinstance(report, dict):
        return {k: _stable(v) for k, v in report.items() if not (k.endswith("_s") or "time" in k)}
    if isinstance(report, list):
        return [_stable(v) for v in report]
    return report


_FRESH_PROCESS = """
import contextlib, io, json, sys, tempfile
from restoration.pipeline import restore
with tempfile.TemporaryDirectory() as out, contextlib.redirect_stdout(io.StringIO()):
    report = restore(sys.argv[1], output_dir=out, save_images=False, efd_closure=True).report
print(json.dumps(report))
"""


@needs_images
def test_restoration_is_deterministic():
    from restoration.pipeline import restore

    image = IMAGES[0]
    reports = []
    with tempfile.TemporaryDirectory() as out:
        for _ in range(2):
            result = restore(os.path.join(ROOT, image), output_dir=out, save_images=False, efd_closure=True)
            reports.append(_stable(json.loads(json.dumps(result.report))))
    fresh = subprocess.run(
        [sys.executable, "-c", _FRESH_PROCESS, os.path.join(ROOT, image)],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    reports.append(_stable(json.loads(fresh.stdout.strip().splitlines()[-1])))

    assert reports[0] == reports[1], "two runs in one process differ"
    assert reports[0] == reports[2], "a run in a fresh process differs"
