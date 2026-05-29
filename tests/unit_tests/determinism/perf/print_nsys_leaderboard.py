# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Print side-by-side det vs nondet ``nsys`` per-NVTX-range comparison.

Reads ``<leaderboard_dir>/nsys-{det,nondet}.csv`` produced by
``nsys stats --report nvtx_sum --format csv`` in the SLURM driver.
Exits non-zero when either CSV is missing or empty so the recipe job
reflects a missing-data condition rather than reporting success.
"""

import csv
import sys
from pathlib import Path


def _load(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open() as fh:
        rows = list(csv.reader(fh))
    # Header detection: substring match on each cell (not list-membership)
    # so a future nsys rename like "NVTX Range" still matches. Pick the
    # FIRST row that has both a Range column and a Total Time column.
    def _is_header(r):
        return any("Range" in c for c in r) and any("Total Time" in c for c in r)

    hdr = next((i for i, r in enumerate(rows) if _is_header(r)), None)
    if hdr is None:
        return {}
    h = rows[hdr]
    ti = next((i for i, c in enumerate(h) if "Total Time" in c), None)
    ri = next((i for i, c in enumerate(h) if "Range" in c), None)
    if ti is None or ri is None:
        return {}
    out: dict[str, float] = {}
    for r in rows[hdr + 1:]:
        if len(r) <= max(ti, ri):
            continue
        name = r[ri].strip()
        if not name or name == "Range":  # skip blank rows + repeated header
            continue
        try:
            out[name] = float(r[ti].replace(",", "")) / 1e6
        except ValueError:
            continue  # non-numeric Total Time (e.g. footer / separator row)
    return out


d = Path(sys.argv[1] if len(sys.argv) > 1 else "logs/perf-leaderboards")
det, nondet = _load(d / "nsys-det.csv"), _load(d / "nsys-nondet.csv")
if not (det and nondet):
    print(f"ERROR: det={len(det)} rows, nondet={len(nondet)} rows; need both for join.")
    sys.exit(1)

print("| Range | det ms | nondet ms | delta % |\n|---|---|---|---|")
for k in sorted(set(det) | set(nondet), key=lambda k: -max(det.get(k, 0), nondet.get(k, 0))):
    dv, nv = det.get(k), nondet.get(k)
    delta = f"{(dv - nv) / nv * 100:+.2f}" if (dv is not None and nv is not None and nv > 0) else "-"
    ds = "-" if dv is None else f"{dv:.3f}"
    ns = "-" if nv is None else f"{nv:.3f}"
    print(f"| {k} | {ds} | {ns} | {delta} |")
