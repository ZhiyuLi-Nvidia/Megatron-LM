# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Determinism-perf pytest plumbing.

Each pytest invocation measures ONE mode (``det`` or ``nondet``) — the env
vars that pin cuBLAS / Transformer Engine to deterministic algorithms are
sticky once kernels run, so we can't compare modes in a single process.
``DETERMINISM_PERF_MODE=det|nondet`` selects the mode at process start;
``tests/unit_tests/determinism/perf/__init__.py`` reads it to gate the
``set_determinism_env_vars`` call.

Rows appended to ``determinism_leaderboard`` look like
``{"kind": "perf-det", "cell": "<id>::<label>", "ms_per_iter": float,
"mem_mb": float}``. ``<label>`` is ``TOTAL.fwd`` or ``TOTAL.bwd`` — the
asserted gate. Per-NVTX-range attribution is produced offline by the
``determinism-perf.yaml`` recipe's ``nsys stats`` + ``print_nsys_leaderboard.py``
pipeline, NOT by this in-process leaderboard.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# Sticky leaderboard JSON path — each pytest invocation writes its own
# mode's file; if both files exist after a run, ``pytest_terminal_summary``
# prints a side-by-side comparison table.
_LEADERBOARD_DIR = Path(os.environ.get("DETERMINISM_PERF_LEADERBOARD_DIR", "/tmp"))


def pytest_configure(config):
    config._determinism_leaderboard = []


@pytest.fixture(scope="session")
def determinism_leaderboard(request):
    return request.config._determinism_leaderboard


def _is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _is_rank0():
        return
    rows = getattr(config, "_determinism_leaderboard", None)
    if not rows:
        return

    mode = os.environ.get("DETERMINISM_PERF_MODE", "det")
    terminalreporter.section(f"determinism perf leaderboard ({mode})")
    terminalreporter.write_line("| Cell | ms/iter | peak MB |")
    terminalreporter.write_line("|---|---|---|")
    for r in rows:
        terminalreporter.write_line(
            f"| {r['cell']} | {r['ms_per_iter']:.3f} | {r['mem_mb']:.1f} |"
        )

    # Persist this mode's leaderboard, then attempt the side-by-side join.
    _LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
    this_file = _LEADERBOARD_DIR / f"det-perf-{mode}.json"
    this_file.write_text(json.dumps(rows, indent=2))

    other_mode = "nondet" if mode == "det" else "det"
    other_file = _LEADERBOARD_DIR / f"det-perf-{other_mode}.json"
    if other_file.exists():
        other_rows = json.loads(other_file.read_text())
        _print_side_by_side(terminalreporter, this_rows=rows, this_mode=mode,
                            other_rows=other_rows, other_mode=other_mode)
    else:
        terminalreporter.write_line("")
        terminalreporter.write_line(
            f"Run DETERMINISM_PERF_MODE={other_mode} next; the join will "
            f"appear automatically when both {_LEADERBOARD_DIR}/det-perf-*.json exist."
        )


def _print_side_by_side(terminalreporter, *, this_rows, this_mode, other_rows, other_mode):
    """Render det vs nondet ms/iter per cell."""
    def _by_cell(rows):
        return {r["cell"]: r["ms_per_iter"] for r in rows}

    this_map = _by_cell(this_rows)
    other_map = _by_cell(other_rows)
    all_cells = sorted(set(this_map) | set(other_map))

    # Column order: det first, then nondet, regardless of which ran last.
    det_map = this_map if this_mode == "det" else other_map
    nondet_map = this_map if this_mode == "nondet" else other_map

    terminalreporter.section("determinism perf — det vs nondet")
    terminalreporter.write_line("| Cell | det ms | nondet ms | delta % |")
    terminalreporter.write_line("|---|---|---|---|")
    for cell in all_cells:
        det = det_map.get(cell)
        non = nondet_map.get(cell)
        delta = (
            f"{(det - non) / non * 100:+.2f}" if (det is not None and non is not None and non > 0)
            else "-"
        )
        det_s = f"{det:.3f}" if det is not None else "-"
        non_s = f"{non:.3f}" if non is not None else "-"
        terminalreporter.write_line(f"| {cell} | {det_s} | {non_s} | {delta} |")
