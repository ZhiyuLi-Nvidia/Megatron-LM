# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Per-module determinism tests — top-level package.

The env vars required for bit-exact reproducibility are NOT set here, because
the perf subpackage needs a path that measures both deterministic and
non-deterministic kernels. Each subpackage (``correctness/``, ``perf/``) sets
its own env-var policy in its own ``__init__.py``:

* ``correctness/__init__.py`` — always sets env vars (bit-exact tests need them).
* ``perf/__init__.py``        — sets env vars only when
  ``DETERMINISM_PERF_MODE != 'nondet'``.

Setting the gate here would mean ``DETERMINISM_PERF_MODE=nondet pytest
tests/unit_tests/determinism/correctness/`` silently runs correctness tests
without the env vars (the parent ``__init__.py`` runs on import of any
subpackage). Keep the policy where the subpackage scope owns it.

``CUDA_DEVICE_MAX_CONNECTIONS=1`` is set here because it must be set BEFORE
the CUDA context is created (the driver reads it at context creation only;
later assignments are no-ops for the existing context). This is the earliest
point in the import graph where we can guarantee that.
"""

import os

# Must be set before CUDA context creation — sticky across the process.
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
