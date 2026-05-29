<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Deterministic Training

Deterministic training guarantees that two runs with identical inputs produce identical outputs at every step. Useful for debugging regressions and for reproducibility studies.

Pass `--deterministic-mode` to any Megatron training entry point (e.g. `pretrain_gpt.py`):

```bash
python pretrain_gpt.py \
  --deterministic-mode \
  <other args ...>
```

When enabled, Megatron applies the env vars and config overrides below via `megatron.training.determinism.apply_determinism_to_args` (called from `validate_args`).

## Environment variables

Populated with `os.environ.setdefault`, so a user-supplied value wins. Must be set before the first cuBLAS / Transformer Engine call — `apply_determinism_to_args` runs early in `validate_args` to guarantee this.

| Variable | Value | Reason |
|---|---|---|
| `NCCL_ALGO` | `Ring` | Disables tree/NVLS collectives that are non-deterministic |
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | `0` | Forces Transformer Engine to use deterministic algorithms |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | Disables cuBLAS heuristic workspace selection |

If you override `NCCL_ALGO` yourself, the value must be a comma-separated subset of `{Ring, Tree, CollnetDirect, CollnetChain}`. NCCL exclusion syntax (`^XXX`) is rejected — it does not pin a deterministic algorithm.

## Config overrides

Applied to the parsed `args` Namespace in `apply_determinism_to_args`:

| Flag | Behavior under `--deterministic-mode` |
|---|---|
| `--cross-entropy-loss-fusion` | Must be off (asserted; fused CE is non-deterministic) |
| `--tp-comm-overlap` | Forced off (the overlap path uses non-deterministic NCCL collectives) |
| `torch.use_deterministic_algorithms` | Set to `True` |

Flash attention is permitted: Transformer Engine's flash-attention backend is deterministic when `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` (see the [Transformer Engine docs](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html)).

## Verifying determinism

The bit-exact correctness suite lives at `tests/unit_tests/determinism/correctness/`. It parametrizes over model presets (GPT-like, Llama-like, Hybrid/Mamba) × parallelism cells (TP, PP, VPP, EP, FSDP, and composites) and asserts that two runs of the same configuration produce bit-identical outputs and gradients. The cost of `--deterministic-mode` (forward / backward / optimizer wall-clock and peak memory) is measured by the leaderboard tests at `tests/unit_tests/determinism/perf/`.
