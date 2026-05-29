# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared config presets for parametrized determinism tests.

Each test in this package parametrizes its model factory with a list of
(name, overrides) pairs from below. Add a new entry here to widen the
determinism net to a new architecture variant — no changes needed in the
test files themselves.

Parallelism is expressed as a single composite dict like
``{"TP": 4, "FSDP": 2}`` or ``{"PP": 2, "VPP": 2, "EP": 4}``; see
``PARALLELISM_CONFIGS``. ``apply_parallelism`` translates the dict into
``Utils.initialize_model_parallel`` kwargs and returns flags for FSDP-wrap
and MoE auto-enable.
"""

import pytest
import torch

# ---------------------------------------------------------------------------
# Base configs — everything below is shared across presets. Override fields
# in the per-preset dict only when they differ from the base.
# ---------------------------------------------------------------------------

_BASE_GPT = dict(
    num_layers=2,
    hidden_size=64,
    ffn_hidden_size=128,  # default is 4*hidden=256; halve for cheaper MLP
    num_attention_heads=8,
    use_cpu_initialization=True,
    bf16=True,
    params_dtype=torch.bfloat16,
    pipeline_dtype=torch.bfloat16,
    sequence_parallel=False,
    hidden_dropout=0.0,
    attention_dropout=0.0,
    deterministic_mode=True,
)

# Hybrid / Mamba layers constrain hidden_size, so the base is wider.
# num_attention_heads must be ≥ max(TP) so the attention layer can shard
# evenly under TP=8 (otherwise: "heads must be divisible by GQA groups").
_BASE_HYBRID = dict(
    num_layers=3,
    # Mamba derives nheads = d_inner / head_dim = (hidden*expand) / 64
    # and requires nheads % ngroups (=8) == 0. hidden=256 → nheads=8 ✓.
    # Smaller hidden_size breaks the divisibility.
    hidden_size=256,
    num_attention_heads=8,
    use_cpu_initialization=True,
    bf16=True,
    params_dtype=torch.bfloat16,
    pipeline_dtype=torch.bfloat16,
    sequence_parallel=False,
    hidden_dropout=0.0,
    attention_dropout=0.0,
    deterministic_mode=True,
)

# Overrides that turn a dense GPT preset into a MoE one. Applied automatically
# by tests when the chosen parallelism dict sets EP > 1.
_MOE_OVERRIDES = dict(
    num_moe_experts=4,
    moe_router_topk=2,
    moe_grouped_gemm=True,
    add_bias_linear=False,
)


def gpt_base() -> dict:
    return dict(_BASE_GPT)


def hybrid_base() -> dict:
    return dict(_BASE_HYBRID)


def moe_overrides(tp: int = 1, ep: int = 1) -> dict:
    """Return MoE overrides. When ``tp > 1`` we must also enable
    ``sequence_parallel`` (MoE+TP without SP raises in moe_layer.py) and
    propagate ``tensor_model_parallel_size`` into the config (otherwise
    the SP validator sees TP=1 in the config and rejects SP=True). When
    ``ep > 1`` we must propagate ``expert_model_parallel_size`` into the
    config — ``parallel_state`` initialising EP groups is not enough;
    ``ColumnParallelLinear``/``RowParallelLinear`` reads
    ``config.expert_model_parallel_size`` to decide whether expert weights
    use the expert tp_group or the dense tp_group."""
    overrides = dict(_MOE_OVERRIDES)
    if tp > 1:
        overrides["sequence_parallel"] = True
        overrides["tensor_model_parallel_size"] = tp
    if ep > 1:
        overrides["expert_model_parallel_size"] = ep
    return overrides


# ---------------------------------------------------------------------------
# Model presets — fed to @pytest.mark.parametrize. Each pytest.param's first
# arg is a dict of TransformerConfig overrides; the `id=` controls the test
# ID pytest prints (handy for `-k <preset-id>`).
# ---------------------------------------------------------------------------

GPT_CONFIGS = [
    # ``gpt-like``    — multi-head attention, LayerNorm, plain MLP (GPT-2 family).
    # ``llama-like``  — grouped-query attention, RMSNorm, gated linear unit
    #                   (Llama / modern-decoder family).
    # The MoE preset (DeepSeek-V3 style) is in ``PERF_GPT_CONFIGS`` below —
    # adding it here would multiply the correctness matrix without adding
    # bit-exactness signal.
    pytest.param({}, id="gpt-like"),
    pytest.param(
        dict(
            num_query_groups=2,
            normalization="RMSNorm",
            gated_linear_unit=True,
            add_bias_linear=False,
        ),
        id="llama-like",
    ),
]


# Perf-only presets. Includes the dense ones above plus an MoE variant
# (``dsv-like``) so the leaderboard captures MoE+EP cost. Not used by
# correctness tests.
#
# ``num_layers`` / ``hidden_size`` / ``ffn_hidden_size`` are bumped so the
# leaderboard has a big enough scale to surface the textbook bwd≈2×fwd
# pattern. At hidden=64 every matmul is launch-bound (~5-15µs dispatch vs
# ~1-5µs compute), so bwd's extra grad_weight matmul doesn't add
# proportional time. ``hidden=512`` makes compute dominate dispatch.
_PERF_NUM_LAYERS = 4
_PERF_HIDDEN_SIZE = 4096
_PERF_FFN_HIDDEN_SIZE = 16384  # 4× hidden, standard transformer ratio
# Keep head_dim = 128 (production-typical). At hidden=4096 num_heads=8
# would give head_dim=512 which hits TE's large-head-dim SWA path with a
# mask-shape mismatch.
_PERF_NUM_ATTENTION_HEADS = _PERF_HIDDEN_SIZE // 128

_PERF_SCALE = dict(
    num_layers=_PERF_NUM_LAYERS,
    hidden_size=_PERF_HIDDEN_SIZE,
    ffn_hidden_size=_PERF_FFN_HIDDEN_SIZE,
    num_attention_heads=_PERF_NUM_ATTENTION_HEADS,
)

PERF_GPT_CONFIGS = [
    # Single perf cell: DeepSeek-V3-style MoE — the configuration whose
    # determinism cost is most interesting (grouped-GEMM with TP + EP
    # combined). gpt-like / llama-like / pure-EP variants were validated
    # during scale-finding but trimmed to keep CI focused.
    pytest.param(
        dict(
            **_PERF_SCALE,
            num_query_groups=2,
            normalization="RMSNorm",
            gated_linear_unit=True,
            add_bias_linear=False,
            num_moe_experts=4,
            moe_router_topk=2,
            moe_grouped_gemm=True,
        ),
        id="dsv-like",
    ),
]


# Mamba / Hybrid SSM perf coverage intentionally absent — the perf bucket
# is focused on the single ``dsv-like-tp2-ep2`` cell. To re-add Mamba perf
# coverage, define ``PERF_HYBRID_CONFIGS`` here AND restore the
# ``test_mamba_perf_breakdown`` method in ``perf/test_gpt_perf.py``.


# Small starter perf parallelism axis. The perf test parametrizes directly
# off this list — extending coverage is a one-line append, no test edits.
# Cells with EP > 1 only run for MoE presets (dense presets ``pytest.skip``
# the combination, see ``test_gpt_perf.py``).
#
# Ordering: EP-first because the MoE expert-parallel cost is the most
# interesting signal in the leaderboard; TP comes last only to round out
# dense-preset coverage. The single-GPU (``tp1``) case carries no
# parallelism to measure and is intentionally dropped.
PERF_PARALLELISM_CONFIGS = [
    # Single perf parallelism cell: TP × EP composite — the most interesting
    # determinism signal lives here (grouped-GEMM on TP-sharded MoE
    # experts). Other cells (ep2/ep4/tp2) were validated during scale-
    # finding but trimmed for CI focus.
    pytest.param({"TP": 2, "EP": 2}, id="tp2-ep2"),
]

HYBRID_CONFIGS = [
    # mamba-attn-mlp covers Mamba + attention + MLP paths. pure-mamba is
    # dropped (Mamba path alone is already exercised here).
    pytest.param("M*-", {}, id="mamba-attn-mlp"),
]


# ---------------------------------------------------------------------------
# Composite parallelism configs.
#
# Each entry is a dict over the shortname keys below. ``apply_parallelism``
# normalises and forwards them to ``Utils.initialize_model_parallel``.
#
#   TP    tensor_model_parallel_size
#   PP    pipeline_model_parallel_size
#   VPP   virtual_pipeline_model_parallel_size
#   CP    context_parallel_size
#   EP    expert_model_parallel_size      (implies MoE preset)
#   FSDP  data-parallel sharding size     (wraps model with fully_shard_model)
#
# A test must skip an entry if Utils.world_size cannot host it; see
# ``required_world_size``.
# ---------------------------------------------------------------------------

PARALLELISM_CONFIGS = [
    # Pure TP.
    pytest.param({"TP": 4}, id="tp4"),
    pytest.param({"TP": 8}, id="tp8"),
    # MoE + EP.
    pytest.param({"EP": 2}, id="ep2"),
    # MoE + TP × EP composites.
    pytest.param({"TP": 2, "EP": 2}, id="tp2-ep2"),
    pytest.param({"TP": 2, "EP": 4}, id="tp2-ep4"),
    # FSDP — pure and EP composite.
    pytest.param({"FSDP": 8}, id="fsdp8"),
    pytest.param({"FSDP": 8, "EP": 4}, id="fsdp8-ep4"),
    # PP — verified via pipeline schedule + NaN-aware equality.
    pytest.param({"PP": 2}, id="pp2"),
    pytest.param({"PP": 4}, id="pp4"),
    pytest.param({"TP": 2, "PP": 2}, id="tp2-pp2"),
    # VPP — _build_gpt forwards vp_stage to GPTModel so each virtual chunk
    # gets the correct layer slice; runner uses num_layers = pp*vpp (one
    # layer per chunk; was bumped 2× before the vp_stage fix landed).
    pytest.param({"PP": 2, "VPP": 2}, id="pp2-vpp2"),
]


_SHORTNAME_TO_INIT_KWARG = {
    "TP": "tensor_model_parallel_size",
    "PP": "pipeline_model_parallel_size",
    "VPP": "virtual_pipeline_model_parallel_size",
    "CP": "context_parallel_size",
    "EP": "expert_model_parallel_size",
}


def apply_parallelism(parallelism: dict) -> tuple[dict, bool, bool]:
    """Translate a composite parallelism dict into init kwargs + flags.

    Returns:
        init_kwargs: kwargs for ``Utils.initialize_model_parallel``.
        needs_fsdp: True if the dict requests FSDP > 1.
        needs_moe:  True if the dict requests EP > 1 (caller should merge
                    ``moe_overrides()`` into the model config).
    """
    init_kwargs = {}
    for shortname, init_key in _SHORTNAME_TO_INIT_KWARG.items():
        if shortname in parallelism:
            init_kwargs[init_key] = parallelism[shortname]
    needs_fsdp = parallelism.get("FSDP", 1) > 1
    needs_moe = parallelism.get("EP", 1) > 1
    return init_kwargs, needs_fsdp, needs_moe


def required_world_size(parallelism: dict) -> int:
    """Total GPUs needed. FSDP and EP both shard along DP, so DP-size is
    ``max(FSDP, EP, 1)``; PP and CP and TP multiply in independently."""
    tp = parallelism.get("TP", 1)
    pp = parallelism.get("PP", 1)
    cp = parallelism.get("CP", 1)
    dp = max(parallelism.get("FSDP", 1), parallelism.get("EP", 1), 1)
    return tp * pp * cp * dp
