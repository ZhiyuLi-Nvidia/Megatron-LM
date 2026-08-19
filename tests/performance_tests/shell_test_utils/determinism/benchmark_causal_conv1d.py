# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Reproduce the determinism and the overhead of the SSM causal conv1d backward.

Backs the numbers in ``docs/developer/determinism/causal-conv1d-overhead.md``. Self-contained:
imports only torch and causal_conv1d, so it runs inside a training container without Megatron
on ``PYTHONPATH``.

Three checks, selected with ``--mode``:

``replay``
    Run one backward repeatedly from identical inputs and count how many replays differ
    bitwise, with the deterministic reduction off and on. This is the determinism claim.

``equivalence``
    Compare the deterministic result against the default one. They are two summation orders
    for the same quantity, so they must agree to fp32 accumulation error -- a large gap would
    mean the deterministic path computes something else.

``speed``
    Device time per iteration, per layout, with the reduction off and on. Uses the torch
    profiler rather than wall clock: at these sizes a python loop is CPU-launch-bound and
    reports the same time for shapes whose work differs 4x.

Example::

    python benchmark_causal_conv1d.py --mode all
    python benchmark_causal_conv1d.py --mode speed --batch 1 --dim 8192 --seqlen 8192
"""

import argparse
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

import causal_conv1d
from causal_conv1d import causal_conv1d_fn

WIDTH = 4
ACTIVATION = "silu"


def chunk_l(seqlen):
    """Sequence tokens per tile in the channel-last backward.

    ``causal_conv1d.cpp``: ``const int kChunkSizeL = seqlen <= 128 ? 64 : 128;``. Each dweight
    element therefore takes ``batch * ceil(seqlen / chunk_l(seqlen))`` partial contributions,
    which is how many values the default path accumulates with atomicAdd in scheduler order.
    """
    return 64 if seqlen <= 128 else 128


def set_deterministic(enabled):
    """Select the kernel's reduction. Read by getenv on every backward call."""
    os.environ["CAUSAL_CONV1D_DETERMINISTIC"] = "1" if enabled else "0"


def make_inputs(batch, dim, seqlen, dtype, channel_last, seed=1234):
    """Build a [B, D, L] conv input in the requested memory layout, plus weight/bias/grad.

    ``channel_last`` (stride(1) == 1) is what the GDP and fused Mamba mixers hand the kernel;
    the channels-first alternative costs a full-activation copy in and another out.
    """
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(seed)
    x_bld = torch.randn(batch, seqlen, dim, device=device, dtype=dtype, generator=gen)
    x = x_bld.transpose(1, 2)
    if not channel_last:
        x = x.contiguous()
    x = x.detach().requires_grad_()
    weight = torch.randn(dim, WIDTH, device=device, dtype=torch.float32, generator=gen)
    weight = weight.requires_grad_()
    bias = torch.randn(dim, device=device, dtype=torch.float32, generator=gen).requires_grad_()
    g_bld = torch.randn(batch, seqlen, dim, device=device, dtype=dtype, generator=gen)
    grad = g_bld.transpose(1, 2)
    if not channel_last:
        grad = grad.contiguous()
    return x, weight, bias, grad


def one_backward(x, weight, bias, grad):
    out = causal_conv1d_fn(x=x, weight=weight, bias=bias, activation=ACTIVATION)
    return torch.autograd.grad(out, (x, weight, bias), grad_outputs=grad)


GRAD_NAMES = ("dx", "dweight", "dbias")


def run_replay(args, dtype):
    """Count replays that differ bitwise from the first, per gradient."""
    print("== replay: same backward repeated, replays differing bitwise / total ==")
    print(
        f"{'B':>3} {'D':>6} {'L':>6} {'layout':>14} {'det':>4} {'contribs':>9} "
        + " ".join(f"{n:>9}" for n in GRAD_NAMES)
        + f" {'max|d dweight|':>15}"
    )
    for batch in args.batch:
        for dim in args.dim:
            for seqlen in args.seqlen:
                for channel_last in (True, False):
                    for det in (False, True):
                        set_deterministic(det)
                        x, weight, bias, grad = make_inputs(
                            batch, dim, seqlen, dtype, channel_last
                        )
                        first, differing, worst = None, dict.fromkeys(GRAD_NAMES, 0), 0.0
                        for _ in range(args.replays):
                            got = one_backward(x, weight, bias, grad)
                            if first is None:
                                first = got
                                continue
                            for name, ref, cur in zip(GRAD_NAMES, first, got):
                                if not torch.equal(ref, cur):
                                    differing[name] += 1
                                    if name == "dweight":
                                        worst = max(worst, (ref - cur).abs().max().item())
                        contribs = batch * (
                            -(-seqlen // chunk_l(seqlen)) if channel_last else 1
                        )
                        layout = "channel_last" if channel_last else "channels_first"
                        total = args.replays - 1
                        print(
                            f"{batch:>3} {dim:>6} {seqlen:>6} {layout:>14} {int(det):>4} "
                            f"{contribs:>9} "
                            + " ".join(f"{differing[n]:>4}/{total:<4}" for n in GRAD_NAMES)
                            + f" {worst:>15.3e}"
                        )
    set_deterministic(False)


def run_equivalence(args, dtype):
    """Compare deterministic against default output: same quantity, different summation order.

    Reports two relative errors per gradient, because they bound different failure modes:

    ``scale``   max|diff| / max|ref| -- how far the tensor moved relative to its own magnitude.
    ``elem``    max(|diff| / |ref|) over elements above a floor -- catches a path that wrecks
                the small entries while leaving the large ones intact, which ``scale`` cannot
                see. The floor excludes entries where the reference itself is at rounding
                level, for which no relative bound is meaningful.
    """
    print("\n== equivalence: deterministic vs default, max relative difference ==")
    print(
        f"{'B':>3} {'D':>6} {'L':>6} {'layout':>14} "
        + " ".join(f"{n + ' ' + kind:>14}" for n in GRAD_NAMES for kind in ("scale", "elem"))
    )
    for batch in args.batch:
        for dim in args.dim:
            for seqlen in args.seqlen:
                for channel_last in (True, False):
                    set_deterministic(False)
                    x, weight, bias, grad = make_inputs(batch, dim, seqlen, dtype, channel_last)
                    default = one_backward(x, weight, bias, grad)
                    set_deterministic(True)
                    det = one_backward(x, weight, bias, grad)
                    cells = []
                    for ref, cur in zip(default, det):
                        ref, cur = ref.float(), cur.float()
                        peak = ref.abs().max().clamp_min(torch.finfo(torch.float32).tiny)
                        diff = (ref - cur).abs()
                        cells.append((diff.max() / peak).item())
                        # Ignore entries whose own magnitude is below 1e-6 of the peak: a
                        # relative bound on those measures the reference's noise, not the gap.
                        big = ref.abs() > peak * 1e-6
                        elem = (diff[big] / ref.abs()[big]).max().item() if big.any() else 0.0
                        cells.append(elem)
                    layout = "channel_last" if channel_last else "channels_first"
                    print(
                        f"{batch:>3} {dim:>6} {seqlen:>6} {layout:>14} "
                        + " ".join(f"{c:>14.3e}" for c in cells)
                    )
    set_deterministic(False)


def gpu_time_us(step, iters, warmup=10):
    """Mean device time per iteration, plus the per-kernel breakdown."""
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            step()
        torch.cuda.synchronize()
    per_kernel = defaultdict(float)
    for evt in prof.key_averages():
        if evt.self_device_time_total:
            per_kernel[evt.key] += evt.self_device_time_total
    return sum(per_kernel.values()) / iters, {k: v / iters for k, v in per_kernel.items()}


def build_steps(batch, dim, seqlen, dtype):
    """One closure per layout, each taking and returning the (b, l, d) layout the mixer uses.

    The transposes each layout needs are therefore inside the measurement, which is the point:
    channels-first has to materialise a copy in and another out, and timing only the kernel
    would hide that.
    """
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(1234)
    x_bld = torch.randn(batch, seqlen, dim, device=device, dtype=dtype, generator=gen)
    x_bld = x_bld.detach().requires_grad_()
    g_bld = torch.randn(batch, seqlen, dim, device=device, dtype=dtype, generator=gen)
    weight = torch.randn(dim, WIDTH, device=device, dtype=torch.float32, generator=gen)
    weight = weight.requires_grad_()
    bias = torch.randn(dim, device=device, dtype=torch.float32, generator=gen).requires_grad_()

    def conv_step(channel_last):
        def step():
            xc = x_bld.transpose(1, 2)
            if not channel_last:
                xc = xc.contiguous()
            out = causal_conv1d_fn(x=xc, weight=weight, bias=bias, activation=ACTIVATION)
            out = out.transpose(1, 2)
            if not channel_last:
                out = out.contiguous()
            torch.autograd.grad(out, (x_bld, weight, bias), grad_outputs=g_bld)

        return step

    w3 = torch.randn(dim, 1, WIDTH, device=device, dtype=dtype, generator=gen).requires_grad_()
    b1 = torch.randn(dim, device=device, dtype=dtype, generator=gen).requires_grad_()

    def torch_step():
        """What gated-delta-net's deterministic branch runs today, for scale."""
        xc = x_bld.transpose(1, 2).contiguous()
        out = F.conv1d(xc, w3, b1, padding=WIDTH - 1, groups=dim)[..., :seqlen]
        out = F.silu(out).transpose(1, 2).contiguous()
        torch.autograd.grad(out, (x_bld, w3, b1), grad_outputs=g_bld)

    return {"channel_last": conv_step(True), "channels_first": conv_step(False)}, torch_step


SPEED_COLUMNS = [
    "channel_last",
    "channel_last_det",
    "channels_first",
    "channels_first_det",
    "F.conv1d",
]


def run_speed(args, dtype):
    """Device time per fwd+bwd, per layout, reduction off and on."""
    print(f"\n== speed: device us per fwd+bwd, {args.iters} iters ==")
    print(
        f"{'B':>3} {'D':>6} {'L':>6} "
        + " ".join(f"{c:>19}" for c in SPEED_COLUMNS)
        + f" {'det delta':>11}"
    )
    for batch in args.batch:
        for dim in args.dim:
            for seqlen in args.seqlen:
                steps, torch_step = build_steps(batch, dim, seqlen, dtype)
                row, breakdowns = {}, {}
                for name, step in steps.items():
                    for det in (False, True):
                        set_deterministic(det)
                        key = name + ("_det" if det else "")
                        row[key], breakdowns[key] = gpu_time_us(step, args.iters)
                set_deterministic(False)
                row["F.conv1d"], breakdowns["F.conv1d"] = gpu_time_us(torch_step, args.iters)
                delta = 100.0 * (row["channel_last_det"] / row["channel_last"] - 1.0)
                print(
                    f"{batch:>3} {dim:>6} {seqlen:>6} "
                    + " ".join(f"{row[c]:>19.1f}" for c in SPEED_COLUMNS)
                    + f" {delta:>10.1f}%"
                )
                if args.breakdown:
                    for key in SPEED_COLUMNS:
                        print(f"    {key}:")
                        for name, value in sorted(breakdowns[key].items(), key=lambda kv: -kv[1]):
                            print(f"        {value:9.1f} us  {name[:110]}")
    set_deterministic(False)


def run_memory(args, dtype):
    """Extra device memory the workspace costs, against the analytic size."""
    print("\n== memory: extra peak allocated across the backward (channel-last) ==")
    print(
        f"{'B':>3} {'D':>6} {'L':>6} {'activation':>12} {'default':>12} {'det':>12} "
        f"{'delta':>12} {'workspace':>12}"
    )
    for batch in args.batch:
        for dim in args.dim:
            for seqlen in args.seqlen:
                peaks = {}
                for det in (False, True):
                    set_deterministic(det)
                    x, weight, bias, grad = make_inputs(batch, dim, seqlen, dtype, True)
                    # Forward outside the window: only the backward differs between modes.
                    out = causal_conv1d_fn(
                        x=x, weight=weight, bias=bias, activation=ACTIVATION
                    )
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    before = torch.cuda.memory_allocated()
                    torch.autograd.grad(out, (x, weight, bias), grad_outputs=grad)
                    torch.cuda.synchronize()
                    peaks[det] = torch.cuda.max_memory_allocated() - before
                    del x, weight, bias, grad, out
                    torch.cuda.empty_cache()
                # dweight workspace (width floats) + dbias workspace (one float) per tile.
                workspace = 4 * batch * (-(-seqlen // chunk_l(seqlen))) * dim * (WIDTH + 1)
                activation = batch * dim * seqlen * dtype.itemsize
                mb = lambda v: f"{v / 2**20:.1f} MB"
                print(
                    f"{batch:>3} {dim:>6} {seqlen:>6} {mb(activation):>12} "
                    f"{mb(peaks[False]):>12} {mb(peaks[True]):>12} "
                    f"{mb(peaks[True] - peaks[False]):>12} {mb(workspace):>12}"
                )
    set_deterministic(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        default="all",
        choices=["all", "replay", "equivalence", "speed", "memory"],
    )
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--dim", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--seqlen", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--breakdown", action="store_true", help="per-kernel split for --mode speed"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this benchmark needs a GPU")
    dtype = getattr(torch, args.dtype)
    version = getattr(causal_conv1d, "__version__", "unknown")
    print(f"torch {torch.__version__}  causal_conv1d {version}  {torch.cuda.get_device_name(0)}")
    print(f"dtype={args.dtype} width={WIDTH} activation={ACTIVATION}\n")

    if args.mode in ("all", "replay"):
        run_replay(args, dtype)
    if args.mode in ("all", "equivalence"):
        run_equivalence(args, dtype)
    if args.mode in ("all", "speed"):
        run_speed(args, dtype)
    if args.mode in ("all", "memory"):
        run_memory(args, dtype)

    os.environ.pop("CAUSAL_CONV1D_DETERMINISTIC", None)


if __name__ == "__main__":
    main()
