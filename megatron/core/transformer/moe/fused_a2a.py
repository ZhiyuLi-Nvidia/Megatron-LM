# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Portions of this code are from DeepSeek DeepEP project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE

from megatron.core.utils import internal_api

try:
    from deep_ep import Buffer
    from deep_ep.utils import EventHandle, EventOverlap

    HAVE_DEEP_EP = True
except ImportError:
    HAVE_DEEP_EP = False

import os
import torch

_buffer = None

# HYBRIDEP_SYNC env knob (default "1" = sync after every dispatch/combine).
# At 8-GPU GB200 the synchronize was needed to avoid cudaErrorIllegalAddress
# in backward after the custom_allgather=False patch unmasked a cross-stream
# race. Set HYBRIDEP_SYNC=0 to test whether the race fires at 256-GPU too,
# or whether it's masked at scale by NCCL-collective barriers / DP averaging.
_HYBRIDEP_SYNC_ENABLED = os.environ.get("HYBRIDEP_SYNC", "1").strip() != "0"


def _hybridep_maybe_sync():
    if _HYBRIDEP_SYNC_ENABLED:
        torch.cuda.synchronize()


def get_hidden_bytes(x: torch.Tensor) -> int:
    """Calculate the number of hidden bytes for a tensor.

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        int: Number of hidden bytes
    """
    return x.size(1) * max(x.element_size(), 2)


def get_buffer(group: torch.distributed.ProcessGroup, hidden_bytes: int):
    """Get or create a buffer for all-to-all communication.

    Args:
        group (torch.distributed.ProcessGroup): Process group for communication
        hidden_bytes (int): Number of hidden bytes needed

    Returns:
        Buffer: Communication buffer
    """
    global _buffer
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        # Split long line for PEP8 compliance
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes
        )

    # Allocate buffer if not existed or not enough buffer
    # NOTES: the adaptive routing configuration of the network **must be off**
    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


class FusedDispatch(torch.autograd.Function):
    """Fused dispatch operation for MoE routing combining computation and communication."""

    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Forward pass of fused dispatch."""
        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())
        # Calculate layout before actual dispatch
        buffer = get_buffer(group, get_hidden_bytes(x))
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = buffer.get_dispatch_layout(
            token_indices,
            num_experts,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        # Do MoE dispatch
        # NOTES: the CPU will wait for GPU's signal to arrive,
        # so this is not compatible with CUDA graph
        (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            num_recv_tokens_per_expert_list,
            handle,
            after_event_overlap,
        ) = buffer.dispatch(
            x,
            topk_idx=token_indices,
            topk_weights=token_probs,  # DeepEP only supports float32 probs
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=event,  # wait in deepep::intra/inter_dispatch
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        # Make sure current stream is synchronized
        if async_finish:
            after_event_overlap.current_stream_wait()

        # Save for backward
        ctx.group = group
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        tokens_per_expert = torch.tensor(num_recv_tokens_per_expert_list)

        return (recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle)

    @staticmethod
    def backward(
        ctx, grad_output, grad_token_indices, grad_token_probs, grad_tokens_per_expert, grad_handle
    ):
        """Backward pass of fused dispatch."""
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        handle = ctx.handle
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        grad_x, grad_token_probs, after_event = buffer.combine(
            grad_output.contiguous(),
            handle,
            topk_weights=grad_token_probs.float(),
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, grad_token_probs, None, None, None, None


class FusedCombine(torch.autograd.Function):
    """Fused combine operation for MoE output combining computation and communication."""

    @staticmethod
    def forward(ctx, x, group, handle, async_finish=False, allocate_on_comm_stream=False):
        """Forward pass of fused combine."""
        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(group, get_hidden_bytes(x))
        combined_x, _, after_event = buffer.combine(
            x,
            handle=handle,
            async_finish=async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if async_finish:
            after_event.current_stream_wait()

        ctx.handle = handle
        ctx.group = group
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        return combined_x, None

    @staticmethod
    def backward(ctx, grad_output, previous_event=None):
        """Backward pass of fused combine."""
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        grad_x, _, _, _, _, after_event = buffer.dispatch(
            grad_output.contiguous(),
            handle=ctx.handle,
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, None, None, None


if HAVE_DEEP_EP:

    def fused_dispatch(
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Perform fused dispatch operation if deep_ep is available.

        Args:
            x: Input tensor [num_tokens, hidden_size]
            token_indices: Token routing indices [num_tokens, topk]
            token_probs: Token routing probabilities [num_tokens, topk]
            num_experts: Number of experts
            group: Process group
            previous_event: Previous CUDA event

        Returns:
            Result of FusedDispatch
        """
        return FusedDispatch.apply(
            x.contiguous(),
            token_indices,
            token_probs,
            num_experts,
            group,
            async_finish,
            allocate_on_comm_stream,
        )

    def fused_combine(x, group, handle, async_finish=False, allocate_on_comm_stream=False):
        """Perform fused combine operation if deep_ep is available.

        Args:
            x: Input tensor
            group: Process group
            handle: Communication handle
            previous_event: Previous CUDA event

        Returns:
            Result of FusedCombine
        """
        return FusedCombine.apply(x, group, handle, async_finish, allocate_on_comm_stream)

    def set_deepep_num_sms(num_sms):
        """Sets the number of SMs to use for DeepEP"""
        Buffer.set_num_sms(num_sms)

else:
    fused_dispatch = None
    fused_combine = None
    set_deepep_num_sms = None


try:
    from deep_ep import HybridEPBuffer

    HAVE_HYBRIDEP = True
except ImportError:
    HAVE_HYBRIDEP = False

# Per-manager keyed buffers (one HybridEPBuffer per _HybridEPManager instance).
# Replaces the prior singleton, which was shared across VP chunks and caused
# DeepEP "negative dimension" / non-determinism when forward and backward A2A
# interleaved on a single buffer in steady-state 1F1B (2602 dsv3 investigation).
_hybrid_ep_buffers: dict = {}


def init_hybrid_ep_buffer(
    buffer_key: int,
    group: torch.distributed.ProcessGroup,
    hidden_dim: int,
    seq_len: int,
    num_local_experts: int,
    num_sms_dispatch_api: int,
    num_sms_combine_api: int,
    fp8_dispatch: bool,
) -> None:
    '''
    Initialize the HybridEP buffer for the given buffer_key.

    Args:
        buffer_key (int): Identity key (typically id(_HybridEPManager)) selecting which
            per-manager buffer to (re)initialize.
        ...
    '''
    assert not fp8_dispatch, "HybridEP dispatcher does not support fp8 dispatch now"
    # enable_custom_allgather: 26.02 container's DeepEP defaulted this to False;
    # 26.04 flipped the default to True. The custom-allgather path is the new
    # culprit for the -4 dim crash under torch.use_deterministic_algorithms(True)
    # — it appears to skip populating num_dispatched_tokens_tensor before
    # executor.cu's `.item<int>()` blocking read, leaving uninitialized memory
    # that reads as -4 (or 0xFFFFFFFC). Force False to match 26.02 behavior
    # where HybridEP+det reached iter 6 successfully (per 2602 doc).
    _hybrid_ep_buffers[buffer_key] = HybridEPBuffer(
        group=group,
        hidden_dim=hidden_dim,
        max_num_of_tokens_per_rank=seq_len,
        num_local_experts=num_local_experts,
        use_fp8=fp8_dispatch,
        num_sms_dispatch_api=num_sms_dispatch_api,
        num_sms_combine_api=num_sms_combine_api,
        enable_custom_allgather=False,
    )


def reset_hybrid_ep_buffer():
    '''Reset all per-manager HybridEP buffers.'''
    _hybrid_ep_buffers.clear()


class HybridEPDispatch(torch.autograd.Function):
    '''
    Fused dispatch operation for permute + dispatch a2a + permute using the HybridEP backend
    '''

    @staticmethod
    def forward(
        ctx,
        x,
        routing_map,
        probs,
        group,
        num_local_experts,
        buffer_key,
        num_sms_dispatch_api=24,
        num_sms_combine_api=24,
        num_permuted_tokens=None,
        pad_multiple=None,
    ):
        '''
        Forward pass of fused dispatch of the HybridEP backend
        '''
        if buffer_key not in _hybrid_ep_buffers:
            seq_len, hidden_dim = x.shape[-2:]
            fp8_dispatch = False  # Currently, we do not support fp8 dispatch
            init_hybrid_ep_buffer(
                buffer_key,
                group,
                hidden_dim,
                seq_len,
                num_local_experts,
                num_sms_dispatch_api,
                num_sms_combine_api,
                fp8_dispatch,
            )
        buffer = _hybrid_ep_buffers[buffer_key]
        non_blocking = num_permuted_tokens is not None
        (
            dispatched_hidden,
            dispatched_probs,
            dispatched_scaling_factor,
            tokens_per_expert,
            handle,
        ) = buffer.dispatch_with_permute(
            hidden=x,
            routing_map=routing_map,
            probs=probs,
            scaling_factor=None,
            num_of_experts_per_rank=num_local_experts,
            pad_multiple=pad_multiple,
            num_permuted_tokens=num_permuted_tokens,
            non_blocking=non_blocking,
        )
        # When non_blocking=True the HybridEP runtime writes the dispatch outputs on
        # its internal comm stream and returns before the write completes.  Any CUDA
        # operation on the returned tensors that is launched on the current (default)
        # stream will race against that still-in-flight write → cudaErrorIllegalAddress.
        # Inserting a full device synchronize here is the simplest safe fix: it ensures
        # every in-flight comm-stream kernel has completed before autograd or the expert
        # matmuls touch the output storage.
        # NOTE: this is intentionally heavier than record_stream — record_stream only
        # prevents the *allocator* from recycling the buffer, it does not prevent the
        # current stream from reading a partially-written tensor.
        if non_blocking:
            _hybridep_maybe_sync()

        ctx.handle = handle
        ctx.pad_multiple = pad_multiple
        ctx.buffer = buffer
        # record_stream: belt-and-suspenders on top of the synchronize above; keeps
        # the allocator from recycling the storage across future stream events.
        _cur = torch.cuda.current_stream()
        for _t in (dispatched_hidden, dispatched_probs, dispatched_scaling_factor, tokens_per_expert):
            if isinstance(_t, torch.Tensor) and _t.is_cuda:
                _t.record_stream(_cur)
        return (
            dispatched_hidden,
            dispatched_probs,
            dispatched_scaling_factor,
            tokens_per_expert,
            handle,
        )

    @staticmethod
    def backward(ctx, grad_x, grad_probs, grad_scaling_factor, grad_tokens_per_expert, grad_handle):
        '''
        Backward pass of fused dispatch of the HybridEP backend
        '''
        handle = ctx.handle
        combined_hidden, combined_probs = ctx.buffer.combine_with_unpermute(
            hidden=grad_x, probs=grad_probs, handle=handle, pad_multiple=ctx.pad_multiple
        )
        # combine_with_unpermute may write on the HybridEP comm stream; synchronize
        # before the upstream autograd nodes (running on current stream) read the result.
        _hybridep_maybe_sync()
        # record_stream: belt-and-suspenders after the synchronize.
        _cur = torch.cuda.current_stream()
        for _t in (combined_hidden, combined_probs):
            if isinstance(_t, torch.Tensor) and _t.is_cuda:
                _t.record_stream(_cur)
        return combined_hidden, None, combined_probs, None, None, None, None, None, None, None, None


@internal_api
class HybridEPCombine(torch.autograd.Function):
    '''
    Fused combine operation for permute + combine a2a + permute using the HybridEP backend
    '''

    @staticmethod
    def forward(ctx, x, handle, buffer_key, num_permuted_tokens=None, pad_multiple=None):
        '''
        Forward pass of fused combine of the HybridEP backend
        '''
        buffer = _hybrid_ep_buffers[buffer_key]
        combined_hidden, _ = buffer.combine_with_unpermute(
            hidden=x, handle=handle, pad_multiple=pad_multiple
        )
        # combine_with_unpermute may write on the HybridEP comm stream; synchronize
        # before downstream computation (running on current stream) reads the result.
        _hybridep_maybe_sync()
        ctx.handle = handle
        ctx.pad_multiple = pad_multiple
        ctx.num_permuted_tokens = num_permuted_tokens
        ctx.buffer = buffer
        # record_stream: belt-and-suspenders after the synchronize.
        if isinstance(combined_hidden, torch.Tensor) and combined_hidden.is_cuda:
            combined_hidden.record_stream(torch.cuda.current_stream())
        return combined_hidden

    @staticmethod
    def backward(ctx, grad_x):
        '''
        Backward pass of fused combine of the HybridEP backend
        '''
        handle = ctx.handle
        _non_blocking = ctx.num_permuted_tokens is not None
        dispatched_hidden, _, _, _, _ = ctx.buffer.dispatch_with_permute(
            hidden=grad_x,
            scaling_factor=None,
            handle=handle,
            pad_multiple=ctx.pad_multiple,
            num_permuted_tokens=ctx.num_permuted_tokens,
        )
        # Same sync rationale as HybridEPDispatch.forward: if dispatch was async, the
        # backward comm stream must be drained before the upstream grad computation
        # (which runs on the current stream) reads dispatched_hidden.
        if _non_blocking:
            _hybridep_maybe_sync()
        if isinstance(dispatched_hidden, torch.Tensor) and dispatched_hidden.is_cuda:
            dispatched_hidden.record_stream(torch.cuda.current_stream())
        return dispatched_hidden, None, None, None, None, None


if HAVE_HYBRIDEP:

    @internal_api
    def hybrid_ep_dispatch(
        x,
        routing_map,
        probs,
        group,
        num_local_experts,
        buffer_key,
        num_sms_dispatch_api=24,
        num_sms_combine_api=24,
        num_permuted_tokens=None,
        pad_multiple=None,
    ):
        '''Perform fused dispatch for the HybridEP backend.

        Args:
            buffer_key (int): Identity key selecting the per-manager HybridEPBuffer.
            (other args unchanged)
        '''
        return HybridEPDispatch.apply(
            x,
            routing_map,
            probs,
            group,
            num_local_experts,
            buffer_key,
            num_sms_dispatch_api,
            num_sms_combine_api,
            num_permuted_tokens,
            pad_multiple,
        )

    @internal_api
    def hybrid_ep_combine(x, handle, buffer_key, num_permuted_tokens, pad_multiple):
        '''Perform fused combine for the HybridEP backend.

        Args:
            buffer_key (int): Identity key selecting the per-manager HybridEPBuffer.
            (other args unchanged)
        '''
        return HybridEPCombine.apply(x, handle, buffer_key, num_permuted_tokens, pad_multiple)

else:
    hybrid_ep_dispatch = None
    hybrid_ep_combine = None
