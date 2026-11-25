"""EAGLE owns admission, requests and outputs. The executor owns graph replay."""

from __future__ import annotations

from contextlib import nullcontext

from sglang.kernels.plugin_kernels.npu.spec import (
    DraftLoopIO,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.speculative.eagle_megakernel import EagleDraftRuntime
from sglang.srt.speculative.eagle_utils import per_step_draft_out_cache_loc


class NpuEagleDraftRuntime(EagleDraftRuntime):
    def __init__(self, worker, binding):
        self.binding = binding
        self.worker = worker
        self.steps = binding.steps

    def propose_chain(self, batch, forward_batch, shape_key=None):
        idle = forward_batch.forward_mode.is_idle()
        req = batch.reqs[0] if len(batch.reqs) == 1 else None
        backend = getattr(
            getattr(self.worker, "cuda_graph_runner", None), "backend", None
        )
        execute = getattr(backend, "execute_draft_graph", None)
        if execute is None:
            # The provider can decline this runner itself (memory saver, a
            # different runner type). That decision comes from process-wide
            # configuration, so every rank takes it.
            return None
        attention_backends = getattr(
            getattr(self.worker, "draft_attn_backend", None), "attn_backends", None
        )
        attention = attention_backends[0] if attention_backends else None
        admitted = False
        if not idle and attention is not None:
            admitted = (
                not self.binding.pd_decode
                or (
                    req is not None
                    and getattr(req, "npu_draft_kv_generation", None)
                    == req.retraction_count
                )
            ) and self.binding.supports_decode(
                batch, forward_batch, self.worker.hot_token_id
            )
        seed = forward_batch.spec_info
        io = None
        if admitted:
            # topk=1 plans a write window of steps+1. The chain writes the seed
            # plus steps-1 recurrent tokens; the last column belongs to verify.
            window = forward_batch.out_cache_loc[: self.steps]
            slots = per_step_draft_out_cache_loc(window, 1, 1, self.steps)
            io = DraftLoopIO(
                forward_batch,
                seed.topk_index.reshape(-1),
                forward_batch.positions,
                seed.hidden_states,
                slots[: self.steps - 1],
            )
        context = (
            forward_context(
                ForwardContext(attn_backend=attention), forward_batch=forward_batch
            )
            if admitted
            else nullcontext()
        )
        # The backend votes once, after its own session lookup and update, so
        # every rank falls back together when one of them cannot replay.
        with context:
            tokens = execute(io, shape_key, admitted=admitted)
        if tokens is None:
            return None
        if tokens.shape != (1, self.steps):
            raise RuntimeError("Invalid NextN suffix width")
        return tokens.to(dtype=seed.topk_index.dtype, copy=True)
