"""Install the registered role after weights and KV are ready, before runner warmup."""

import logging

import torch.distributed as dist

from sglang.kernels.plugin_kernels.npu.spec import Decline, PreparationView, find_spec
from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)


def install_megakernels(runner):
    architecture = type(runner.model).__name__
    group = get_parallel().tp_group.cpu_group
    binding = error = decline = None
    signature = None
    try:
        spec = find_spec(architecture)
        if spec is None:
            decline = f"no megakernel spec for {architecture}"
        elif (spec.role == "nextn") != runner.is_draft_worker:
            decline = (
                f"spec role {spec.role} does not match "
                f"draft_worker={runner.is_draft_worker}"
            )
        else:
            cfg = resolved_view(runner.server_args)
            resources = PreparationView(
                model=runner.model,
                token_to_kv_pool=runner.token_to_kv_pool,
                physical_slots=not runner.runner_ids.is_translating
                and not runner.token_to_kv_pool_allocator.hands_out_virtual_ids,
            )
            outcome = spec.prepare(resources, cfg)
            if isinstance(outcome, Decline):
                decline = outcome
            elif getattr(outcome, "role", None) is not None:
                binding = outcome
                signature = (
                    architecture,
                    binding.role,
                    binding.mode,
                    getattr(cfg, "speculative_num_steps", None),
                    getattr(cfg, "speculative_num_draft_tokens", None),
                )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    ranks = [None] * dist.get_world_size(group)
    dist.all_gather_object(ranks, (signature, error, decline), group=group)
    if any(peer_error for _, peer_error, _ in ranks):
        raise RuntimeError(f"NPU megakernel preflight failed: {ranks}")

    refusals = [reason for peer, _, reason in ranks if peer is None]
    if refusals:
        expected = all(
            isinstance(reason, Decline) and reason.expected for reason in refusals
        )
        log = logger.info if expected else logger.warning
        log(
            "NPU megakernel declined %s: %s",
            architecture,
            [
                reason.reason if isinstance(reason, Decline) else reason
                for reason in refusals
            ],
        )
        return None
    if any(peer != signature for peer, _, _ in ranks):
        raise RuntimeError(f"NPU megakernel preflight failed: {ranks}")

    binding.build(runner.model)
    logger.info("NPU megakernel architecture=%s role=%s mode=%s", *signature[:3])
    return binding
