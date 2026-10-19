"""NPU scheduling parity and optional real-device graph capture coverage."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.kernels.ops.speculative.dspark import dspark_schedule as schedule
from sglang.srt.speculative.dspark_components.dspark_planner import DSparkScheduleConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("bs,gamma", [(0, 8), (1, 8), (2, 3), (17, 8), (65, 5)])
@pytest.mark.parametrize("pattern", ["random", "ones", "zeros", "coarse"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_npu_schedule_matches_reference_for_budgets_and_ties(bs, gamma, pattern, dtype):
    generator = torch.Generator().manual_seed(123)
    # Noncontiguous confidence exercises the same layouts as confidence relay.
    confidence = torch.rand(bs, gamma * 2, generator=generator).to(dtype)[:, ::2]
    if pattern == "ones":
        confidence.fill_(1)
    elif pattern == "zeros":
        confidence.zero_()
    elif pattern == "coarse":
        confidence = (confidence * 4).floor() / 4
    for min_len, max_len, eps in ((0, 0, 1e-6), (1, 0, 0), (1, 1, 0.5), (1, 0, 1.1)):
        cfg = DSparkScheduleConfig(
            gamma=gamma,
            min_verify_len=min_len,
            max_verify_len=max_len,
            survival_eps=eps,
        )
        for budget in (-1, 0, 1, 3, bs * gamma // 2, bs * gamma, bs * gamma + 3):
            reference = schedule.schedule_verify_lens_topk(
                confidence=confidence, budget=budget, cfg=cfg
            )
            actual = schedule.schedule_verify_lens_topk_npu(
                confidence=confidence, budget=budget, cfg=cfg
            )
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            assert actual.dtype == torch.int32


def test_npu_schedule_ties_order_position_before_request():
    cfg = DSparkScheduleConfig(gamma=8)
    confidence = torch.ones(3, 8)
    # An unstable top-k can give all five equal candidates to one request.
    # Reference ordering gives one to each request, then to requests 0 and 1.
    assert schedule.schedule_verify_lens_topk_npu(
        confidence=confidence, budget=5, cfg=cfg
    ).tolist() == [3, 3, 2]


def test_npu_dispatch_precedes_the_cuda_alias():
    confidence = Mock(device=SimpleNamespace(type="npu"))
    with (
        patch.object(schedule, "inputs_on_cuda", return_value=True) as cuda_predicate,
        patch.object(schedule.ScheduleVerifyLensTopk, "npu", return_value="npu") as npu,
        patch.object(schedule.ScheduleVerifyLensTopk, "triton") as triton,
    ):
        assert (
            schedule.ScheduleVerifyLensTopk.execute(
                confidence=confidence, budget=3, cfg=None
            )
            == "npu"
        )
        npu.assert_called_once_with(confidence=confidence, budget=3, cfg=None)
        cuda_predicate.assert_not_called()
        triton.assert_not_called()


@pytest.mark.parametrize("cuda", [False, True])
def test_existing_cpu_and_cuda_dispatch_unchanged(cuda):
    confidence = torch.ones(2, 8)
    cls = schedule.ScheduleVerifyLensTopk
    with (
        patch.object(schedule, "inputs_on_cuda", return_value=cuda),
        patch.object(cls, "npu") as npu,
        patch.object(cls, "torch", return_value="torch") as torch_impl,
        patch.object(cls, "triton", return_value="triton") as triton,
    ):
        assert cls.execute(confidence=confidence, budget=3, cfg=None) == (
            "triton" if cuda else "torch"
        )
        npu.assert_not_called()
        (triton if cuda else torch_impl).assert_called_once()


def test_schedule_does_not_read_tensor_values_on_the_host():
    confidence = torch.tensor([[0.9, 0.8, 0.7], [0.2, 0.1, 0.1]])
    cfg = DSparkScheduleConfig(gamma=3)
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("CPU transfer")),
        patch.object(torch.Tensor, "item", side_effect=AssertionError("scalar read")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("list read")),
        patch.object(
            torch.Tensor, "__bool__", side_effect=AssertionError("tensor branch")
        ),
    ):
        actual = schedule.schedule_verify_lens_topk_npu(
            confidence=confidence, budget=2, cfg=cfg
        )
    assert actual.tolist() == [3, 1]


_NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


@pytest.mark.skipif(not _NPU_AVAILABLE, reason="requires an Ascend NPU")
@pytest.mark.parametrize("budget", [2, 100])
def test_npu_capture_replay_reads_updated_confidence(budget):
    cfg = DSparkScheduleConfig(gamma=3)
    static_input = torch.ones(2, 3, device="npu")
    # Never fall through to Triton, even if torch_npu aliases is_cuda to True.
    with patch.object(
        schedule.ScheduleVerifyLensTopk, "triton", side_effect=AssertionError
    ):
        for _ in range(2):
            schedule.ScheduleVerifyLensTopk.execute(
                confidence=static_input, budget=budget, cfg=cfg
            )
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            static_output = schedule.ScheduleVerifyLensTopk.execute(
                confidence=static_input, budget=budget, cfg=cfg
            )
        for values in (
            [[0.9, 0.8, 0.7], [0.2, 0.1, 0.1]],
            [[0.2, 0.1, 0.1], [0.9, 0.8, 0.7]],
            [[1.0] * 3] * 2,
            [[0.0] * 3] * 2,
        ):
            confidence = torch.tensor(values)
            static_input.copy_(confidence)
            graph.replay()
            torch.npu.synchronize()
            reference = schedule.schedule_verify_lens_topk(
                confidence=confidence, budget=budget, cfg=cfg
            )
            torch.testing.assert_close(static_output.cpu(), reference, rtol=0, atol=0)
