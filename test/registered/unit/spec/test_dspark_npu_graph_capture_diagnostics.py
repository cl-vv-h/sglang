"""Exercise capture-phase reporting without an NPU or changing graph behavior."""

import logging
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.hardware_backend.npu.graph_runner import npu_cudagraph_backend as module
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "failing_call,phase", [(1, "warmup-1"), (2, "warmup-2"), (3, "capture")]
)
@pytest.mark.parametrize("debug", [False, True])
def test_original_error_keeps_phase_and_traceback(failing_call, phase, debug, caplog):
    backend = module.NPUCudaGraphBackend.__new__(module.NPUCudaGraphBackend)
    device = Mock()
    device.mem_get_info.return_value = (5 << 30, 64 << 30)
    device.memory_allocated.return_value = 50 << 30
    device.memory_reserved.return_value = 55 << 30
    device.current_stream.return_value = "capture-stream"
    backend._device_module = device
    backend._tp_group = Mock()
    backend._pool = (1, 2)
    backend._capture_stream = "capture-stream"
    backend._enable_torch_compile = False
    backend._memory_saver_adapter = None
    backend._graphs = {}
    backend._outputs = {}
    error = RuntimeError("original HCCL allocation error")
    forward = Mock(side_effect=[None] * (failing_call - 1) + [error])
    npu = SimpleNamespace(NPUGraph=Mock(), graph=lambda *args, **kwargs: nullcontext())
    with (
        patch.dict(sys.modules, {"torch_npu": SimpleNamespace()}),
        patch.object(torch, "npu", npu, create=True),
        patch.object(module, "get_bool_env_var", return_value=debug),
        caplog.at_level(logging.INFO),
        pytest.raises(RuntimeError) as caught,
    ):
        backend.capture_one("tokens-36", forward)
    assert caught.value is error
    assert f"failed phase={phase} key=tokens-36" in caplog.text
    assert not backend._graphs and not backend._outputs
    assert forward.call_count == failing_call
    # Diagnostics sample once BEFORE each phase, never after the failure.
    assert device.mem_get_info.call_count == (failing_call if debug else 0)
    assert device.synchronize.call_count == min(failing_call, 2)
    device.empty_cache.assert_not_called()


def test_capture_success_retains_two_warmups_and_publishes_graph():
    backend = module.NPUCudaGraphBackend.__new__(module.NPUCudaGraphBackend)
    backend._device_module = Mock()
    backend._tp_group = Mock()
    backend._pool = (1, 2)
    backend._capture_stream = "capture-stream"
    backend._enable_torch_compile = False
    backend._memory_saver_adapter = None
    backend._graphs = {}
    backend._outputs = {}
    graph = object()
    forward = Mock(side_effect=["warmup1", "warmup2", "output"])
    hook = Mock()
    graph_context = Mock(side_effect=lambda *args, **kwargs: nullcontext())
    with (
        patch.dict(sys.modules, {"torch_npu": SimpleNamespace()}),
        patch.object(
            torch,
            "npu",
            SimpleNamespace(NPUGraph=lambda: graph, graph=graph_context),
            create=True,
        ),
        patch.object(module, "get_bool_env_var", return_value=False),
    ):
        backend.capture_one("tokens-36", forward, post_warmup_hook=hook)
    assert forward.call_count == 3 and hook.call_count == 2
    assert backend._graphs["tokens-36"] is graph
    assert backend._outputs["tokens-36"] == "output"
    graph_context.assert_called_once_with(
        graph, pool=(1, 2), stream="capture-stream", auto_dispatch_capture=True
    )
    backend._device_module.mem_get_info.assert_not_called()
