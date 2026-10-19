"""CPU checks for the DeepEP payload / MXFP8 GMM1 input contract."""

import os
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import sglang.srt.layers.quantization  # noqa: F401, initialize quantization first
from sglang.srt.hardware_backend.npu.quantization import moe_methods
from sglang.srt.layers.moe.token_dispatcher import deepep
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_INT8_ENV = "DEEP_NORMAL_MODE_USE_INT8_QUANT"


class _LegacyEnvBuffer:
    """Emulate the vendor fallback when no explicit quant mode is active."""

    def __init__(self, fail=False):
        self.fail = fail
        self.observed_env = None

    def get_dispatch_layout(self, *args, **kwargs):
        return None, None, None, None, None

    def _payload(self, x):
        self.observed_env = os.environ.get(_INT8_ENV)
        if self.fail:
            raise RuntimeError("dispatch failed")
        if self.observed_env == "1":
            return x.to(torch.int8), torch.ones(x.shape[0])
        return x

    def dispatch(self, x, **kwargs):
        return self._payload(x), None, None, [x.shape[0]], object(), None

    def low_latency_dispatch(self, x, *args, use_fp8=False, **kwargs):
        return self._payload(x), None, object(), None, None


def _dispatch(mode, *, is_npu=True, use_int8=False, fail=False):
    cls = (
        deepep._DeepEPDispatcherImplNormal
        if mode == "normal"
        else deepep._DeepEPDispatcherImplLowLatency
    )
    dispatcher = object.__new__(cls)
    dispatcher.num_experts = 1
    dispatcher.async_finish = False
    dispatcher.return_recv_hook = False
    dispatcher.num_max_dispatch_tokens_per_rank = 8
    dispatcher.quant_config = {}
    dispatcher.use_fp8 = use_int8
    dispatcher.use_mxfp4 = False
    dispatcher.use_mxfp8 = False
    dispatcher.use_nvfp4 = False
    buffer = _LegacyEnvBuffer(fail=fail)
    dispatcher._get_buffer = lambda: buffer
    x = torch.ones(2, 64, dtype=torch.bfloat16)
    ids = torch.zeros(2, 1, dtype=torch.int64)
    weights = torch.ones(2, 1)
    with (
        patch.object(deepep, "_is_npu", is_npu),
        patch.object(deepep, "_deepep_precompile_tp_barrier"),
        patch.object(
            deepep.DeepEPConfig,
            "get_instance",
            return_value=SimpleNamespace(normal_dispatch_config=None),
        ),
        patch.object(deepep, "get_global_expert_distribution_recorder"),
    ):
        args = (x, ids, weights, None) if mode == "normal" else (x, ids, weights)
        output = dispatcher._dispatch_core(*args)
    return output[0], buffer


@pytest.mark.parametrize("mode", ["normal", "low_latency"])
@pytest.mark.parametrize(
    "is_npu,use_int8", [(True, False), (True, True), (False, False)]
)
def test_bf16_contract_overrides_only_npu_bf16_dispatch(mode, is_npu, use_int8):
    # Reproduce the failure: the process has INT8 enabled, but this particular
    # layer requests BF16 for local MXFP8 activation quantization / an FP draft.
    with patch.dict(os.environ, {_INT8_ENV: "1"}):
        payload, buffer = _dispatch(mode, is_npu=is_npu, use_int8=use_int8)
        assert os.environ[_INT8_ENV] == "1"
    should_quantize = not is_npu or use_int8
    assert isinstance(payload, tuple) == should_quantize
    if not should_quantize:
        assert payload.dtype == torch.bfloat16
        assert buffer.observed_env == "0"


@pytest.mark.parametrize("mode", ["normal", "low_latency"])
@pytest.mark.parametrize("previous", [None, "0", "1"])
def test_dispatch_exception_restores_the_exact_environment(mode, previous):
    with patch.dict(os.environ):
        os.environ.pop(_INT8_ENV, None)
        if previous is not None:
            os.environ[_INT8_ENV] = previous
        with pytest.raises(RuntimeError, match="dispatch failed"):
            _dispatch(mode, fail=True)
        assert os.environ.get(_INT8_ENV) == previous


@pytest.mark.parametrize("mode", ["normal", "low_latency"])
def test_alternating_bf16_and_int8_layers_keep_separate_contracts(mode):
    with patch.dict(os.environ, {_INT8_ENV: "1"}):
        bf16, _ = _dispatch(mode)
        int8, _ = _dispatch(mode, use_int8=True)
        assert bf16.dtype == torch.bfloat16
        assert int8[0].dtype == torch.int8
        assert int8[1].ndim == 1
        assert os.environ[_INT8_ENV] == "1"


def _run_gmm1(x, scale, *, quantized=None):
    # Patch only NPU kernels. The dispatch and quantization methods themselves
    # run unchanged; no torch_npu installation is needed for these contracts.
    with ExitStack() as stack:
        quant = stack.enter_context(
            patch.object(
                torch.ops.npu,
                "npu_dynamic_mx_quant",
                create=True,
                return_value=quantized,
            )
        )
        gmm = stack.enter_context(
            patch.object(
                torch.ops.npu,
                "npu_grouped_matmul_swiglu_quant_v2",
                create=True,
                return_value=(torch.empty(2, 32), torch.empty(2, 1, 2)),
            )
        )
        stack.enter_context(
            patch.object(moe_methods, "_require_e8m0_dtype", return_value=293)
        )
        method = moe_methods.NPUMXFP8MoEMethod("w13")
        info = SimpleNamespace(
            w13_weight=torch.empty(1, 64, 64), w13_weight_scale=torch.empty(1)
        )
        output = method.apply_fused_gmm1_swiglu(
            info, x, torch.tensor([x.shape[0]]), scale, group_list_type=1
        )
        return output, quant, gmm


@pytest.mark.parametrize("rows", [0, 2])
def test_bf16_dispatch_generates_real_mx_block_scales(rows):
    x = torch.ones(rows, 64, dtype=torch.bfloat16)
    mx = torch.empty(rows, 64, dtype=torch.float8_e4m3fn)
    scale = torch.empty(rows, 1, 2, dtype=torch.uint8)
    _, quant, gmm = _run_gmm1(x, None, quantized=(mx, scale))
    quant.assert_called_once_with(x, dst_type=torch.float8_e4m3fn)
    assert gmm.call_args.kwargs["x"] is mx
    assert gmm.call_args.kwargs["x_scale"] is scale


@pytest.mark.parametrize("flat", [False, True])
def test_prequantized_mx_block_scales_keep_their_storage(flat):
    x = torch.empty(2, 128, dtype=torch.float8_e4m3fn)
    scale = torch.arange(8, dtype=torch.uint8).view((2, 4) if flat else (2, 2, 2))
    _, quant, gmm = _run_gmm1(x, scale)
    quant.assert_not_called()
    packed = gmm.call_args.kwargs["x_scale"]
    assert packed.shape == (2, 2, 2)
    assert packed.data_ptr() == scale.data_ptr()
    assert torch.equal(packed.flatten(), scale.flatten())


@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn])
def test_per_token_scale_is_rejected_before_cann(dtype):
    with pytest.raises(ValueError, match="3D E8M0 block scales"):
        _run_gmm1(torch.empty(2, 64, dtype=dtype), torch.ones(2))
