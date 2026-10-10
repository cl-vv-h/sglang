"""CPU contract tests for the actual eagle_sample body with mocked NPU calls.

Run: python -m pytest test/manual/test_npu_selected_row_dispatch.py -q
These validate routing/arguments/RNG/broadcasts, not device kernel arithmetic.
"""

import ast
import os
import sys
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


def install(monkeypatch, name, **members):
    module = ModuleType(name)
    module.__dict__.update(members)
    monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize(
    "case",
    [
        "enabled",
        "top_k",
        "disabled",
        "native",
        "triton_off",
        "branching",
        "rejection",
        "no_top_p",
        "bf16",
    ],
)
def test_eagle_sampling_contract(monkeypatch, case):
    calls = []
    selected = case in ("enabled", "top_k")
    monkeypatch.setenv(
        "SGL_KERNEL_NPU_SAMPLING_SELECTED_ROW", "0" if case == "disabled" else "1"
    )
    monkeypatch.setenv(
        "SGL_KERNEL_NPU_SAMPLING_TRITON", "0" if case == "triton_off" else "1"
    )
    monkeypatch.setenv(
        "SGL_KERNEL_NPU_SAMPLING_TRITON_DIV", "native" if case == "native" else "rn"
    )

    def top_k(probs, thresholds):
        calls.append("top_k")
        return probs

    def top_p(probs, thresholds):
        calls.append("top_p")
        return probs

    def sample(**kwargs):
        calls.append("sample")
        assert kwargs["target_probs"].ndim == (2 if selected else 3)
        if selected:
            assert kwargs["top_ps"].shape == (8,)
            assert kwargs["tree_topk"] == 1
            assert "draft_probs" not in kwargs and "deterministic" not in kwargs
        else:
            assert kwargs["draft_probs"].shape == (2, 4, 17)
            assert kwargs["deterministic"] is True
        kwargs["accept_token_num"].fill_(2)

    def coins(**kwargs):
        calls.append("coins")
        return torch.rand(2, 4), torch.rand(2)

    group = SimpleNamespace(
        world_size=2, broadcast=lambda *a, **kw: calls.append("broadcast")
    )
    install(
        monkeypatch,
        "sglang.srt.layers.dp_attention",
        is_dp_attention_enabled=lambda: True,
    )
    install(
        monkeypatch,
        "sglang.srt.runtime_context",
        get_parallel=lambda: SimpleNamespace(attn_tp_group=group, tp_group=group),
    )
    install(
        monkeypatch,
        "sglang.srt.sampling.penaltylib.repetition_penalty",
        apply_scaling_penalties=lambda *a: None,
    )
    install(
        monkeypatch,
        "sglang.srt.speculative.spec_utils",
        SIMULATE_ACC_LEN=0,
        SIMULATE_ACC_TOKEN_MODE="fixed",
        generate_simulated_accept_index=None,
    )
    install(
        monkeypatch,
        "sglang.srt.utils.async_probe",
        maybe_detect_nan=lambda *a: None,
        sanitize_nan_logits=lambda *a: None,
    )
    install(
        monkeypatch,
        "sgl_kernel_npu.sample",
        chain_speculative_sampling_triton=sample,
        top_k_renorm_prob=top_k,
        top_p_renorm_prob=top_p,
        tree_speculative_sampling_target_only=sample,
    )
    install(
        monkeypatch,
        "sgl_kernel_npu.sample.selected_row",
        top_p_tree_sampling_selected_row=sample,
    )

    source = (
        Path(__file__).resolve().parents[2]
        / "python/sglang/srt/speculative/eagle_utils.py"
    )
    tree = ast.parse(source.read_text())
    func = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "eagle_sample"
    )
    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            func,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(code)
    spec = SimpleNamespace(
        speculative_use_rejection_sampling=case == "rejection",
        speculative_use_block_verification=False,
        speculative_accept_threshold_single=1.0,
        speculative_accept_threshold_acc=1.0,
    )
    ns = dict(
        torch=torch,
        os=os,
        partial=partial,
        _is_npu=True,
        _is_cuda=False,
        _is_hip=False,
        _is_cpu=False,
        _is_xpu=False,
        get_spec=lambda: spec,
        _verify_coins=coins,
        _verify_uses_greedy=lambda **kw: False,
        _can_use_sparse_uno_tree_target_sampling=lambda *a: False,
        sampling_softmax=lambda x, temperatures: torch.softmax(
            x / temperatures.to(x.dtype), -1
        ),
    )
    exec(compile(code, str(source), "exec"), ns)
    info = SimpleNamespace(
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        logit_bias=None,
        is_all_greedy=False,
        temperatures=torch.ones(2, 1),
        need_top_k_sampling=case == "top_k",
        top_ks=torch.tensor([5, 5]),
        need_top_p_sampling=case != "no_top_p",
        top_ps=torch.tensor([0.95, 0.95]),
    )
    batch = SimpleNamespace(
        device="cpu",
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        seq_lens=torch.tensor([3, 3]),
        sampling_info=info,
    )
    verify = SimpleNamespace(
        draft_token=torch.zeros(8, dtype=torch.int64),
        draft_token_num=4,
        max_tree_depth=4,
        tree_topk=2 if case == "branching" else 1,
        retrieve_index=torch.arange(8).reshape(2, 4),
        retrieve_next_token=torch.zeros(2, 4, dtype=torch.int64),
        retrieve_next_sibling=torch.full((2, 4), -1),
        draft_probs=torch.zeros(2, 4, 17),
    )
    logits = SimpleNamespace(
        next_token_logits=torch.randn(
            8, 17, dtype=torch.bfloat16 if case == "bf16" else torch.float32
        )
    )
    _, lengths, _ = ns["eagle_sample"](verify, batch, logits)
    assert calls.count("coins") == calls.count("sample") == 1
    assert calls.count("broadcast") == 3
    assert calls.count("top_p") == int(not selected and case != "no_top_p")
    assert calls.count("top_k") == int(case == "top_k")
    if case == "top_k":
        assert calls.index("top_k") < calls.index("sample")
    assert torch.equal(lengths, torch.tensor([3, 3], dtype=torch.int32))
