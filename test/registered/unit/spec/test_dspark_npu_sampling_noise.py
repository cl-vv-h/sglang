"""CPU checks for the A5 noise transform; graph capture needs NPU validation."""

from unittest.mock import patch

import torch

from sglang.kernels.ops.speculative.dspark import markov_sample_npu_legacy as sampler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def test_arch35_noise_uses_uniform_transform_in_place():
    torch.manual_seed(19)
    expected = -torch.empty(4, 97).uniform_(1e-8, 1.0).log()
    torch.manual_seed(19)
    actual = torch.empty_like(expected)
    pointer = actual.data_ptr()
    with (
        patch.object(sampler, "is_npu_arch35", return_value=True),
        patch.object(
            torch.Tensor, "exponential_", side_effect=AssertionError("capture unsafe")
        ),
    ):
        sampler._fill_exponential_noise(actual)
    assert actual.data_ptr() == pointer
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(actual).all() and (actual > 0).all()


def test_other_architectures_keep_exponential_rng():
    torch.manual_seed(19)
    expected = torch.empty(4, 97).exponential_(1.0)
    torch.manual_seed(19)
    actual = torch.empty_like(expected)
    with patch.object(sampler, "is_npu_arch35", return_value=False):
        sampler._fill_exponential_noise(actual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
