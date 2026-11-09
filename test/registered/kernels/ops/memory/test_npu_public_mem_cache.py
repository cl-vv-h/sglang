"""NPU integration checks for SGLang's public mem-cache operator calls."""

import pytest
import torch

pytest.importorskip("torch_npu")

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=15, suite="base-b-test-1-npu-a3")

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Requires an NPU")


@pytest.mark.parametrize(
    "prefix,sequence,last,expected",
    [
        pytest.param(
            [0, 8, 5],
            [3, 8, 12],
            [-1, 31, 44],
            [80, 81, 82, 45, 46, 47, 96, 97, 98, 99],
            id="mixed-prefix-and-tail",
        ),
        pytest.param(
            [0, 8],
            [8, 16],
            [-1, 71],
            list(range(80, 88)) + list(range(96, 104)),
            id="full-page-end",
        ),
    ],
)
@pytest.mark.parametrize(
    "num_new_pages", [None, 2], ids=["computed-pages", "provided-pages"]
)
@pytest.mark.parametrize("input_dtype", [torch.int32, torch.int64])
def test_alloc_extend_preserves_allocator_state(
    prefix, sequence, last, expected, num_new_pages, input_dtype
):
    from sglang.srt.hardware_backend.npu.allocator_npu import (
        NPUPagedTokenToKVPoolAllocator,
    )

    allocator = NPUPagedTokenToKVPoolAllocator.__new__(NPUPagedTokenToKVPoolAllocator)
    allocator.page_size = 8
    allocator.roundup = 7
    allocator.device = "npu"
    allocator.debug_mode = False
    allocator.free_pages = torch.tensor(
        [10, 12, 14, 16], dtype=torch.int64, device="npu"
    )
    prefix_cpu = torch.tensor(prefix, dtype=torch.int64, device="cpu")
    sequence_cpu = torch.tensor(sequence, dtype=torch.int64, device="cpu")
    result = allocator.alloc_extend(
        prefix_cpu.to(device="npu", dtype=input_dtype),
        prefix_cpu,
        sequence_cpu.to(device="npu", dtype=input_dtype),
        sequence_cpu,
        torch.tensor(last, dtype=input_dtype, device="npu"),
        len(expected),
        num_new_pages,
    )
    torch.npu.synchronize()
    assert result.dtype == torch.int32
    torch.testing.assert_close(
        result.cpu(),
        torch.tensor(expected, dtype=torch.int32, device="cpu"),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        allocator.free_pages.cpu(),
        torch.tensor([14, 16], dtype=torch.int64, device="cpu"),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
def test_cache_locations_roundtrip_with_zero_length_request(slot_dtype):
    from sglang.kernels.ops.speculative.cache_locs import assign_extend_cache_locs_func
    from sglang.srt.mem_cache.allocation import assign_req_to_token_pool_func

    pool = torch.full((3, 8), -1, dtype=torch.int32, device="npu")
    requests = torch.tensor([2, 0, 1], dtype=torch.int64, device="npu")
    start = torch.tensor([1, 3, 0], dtype=torch.int64, device="npu")
    end = torch.tensor([3, 3, 1], dtype=torch.int64, device="npu")
    slots = torch.tensor([41, 42, 43], dtype=slot_dtype, device="npu")
    result = assign_req_to_token_pool_func(requests, pool, start, end, slots, 3)
    assert result is None
    gathered = assign_extend_cache_locs_func(requests, pool, start, end, 3, 1, "npu")
    torch.npu.synchronize()
    expected = torch.full((3, 8), -1, dtype=torch.int32, device="cpu")
    expected[2, 1:3] = torch.tensor([41, 42], dtype=torch.int32, device="cpu")
    expected[1, 0] = 43
    torch.testing.assert_close(pool.cpu(), expected, atol=0, rtol=0)
    torch.testing.assert_close(
        gathered.cpu(), slots.cpu().to(torch.int32), atol=0, rtol=0
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
