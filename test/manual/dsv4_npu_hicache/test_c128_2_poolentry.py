"""Task C128-2: C128 PoolEntry real P geometry + device alloc/free override + component binding.

Verifies the NPU C128 independent-index pool wiring in hybrid_pool_assembler:

  1. _COMPONENT_HOST_ATTR carries C128 -> ("c128_kv_pool_host", "_c128_kv_pool_host").
  2. _deepseek_v4_num_host_pages sizes the C128 host budget from the C128 device
     pool at hicache_ratio (decision §11.1) — not the FULL page count. GPU (no
     c128_attn_allocator) keeps FULL's count for the KV-derived sidecar.
  3. build_pool_entry carries device_alloc_fn / device_free_fn (bare c128
     allocator methods).
  4. _apply_stack_result binds _c128_kv_pool_host onto the C128 component.
  5. (NPU only) a real DeepSeekV4PagedHostPool mirrors slot_page_size == P.

Run inside container cx-dsv4:
    python3 test/manual/dsv4_npu_hicache/test_c128_2_poolentry.py
"""
import torch
from unittest.mock import MagicMock

from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _COMPONENT_HOST_ATTR,
    StackBuildResult,
    _apply_stack_result,
    _deepseek_v4_num_host_pages,
    build_pool_entry,
)
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.unified_cache.components import ComponentType

C128 = ComponentType.C128


def _num_host_pages(params, server_args, kvcache):
    return _deepseek_v4_num_host_pages(
        params=params,
        server_args=server_args,
        kvcache=kvcache,
        page_size=256,
        swa_page_size=32,
    )


def main():
    # ---- 1. _COMPONENT_HOST_ATTR completeness contract ----
    assert C128 in _COMPONENT_HOST_ATTR, "C128 missing from _COMPONENT_HOST_ATTR"
    assert _COMPONENT_HOST_ATTR[C128] == (
        "c128_kv_pool_host",
        "_c128_kv_pool_host",
    ), _COMPONENT_HOST_ATTR[C128]

    # ---- 2. C128 host budget scales with the C128 device pool (§11.1) ----
    params = MagicMock()
    server_args = MagicMock()
    server_args.hicache_size = 0
    server_args.hicache_ratio = 0.1
    kvcache = MagicMock()
    kvcache.size = 1000
    kvcache.swa_size = 200

    c128_alloc = MagicMock()
    c128_alloc.num_pages = 500
    params.token_to_kv_pool_allocator = MagicMock()
    params.token_to_kv_pool_allocator.size_full = 1000
    params.token_to_kv_pool_allocator.c128_attn_allocator = c128_alloc

    full_host, swa_host, c128_host = _num_host_pages(params, server_args, kvcache)
    assert c128_host == int(500 * 0.1), f"c128_host_pages={c128_host}, expected 50"
    assert c128_host != full_host, "C128 host budget must not reuse the FULL page count"

    # GPU path: no c128_attn_allocator -> C128 sidecar keeps FULL's budget.
    params.token_to_kv_pool_allocator = MagicMock()
    params.token_to_kv_pool_allocator.size_full = 1000
    params.token_to_kv_pool_allocator.c128_attn_allocator = None
    _, _, c128_host_gpu = _num_host_pages(params, server_args, kvcache)
    assert c128_host_gpu == full_host, "GPU C128 sidecar must reuse FULL host budget"

    # ---- 3. PoolEntry carries the device overrides ----
    alloc_fn, free_fn = MagicMock(), MagicMock()
    entry = build_pool_entry(
        name=PoolName.DEEPSEEK_V4_C128,
        host_pool=MagicMock(),
        device_pool=MagicMock(),
        layer_mapping={0: 0},
        transfer_layer_num=1,
        device_alloc_fn=alloc_fn,
        device_free_fn=free_fn,
    )
    assert entry.device_alloc_fn is alloc_fn, "device_alloc_fn not carried onto entry"
    assert entry.device_free_fn is free_fn, "device_free_fn not carried onto entry"

    # ---- 4. _apply_stack_result binds the C128 host pool onto the component ----
    cache = MagicMock()
    cache.components = {C128: MagicMock()}
    host_pool = MagicMock()
    kvcache = MagicMock()
    controller = MagicMock()
    result = StackBuildResult(
        host_pool_group=MagicMock(),
        cache_controller=controller,
        component_host_pools={C128: host_pool},
        sidecars=[],
        transfer_layer_num=1,
        pools_desc="KV + SWA + C128",
    )
    _apply_stack_result(cache, kvcache, params, result)
    assert cache.c128_kv_pool_host is host_pool, "cache.c128_kv_pool_host not bound"
    assert (
        cache.components[C128]._c128_kv_pool_host is host_pool
    ), "component._c128_kv_pool_host not bound"

    # ---- 5. (NPU) real host pool geometry mirrors slot_page_size == P ----
    from sglang.srt.utils import is_npu

    if is_npu():
        from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool

        P = 16
        layer_num = 2
        dim = 64
        device_buffers = [
            torch.zeros(8, P, 1, dim, dtype=torch.bfloat16, device="npu")
            for _ in range(layer_num)
        ]
        item_bytes = P * 1 * dim * torch.bfloat16.itemsize
        real_pool = DeepSeekV4PagedHostPool(
            pool_name="c128_geom_check",
            device_buffers=device_buffers,
            item_bytes=item_bytes,
            num_host_pages=c128_host,
            slot_page_size=P,
        )
        assert real_pool.slot_page_size == P, real_pool.slot_page_size
        assert real_pool.num_host_pages == c128_host, real_pool.num_host_pages
        assert real_pool.size == c128_host * P, real_pool.size
        print("  NPU real pool: slot_page_size=P, num_host_pages=c128_host*ratio OK")
    else:
        print("  SKIP NPU real-pool geometry check (not on NPU)")

    print("PASS: C128 PoolEntry geometry, overrides, and component binding")


if __name__ == "__main__":
    main()
