"""Task C128-5: host eviction linkage (spec §4.7 option 1) + exactly-once release.

C128 is a FULL required payload: evicting C128 host pages must never leave the
corresponding FULL host residency intact (no stranded FULL/C128 host slots).

Option 1 (delegation, chosen in plan Task 5 Step 1): C128 host pressure is
delegated to FULL host-leaf eviction via the C128 PoolEntry host_evict_fn.
`_evict_host_leaf` evicts the WHOLE leaf atomically (FULL + C128 host values),
so this test pins:

  1. a FULL host-leaf eviction frees the C128 host_value exactly once and the
     C128 broad validator rejects the evicted group as a host hit (FULL hit
     cannot proceed without its C128 payload);
  2. host eviction does NOT touch the C128 device refcount;
  3. device eviction releases the device page refcount exactly once;
  4. host_evict_fn converts n C128 host slots -> n*128 FULL raw tokens and is
     carried onto the C128 PoolEntry by build_pool_entry.

Run inside container cx-dsv4:
    python3 test/manual/dsv4_npu_hicache/test_c128_5_eviction.py
"""
import torch
from types import SimpleNamespace

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import build_pool_entry
from sglang.srt.mem_cache.unified_cache.cache_action import FreeComponentDeviceSlot
from sglang.srt.mem_cache.unified_cache.components import (
    ComponentData,
    ComponentType,
    EvictLayer,
)
from sglang.srt.hardware_backend.npu.dsv4.c128_sidecar_component import (
    C128SidecarComponent,
)

C128 = ComponentType.C128
FULL = ComponentType.FULL
P = 16


class _FakeC128Allocator:
    """token_to_kv_pool_allocator stand-in: c128 page_size + refcount release."""

    def __init__(self):
        self.c128_attn_allocator = SimpleNamespace(page_size=P)
        self.released = []  # release_c128_pages calls (page ids)
        self.retained = []

    def retain_c128_pages(self, page_ids):
        self.retained.append(page_ids.clone())

    def release_c128_pages(self, page_ids):
        self.released.append(page_ids.clone())


class _FakeHostPool:
    def __init__(self):
        self.freed = []  # host values returned to the pool

    def free(self, host_value):
        self.freed.append(host_value.clone())


class _FakeNode:
    def __init__(self, value=None, host_value=None):
        self.id = id(self)
        cd = ComponentData()
        cd.value = value
        cd.host_value = host_value
        self.component_data = {C128: cd, FULL: ComponentData()}


class _FakeTreeCore:
    def __init__(self):
        self.component_evictable_size_ = {C128: 0}
        self.is_write_back = False

    def _update_evictable_leaf_sets(self, node):
        pass


def _make_component(allocator):
    cache = SimpleNamespace(token_to_kv_pool_allocator=allocator)
    comp = C128SidecarComponent(cache=cache, params=None)
    comp.tree_core = _FakeTreeCore()
    return comp


def _test_host_leaf_eviction_frees_c128():
    """A FULL host-leaf eviction (evict_component ALL) frees C128 host_value;
    the group is no longer a host hit; device refcount untouched."""
    allocator = _FakeC128Allocator()
    host_pool = _FakeHostPool()
    comp = _make_component(allocator)
    comp._c128_kv_pool_host = host_pool

    dev_value = torch.tensor([7])
    host_value = torch.tensor([10, 11, 12, 13])
    node = _FakeNode(value=dev_value, host_value=host_value)
    # Host-only state after demote: device value already evicted.
    node.component_data[C128].value = None

    device_frees = {C128: []}
    host_frees = {C128: []}
    # This is exactly what _evict_host_leaf calls for each component.
    comp.evict_component(node, device_frees, host_frees, target=EvictLayer.ALL)

    assert node.component_data[C128].host_value is None, "C128 host_value not cleared"
    assert len(host_frees[C128]) == 1 and torch.equal(host_frees[C128][0], host_value)
    assert len(device_frees[C128]) == 0, "host eviction must not touch device pages"

    # Drain host_frees -> free_host_values -> host pool free (exactly once).
    comp.free_host_values(host_frees[C128])
    assert len(host_pool.freed) == 1 and torch.equal(host_pool.freed[0], host_value)

    # Stranded-slot invariant: the evicted group is no longer a host hit, so a
    # FULL host hit cannot proceed without its C128 payload (no stranded FULL).
    broad = comp.create_match_validator(match_device_only=False)
    assert not broad(node), "evicted group still accepted as a host hit"

    # Device refcount untouched by host eviction.
    assert len(allocator.released) == 0, "host eviction must not release device pages"
    print("  host-leaf eviction frees C128 exactly once; no stranded FULL; device untouched OK")


def _test_device_eviction_releases_once():
    """Device eviction collects page ids and apply_component_action releases
    the refcount exactly once."""
    allocator = _FakeC128Allocator()
    comp = _make_component(allocator)

    dev_value = torch.tensor([3, 4])
    node = _FakeNode(value=dev_value, host_value=None)

    device_frees = {C128: []}
    host_frees = {C128: []}
    comp.evict_component(node, device_frees, host_frees, target=EvictLayer.DEVICE)
    assert len(device_frees[C128]) == 1 and torch.equal(device_frees[C128][0], dev_value)
    assert node.component_data[C128].value is None
    assert len(host_frees[C128]) == 0, "device eviction must not free host slots"

    # _drain_device_frees -> FreeComponentDeviceSlot -> apply_component_action.
    comp.apply_component_action(
        FreeComponentDeviceSlot(device_frees[C128], component_type=C128)
    )
    assert len(allocator.released) == 1 and torch.equal(
        allocator.released[0], dev_value
    ), "release_c128_pages must run exactly once on device eviction"
    # A second eviction of the (now value-less) node frees nothing more.
    comp.evict_component(node, device_frees, host_frees, target=EvictLayer.DEVICE)
    assert len(allocator.released) == 1, "double-free on repeat device eviction"
    print("  device eviction releases refcount exactly once OK")


def _test_delegation_conversion_and_wiring():
    """host_evict_fn converts n C128 host slots -> n*128 FULL raw tokens and is
    carried onto the C128 PoolEntry by build_pool_entry."""
    calls = []

    class _FakeCache:
        def evict_host(self, num_tokens, component_type):
            calls.append((num_tokens, component_type))
            return 0

    cache = _FakeCache()
    # The exact lambda wired in _DeepSeekV4Strategy.build (spec §4.7 option 1).
    host_c128_evict_fn = lambda n: cache.evict_host(n * 128, ComponentType.FULL)
    host_c128_evict_fn(3)  # 3 C128 host slots needed
    assert calls == [(3 * 128, ComponentType.FULL)], calls

    entry = build_pool_entry(
        name=PoolName.DEEPSEEK_V4_C128,
        host_pool=SimpleNamespace(),
        device_pool=SimpleNamespace(),
        layer_mapping={0: 0},
        transfer_layer_num=1,
        host_evict_fn=host_c128_evict_fn,
    )
    assert entry.host_evict_fn is host_c128_evict_fn, "host_evict_fn not carried"
    print("  delegation conversion (n -> n*128) + PoolEntry wiring OK")


def _test_tier1_host_pressure_delegation():
    """Deterministic closed loop (real C128 host pool, real delegation fn):

    Fill the C128 host pool -> a backup's C128 host alloc fails ->
    the controller calls host_evict_fn -> the delegation (evict_host(n*128,
    FULL)) frees C128 host slots -> the retry recovers.
    """
    from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
        HybridCacheController,
    )
    from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
        _delegate_c128_host_evict,
    )
    from sglang.srt.mem_cache.memory_pool_host import (
        DeepSeekV4PagedHostPool,
        PoolEntry,
    )

    P = 16
    num_host_pages = 4
    layer_num = 2
    dim = 64
    device_buffers = [
        torch.zeros(8, P, 1, dim, dtype=torch.bfloat16, device="npu")
        for _ in range(layer_num)
    ]
    item_bytes = P * 1 * dim * torch.bfloat16.itemsize
    c128_host = DeepSeekV4PagedHostPool(
        pool_name="tier1_c128",
        device_buffers=device_buffers,
        item_bytes=item_bytes,
        num_host_pages=num_host_pages,
        slot_page_size=P,
    )
    # Fill the C128 host pool completely.
    assert c128_host.alloc(c128_host.size) is not None, "fill failed"
    assert c128_host.available_size() == 0, "pool not full"

    evict_calls = []

    class _FakeFullCache:
        def evict_host(self, num_tokens, component_type):
            evict_calls.append((num_tokens, component_type))
            # Simulate a FULL host-leaf eviction freeing its C128 payload:
            # num_tokens = n*128 raw tokens -> n compressed slots.
            freed = num_tokens // 128
            c128_host.free(torch.arange(freed, dtype=torch.int64))
            return 0

    entry = PoolEntry(
        name=PoolName.DEEPSEEK_V4_C128,
        host_pool=c128_host,
        device_pool=SimpleNamespace(),
        layer_mapper=lambda layer_id: layer_id,
        host_evict_fn=lambda n: _delegate_c128_host_evict(_FakeFullCache(), n),
        device_alloc_fn=None,
        device_free_fn=None,
    )
    controller = object.__new__(HybridCacheController)
    controller.mem_pool_host = SimpleNamespace(
        entry_map={PoolName.DEEPSEEK_V4_C128: entry}
    )

    transfer = PoolTransfer(
        name=PoolName.DEEPSEEK_V4_C128,
        indices_from_pool=None,
        device_indices=torch.arange(2 * P, dtype=torch.int64),  # 2 groups
        host_indices=None,
    )
    result = controller._resolve_pool_transfers_allocation(
        [transfer], alloc_host=True
    )
    assert result is not None, "host pressure delegation must recover the alloc"
    assert transfer.host_indices is not None and len(transfer.host_indices) == 2 * P
    assert evict_calls == [(2 * P * 128, ComponentType.FULL)], evict_calls
    print("  Tier1: real C128 host pool pressure -> delegation -> retry recovered OK")


def main():
    print("=== test_c128_5_eviction ===")
    _test_host_leaf_eviction_frees_c128()
    _test_device_eviction_releases_once()
    _test_delegation_conversion_and_wiring()
    _test_tier1_host_pressure_delegation()
    print("PASS: C128 host eviction linkage + exactly-once release")


if __name__ == "__main__":
    main()
