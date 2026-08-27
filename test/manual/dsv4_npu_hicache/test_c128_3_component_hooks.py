"""Task C128-3: host-aware C128 component hooks (validator / evict / free / lock).

Covers:
  1. create_match_validator accepts a device-only and a host-only node; with
     match_device_only=True only the device node is accepted.
  2. evict_component HOST / DEVICE / ALL: host_value lands in host_frees[C128]
     and is cleared; device page IDs stay in the refcount collection path;
     ALL collects both.
  3. free_host_values returns host values to the bound C128 host pool.
  4. host lock (acquire/release with lock_host=True) uses host_lock_ref.

Run inside container cx-dsv4:
    python3 test/manual/dsv4_npu_hicache/test_c128_3_component_hooks.py
"""
import torch
from types import SimpleNamespace

from sglang.srt.mem_cache.unified_cache.components import (
    ComponentData,
    ComponentType,
    EvictLayer,
)
from sglang.srt.hardware_backend.npu.dsv4.c128_sidecar_component import (
    C128SidecarComponent,
)

C128 = ComponentType.C128


def _cd(value=None, host_value=None):
    cd = ComponentData()
    cd.value = value
    cd.host_value = host_value
    return cd


class _FakeNode:
    def __init__(self, value=None, host_value=None):
        self.id = id(self)
        self.component_data = {C128: _cd(value, host_value)}


class _FakeTreeCore:
    def __init__(self):
        # Mirrors UnifiedTreeCore.__init__: pre-populated per component type.
        self.component_evictable_size_ = {C128: 0}
        self.is_write_back = False

    def _update_evictable_leaf_sets(self, node):
        pass


class _FakeHostPool:
    def __init__(self):
        self.freed = []

    def free(self, host_value):
        self.freed.append(host_value)


def _make_component():
    cache = SimpleNamespace(token_to_kv_pool_allocator=SimpleNamespace())
    comp = C128SidecarComponent(cache=cache, params=None)
    comp.tree_core = _FakeTreeCore()
    return comp


def main():
    dev_value = torch.tensor([3, 7])
    host_value = torch.tensor([1, 9])

    comp = _make_component()

    # ---- 1. host-aware validator ----
    dev_only = _FakeNode(value=dev_value, host_value=None)
    host_only = _FakeNode(value=None, host_value=host_value)

    broad = comp.create_match_validator(match_device_only=False)
    assert broad(dev_only), "device-only node rejected by broad validator"
    assert broad(host_only), "host-only node rejected by broad validator"

    device_only = comp.create_match_validator(match_device_only=True)
    assert device_only(dev_only), "device-only node rejected by match_device_only validator"
    assert not device_only(host_only), (
        "host-only node accepted with match_device_only=True"
    )

    # ---- 2. evict_component HOST / DEVICE / ALL ----
    # HOST eviction collects host_value only, leaves device value intact.
    device_frees = {C128: []}
    host_frees = {C128: []}
    host_node = _FakeNode(value=dev_value, host_value=host_value)
    comp.evict_component(host_node, device_frees, host_frees, target=EvictLayer.HOST)
    assert len(host_frees[C128]) == 1 and host_frees[C128][0] is host_value
    assert host_node.component_data[C128].host_value is None
    assert len(device_frees[C128]) == 0, "HOST eviction must not touch device value"
    assert host_node.component_data[C128].value is dev_value

    # DEVICE eviction collects page IDs, clears value, leaves host_value.
    device_frees = {C128: []}
    host_frees = {C128: []}
    host_node = _FakeNode(value=dev_value, host_value=host_value)
    comp.evict_component(host_node, device_frees, host_frees, target=EvictLayer.DEVICE)
    assert len(device_frees[C128]) == 1 and device_frees[C128][0] is dev_value
    assert host_node.component_data[C128].value is None
    assert len(host_frees[C128]) == 0
    assert host_node.component_data[C128].host_value is host_value

    # ALL eviction collects both.
    device_frees = {C128: []}
    host_frees = {C128: []}
    host_node = _FakeNode(value=dev_value, host_value=host_value)
    comp.evict_component(host_node, device_frees, host_frees, target=EvictLayer.ALL)
    assert len(device_frees[C128]) == 1
    assert len(host_frees[C128]) == 1
    assert host_node.component_data[C128].value is None
    assert host_node.component_data[C128].host_value is None

    # ---- 3. free_host_values returns values to the host pool ----
    pool = _FakeHostPool()
    comp._c128_kv_pool_host = pool
    comp.free_host_values([host_value])
    assert pool.freed == [host_value], "free_host_values must call host pool free"

    # ---- 4. host lock uses host_lock_ref ----
    host_node = _FakeNode(value=dev_value, host_value=host_value)
    cd = host_node.component_data[C128]
    comp.acquire_component_lock(host_node, None, lock_host=True)
    assert cd.host_lock_ref == 1, cd.host_lock_ref
    comp.release_component_lock(host_node, None, lock_host=True)
    assert cd.host_lock_ref == 0, cd.host_lock_ref

    print("PASS: C128 host-aware validator / evict / free / lock hooks")


if __name__ == "__main__":
    main()
