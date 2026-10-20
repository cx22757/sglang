"""DSV4 coarse C128 storage objects and Mooncake hybrid-prefix recovery."""

import unittest
from queue import Queue
from types import SimpleNamespace

import torch

from sglang.srt.hardware_backend.npu.dsv4.c128_sidecar_component import (
    C128SidecarComponent,
)
from sglang.srt.mem_cache.base_prefix_cache import InsertResult
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
    resolve_pool_transfer_hits,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    PPPrefetchPoolSpec,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.srt.mem_cache.unified_cache.components import (
    CacheTransferPhase,
    ComponentData,
    ComponentType,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _backend(existing):
    backend = MooncakeStore.__new__(MooncakeStore)
    backend.mem_pool_host = SimpleNamespace(kv_buffer=None)
    backend.registered_pools = {
        name: SimpleNamespace(page_size=16) for name in PoolName
    }
    backend.config_prefix = "tenant"
    backend.mla_suffix = "0"
    backend.mha_suffix = "0"
    backend.is_mla_backend = True
    backend._batch_exist = lambda keys, extra_info=None: [
        int(key in existing) for key in keys
    ]
    return backend


def _key(index, pool):
    return f"tenant_h{index}_0_{pool}"


def _coarse():
    return PoolTransfer(
        name=PoolName.DEEPSEEK_V4_C128,
        keys=["__placeholder__"] * 2,
        logical_pages_per_object=16,
    )


def _component(page_size=16):
    component = C128SidecarComponent.__new__(C128SidecarComponent)
    component.cache = SimpleNamespace(
        token_to_kv_pool_allocator=SimpleNamespace(
            c128_attn_allocator=SimpleNamespace(page_size=page_size)
        )
    )
    root = SimpleNamespace(id=0, key=[], parent=None)
    tail = SimpleNamespace(
        id=1,
        key=list(range(128 * page_size)),
        parent=root,
        hash_value=[f"h{i}" for i in range(page_size)],
        component_data={ComponentType.C128: ComponentData()},
    )
    nodes = {0: root, 1: tail}
    resident = set()
    component.tree_core = SimpleNamespace(
        page_size=128,
        root_node=root,
        node_by_id=nodes.__getitem__,
        host_lru_lists={
            ComponentType.C128: SimpleNamespace(
                in_list=lambda node: node.id in resident,
                insert_mru=lambda node: resident.add(node.id),
            )
        },
        _update_evictable_leaf_sets=lambda node: None,
    )
    component._c128_kv_pool_host = SimpleNamespace(
        alloc=lambda size: torch.arange(size),
    )
    return component, root, tail, resident


class TestDSV4MooncakeL3(CustomTestCase):
    def test_coarse_objects_use_namespaced_endpoint_hashes(self):
        backend = _backend({_key(15, PoolName.DEEPSEEK_V4_C128)})
        result = backend.batch_exists_v2([f"h{i}" for i in range(32)], [_coarse()])
        self.assertEqual(result.kv_hit_pages, 16)
        self.assertEqual(result.restorable_prefix_pages, [16])
        self.assertEqual(result.extra_pool_hit_pages[PoolName.DEEPSEEK_V4_C128], 1)

    def test_missing_and_complete_coarse_objects(self):
        for objects, expected in (([], 0), ([15, 31], 32)):
            with self.subTest(objects=objects):
                backend = _backend(
                    {_key(i, PoolName.DEEPSEEK_V4_C128) for i in objects}
                )
                result = backend.batch_exists_v2(
                    [f"h{i}" for i in range(35)], [_coarse()]
                )
                self.assertEqual(result.kv_hit_pages, expected)
                self.assertEqual(
                    result.restorable_prefix_pages, list(range(16, expected + 1, 16))
                )

    def test_coarse_pool_intersects_trailing_windows_in_one_query(self):
        existing = {
            _key(15, PoolName.DEEPSEEK_V4_C128),
            _key(31, PoolName.DEEPSEEK_V4_C128),
            _key(14, PoolName.SWA),
            _key(15, PoolName.SWA),
            _key(30, PoolName.SWA),
        }
        backend = _backend(existing)
        queries = []

        def exists(keys, extra_info=None):
            queries.append(keys)
            return [int(key in existing) for key in keys]

        backend._batch_exist = exists
        trailing = PoolTransfer(
            name=PoolName.SWA,
            keys=["placeholder"] * 2,
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        result = backend.batch_exists_v2(
            [f"h{i}" for i in range(32)], [_coarse(), trailing]
        )
        self.assertEqual(result.restorable_prefix_pages, [16])
        self.assertEqual(len(queries), 1)
        self.assertEqual(len(queries[0]), 34)

    def test_physical_kv_prefix_limits_coarse_recovery(self):
        backend = _backend(
            {_key(15, PoolName.DEEPSEEK_V4_C128), _key(31, PoolName.DEEPSEEK_V4_C128)}
        )
        backend.mem_pool_host.kv_buffer = object()
        backend.batch_exists = lambda keys, extra_info: 24
        result = backend.batch_exists_v2([f"h{i}" for i in range(32)], [_coarse()])
        self.assertEqual(result.restorable_prefix_pages, [16])

    def test_short_backend_response_cannot_create_false_hits(self):
        backend = _backend(set())
        backend._batch_exist = lambda keys, extra_info: [1]
        result = backend.batch_exists_v2([f"h{i}" for i in range(32)], [_coarse()])
        self.assertEqual(result.restorable_prefix_pages, [16])

    def test_trailing_pools_preserve_common_nonmaximum_endpoint(self):
        transfers = [
            PoolTransfer(
                name=name, keys=["placeholder"], hit_policy=PoolHitPolicy.TRAILING_PAGES
            )
            for name in (
                PoolName.DEEPSEEK_V4_C4_STATE,
                PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
            )
        ]
        result = resolve_pool_transfer_hits(
            4, transfers, [[False, True, False, True], [False, True, True, False]]
        )
        self.assertEqual(result.restorable_prefix_pages, [2])
        self.assertEqual(result.kv_hit_pages, 2)

    def test_invalid_geometry_rejected(self):
        transfer = _coarse()
        for coverage in (0, -1):
            transfer.logical_pages_per_object = coverage
            with self.subTest(coverage=coverage), self.assertRaises(ValueError):
                _backend(set()).batch_exists_v2(["h0"], [transfer])
        with self.assertRaises(ValueError):
            resolve_pool_transfer_hits(1, [transfer], [])
        for slots in (15, 17, 32):
            transfer.host_indices = torch.arange(slots)
            transfer.keys = ["h15"]
            with (
                self.subTest(slots=slots),
                self.assertRaisesRegex(ValueError, "key/page mismatch"),
            ):
                _backend(set())._batch_io_v2([transfer], is_set=False)

    def test_c128_backup_and_prefetch_use_runtime_geometry(self):
        for page_size in (16, 32):
            with self.subTest(page_size=page_size):
                component, root, tail, _ = _component(page_size)
                tail.component_data[ComponentType.C128].host_value = torch.arange(
                    page_size
                )
                backup = component.build_hicache_transfers(
                    tail, CacheTransferPhase.BACKUP_STORAGE
                )[0]
                self.assertEqual(backup.keys, [f"h{page_size - 1}"])
                self.assertEqual(backup.logical_pages_per_object, page_size)
                prep = component.prepare_prefetch(
                    root.id, prefetch_tokens=128 * page_size
                )
                transfer = component.build_hicache_transfers(
                    root,
                    CacheTransferPhase.PREFETCH,
                    staging_tokens=prep.staging_tokens,
                )[0]
                self.assertEqual(transfer.keys, ["__placeholder__"])
                self.assertIsNone(transfer.host_indices)
                self.assertEqual(transfer.logical_pages_per_object, page_size)
                self.assertEqual(
                    component.align_storage_prefetch_length(
                        root.id, 128 * page_size + 127
                    ),
                    128 * page_size,
                )
                tail.key = tail.key[:-128]
                self.assertEqual(
                    component.align_storage_prefetch_length(tail.id, 128 * page_size), 0
                )

    def test_c128_commit_publishes_complete_group_and_frees_duplicate(self):
        component, root, tail, resident = _component()
        transfer = _coarse()
        transfer.host_indices = torch.arange(16)
        result = PoolTransferResult(16, {PoolName.DEEPSEEK_V4_C128: 1})
        actions = []
        for _ in range(2):
            component.commit_hicache_transfer(
                root,
                CacheTransferPhase.PREFETCH,
                [transfer],
                cache_actions=actions,
                insert_result=InsertResult(
                    prefix_len=0, total_len=2048, inserted_host_node=tail.id
                ),
                pool_storage_result=result,
            )
        self.assertTrue(
            torch.equal(
                tail.component_data[ComponentType.C128].host_value, torch.arange(16)
            )
        )
        self.assertEqual(resident, {tail.id})
        self.assertEqual(len(actions), 1)
        freed = []
        component._c128_kv_pool_host.free = lambda indices: freed.extend(
            indices.tolist()
        )
        component.apply_component_action(actions[0])
        self.assertEqual(freed, list(range(16)))

    def test_hit_time_staging_allocates_objects_not_logical_pages(self):
        component, _, _, _ = _component()
        transfer = _coarse()
        cache = SimpleNamespace(
            page_size=128,
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(
                    entry_map={
                        PoolName.DEEPSEEK_V4_C128: SimpleNamespace(
                            host_pool=SimpleNamespace(page_size=16)
                        )
                    }
                )
            ),
            components={ComponentType.C128: component},
        )
        info = SimpleNamespace(comp_xfers={ComponentType.C128: [transfer]})
        self.assertTrue(
            UnifiedRadixCache._alloc_prefetch_aux_staging(cache, info, 2048)
        )
        self.assertEqual(transfer.host_indices.numel(), 16)

    def test_pp_reconstructs_coarse_geometry_and_allocates_hit_objects(self):
        transfer = _coarse()
        spec = PPPrefetchPoolSpec.from_transfer(transfer)
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.page_size = 128
        controller.prefetch_tokens_occupied = 0
        controller.mem_pool_host = SimpleNamespace(
            alloc=lambda size, pool: torch.arange(size),
            get_pool=lambda name: SimpleNamespace(page_size=16),
        )
        ticket = SimpleNamespace(storage_hit_count=2048, pool_specs=[spec])
        operation = SimpleNamespace(pool_transfers=[transfer])
        self.assertTrue(controller._allocate_pp_prefetch_buffers(ticket, operation))
        restored = operation.pool_transfers[0]
        self.assertEqual(restored.logical_pages_per_object, 16)
        self.assertEqual(restored.host_indices.numel(), 16)
        self.assertEqual(operation.host_indices.numel(), 2048)

    def test_shared_arena_sizes_coarse_pool_after_hit_reduction(self):
        transfer = _coarse()
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.page_size = 128
        controller.mem_pool_host = SimpleNamespace(
            anchor_entry=SimpleNamespace(name=PoolName.KV),
            entry_map={
                transfer.name: SimpleNamespace(host_pool=SimpleNamespace(page_size=16))
            },
        )
        operation = SimpleNamespace(pool_transfers=[transfer], sidecar_hash_values=None)
        requests, _ = controller._shared_prefetch_requests(operation, 2048)
        self.assertEqual(requests, [(PoolName.KV, 2048), (transfer.name, 16)])

    def test_failed_c128_read_releases_staging_without_publication(self):
        component, root, tail, resident = _component()
        transfer = _coarse()
        transfer.host_indices = torch.arange(16)
        actions = []
        component.commit_hicache_transfer(
            root,
            CacheTransferPhase.PREFETCH,
            [transfer],
            cache_actions=actions,
            insert_result=InsertResult(
                prefix_len=0, total_len=2048, inserted_host_node=tail.id
            ),
            pool_storage_result=PoolTransferResult(16, {}),
        )
        self.assertIsNone(tail.component_data[ComponentType.C128].host_value)
        self.assertFalse(resident)
        self.assertEqual(len(actions), 1)
        self.assertTrue(torch.equal(actions[0].host_indices[0], transfer.host_indices))

    def test_read_keys_follow_reduced_hit_hash_chain(self):
        transfer = _coarse()
        transfer.host_indices = torch.arange(16)
        operation = SimpleNamespace(
            pool_transfers=[transfer],
            hash_value=[f"tenant-salt-h{i}" for i in range(16)],
            sidecar_hash_values=None,
            prefix_keys=None,
            request_id="request",
            is_terminated=lambda: False,
        )
        backend = SimpleNamespace()

        def read(transfers, extra_info=None):
            self.assertEqual(transfers[0].keys, ["tenant-salt-h15"])
            return {PoolName.DEEPSEEK_V4_C128: [True]}

        backend.batch_get_v2 = read
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.storage_backend = backend
        controller.prefetch_sync_queue = Queue()
        controller._page_transfer_sidecar(operation, 16)
        self.assertEqual(
            controller.prefetch_sync_queue.get_nowait().pool_hits,
            {PoolName.DEEPSEEK_V4_C128: 1},
        )


if __name__ == "__main__":
    unittest.main()
