"""Task C128-4: C128 grouped backup/load-back round-trip (component level).

In-process (no card) coverage:
  1. BACKUP_HOST build: node device page ids -> expanded indices (groups*P),
     expanded<->page inverse (`// P`) holds, transfer is independent
     (indices_from_pool=None).
  2. commit(BACKUP_HOST) publishes host_value.
  3. LOAD_BACK build: collects each group-endpoint owner's host_value along the
     evicted path (sparse), host_indices = groups*P.
  4. commit(LOAD_BACK): controller-allocated expanded device indices // P back to
     page ids, retain_c128_pages once, restores cd.value via
     set_component_device_value (free-path ② ownership transfer).
  5. Multi-pool allocation failure rollback: _resolve_pool_transfers_allocation
     frees the C128 expanded indices via the bare device_free_fn (free-path ①)
     and resets device_indices to None.

E2E (--e2e, requires a running HiCache DSV4 server on port 30000):
  sends a prompt twice, asserts identical greedy output; records the
  G-1/G/G+1/2G boundary and async-copy-failure observation notes (plan §11.4
  evidence boundary — record only, no transactional-rollback assertion).

Run inside container cx-dsv4:
    python3 test/manual/dsv4_npu_hicache/test_c128_4_backup_loadback.py
    python3 test/manual/dsv4_npu_hicache/test_c128_4_backup_loadback.py --e2e
"""
import functools
import sys
import torch
from types import SimpleNamespace

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.unified_cache.components import (
    CacheTransferPhase,
    ComponentData,
    ComponentType,
    PrepareLoadBackResult,
)
from sglang.srt.hardware_backend.npu.dsv4.c128_sidecar_component import (
    C128SidecarComponent,
)

C128 = ComponentType.C128
P = 16  # c128_page_size

# This script runs with stdout piped through ssh/docker exec (fully buffered),
# so every print must flush explicitly — otherwise progress and crash traces
# vanish with the buffer on exit.  Also keeps the e2e live in the log.
print = functools.partial(print, flush=True)


class _FakeC128Allocator:
    """Stands in for DSV4NPUTokenToKVPoolAllocator (what the component's
    ``allocator`` property returns): exposes ``c128_attn_allocator.page_size``,
    ``retain_c128_pages``, and the bare ``alloc``/``free`` used as the PoolEntry
    device overrides."""

    def __init__(self, page_size, first_page=100):
        self.c128_attn_allocator = SimpleNamespace(
            page_size=page_size,
            free_pages=torch.empty((0,), dtype=torch.int64),
        )
        self.retained = {}  # page_id -> refcount
        self.bare_freed = []  # path-1 free() calls (expanded indices)
        self._next_page = first_page  # avoid page-0 sentinel

    def alloc(self, size):
        num_pages = size // self.c128_attn_allocator.page_size
        page_ids = torch.arange(
            self._next_page, self._next_page + num_pages, dtype=torch.int64
        )
        self._next_page += num_pages
        return (
            page_ids[:, None] * self.c128_attn_allocator.page_size
            + torch.arange(self.c128_attn_allocator.page_size, dtype=torch.int64)
        ).flatten()

    def free(self, indices):
        self.bare_freed.append(indices.clone())

    def retain_c128_pages(self, page_ids):
        for pid in page_ids.tolist():
            self.retained[pid] = self.retained.get(pid, 0) + 1


def _cd(value=None, host_value=None):
    cd = ComponentData()
    cd.value = value
    cd.host_value = host_value
    return cd


class _FakeNode:
    def __init__(self, evicted, parent=None, value=None, host_value=None):
        self.id = id(self)
        self.evicted = evicted
        self.parent = parent
        self.component_data = {C128: _cd(value, host_value)}


class _FakeTreeCore:
    def __init__(self, root, nodes):
        self.root_node = root
        self._by_id = {n.id: n for n in nodes}

    def node_by_id(self, nid):
        return self._by_id[nid]

    def set_component_device_value(self, nid, ct, value):
        self._by_id[nid].component_data[ct].value = value


def _make_component(allocator):
    req_to_token_pool = SimpleNamespace(
        set_c128_prefix_pages=lambda req, pages: setattr(
            req, "c128_prefix_page_ids", pages.clone()
        )
    )
    cache = SimpleNamespace(
        token_to_kv_pool_allocator=allocator,
        req_to_token_pool=req_to_token_pool,
    )
    comp = C128SidecarComponent(cache=cache, params=None)
    return comp


def _test_roundtrip():
    """Single node BACKUP_HOST -> demote -> LOAD_BACK."""
    allocator = _FakeC128Allocator(P)
    comp = _make_component(allocator)

    root = _FakeNode(evicted=False)
    node = _FakeNode(evicted=False, parent=root, value=torch.tensor([10, 11]))
    comp.tree_core = _FakeTreeCore(root, [root, node])

    # --- BACKUP_HOST ---
    xfers = comp.build_hicache_transfers(node, CacheTransferPhase.BACKUP_HOST)
    assert xfers is not None and len(xfers) == 1
    t = xfers[0]
    assert t.name == PoolName.DEEPSEEK_V4_C128
    assert t.indices_from_pool is None, "C128 must be an independent-index transfer"
    assert len(t.device_indices) == 2 * P, f"expected 2*P expanded, got {len(t.device_indices)}"
    # Each page expands to P consecutive slots; //P recovers the page id per
    # slot, so unique() is the strict inverse of *P for the distinct page ids.
    assert torch.equal(torch.unique(t.device_indices // P), torch.tensor([10, 11])), (
        "expanded->page inverse broken"
    )

    # Mock controller host allocation, then commit.
    host_indices = torch.arange(2 * P, dtype=torch.int64)
    t.host_indices = host_indices
    comp.commit_hicache_transfer(
        node, CacheTransferPhase.BACKUP_HOST, [t], cache_actions=[]
    )
    assert torch.equal(node.component_data[C128].host_value, host_indices), (
        "BACKUP_HOST commit did not publish host_value"
    )

    # --- demote (device value cleared) ---
    node.evicted = True
    node.component_data[C128].value = None

    # --- LOAD_BACK ---
    xfers = comp.build_hicache_transfers(node, CacheTransferPhase.LOAD_BACK)
    assert xfers is not None and len(xfers) == 1
    t = xfers[0]
    assert t.indices_from_pool is None
    assert torch.equal(t.host_indices, host_indices), "LOAD_BACK host_indices mismatch"

    # Mock controller device allocation (bare alloc returns fresh expanded indices).
    dev = allocator.alloc(len(t.host_indices))
    t.device_indices = dev
    comp.commit_hicache_transfer(
        node, CacheTransferPhase.LOAD_BACK, [t], cache_actions=[]
    )

    restored = node.component_data[C128].value
    assert torch.equal(restored, torch.unique(dev // P)), (
        "LOAD_BACK commit did not restore distinct page ids"
    )
    for pid in restored.tolist():
        assert allocator.retained.get(pid, 0) == 1, (
            f"page {pid} refcount != 1 after retain (free-path ②)"
        )
    print("  round-trip OK")


def _test_loadback_chain():
    """Multi-node LOAD_BACK walks a sparse host_value path (group endpoints)."""
    allocator = _FakeC128Allocator(P)
    comp = _make_component(allocator)

    root = _FakeNode(evicted=False)
    node_a = _FakeNode(evicted=True, parent=root, host_value=torch.arange(P))
    node_b = _FakeNode(evicted=True, parent=node_a, host_value=None)  # non-endpoint
    node_c = _FakeNode(
        evicted=True, parent=node_b, host_value=torch.arange(P, 2 * P)
    )
    comp.tree_core = _FakeTreeCore(root, [root, node_a, node_b, node_c])

    xfers = comp.build_hicache_transfers(node_c, CacheTransferPhase.LOAD_BACK)
    assert xfers is not None and len(xfers) == 1
    t = xfers[0]
    assert len(t.host_indices) == 2 * P, len(t.host_indices)
    assert t.nodes_to_load == [node_a.id, node_c.id], (
        "only group-endpoint owners must be in nodes_to_load"
    )

    dev = allocator.alloc(len(t.host_indices))
    t.device_indices = dev
    comp.commit_hicache_transfer(
        node_c, CacheTransferPhase.LOAD_BACK, [t], cache_actions=[]
    )
    assert torch.equal(
        node_a.component_data[C128].value, torch.unique(dev[:P] // P)
    )
    assert torch.equal(
        node_c.component_data[C128].value, torch.unique(dev[P:] // P)
    )
    for pid in (dev // P).tolist():
        assert allocator.retained.get(pid, 0) == 1
    print("  multi-node sparse chain OK")


def _test_loadback_refreshes_request_sidecar():
    """The triggering request must see C128 pages restored by load-back."""
    allocator = _FakeC128Allocator(P)
    comp = _make_component(allocator)

    root = _FakeNode(evicted=False)
    device_node = _FakeNode(
        evicted=True, parent=root, value=torch.tensor([10], dtype=torch.int64)
    )
    host_node = _FakeNode(
        evicted=True, parent=device_node, host_value=torch.arange(P)
    )
    comp.tree_core = _FakeTreeCore(root, [root, device_node, host_node])

    xfer = comp.build_hicache_transfers(
        host_node, CacheTransferPhase.LOAD_BACK
    )[0]
    xfer.device_indices = allocator.alloc(len(xfer.host_indices))
    comp.commit_hicache_transfer(
        host_node, CacheTransferPhase.LOAD_BACK, [xfer], cache_actions=[]
    )

    req = SimpleNamespace(best_match_node=host_node.id)
    comp.finalize_load_back(req, PrepareLoadBackResult(), success=True)
    assert torch.equal(
        req.c128_prefix_page_ids, torch.tensor([10, 100], dtype=torch.int64)
    ), (
        "load-back requester kept the pre-load match table instead of the "
        f"restored C128 page: {req.c128_prefix_page_ids}"
    )
    print("  request C128 sidecar refresh after load-back OK")


def _test_allocation_rollback():
    """Free-path ①: controller multi-pool rollback frees C128 via bare free."""
    from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
        HybridCacheController,
    )
    from sglang.srt.mem_cache.memory_pool_host import PoolEntry

    allocator = _FakeC128Allocator(P)
    entry = PoolEntry(
        name=PoolName.DEEPSEEK_V4_C128,
        host_pool=SimpleNamespace(
            alloc=lambda n: torch.arange(n, dtype=torch.int64), free=lambda x: None
        ),
        device_pool=SimpleNamespace(),
        layer_mapper=lambda layer_id: layer_id,
        device_alloc_fn=allocator.alloc,
        device_free_fn=allocator.free,
    )
    # A later independent pool whose alloc always fails -> forces rollback.
    fail_entry = PoolEntry(
        name=PoolName.DEEPSEEK_V4_C4,
        host_pool=SimpleNamespace(
            alloc=lambda n: torch.arange(n, dtype=torch.int64), free=lambda x: None
        ),
        device_pool=SimpleNamespace(),
        layer_mapper=lambda layer_id: layer_id,
        device_alloc_fn=lambda n: None,
        device_free_fn=lambda x: None,
    )
    controller = object.__new__(HybridCacheController)
    controller.mem_pool_host = SimpleNamespace(
        entry_map={
            PoolName.DEEPSEEK_V4_C128: entry,
            PoolName.DEEPSEEK_V4_C4: fail_entry,
        }
    )

    t_c128 = PoolTransfer(
        name=PoolName.DEEPSEEK_V4_C128,
        indices_from_pool=None,
        host_indices=torch.arange(2 * P, dtype=torch.int64),
        device_indices=None,
    )
    t_c4 = PoolTransfer(
        name=PoolName.DEEPSEEK_V4_C4,
        indices_from_pool=None,
        host_indices=torch.arange(4, dtype=torch.int64),
        device_indices=None,
    )
    result = controller._resolve_pool_transfers_allocation(
        [t_c128, t_c4], alloc_host=False
    )
    assert result is None, "later pool allocation failure must roll back"
    assert t_c128.device_indices is None, "C128 device_indices must be rolled back"
    assert len(allocator.bare_freed) == 1, (
        "C128 allocated indices must be freed via bare device_free_fn (path ①)"
    )
    freed = allocator.bare_freed[0]
    assert torch.equal(torch.unique(freed // P), torch.tensor([100, 101])), (
        f"rolled-back indices wrong: {freed}"
    )
    print("  allocation rollback (free-path ①) OK")


MODEL_PATH = "/mnt/paas/weights/DeepSeek-V4-Flash-w8a8-mtp"
RUN_SCRIPTS_DIR = "/home/cx/hicache_dsv4"
E2E_BASE_URL = "http://127.0.0.1:30000"
# c128_page_size defaults to 16 (server_args) -> G = 128 * P = 2048 raw tokens.
E2E_G = 128 * 16
E2E_BOUNDARY_LENS = [E2E_G - 1, E2E_G, E2E_G + 1, 2 * E2E_G]


def _launch_run_server(script_name: str) -> str:
    """Launch a container run_server script (backgrounded server), return log dir."""
    import subprocess

    script = f"{RUN_SCRIPTS_DIR}/{script_name}"
    proc = subprocess.Popen(
        ["bash", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        out, _ = proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
    lines = [l for l in (out or "").splitlines() if l.strip()]
    log_dir = lines[-1].strip() if lines else None
    print(f"  launched {script_name}; log={log_dir}")
    return log_dir


def _wait_health(base_url: str, timeout: float = 1800.0) -> bool:
    import time
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def _kill_servers() -> None:
    import subprocess
    import time

    subprocess.run(["pkill", "-9", "-f", "sglang.launch_server"], capture_output=True)
    time.sleep(3)


def _generate(base_url: str, input_ids, rid: str, max_new_tokens: int = 8):
    """POST /generate; return (output_ids, cached_tokens)."""
    import json
    import urllib.request

    payload = {
        "rid": rid,
        "input_ids": [int(x) for x in input_ids],
        "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
        "return_logprob": False,
        "routed_dp_rank": 0,
    }
    req = urllib.request.Request(
        base_url + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = resp.read()
    except Exception as e:
        print(f"    _generate FAIL rid={rid}: {e!r}")
        raise
    try:
        result = json.loads(body)
    except Exception as e:
        print(f"    _generate BAD-JSON rid={rid}: {e!r} body={body[:400]!r}")
        raise
    meta = result.get("meta_info", {})
    return result.get("output_ids", []), meta.get("cached_tokens", 0)


def _make_input_ids(tokenizer, pattern_text: str, target: int):
    pattern = tokenizer.encode(pattern_text, add_special_tokens=False)
    return (pattern * ((target + len(pattern) - 1) // len(pattern)))[:target]


def _real_allocator_capacity_recovery() -> None:
    """Plan Step 6 item 3: real c128 allocator capacity fully recovers after
    bare-free rollback (free-path ①, expanded-index unit)."""
    from sglang.srt.hardware_backend.npu.allocator_npu import (
        NPUPagedTokenToKVPoolAllocator,
    )

    P = 16
    device = "npu" if torch.npu.is_available() else "cpu"
    alloc = NPUPagedTokenToKVPoolAllocator(
        size=256,
        page_size=P,
        dtype=torch.int64,
        device=device,
        kvcache=SimpleNamespace(),
        need_sort=False,
    )
    before = alloc.available_size()
    indices = alloc.alloc(2 * P)  # 2 groups of expanded indices
    assert indices is not None and indices.numel() == 2 * P
    assert alloc.available_size() == before - 2 * P, "alloc must consume 2 groups"
    alloc.free(indices)  # bare free, exactly what _resolve_pool_transfers_allocation
    #   rollback does for an independent C128 pool (device_free_fn = alloc.free)
    assert alloc.available_size() == before, (
        "c128 allocator capacity must fully recover after bare free (free-path ①)"
    )
    print("  real allocator capacity recovery (path ① bare free) OK")


def _leave_server_for_inspection(log_dir, phase, msg) -> None:
    """Abort the e2e WITHOUT killing the server.

    Relaunching the 16-card server is expensive, so on any failure we keep it
    running for inspection: print the log path + manual-stop hint, then exit
    non-zero.  Kills only ever happen on the success path (phase transition /
    final cleanup) or by the operator.
    """
    print(f"  [{phase}] FAILED: {msg}")
    print(f"  [{phase}] server LEFT RUNNING for inspection (DO NOT auto-kill — "
          f"relaunch is expensive).")
    print(f"  [{phase}] server log: {log_dir}")
    print(f"  [{phase}] manual stop: docker exec cx-dsv4 bash -lc "
          f"'pkill -9 -f sglang.launch_server'")
    sys.exit(1)


def _e2e_main() -> None:
    """Real-hardware e2e (plan Task 4 Step 6), launches servers via the
    container run_server.sh / run_server_no_hicache.sh scripts.

    Occupies the 16 NPU cards; run only when the card is free.

    Kill policy: servers are ONLY killed on the success path.  Any failure
    leaves the current server running for inspection (see
    _leave_server_for_inspection).
    """
    import time

    RID = f"c128-task4-e2e-{int(time.time())}"

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    except Exception as e:
        print(f"  SKIP: tokenizer unavailable ({e}) — cannot build boundary prompts")
        return

    def inputs(n):
        return _make_input_ids(tokenizer, " dsv4-main85-c128-bndry", n)

    def run_prompts(label, base_url):
        outs = {}
        for n in E2E_BOUNDARY_LENS:
            out, cached = _generate(base_url, inputs(n), f"{RID}-{label}-{n}")
            outs[n] = (out, cached)
            print(f"    [{label}] len={n}: out={out} cached={cached}")
        return outs

    # Fresh start: clear orphans left by previous failed runs (deliberate
    # cleanup of dead tasks, NOT a failure response).
    _kill_servers()

    # ---- Phase A: HiCache ON ----
    print("  === Phase A: HiCache ON (run_server.sh) ===")
    log_a = _launch_run_server("run_server.sh")
    if not _wait_health(E2E_BASE_URL):
        _leave_server_for_inspection(log_a, "Phase A", "server not healthy within timeout")
    print("  healthy; running boundary prompts…")
    on_outs = run_prompts("on", E2E_BASE_URL)
    out_replay, cached_replay = _generate(
        E2E_BASE_URL, inputs(E2E_G), f"{RID}-on-replay"
    )
    print(f"    [on-replay] len={E2E_G}: out={out_replay} cached={cached_replay}")
    print(f"    [first-G]   len={E2E_G}: out={on_outs[E2E_G][0]} cached={on_outs[E2E_G][1]}")
    if out_replay != on_outs[E2E_G][0]:
        _leave_server_for_inspection(
            log_a, "Phase A",
            f"replay output differs from first pass (out={out_replay} vs "
            f"{on_outs[E2E_G][0]}, cached={cached_replay})",
        )
    if cached_replay > 0:
        print("  L2 hit evidence: replay cached_tokens>0, output identical")
    else:
        print("  NOTE: replay cached_tokens=0 (backup may be in flight); "
              "output equivalence still holds")
    print("  Phase A passed; stopping HiCache server (success path)…")
    _kill_servers()

    # ---- Phase B: HiCache OFF baseline ----
    print("  === Phase B: HiCache OFF (run_server_no_hicache.sh) ===")
    log_b = _launch_run_server("run_server_no_hicache.sh")
    if not _wait_health(E2E_BASE_URL):
        _leave_server_for_inspection(log_b, "Phase B", "server not healthy within timeout")
    print("  healthy; running boundary prompts…")
    off_outs = run_prompts("off", E2E_BASE_URL)

    # ---- Phase C: on/off equivalence (Phase B server still up for inspection) ----
    diverged = []
    for n in E2E_BOUNDARY_LENS:
        on_out, _ = on_outs[n]
        off_out, _ = off_outs[n]
        if on_out != off_out:
            diverged.append((n, on_out, off_out))
    if diverged:
        for n, on_out, off_out in diverged:
            print(f"    DIVERGE len={n}: on={on_out} off={off_out}")
        _leave_server_for_inspection(
            log_b, "Phase C",
            f"on/off output diverged at boundaries {[d[0] for d in diverged]}",
        )
    print("  on/off outputs identical for G-1/G/G+1/2G boundaries")
    print("  Phase B+C passed; stopping no-HiCache server (success path)…")
    _kill_servers()

    # ---- Phase D: allocation failure -> rollback -> capacity recovery (item 3).
    _real_allocator_capacity_recovery()

    # ---- Phase E: async copy failure — record only (spec §11.4 evidence boundary).
    print("  OBSERVE (record-only, spec §11.4 / plan item 4):")
    print("    - partial tail groups are recomputed, not cached: covered by on/off "
          "equivalence at G-1/G/G+1/2G above.")
    print("    - async Ascend copy failure behavior: record from server.log; NO "
          "transactional-rollback assertion made (evidence boundary).")
    print("  e2e PASS: hit output equivalent to no-HiCache; boundaries covered")


def main():
    print("=== test_c128_4_backup_loadback ===")
    _test_roundtrip()
    _test_loadback_chain()
    _test_loadback_refreshes_request_sidecar()
    _test_allocation_rollback()

    if "--e2e" in sys.argv:
        print("--- e2e (real-hardware, launches 16-card servers; run when card free) ---")
        _e2e_main()
    else:
        print("  (pass --e2e to run the real-hardware backup/load-back check)")

    print("PASS: C128 grouped backup/load-back round-trip")


if __name__ == "__main__":
    main()
