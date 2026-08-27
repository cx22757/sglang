# DSV4 NPU HiCache 最近两份提交功能讲解（供代码理解参考）

> 日期 2026-08-17 · 分支 `dev/main-8.5` · 工作树 HEAD = `4c26f9b005`
> 目的：按功能梳理最近两份提交改了什么、为什么，作为后续提问的参考。**不含测试代码与文档**。
> 关联：实现计划 `plans/2026-08-15-dsv4-npu-hicache-plan2-c128-l1l2.md`；LOADS_BACK 触发问题见 `plans/2026-08-17-dsv4-npu-loadback-handoff.md`。

```text
735ffc045b  feat(dsv4-npu-hicache): DSV4 NPU hierarchical KV cache — L1<->L2 data plane + C128 independent-index pool
            （Plan-1 数据面 + Plan-2 C128 L1↔L2，18 个 commit squash，8/16）
4c26f9b005  feat: add C128 load back test
            （实机 e2e 暴露的 C4 索引修复 + C128 host-LRU 修复 + 调试日志，8/17）
```

---

## 提交 1 `735ffc045b`

### A. 数据面（Plan-1）

#### A1. `kernel_ascend` 搬运后端（`memory_pool_host.py`）
给两类 DSV4 host pool 增加 `io_backend="kernel_ascend"` 分支，调用 NPU 通用搬运 op `sgl_kernel_npu.kvcacheio.transfer_kv_dim_exchange`（复用，不新增 kernel）。

- **Paged pool**（SWA/C4/C128 KV，`memory_pool_host.py:1147-1289`）：
  - D2H（`backup_from_device_all_layer`）逐层循环提交；H2D（`load_to_device_per_layer`）只处理当前 `layer_id`。
  - 新增 `_host_page_view(l)`：把 host buffer 转成 op 要求的 5-D view `[num_host_pages, 1, slot_page_size, 1, kv_dim]`。
  - V 传 empty tensor：DSV4 paged pool 只存 K，无 K/V split。
- **State pool**（C4 attention/indexer state，`memory_pool_host.py:1675-1797`）：
  - state 是 ring 结构，新增 `_ring_op_indices(rows)`：把 SWA 页号展开成 ring-row 索引（`row * R + arange(R)`，每页 R 个），以 `page_size=ring_size` 调 op。
  - 直接传 SWA loc 会让 `index // ring_size` 选错 state 行，且越过 `num_dev_pages`。

#### A2. C4 Indexer split-buffer 布线（`memory_pool_host.py`）
NPU 上 C4 indexer 的 int8 K 与 fp16 scale 存在**独立 buffer**（`index_k_buffer` / `index_scale_buffer`），不是 GPU 的 packed `index_k_with_scale_buffer`。

- 新增 `attach_scale_buffers()`（:868）为 host pool 附带 fp16 scale buffer，`_host_scale_page_view()`（:908）转 5-D view。
- 搬运时 K、scale **两次独立调用**（同一 host/device indices、同一流，K→scale）：op 用 K 的 `element_size` 算 V 宽度，混合 dtype 会损坏 scale 数据。

#### A3. host pin-memory allocator 解析键修复（item C，`memory_pool_host.py`）
`ALLOC_MEMORY_FUNCS` 按设备**类型字符串** `"npu"` 建表，但 DSV4 paged/state 池用 `torch.device` 对象查表会 miss → 静默 fallback 到 `cudaHostRegister` → NPU 上失败。

- Paged 池（:771）与 State 池（:1412）都改为按 `gpu_device.type` 解析 allocator。

#### A4. NPU KV getter 加逐层可见性门禁（item D2，`dsv4_memory_pool.py:418/445/466`）
`get_key_buffer` / `get_swa_buffer` / `get_compress_buffer` 三个 KV getter 增加 `self.wait_layer_transfer(layer_id)`。

- 之前只有 compress-state getter 会 gate，KV getter 不 gate → forward 会在该层 H2D event 未完成时读到 buffer。
- 补上后 consumer（NPU attention backend）在 getter 内部等对应层拷贝完成再读。

#### A5. C128 device 池暴露 `bytes_per_page_padded`（`dsv4_memory_pool.py:56`）
NPU bf16 路径在 `create_buffer` 里设置该字段（GPU 在 fp8 路径设置），assembler 据此计算 C128 host pool 的 `item_bytes`。

### B. C128 适配（Plan-2）

#### B1. strategy 放宽 + sidecar 分流（`hybrid_pool_assembler.py`）
- `_DeepSeekV4Strategy.matches`（:1191）由严格 `components == {FULL, SWA}` 放宽为接受可选的 C128：`{FULL,SWA}` 或 `{FULL,SWA,C128}`。
- sidecar 列表（:1242-1253）在 C128 是 tree component（NPU 路线）时**排除** `DEEPSEEK_V4_C128` / `DEEPSEEK_V4_C128_STATE` 的 KV/SWA-derived sidecar，避免同名 pool 双 transfer（component transfer + KV-derived sidecar transfer 同时出现）。

#### B2. C128 host pool 真实几何 + 独立预算（`hybrid_pool_assembler.py:337-354`）
- `slot_page_size`：由 `page_size` 改为 `c128_kv_pool.kernel_page_size`（真实 `P`）。
- `num_host_pages`：由沿用 FULL 页数改为 `c128_attn_allocator.num_pages * ratio`（独立 host 预算，决策 §11.1）。

#### B3. C128 PoolEntry device override（`hybrid_pool_assembler.py:638-655`）
- `device_alloc_fn` / `device_free_fn` = 裸 `c128_attn_allocator.alloc/free`（吃 expanded 压缩索引，对应 free-path ①：controller rollback 直接裸 free）。
- `host_evict_fn` = `_delegate_c128_host_evict`（§4.7 方案 1）：C128 host 压力换算成 `n * 128` raw tokens，委托 FULL host-leaf eviction（`cache.evict_host(n*128, FULL)`），避免 stranded FULL/C128 host slots。

#### B4. `_COMPONENT_HOST_ATTR` + `component_host_pools` 加 C128（`hybrid_pool_assembler.py:1153/1293`）
- `_COMPONENT_HOST_ATTR[C128] = ("c128_kv_pool_host", "_c128_kv_pool_host")`。
- NPU 路线把 C128 host pool 绑定进 `component_host_pools`，经 `_apply_stack_result` 挂到 component 的 `_c128_kv_pool_host` 槽位。

#### B5. C128 host-aware hooks（`c128_sidecar_component.py`）
| Hook | 实现 |
|---|---|
| `create_match_validator`（:114-127） | 普通 match 认 `value is not None or host_value is not None`；`match_device_only=True` 只认 device |
| `evict_component`（:258-303） | `DEVICE` 分支收 page ID 进 device_frees（refcount 语义，free-path ②）；`HOST` 分支直接归还 host pool |
| `acquire/release_component_lock`（:336-369） | `lock_host=True` 操作 `host_lock_ref`，不新建锁账本 |
| `free_host_values`（:371-380） | 调 `self._c128_kv_pool_host.free` |
| `drive_host_eviction` | no-op（压力已委托 FULL，见 B3） |
| `_expand_page_indices`（:420-431） | `page_id * P + arange(P)`（expanded 索引） |
| `build_hicache_transfers`（:437-497） | `BACKUP_HOST` 产出 expanded `device_indices`；`LOAD_BACK` 沿 evicted 祖先收集各 group endpoint 的 `host_value` |
| `commit_hicache_transfer`（:503-535） | `BACKUP_HOST` 发布 `host_value`；`LOAD_BACK` 用 `//P → unique → retain_c128_pages + set_component_device_value` 恢复（free-path ②） |

- `[C128-HiCache]` 观测日志前缀（`_OBS`）。

> ⚠️ **C4 `slot_page_size` 也在本提交从 `page_size` 改为 `kernel_page_size`=32**（assembler:497）——**这是提交 2 越界 bug 的引入点**：改了 slot_page_size 却没同步换算继承的 FULL 索引。

---

## 提交 2 `4c26f9b005`

### C1. C4 kernel_ascend D2H 索引越界崩溃（`hybrid_cache_controller.py:945-966`）
- **现象**：实机 fill 填满设备池触发 host eviction 后，scheduler 崩溃（`device_page_index must be less than the 2nd dim of device_k`）。崩溃 pool 是 C4，不是 C128。
- **根因**：提交 1 把 C4 `slot_page_size` 改成 32 后，KV-derived 分支把 FULL **raw-token** 索引直接透传给 C4，kernel 按 `index // 32` 算页号 → 页号虚高 4 倍 → 越界（FULL 索引 421119 → `//32=13159 > 13150`）。
- **修复**：KV-derived 分支按 `ratio = page_size // slot_page_size` 换算索引；`ratio > 1` 才 `//ratio`，GPU / `slot_page_size == page_size` 的池 `ratio == 1` 原样透传。
- **覆盖**：D2H backup 与 H2D load-back 都从 `_resolve_pool_transfers_allocation` 拿索引，一条路径同时修好两个方向；C4（32）与 C4_INDEXER（`_kp`=32）自动换算。

### C2. C128 host-LRU 记账修复（`c128_sidecar_component.py`）
- **evict_component DEVICE tombstone 后提升 host LRU**（:278-288）：tombstone 后若 `host_value` 仍在 → 节点变 host-only（S3 态）→ 镜像 SWA（`swa_component.py:500-507`）提升进 host LRU，修 sanity check `+S3` 崩溃。
- **commit BACKUP_HOST 异步落地补提升**（:563-573）：若 backup 落地前 device 已被 tombstone → 同样提升（异步顺序洞，否则稍后 sanity 崩）。
- **`_evict_device_start/next/end` 补成完整 walk**（:335-407）：实现 C128 自持的设备驱逐 walk（tail 节点交 driver 走正常 evict/backup/demote；boundary 节点内联 tombstone）。**但当前是死代码**：driver 硬编码 `request_by_type[ComponentType.C128] = 0`（`unified_radix_cache.py:457`）跳过 walk；**保持 disabled 是正确的**——boundary 节点被 tombstone 后无法 LOAD_BACK（build 只走 evicted 节点）也无法重挂（commit_insert 只挂 new leaf），启用会读垃圾。

### C3. D2H 失败调试日志（`memory_pool_host.py:1157-1236`）
- try/except 包裹 `kernel_ascend` D2H 两次调用（K + scale），失败时打印 pool/layer/shape/索引范围。
- 定位 C1 崩溃用，`DBG kernel_ascend D2H FAIL` 前缀，**待清理**。

---

## 关键交叉点与待办

| 事项 | 状态 |
|---|---|
| C4 `slot_page_size=32` 是提交 2 bug 的引入点 | 已由 C1 修复（索引 `//ratio`） |
| `memory_pool_host.py` 的 `DBG kernel_ascend D2H FAIL` 日志（2 处 try/except） | **待清理** |
| C128 `_evict_device` walk 死代码（`request_by_type[C128]=0`） | 保持 disabled 正确；`request_by_type[C128]=0` 语义待决策 |
| C128 LOAD_BACK 从未在真实 server 触发 | 见 `plans/2026-08-17-dsv4-npu-loadback-handoff.md`（触发路径问题，非实现问题） |
| 2G target 重复命中 `cached=2048` 截断 | 已知接受行为：FULL 异步 backup 时序导致，与 C128 无关 |
