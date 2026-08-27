# DSV4 NPU HiCache L1↔L2 适配 PR 讲解（l1l2-pass）

> 日期 2026-08-24 · 分支 `l1l2-pass`（= `dev/main-8.5` 最新内容、干净 2 提交形态）
> 目的：按"请求的一生"主干（服务初始化 → 请求到达 → 前缀匹配 → 插树 → 缓存操作 → 数据面）把整个 PR 的代码串起来，供理解与提问参考。**不含测试代码细节**。
> 行号均指 `l1l2-pass` 分支当前代码。

## PR 总览

| 提交 | 内容 | 说明 |
|---|---|---|
| `4e05ad03a5` feat: adapt Hicache L1-L2 for DSV4 | 主体实现（+891/-46） | 9 个源码文件 |
| `c3bbabe710` feat: add DSV4 test of Hicache L1-L2 | 测试（+3014） | 6 个保留的组件/单元测试 |

- **基线** `b915d68d81`（Talantan1102 的 DSV4 backend review）。
- **改动面**：`c128_sidecar_component.py`(+342)、`memory_pool_host.py`(+361)、`hybrid_pool_assembler.py`(+184)、`dsv4_allocator.py`(+46)、`schedule_batch.py`(+43)、`unified_radix_cache.py`(+31)、`hybrid_cache_controller.py`(+22)、`dsv4_memory_pool.py`(+7)、`allocation.py`(+8)。
- ⚠️ **`unified_tree_core.py` 不在改动中**——树的匹配/插树/驱逐是基线通用框架，PR 不碰；C128 的一切适配都发生在**组件层**（`c128_sidecar_component.py`）和**装配层**（assembler）。

### 架构前提（DSV4 池结构）

- **无物理 FULL KV 池**：FULL 只是逻辑容量（记账/驱逐分母），物理池只有 SWA / C4 / C128 / compress-state。
- **C128 = 独立索引池**：每 128 个压缩位置 = 一个物理页（P=16 slot），一页压缩 G=2048 个原始 token（一个 group）。页号由 `c128_attn_allocator` 独立分配。
- **三种索引单位**：FULL 逻辑 token 索引（KV 池 slot）/ C128 物理页号（树 value、sidecar、refcount 的主键）/ expanded 槽位（`page_id × P + arange(P)`，只在 allocator/搬运边界出现）。

---

## ① 服务初始化（assembler 装配）

起点：`hybrid_pool_assembler.py` 的 `attach_hybrid_pool_to_unified_cache` / `_DeepSeekV4Strategy`。DSV4 组装路径把"组件声明"变成"真实 host 池 + 绑定"。

### 1.1 strategy 放宽接受 C128

`_DeepSeekV4Strategy.matches` 由严格 `{FULL, SWA}` 放宽为接受可选 C128：`{FULL,SWA}` 或 `{FULL,SWA,C128}`。C128 成为树的合法组件，参与匹配/插树/驱逐全流程。

### 1.2 C128 host pool 真实几何 + 独立预算

C128 的 host 池（L2）不再沿用 FULL 页数，而是：

- `slot_page_size` = `c128_kv_pool.kernel_page_size`（真实 P，而非 FULL page_size）；
- `num_host_pages` = `c128_attn_allocator.num_pages × ratio`——**独立 host 预算**，按 C128 自己的物理页数定大小。

### 1.3 C4 层组装门控（`if c4_layer_mapping:`）

`hybrid_pool_assembler.py:492` `if c4_layer_mapping:` 是 C4 系列 pool 的**入口门控**——只有模型配置含 C4 压缩层（layer_mapping 非空）才构建，块内三组：

- **C4 KV pool**（:546 `DEEPSEEK_V4_C4`）：`slot_page_size` 取 `kernel_page_size`（:498，NPU = page_size/4 如 32；GPU 无此属性 fallback 到 page_size）。⚠️ 这是 C4 索引换算的敏感点——slot 尺寸变化必须同步换算 FULL 继承索引（否则 `index // slot_page_size` 页号虚高 4 倍，见 ⑤ 数据面）。
- **C4 indexer pool**（`DEEPSEEK_V4_C4_INDEXER`）：NPU 分支 split buffer——int8 K（`index_k_buffer`）+ fp16 scale（`index_scale_buffer`，`dsv4_memory_pool.py:172/183`），构建后 `attach_scale_buffers`（:529）为 host 附带 fp16 scale buffer；GPU 分支用 packed `index_k_with_scale_buffer`（详见 ⑥）。
- **C4 STATE pools**（:562 `if not is_unified_kv`）：`DEEPSEEK_V4_C4_STATE`（:563）+ `DEEPSEEK_V4_C4_INDEXER_STATE`（:574），用 `DeepSeekV4StateHostPool`、`num_host_pages = swa_num_host_pages`、`swa_page_size`——**C4 state 与 SWA 1:1 绑定的装配落地**：压缩窗口恰好一个 ring 宽，state 按 SWA 页组织，host 侧复用 SWA 页数/页大小、无独立 allocator（`alloc/free` 均 raise）。

### 1.4 C128 PoolEntry override

C128 的 device 分配不走 FULL 的 allocator 路径：

- `device_alloc_fn` / `device_free_fn` = 裸 `c128_attn_allocator.alloc/free`（吃 expanded 压缩索引，free-path ①：controller rollback 直接裸 free）；
- `host_evict_fn` = `_delegate_c128_host_evict`（`hybrid_pool_assembler.py:1191`）：C128 host 压力换算成 `n × 128` raw token，委托 **FULL** 的 host-leaf eviction（详见 ⑤）。

### 1.5 sidecar 分流 + 组件绑定

- sidecar 列表在 C128 是 tree component 时**排除** `DEEPSEEK_V4_C128` 的 KV/SWA-derived sidecar，避免同一 pool 双 transfer（component transfer + sidecar transfer 同时出现）。
- `_COMPONENT_HOST_ATTR[C128] = ("c128_kv_pool_host", ...)`，经 `_apply_stack_result` 把 C128 host pool 挂到组件槽位，供 `free_host_values` 使用。

**产出**：`_c128_kv_pool_host` 就位、controller/组件互相引用、LayerDoneCounter 就位——① 完成，服务可接请求。

---

## ② 请求到来 → prefill 写 KV

### 2.1 逐层可见性门禁（getter gate）

NPU 的 KV getter——`get_key_buffer`（`dsv4_memory_pool.py:420`）、`get_swa_buffer`（:440）、`get_compress_buffer`（:456）——第一行都调 `wait_layer_transfer(layer_id)`（定义在 `deepseek_v4_memory_pool.py`，`layer_transfer_counter.wait_until(layer - start_layer)`）。

- **写路径**（新 token KV）：设备端计算直接写池，同 stream 天然有序，无需 gate；
- **读路径**（命中前缀回载）：H2D 在后台 `load_stream` 上**逐层**提交（`cache_controller.start_loading` 每层 `producer_event.complete(i)`），getter 里 `wait_until` 让当前流 `wait_event(该层)`——**forward 推进到哪层就只等哪层**，层间与拷贝重叠。

### 2.2 C128 容量预留（新增修复）

`allocation.py` 的 `alloc_paged_token_slots_extend/decode` 在 DSV4 分支里，先算本 step 需要的 C128 页数、再确保容量（见"新增修复专题"）：

```python
c128_num_pages = allocator.c128_num_pages_needed(prefix_lens_cpu, seq_lens_cpu)
allocator.ensure_c128_capacity(tree_cache, c128_num_pages)
```

**含义**：prefill/decode 分配 KV 槽之前，先保证 C128 物理页够用——不够则**驱逐 FULL device 叶子**腾空间（FULL 驱逐级联释放其 C128 载荷），避免 C128 分配失败。

---

## ③ 请求再次命中 → 查前缀树

前缀树是 **FULL 的树**（key = 原始 token），所有组件共享、各自挂 value。匹配主循环在基线的 `unified_tree_core._match_prefix_helper`，C128 以组件回调参与。

### 3.1 C128 validator（`c128_sidecar_component.py:111`）

```python
def _valid(node):
    if match_device_only:
        return cd.value is not None                      # device-only 轨：只认 L1 页
    return cd.value is not None or cd.host_value is not None   # 默认轨：L1 或 L2 任一
```

- **命中长度只由 FULL key 决定**（`child.key.match`），组件只做 AND 投票（veto）；
- C128 页只挂在"完整 group 结束的节点"上，validator 检查的就是这些端点有没有 C128 数据；
- HiCache 双轨：`best_match_node`（device-or-host，回载起点）/ `best_match_device_node`（device-only，scheduler 的 device 前缀）。

### 3.2 match 收尾收集命中页（`c128_sidecar_component.py:142`）

`finalize_match_result_in_cache` 沿 `best_match_node` → root 收集 device 页（`_collect_device_pages`，:126 抽出），按 FULL 设备覆盖推算 expected 页数做断言（`:157-165`：`device_indices` 只到 `last_device_node`，要补 best→last_device 间 FULL values——SWA 可先于 FULL 被驱逐），最后 `set_c128_prefix_pages(req, pages)` 把命中页写进请求。

### 3.3 load-back 后刷新页表（`c128_sidecar_component.py:180`，新增修复）

```python
def finalize_load_back(self, req, prep, success):
    # match_prefix 发生在 load-back 之前，host-only 的 C128 端点在 match 时
    # 不在请求页表里；commit_load_back 把页恢复到树后，重新收集再刷新。
    pages = self._collect_device_pages(req.best_match_node)
    self.cache.req_to_token_pool.set_c128_prefix_pages(req, pages)
```

**问题**：`finalize_match_result_in_cache` 只收集 **device** 页；若命中前缀的 C128 页在 L1 被驱逐、只有 host，match 时收集不到 → 请求页表缺页。load-back 从 L2 恢复后必须**重刷**，否则 prefill 的 C128 attention 拿不到恢复的页。

---

## ④ 插树：C128 按 group 边界挂页

### 4.1 读回页 + 截断（`c128_sidecar_component.py:330` `prepare_for_caching_req`）

```python
cache_len = logical_len // G * G                    # 只缓存完整 group 的倍数
num_pages = cache_len // G
insert_params.c128_value = req_to_c128_sidecar[req_pool_idx, :num_pages].clone()
return cache_len                                    # 参与上层 min，FULL 也被截到 group 边界
```

### 4.2 group 边界拆节点 + attach（`c128_sidecar_component.py:260` `commit_insert_component_data`）

```python
for boundary in range(first_boundary, len(key)+1, G):
    boundary_node = self._ensure_boundary_node(node, boundary, cache_actions)
    self._attach(boundary_node, c128_value[page_index])   # 每边界挂一页
```

- **组件会拆 radix 节点**：`_ensure_boundary_node`（:236）找到覆盖 boundary 的节点，必要时 `tree_core._split_node` 在 `boundary - node_start` 处切——**不改 key 内容，只插入"结束于 group 边界"的端点**，让 C128 页有挂点、后续匹配能精确对齐完整 group。
- `_attach`（:101）：`pages.clone()` + `set_component_device_value` + `retain_c128_pages`（refcount +1）。clone 是为不共享请求级 `req_to_c128_sidecar`（会被后续请求覆盖）。

---

## ⑤ 缓存操作：backup / load-back / evict / host 委托

### 5.1 Write-through backup（L1→L2）

触发：节点命中超 `write_through_threshold`（基线 `_inc_hit_count_and_check`）→ `BackupKV` action → `_execute_and_commit_kv_backup`（`unified_radix_cache.py`）→ `cache_controller.write`（分配 host 槽 + 异步 D2H）。

C128 的 transfer（`c128_sidecar_component.py:487` BACKUP_HOST）：取节点挂的 C128 页 → `_expand_page_indices`（:471，`page_id × P + arange(P)`）展开成 slot → `PoolTransfer(indices_from_pool=None, device_indices=expanded)`——**独立索引**，控制器按 `len = groups × P` 分配 C128 host 槽。

提交（:548 BACKUP_HOST）：发布 `host_value`；若 device 页已在 backup 落地前被 tombstone → 节点变 host-only → **补提升进 host LRU**（异步顺序洞修复）。

### 5.2 Load-back（L2→L1）

transfer（:517 LOAD_BACK）：沿 **evicted 祖先**收集各 group 端点的 `host_value`（稀疏）→ `PoolTransfer(host_indices=cat, device_indices=None)`。

提交（:577）：`device_indices` 切片 → `// P` → `unique` 还原页号 → `retain_c128_pages` + `set_component_device_value`——**所有权转移（free-path ②）**：retain 后由树在驱逐时 release，绝不由 controller 裸 free。

**新增**：`unified_radix_cache.py:96` `_c128_transfer_num_pages` 校验 load-back transfer 含完整物理页（`num_slots % P == 0`），并在 H2D 前 `ensure_c128_capacity`（见修复专题）。

### 5.3 Evict（设备驱逐）

FULL device 叶子被驱逐（`_demote`/`_cascade_evict`）时级联触发 C128 `evict_component`（:295）：DEVICE 分支把 `cd.value` 收进 device_frees → `FreeComponentDeviceSlot` → `release_c128_pages`（**refcount −1，归零才真 free**）。device 释放依赖 FULL 触发、但走 refcount——因为 C128 device 页被多请求共享（各自 retain）。

### 5.4 Lock hooks（`c128_sidecar_component.py:438/451`）

- **device 路径 no-op**：C128 device 页靠 retain/release refcount 天然保活，不需要 FULL 那种 path-lock；
- **host 路径镜像 FULL 单节点锁**：load-back 正从 L2 读走 host 槽时，`host_lock_ref++` 防 host eviction 回收。

### 5.5 Host eviction 委托（`hybrid_pool_assembler.py:1191`）

```python
def _delegate_c128_host_evict(cache, n):
    return cache.evict_host(n * 128, ComponentType.FULL)
```

C128 host 池分配失败 → 委托 FULL 驱逐 host leaf → `_evict_host_leaf`（基线）遍历所有组件原子连带释放 C128 host_value → `free_host_values`（:462）→ `_c128_kv_pool_host.free`。**C128 页是 FULL 前缀的载荷，必须以 FULL leaf 为单位整体驱逐、不留孤儿**；`drive_host_eviction`（:423）是 no-op（已委托）。

### 5.6 host-LRU 不变量（异步时序兜底）

backup 是"enqueue D2H + commit 记账"两步异步、evict 是独立内存压力流程，二者在 Scheduler 主循环内并行、顺序不保证（非线程竞态——所有 commit 都在主循环同步做）。"反序"（evict 先 tombstone C128 value、backup 后发布 host_value）时节点瞬变 host-only，靠 **backup commit 里发现 value is None 就补提升进 host LRU** 维持"所有 host-only 节点 ∈ host LRU"的 sanity 不变量。

---

## ⑥ 数据面：`kernel_ascend` 搬运

所有缓存操作的最终落地都是同一个 NPU 搬运 op `transfer_kv_dim_exchange`（`sgl_kernel_npu.kvcacheio`）：

- **5-D dim-exchange**：device `[layer,page,P,head,dim]` vs host `[page,layer,P,head,dim]`，`aclrtMemcpy2dAsync` 一次拷一页跨所有 layer；`device_indices`/`host_indices` 定页码（`index // page_size`）；
- **K / scale 两次调用**：op 用 K 的 `element_size` 算 V 宽度，混合 dtype 会损坏 → 同 indices 同 stream 分两次。

`memory_pool_host.py` 的 `kernel_ascend` 分支：

| pool | D2H（backup） | H2D（load-back） | 视图 |
|---|---|---|---|
| Paged（SWA/C4/C128 KV） | :1082→:1150 | :1193→:1256 | `_host_page_view` → `[num_host_pages,1,P,1,dim]` |
| C4 indexer | 同 paged + scale | 同 paged | `attach_scale_buffers` :871 / `_host_scale_page_view` :945 |
| State（C4 压缩状态） | :1624→:1678 | :1717→:1766 | `_ring_op_indices` :1535（`row×R + arange(R)`，以 `page_size=R` 调 op） |

- **`attach_scale_buffers`（:871）**：为 host pool 附带 fp16 scale buffer（`scale_kv_buffer`），与 K 同页数同布局；`_host_scale_page_view`（:945）转 5-D（layer_first 每层一块 / page_first 非连续 slice）。
- **`ALLOC_MEMORY_FUNCS`**：host pin-memory 分配函数注册表——默认 CUDA 用 `cudaHostRegister`，NPU 覆盖成 `torch.empty(pin_memory=True)`（键按设备类型字符串查，用 `torch.device` 对象会 miss）。分配目标 `self.device` 恒为 `"cpu"`（L2 是 host 内存），NPU 键只决定"怎么 pin"。

---

## 新增修复专题（`5c9b6a25be`，已并入 l1l2-pass 代码）

这一个提交对应 commit message "pass L1+L2 precise and performance testing"，三类修复：

### ① load-back 后刷新请求页表（精确性）

- `finalize_match_result_in_cache` 的收集逻辑抽成 `_collect_device_pages`（`c128_sidecar_component.py:126`）；
- 新增 `finalize_load_back`（:180）：match 早于 load-back，host-only C128 端点不在 match finalizer 建的页表里；commit_load_back 恢复页到树后重刷 `set_c128_prefix_pages`。**没有它，首次 load-back 的请求 C128 attention 会读缺页**。

### ② C128 容量预留（性能/正确性）

- `dsv4_allocator.py:267` `c128_num_pages_needed`：算本 step 需要的 C128 物理页数（`(seq_len//128 → ceil到P) − (prefix_len//128 → ceil到P)`）；
- `:279` `ensure_c128_capacity`：C128 页不够 → 循环 `tree_cache.evict(EvictParams(num_tokens=page_size))` 驱逐 FULL device 叶子（**FULL 驱逐级联释放其 C128 载荷**，避免独立驱逐 C128 留下"FULL 前缀中间有洞"）；
- 接入点：
  - `schedule_batch.py:2817` `check_decode_mem`：decode 前同时检查 FULL/SWA（`full_swa_available_size` :520）与 C128 容量；
  - `allocation.py` `alloc_paged_token_slots_extend/decode`：分配前 ensure；
  - `unified_radix_cache.py:96` `_c128_transfer_num_pages` + load-back 前 ensure：回载前保证 C128 device 页放得下。

### ③ 性能指标

`benchmark/serving.py` +8（仅 dev/main-8.5，l1l2-pass 未含）。

---

## 设计主线

1. **C128 是"独立索引池、载荷语义"**：页号当所有权主键（树 value / sidecar / refcount 全用页号），槽位只在 allocator/搬运边界展开；device 用 refcount 保活、host 委托 FULL 驱逐、lock no-op——每一步都源于"一页压一个 FULL group、被多请求共享"这个几何。
2. **正确性靠不变量兜底、不靠串行化**：异步 backup 与 evict 交错时，"host-only 必在 host LRU"不变量在任何 action 顺序下保持；load-back 后重刷页表保证 C128 页表与树状态一致。
3. **树的框架不动，一切适配在组件层**：`unified_tree_core.py` 零改动，C128 通过组件 hooks（validator / insert / evict / transfer / lock）接入，印证了 unified-cache 组件化设计的可扩展性。

---

## 测试矩阵（c3bbabe710 保留项）

| 测试 | 覆盖 |
|---|---|
| `test_c128_2_poolentry.py` | PoolEntry 几何/override/绑定 |
| `test_c128_3_component_hooks.py` | 组件 hooks（validator/insert/evict） |
| `test_c128_4_backup_loadback.py` | backup/load-back 往返（含 5c9b6a25be 补充） |
| `test_c128_5_eviction.py` | 驱逐 + refcount |
| `test_d_layer_gate.py` | getter 逐层可见性门禁 |
| `test_strategy_reachable.py` | assembler strategy 可达性 |
