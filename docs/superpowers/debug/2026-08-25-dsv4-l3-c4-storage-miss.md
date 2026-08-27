# DSV4 512K L3 C4 / C4-indexer 整池 miss —— 定位报告

> 2026-08-25。fresh 环境下执行 512K、100% 目标命中的同机 L3 测试，16 个并发请求中仅 1 个命中；其余 15 个请求的主 KV、SWA、C128 和压缩状态池均完整回载，但 C4 与 C4-indexer 查询结果均为 `0/4096`，导致 hybrid prefetch（混合预取）按 all-or-nothing 规则整体丢弃。**根因尚未定位**，当前需继续区分“写入侧未写入 L3”和“写入/查询 key 不一致”。

---

## 1. 摘要

fresh 512K 诊断组的实际命中率为 **6.2%**：

- 1/16 请求成功从 L3 回载；
- 15/16 请求触发 hybrid prefetch 整体丢弃，随后退化为完整 prefill；
- 15 条失败请求的附加池结果完全一致；
- 查询侧已为 C4 和 C4-indexer 各构造 4096 个 storage key，但 Mooncake 返回整池 0 命中；
- Mooncake 峰值容量仅约 52.1%，没有 eviction（驱逐）、allocation failure（分配失败）或显式 Put failure（写入失败）。

当前故障已经收敛到 `deepseek_v4_c4` 与 `deepseek_v4_c4_indexer` 的 L3 写入/查询路径。

## 2. 环境与配置

- NPU 远程机：`root@113.46.13.20`
- 容器：`cx-dsv4`
- 模型：`/mnt/paas/weights/DeepSeek-V4-Flash-w8a8-mtp`
- SGLang：16 TP、16 DP、DP attention
- Mooncake master：同机 `127.0.0.1:58051`
- 请求：512K（`524288` token）
- 请求数：16
- populate 并发：16
- measure 并发：16
- 目标命中：100%
- 关键参数：
  ```text
  --page-size 128
  --max-total-tokens 532480
  --swa-full-tokens-ratio 0.5
  --hicache-ratio 2.0
  --enable-hierarchical-cache
  --hicache-io-backend kernel_ascend
  --hicache-storage-backend mooncake
  --hicache-write-policy write_through
  --hicache-storage-prefetch-policy wait_complete
  ```

测试前执行 `docker restart cx-dsv4`，随后重新启动 Mooncake 与 SGLang，未复用前一组的缓存或 Mooncake 数据。

## 3. fresh 512K 结果

### 3.1 整体结果

```text
目标命中率：100%
实际命中率：6.2%
成功回载：  1/16
预取丢弃： 15/16
```

### 3.2 失败请求的多池诊断

15 条失败请求均输出：

```text
completed_local=524288
completed_synced=524288
requested=524288
pools=[
  ('swa', True, 1, 1, 1),
  ('deepseek_v4_c128', True, 256, 256, 256),
  ('deepseek_v4_c4', True, 0, 0, 4096),
  ('deepseek_v4_c4_indexer', True, 0, 0, 4096),
  ('deepseek_v4_c4_state', True, 1, 1, 1),
  ('deepseek_v4_c4_indexer_state', True, 1, 1, 1)
]
```

元组字段含义：

```text
(pool_name, keys_present, hit_local, hit_synced, expected)
```

整理如下：

| Pool | 查询 key 已构造 | local hit | 跨组同步后 hit | 期望页数 | 状态 |
|---|---:|---:|---:|---:|---|
| 主 KV | 是 | 524288 token | 524288 token | 524288 token | 完整 |
| SWA | 是 | 1 | 1 | 1 | 完整 |
| C128 | 是 | 256 | 256 | 256 | 完整 |
| C4 | 是 | 0 | 0 | 4096 | **整池 miss** |
| C4-indexer | 是 | 0 | 0 | 4096 | **整池 miss** |
| C4 state | 是 | 1 | 1 | 1 | 完整 |
| C4-indexer state | 是 | 1 | 1 | 1 | 完整 |

`completed_local == completed_synced == requested`，且 C4 两池的 local/synced 均为 0，说明：

1. 主 KV 在当前 rank 本地已经完整回载；
2. 不是 attention group（注意力组）间 `MIN all-reduce` 把本地成功结果压成 0；
3. 查询侧确实构造了 C4/C4-indexer 的 keys（`keys_present=True`）；
4. 失败发生在这两组 key 的 L3 查询结果上。

## 4. 为什么最终仍是 0 命中

DSV4 L3 hybrid prefetch 对主 KV 和所有附加池执行完整性检查。即使主 KV 已完整回载，只要任一必需附加池不完整，就会：

1. 释放本次预取分配的 host/device staging；
2. 将该请求的 loaded tokens 记为 0；
3. 回退到完整 prefill。

因此 C4 和 C4-indexer 的 `0/4096` 会使已经成功回载的主 KV、SWA、C128 与 state 数据全部作废。这解释了失败请求最终观测到 0% cache hit，而不是“主 KV 命中、C4 缺失”的部分命中。

## 5. 已排除方向

### 5.1 不是主 KV 回载失败

15 条失败请求均为：

```text
completed_local=524288
completed_synced=524288
requested=524288
```

主 KV 已完整回载。

### 5.2 不是 SWA、C128 或 state pool 缺失

这些池均达到期望页数：

```text
swa                           1/1
deepseek_v4_c128            256/256
deepseek_v4_c4_state          1/1
deepseek_v4_c4_indexer_state  1/1
```

### 5.3 不是 Mooncake 总容量不足

Mooncake master 观测：

```text
峰值容量：约 416.61 / 800 GB（52.1%）
Eviction：0/0
AllocFail：0
PutRevoke：0
```

没有到达 high watermark（高水位），也没有驱逐或分配失败证据。

### 5.4 不是 storage batch size 的总页数上限

Mooncake storage 路径中的 `STORAGE_BATCH_SIZE=128` 是循环分批大小，不是一次操作的最大总页数。4096 页应被拆成多个批次处理，不能仅凭该常量解释整池 0 命中。

### 5.5 不是测试组间污染

该组在 `docker restart cx-dsv4` 后重新启动 Mooncake 和 SGLang，为 fresh 独立环境。此前已确认复用服务会严重影响 256K/512K 结果，因此本组没有复用上一组服务。

## 6. 当前根因范围

尚需在以下可能性之间进一步区分：

1. **写入侧没有提交 C4/C4-indexer**：populate backup 时 PoolTransfer 未包含这两个池，或被条件分支跳过；
2. **写入页数被截断**：PoolTransfer 存在，但 host indices、keys 或 value buffers 的长度/几何不一致，导致 put 未覆盖 4096 页；
3. **写入与查询 key 不一致**：populate 和 measure 为同一逻辑前缀生成了不同的 pool storage key；
4. **异步生命周期问题**：写入任务提交成功，但 key/value buffer 在 Mooncake 消费前被复用或释放；
5. **成功统计只覆盖其他池**：Mooncake master 的 Batch Put success/total 一致，但成功项可能来自主 KV、C128 和 state pool，尚不能证明 C4 两池写入成功；
6. **512K 边界条件**：C4/C4-indexer 在 4096 个 128-token page 的边界上存在 key 构造、分批或索引范围问题。

当前最重要的判别是：

> C4/C4-indexer 的 4096 个 key 是否实际进入 L3 put；若进入，put 和 get 两侧生成的 key 是否一致。

## 7. 加入debug日之后测试
 写入与查询 key 均正常
                                                                                                                                                                                                                                                                                              
  16 个 rank 的两个池都完成了完整写入： 
  deepseek_v4_c4:                                                                                                                                                                                                                                                                                                   
    total_keys=4096                                                                                                                                                                                                                                                                                                 
    successful_objects=4096                                                                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                                                                    
  deepseek_v4_c4_indexer:                                                                                                                                                                                                                                                                                           
    total_keys=4096                                                                                                                                                                                                                                                                                                 
    successful_objects=409
- 16/16 rank 全部达到 4096/4096；                                                                                                                                                                                                                                                                                 
  - 没有 set failure；                                                                                                                                                                                                                                                                                              
  - measure 阶段 exists 也全部为：                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                                    
  objects=4096                                                                                                                                                                                                                                                                                                      
  existing_objects=4096                                                                                                                                                                                                                                                                                             
  page_hits=4096                                                                                                                                                                                                                                                                                                    
  prefix_hits=4096                                                                                                                                                                                                                                                                                                  
  exist_codes={1: 4096}                                                                                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                                                                    
  - set 与 exists/get 的首尾 key 指纹完全一致。 
  因此已明确排除：                                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                                    
  - C4/C4-indexer 没有写入 L3；                                                                                                                                                                                                                                                                                     
  - 写入被截断；                                                                                                                                                                                                                                                                                                    
  - put/get key 不一致；                                                                                                                                                                                                                                                                                            
  - Mooncake 中对象不存在。                                                                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                                                                    
  真正失败发生在数据传输阶段
get 结果：                                                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                                                    
  ┌────────────┬────────┬─────────┐                                                                                                                                                                                                                                                                                 
  │    Pool    │  成功  │  失败   │                                                                                                                                                                                                                                                                                 
  ├────────────┼────────┼─────────┤                                                                                                                                                                                                                                                                                 
  │ C4         │ 1 rank │ 15 rank │                                                                                                                                                                                                                                                                                 
  ├────────────┼────────┼─────────┤                                                                                                                                                                                                                                                                                 
  │ C4-indexer │ 0 rank │ 16 rank │                                                                                                                                                                                                                                                                                 
  └────────────┴────────┴─────────┘                                                                                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                                                                                    
  失败返回码完全一致：                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                    
  io_codes={-707: 4096}                                                                                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                                                                    
  Mooncake v0.3.11.post1 源码中：                                                                                                                                                                                                                                                                                   
                                                                                                                                                                                                                                                                                                                    
  LEASE_EXPIRED = -707
含义是：                                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                    
  ▎ 数据传输完成前，Master 授予的读取 lease（租约）已经过期。
                                                                                                                                                                                                                                                                                                                    
  Mooncake master 的默认 KV lease TTL 为：                                                                                                                                                                                                                                                                                                                     
  default_kv_lease_ttl = 5000  // 5 秒
  这与现场结果完全吻合： 
  - DP1 的 C4 在约 5 秒内完成，4096/4096 成功；
  - 其余 rank 在并发竞争下超过 5 秒，整批返回 -707；
  - DP1 随后读取 C4-indexer 时也已超过租约窗口，因此 C4-indexer 返回 -707；
  - 最终 16/16 请求均触发 hybrid prefetch discarded。 

关键日志：

```text
/home/cx/data/dsv4_npu_hicache_l3_20260825_154022/server.log
/home/cx/data/dsv4_npu_hicache_l3_20260825_154022/mooncake_master.log
```


