# DSV4 C128 write-through backup 偶发卡死 prefill —— Bug 报告

> 2026-08-18。性能测试探针（wstress）中发现：L1+L2（write-through）配置下，C128 backup（L1→L2）路径在**中低负载、无驱逐**时偶发把单个 prefill 请求卡死 1–6 分钟。已阻塞 perf 测试（populate 阶段即触发，W0.5 已接近卡点）。**根因未定位**，本文给出完整证据链 + 代码位点，供另一 agent 定位修复。

---

## 1. 摘要

DSV4 C128 分层缓存（`--enable-hierarchical-cache --hicache-io-backend kernel_ascend`，默认 write-through）下，**串行 populate 累计 ~488 个 C128 group（~1M token）的 backup 后，下一个 prefill 请求被卡死**：单条 chunk 的 prefill 在 `input throughput 0.78 tok/s` 下跑 100+ 秒（正常 ~4000 tok/s）。server `/health` 仍 200（非全挂），探针进程存活但等待 300s timeout。

**关键排除**：不是容量（host 池仅 2.5%、device 池 1–4% 占用）、不是驱逐（evict DEVICE/HOST 均为 0）、不是并发（串行也卡）、不是全挂（health OK）。是 backup 管线在中等负载下的**偶发中断/死锁类 bug**。

## 2. 环境与配置

- NPU：`root@113.46.13.20`，容器 `cx-dsv4`，16 卡。
- Server：`/home/cx/hicache_dsv4/run_server_probe.sh`（= run_server.sh + `--context-length 1048576` + `--hicache-ratio 1.5` + `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`）。
  - 关键参数：`--mem-fraction-static 0.7 --tp-size 16 --dp-size 16 --enable-dp-attention --prefill-max-requests 1 --chunked-prefill-size 32768 --page-size 128 --enable-hierarchical-cache --hicache-io-backend kernel_ascend --hicache-ratio 1.5`。
- 探针：`test/manual/dsv4_npu_hicache/probe_perf_assumptions.py --only-wstress --route roundrobin`（**串行** populate 16K 前缀，round-robin 路由 `routed_dp_rank = i % 16`，单请求 timeout 300s）。
- 池几何（server.log 实测）：device `full=1676928, swa=167680, c4=419232, c128=13101`；host c128 = 1227 pages/rank（`--hicache-ratio 1.5` → host=1.5×device）。

## 3. 复现步骤

```bash
# 1) 干净起服
ssh root@113.46.13.20 "docker restart cx-dsv4"
ssh root@113.46.13.20 "docker exec -d cx-dsv4 bash -lc 'nohup bash /home/cx/hicache_dsv4/run_server_probe.sh > /home/cx/hicache_dsv4/probe_launch.out 2>&1 &'"
# 等 /health 200
# 2) 串行 populate 16K 前缀，round-robin，累计 ~1M token 后卡
ssh root@113.46.13.20 "docker exec cx-dsv4 bash -lc 'cd /sgl-workspace/sglang && python3 -u test/manual/dsv4_npu_hicache/probe_perf_assumptions.py --only-wstress --route roundrobin'"
```

探针输出（卡点）：
```
  ws-canary: 16/16 done (82s)            # 16×16K 正常，~5s/条
  ws-1M: 32/32 done (158s)               # 又 32 条正常
  ws-1M: 8/13 done (39s)                 # 第 9 条起卡死
```

## 4. 症状与证据

### 4.1 串行复现（probe `20260818_024714`，干净 server）

- `build BACKUP_HOST` = **488**，`commit BACKUP_HOST` = **488**（注意：**一行组 build + 一行 commit，别按单个关键字数成 976**）。
- `evict DEVICE` = 0，`evict HOST` = 0，`LOAD_BACK` = 0。
- 卡点前日志（02:54，DP11/DP12 快速 chunk，~4000 tok/s，2048-token chunk 逐组 backup）：
  ```
  [02:54:26 DP12] Prefill batch #new-token: 2048, #pending-token: 14336, input throughput: 27.45   ← 单个 chunk 已卡 74s
  [02:54:29 DP12] [C128-HiCache] build BACKUP_HOST(L1->L2) node=88 page_ids=[32] groups=1 expanded_slots=16
  [02:54:30 DP12] [C128-HiCache] commit BACKUP_HOST(L1->L2) node=88 host_slots=16 published
  [02:56:21 DP1 ] Prefill batch #new-token: 128, input throughput: 0.78                            ← 128-token chunk 卡 ~164s（0.78 tok/s）
  ```
- 卡死时 server health 200，探针进程存活（等在 300s timeout 窗口内）。

### 4.2 并发复现（probe `20260818_010913`，非干净缓存）

- 16-way `ThreadPoolExecutor` 并发 populate（**探针早期写法，偏离 Qwen3 串行方法论**），rank `DP5` 一条 128-token prefill 卡 ~6 分钟（`input throughput 0.35 tok/s`）。
- 该 server 当时缓存已接近 FULL（残留 ~1.7M），但与串行复现（低占用也卡）合并看，**容量不是必要触发条件**。

### 4.3 占用核算（证明非容量）

round-robin 把 ~488 组摊到 16 个 rank：
- 每 rank ≈ 30.5 组 backup → host c128 池 `30.5/1227 ≈ 2.5%`。
- 每 rank ≈ 62.5K FULL token → device FULL 池 `62.5K/1676K ≈ 3.7%`（server.log 每批 `full token usage: 0.01`）。
- **设备/主机都在 ~1–4% 占用时就卡**，且零驱逐 → 与容量、驱逐无关。

## 5. 根因假设（供定位，未证实）

C128 write-through backup（D→H）管线在累计 ~488 次传输后偶发卡住单个请求。候选位点：

| 位点 | 文件:行 | 角色 |
|---|---|---|
| C128 backup build | `hardware_backend/npu/dsv4/c128_sidecar_component.py:518` | 发 BACKUP_HOST D→H 传输（`expanded_slots=16`） |
| C128 backup commit | 同上 `:600` | host 槽发布（`host_slots=16`） |
| 传输编排 | `mem_cache/unified_radix_cache.py:945` | BACKUP_HOST 阶段执行 kv_xfer/comp_xfers |
| 各组件 D→H handler | `unified_cache/components/{full,swa,mamba,tree}_component.py` | 传输执行 |

推测方向（都未验证）：
1. **D2H 传输引擎/流泄漏**：累计 ~488 次 transfer 后某个 stream/kernel 挂起，阻塞后续 prefill 的该 chunk。
2. **锁/队列积压**：backup 管线有全局锁或固定队列，偶发竞争死锁。
3. **页/节点索引条件**：卡点页索引到特定范围（node≈168, page_ids≈72）后偶发。
4. **write-through 异步 backup 时序**：`expanded_slots`/`host_slots` 分配与发布之间的竞态。

## 实际debug测试
本轮新增关键变化                                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                                    
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