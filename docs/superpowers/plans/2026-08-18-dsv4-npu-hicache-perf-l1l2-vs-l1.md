# DSV4 NPU HiCache 性能测试计划 —— L1+L2 vs 只开 L1（含全关基线）

> 2026-08-18。基于 `test/manual/dsv4_npu_hicache/probe_perf_assumptions.py` 的探针实测结论（见 §3），回答：**工作集越过 L1 后，L2（host C128 池）能挽回多少 TTFT / Prefill TPS**。
> 前置：C128 write-through backup 卡死 bug 已规避（`docs/superpowers/debug/2026-08-18-dsv4-c128-writethrough-backup-stall.md`），**假设并发 populate 可用**（写计划时点，执行前复验）。

---

## 0. 目标

在**相同 L1 容量（钉 384K/rank）、L2=2×L1、相同请求/路由/并发**下，只改变缓存层级配置，测量：

**主目标（P1 核心）：L2 收益**（W1.5 档，>L1、<L2）：**C2 只开 L1 vs C1 L1+L2** → L2（host）把被 L1 驱逐的 KV 兜住带来的增量收益（TTFT 降幅 / Prefill TPS 升幅）。

> **L1 收益（C3 全关 vs C2 只开 L1，W0.5 档）是确定的**（radix 开 vs 关必然有收益），本次不专门测——最多作冷参照 sanity（可选，见 §4.3 P3）。

> W 档与对比配对：W1.5 溢出 L1 → C3 只是冷全量，测 C3 无意义。因此 **核心矩阵只跑 C2 vs C1 在 W1.5**；C3 仅作可选冷参照锚点（0% 命中校验）。

配置（按执行顺序 **全关 → 只开 L1 → L1+L2**）：

| 配置 | 缓存 | 启动脚本 | 关键差异 |
|---|---|---|---|
| **C3 全关** | ✗ | no_hicache + `--disable-radix-cache` | 0% 命中全量 prefill |
| **C2 只开 L1** | radix（device） | no_hicache | radix 默认开 |
| **C1 L1+L2** | radix + host C128 | run_server_probe_pin.sh 同款 | + `--enable-hierarchical-cache --hicache-io-backend kernel_ascend --hicache-ratio 2.0` |

**禁止**把 C3 数据当作 C1 的 L2 增量结论（那是总缓存收益）；L2 增量必须 C2 vs C1，L1 增量必须 C3 vs C2。

---

## 1. 公共配置（三份脚本必须一致）

```
--model-path /mnt/paas/weights/DeepSeek-V4-Flash-w8a8-mtp \
--page-size 128 --tp-size 16 --trust-remote-code --device npu \
--attention-backend ascend --watchdog-timeout 9000 \
--host 0.0.0.0 --port 30000 --mem-fraction-static 0.7 \
--prefill-max-requests 1 --chunked-prefill-size 32768 \
--max-running-requests 16 --dp-size 16 --enable-dp-attention \
--moe-a2a-backend deepep --deepep-mode auto \
--quantization modelslim --enable-dp-lm-head \
--kv-cache-dtype auto --random-seed 20260807 --context-length 1048576 \
--max-total-tokens 384000 \
--enable-hierarchical-cache --hicache-io-backend kernel_ascend --hicache-ratio 2.0   # 仅 C1
```

- `--max-total-tokens 384000` 钉**每 rank** L1 FULL = 384K token（page 对齐 384000/128=3000），三份配置相同 → L1 严格一致。
- C1 的 host 池 = **2×L1**（`--hicache-ratio 2.0`），每 rank host c128 ≈ 375 组 ≈ 768K raw。**L2(768K) > W1.5(576K)**，W1.5 的全部 KV 都能装进 L2，余量 33%（L2 不溢出）。
- **host DRAM 预算**：384K pin + ratio 2.0 下全部 host 池约 **8.1 GB/rank × 16 ≈ 130 GB**（swa 3.4 + c4 4.1 + c4_indexer 0.5 + c128 0.1 GB/rank），机器 1.5TB 余量巨大。
- `--context-length 1048576`（DSV4-Flash config 声明 1M，`SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` 防御性设）。
- 每次启动后从 server.log 记录实际 `full=` / host 池 pages（见 §6 硬校验）。

---

## 2. 请求生成（借鉴 bench_ids.py，DSV4 适配）

沿用 `test_plan/multi_instance/scripts/bench_ids.py` 的方法，做 DSV4 适配：

- **前缀池**：真实文本 token 池（`gen_pool.py`）滑窗切片，首 token 各异 → radix 不误撞。
- **对齐单位**：`PAGE=128` → **C128 group G=2048**，前缀 P = N×G（N 完整组）。
- **100% 命中档**：`input_len = N×G + 1`（多 1 个 token 垫掉 sglang「最长命中−1」，`cached = N×G` 整组全中——探针实证）。
- **50% 档（可选扩展）**：前缀 P=N×G + 新后缀 Q，`key_limit=P+Q−1 ≥ P` 不受 −1 影响。
- **路由**：populate 与 measure **同一 round-robin 路由**（前缀 i → `routed_dp_rank = i%16`），保证重放命中缓存它的 rank。
  - ⚠️ 原生 bench_serving 不设 routed_dp_rank（free 路由=server 内置 round_robin）。**需先验证 free 路由下同序重放的命中一致性**（见 §8 风险 3）；若不稳，measure 用 c=1 保序或 loader 显式钉路由。
- **populate**：并发（假设 bug 已修），单请求 timeout 300s，每批打进度。populate 与 measure 用同一批前缀/同一 seed。
- **measure**：原生 `bench_serving`（monkeypatch `get_dataset`，`request_rate=inf, max_concurrency=N`），TTFT/TPS 全走原生，不自己统计。

---

## 3. 探针实测前提（不重测，直接进矩阵）

| 项 | 结论 |
|---|---|
| L1 FULL | 钉后 = 384K/rank（pin-check 实证 `--max-total-tokens` 生效、per-rank） |
| L2 = 2×L1 | ratio 2.0；pin-check 在 ratio 1.5 时实测 1.485×，ratio 2.0 后约 2×（执行前以日志实测为准） |
| 长度可行性 | 32K–256K 全过（context 1M）；**256K 需 L1≥256K，钉 384K 后 256K 占 67%** |
| N×G+1 trick | `N×G+1` → cached=N×G；`N×G` → (N−1)×G（A/B 段实证） |
| 路由 | round-robin 必须（rank0 钉死 → 并发恒平假象）；c1→c16 扩展 5.7× |
| 并发扩展 | c=1/2/4/8/16 近似线性到 c4，c4→c8 平台，c16 5.7×（D 段实证） |
| 命中正确性 | populate 首轮 vs 重放 output==cold 全 True（A/B 段实证） |
| HBM | 静态 ~981/1048 GB（权重+池预分配），增量 KV 极小 |

---

## 4. 测试矩阵

### 4.1 矩阵总览（按"对比维度"组织，配置配对见 §0）

| 对比 | 配置对 | out | 长度 | W 档 | 并发 | rep | 收益 |
|---|---|---|---|---|---|---|---|
| **L2 收益（prefill，核心）** | C2 vs C1 | 1 | 32K..256K | **W1.5** | **16**（可补 c1） | ≥3 | `(C2−C1)/C2` |
| **L2 收益（decode）** | C2 vs C1 | 1024 | 32K..256K | W1.5 | 16 | 1 | E2E / Output TPS |
| **L1 收益（prefill，可选 sanity）** | C3 vs C2 | 1 | {32K,128K} | W0.5 | 16 | ≥3 | `(C3−C2)/C3` |

> **并发收敛为 c=16 主**（DP 满利用；Prefill TPS 为干净信号，TTFT 排队主导但相对差异仍有效）。**c=1 可选补干净 TTFT**（代表长度 32K/128K，验证收益非排队假象）。c=4/8 由探针实证曲线，矩阵不复测。
> **L1 收益（C3 vs C2）仅作 sanity**（结果确定），跑最小集 {32K,128K}×W0.5 即可；若不要冷参照锚点可直接跳过 C3。

- **W0.5 = 192K/rank**（L1=384K/rank 的一半，全进 L1）：C3 0% 命中、C2 ~100% 命中 → 隔离 **L1 收益**。
- **W1.5 = 576K/rank**（=1.5×L1，>L1=384K、< **L2=768K** 且 < L1+L2=1152K/rank）：C2 命中崩到 ~67%（576/384 LRU 覆盖）、C1 ~100%（L2=768K > W1.5，全部 KV 兜住且 L2 不溢出）→ 隔离 **L2 收益**。
- **C2 两档都跑**（W0.5 作 L1 基线、W1.5 作 L2 基线）；C3 只 W0.5（W1.5 下是冷全量无意义）；C1 只 W1.5（W0.5 下 ≈C2 无意义）。
- **量纲统一为 per-rank token**（请求 round-robin 摊到 16 个 rank，每 rank 只装自己的 N/16 条）：L1=384K/rank、L2=768K/rank、W0.5=192K/rank、W1.5=576K/rank、L1+L2=1152K/rank。**L2(768K) > W1.5(576K)** → W1.5 的全部 KV 在 L1+L2 配置下都能进 L2，L2 不溢出（余量 33%）。总量 = 16×per-rank（W1.5 总量 9.2M 用于估 populate 时间；若退回串行 populate ≈48min/配置，见 §8 风险 1）。

### 4.2 请求数（N = ceil(16 × W_per_rank / 长度)）

| 长度 | W0.5（192K/rank） | W1.5（576K/rank） | 每 rank 实际占池 |
|---|---|---|---|
| 32K | 94 | 282 | W0.5: ~6条=192K=50% · W1.5: ~18条=576K=150%(溢出) |
| 64K | 47 | 141 | W0.5: ~3条=192K=50% · W1.5: ~9条=576K=150%(溢出) |
| 128K | 24 | 71 | W0.5: ~2条=256K=67% · W1.5: ~4.5条=576K=150%(溢出) |
| 256K | 12 | 36 | W0.5: ~1条=256K=67% · W1.5: ~2.25条=576K=150%(溢出) |

> 请求按 round-robin（前缀 i → rank i%16）摊到 16 个 rank，每 rank 只装 N/16 条，KV 落进该 rank 自己的 384K FULL 池。W0.5=每 rank 半满（长档因取整到 67%），W1.5=每 rank 150%（溢出 → L1-only 命中崩、L1+L2 靠 host 兜住）。256K 档单条已占池 67%，并发受限（见 §8 风险 2）。

### 4.3 优先级分批（前一批过再跑下一批）

| 批次 | 内容 | 判读 |
|---|---|---|
| **P1 核心（L2 收益）** | C2 vs C1 · W1.5 · out=1 × 长度全 × c=16 | **L2 增量收益**（TPS 主信号） |
| **P2 扩展** | 补 **c=1**（长度{32K,128K}，干净 TTFT，验证收益非排队假象） | 收益稳健性 |
| **P3 decode** | C2 vs C1 · W1.5 · out=1024 × 长度全 × c=16（单 rep） | E2E / Output TPS |
| **P4 可选 sanity** | C3 vs C2 · W0.5 · out=1 × {32K,128K} × c=16（L1 收益确定，仅冷参照锚点） | L1 收益 sanity / C3 锚点 |
| **P5 可选** | 50% 部分命中、C1@W0.5（L2 空闲附加开销） | 扩展 |

---

## 5. 正确性校验（性能 + 正确性一起测）

每 case 三步骤（正确性重放**在 measure 之后**，不污染性能数据）：

```
① populate：逐条写前缀（100% 档=完整 N×G+1；50% 档=前缀 P），
            用 /generate 捕获 output_ids → 首轮输出
② measure：原生 bench_serving（纯性能）
③ 重放对比：measure 后，用与①相同输入重放，捕获 output_ids，
            与①逐 token 比较 → PASS/FAIL + 不一致条数
```

- 100% 档 populate 写完整 `N×G+1`，重放同输入 → 同位置逐 token 比；
- ③ 发生在 measure 之后，重放触发的 load-back（C1 的 W1.5 档）顺带验证 L2→L1 回载正确性；每条约 8 token 输出，开销可忽略；
- C3 无缓存，populate 与重放都是冷算 → 逐 token 一致是确定性控制组。

---

## 6. 硬校验（任一不过 → 该 case INVALID）

1. **C2/C1 实际 L1 FULL 容量必须相同**（server.log `DSV4 pool sizes` 取**最后一条**，约束后值）且 ≈384K；不一致整轮作废。
2. C1 的 L2（host c128 span）≈ 2×L1（±5%，ratio 2.0）。
3. **命中率按观测值报告**，不强求精确目标；但 populate 后 measure 前先做一次前缀重放，确认 `cached_tokens` 达到预期（100% 档 ≈ N×G，因 −1 影响 ±1 组内）。
4. 请求未被 context 拒绝（`Successful == 请求数`）。
5. **W1.5 溢出校验**：C2（只开 L1）在 W1.5 的 achieved_hit 应明显低于其在 W0.5 的值（~67% vs ~100%），验证工作集真的溢出 L1；若仍 ~100%，工作集没溢出，该档重查（加大 W 或检查路由）。
6. **C3 校验**：C3（全关）在 W0.5 的 achieved_hit 应 ≈ 0%（验证 `--disable-radix-cache` 生效）。
7. 服务日志该窗口无 `Traceback`/`OOM`/backup 卡死（见 §8 风险 1）。
8. `achieved_hit` 与 `cached-token` 交叉验证（不能只靠 `Successful`）。

---

## 7. 指标与收益计算

bench 汇总取 **Mean/P50/P90 TTFT**、**Input token throughput**（out=1 档另留 E2E；out=1K 档加 Output TPS）。

```
# L1 收益（W0.5 档，C3 全关 vs C2 只开 L1）
L1 TTFT 收益 = (TTFT_C3 − TTFT_C2) / TTFT_C3
L1 TPS 收益  = (TPS_C2 − TPS_C3) / TPS_C3

# L2 收益（W1.5 档，C2 只开 L1 vs C1 L1+L2）
L2 TTFT 收益 = (TTFT_C2 − TTFT_C1) / TTFT_C2
L2 TPS 收益  = (TPS_C1 − TPS_C2) / TPS_C2
```

- **主结论 = L2 收益（W1.5 档 C2 vs C1）**。L1 收益（W0.5 档 C3 vs C2）仅作 sanity，结果确定不重点解读。禁止用 C3 vs C1 冒充 L2 结论。
- 每 case ≥3 rep 取中位数（P3 单 rep 注明）。
- 低并发/高请求数档 TTFT 受排队主导，Prefill TPS 是全程干净信号。
- 每条记录：TTFT、E2E、TPS、achieved_hit、`[C128-HiCache]` backup/evict/load-back 事件计数（解释命中来源）。

---

## 8. 风险与已知问题

1. **backup 卡死 bug**（`docs/superpowers/debug/2026-08-18-...`）：C128 write-through backup 路径偶发卡 prefill。已规避，**写计划时假设并发 populate 可用——执行前先用并发 populate 复验**（探针 `--pin-check` 的 sanity 段已覆盖串行；需加一段并发 populate 冒烟）。若复现卡死，退回串行 populate（W1.5=9.2M ≈ 48min/配置）。
2. **256K 请求占池 67%**：钉 L1=384K 后，单条 256K 请求占单 rank 池 67%。round-robin 下 **c≤16 → 每 rank 只 1 条** → c=16 仍成立（16 rank × 1 × 67%）。仅当某 rank 被路由 ≥2 条 256K 时才超池（即 N>16 时，如 W1.5 的 36 条 → 部分 rank 2 条 = 134% → 溢出，这正是 W1.5 的测点）。measure 若出现 OOM/拒绝则该 case 作废并注明。
3. **free 路由命中一致性**：原生 bench_serving 不设 `routed_dp_rank`，同前缀重放能否落同一 rank 依赖 server 内置 round_robin 的到达序。**执行前用探针验证**：populate 顺序写、measure 顺序重放，检查 achieved_hit ≈ 预期；若并发下命中掉，measure 用 c1 或 loader 显式钉路由。
4. **populate 并发会触发 W1.5 驱逐+backup**（工作集超池），与 bug 规避后是否仍安全，需在 P1 第一 case 观察。

---

## 9. 执行流程（Runbook 概要）

```bash
# 0) 三份启动脚本（从 run_server.sh 派生）：
#    run_hicache_perf_pin.sh   # C1: + --max-total-tokens 384000 --hicache-ratio 2.0
#    run_l1only_perf_pin.sh    # C2: + --max-total-tokens 384000（无 hicache）
#    run_off_perf_pin.sh       # C3: C2 + --disable-radix-cache
# 1) C2（只开L1）: 起服 → 等 READY → W1.5(populate→measure→重放) → 拉数据   ← 基线（P1 必跑）
# 2) C1（L1+L2）: 起服 → 等 READY → W1.5(populate→measure→重放) → 拉数据     ← 被测（P1 必跑）
# 3)（可选）C3（全关）: 起服 → 等 READY → W0.5 冷全量（skip-populate）→ 重放 → 拉数据  ← P4 sanity 锚点
# 4) 算收益：主 = L2收益 C2 vs C1（W1.5）；可选 = L1收益 C3 vs C2（W0.5）
# 每轮记录：npu-smi HBM、server.log 尾部、[C128-HiCache] 事件计数
```

执行顺序 C3→C2→C1（用户指定）。每配置一 server 跑到底，不做按 case bounce（16 卡启动 ~5min 很贵）。

---

## 10. 探针/脚本清单

| 脚本 | 位置 | 状态 |
|---|---|---|
| `run_l1l2_perf_pin.sh`（C1）| `test/manual/dsv4_npu_hicache/`（已部署远程 + 容器） | ✅ pin 384K + hicache + ratio 2.0 |
| `run_l1only_perf_pin.sh`（C2）| 同上 | ✅ pin 384K，无 hicache |
| `run_off_perf_pin.sh`（C3）| 同上 | ✅ C2 + `--disable-radix-cache` |
| `bench_ids_dsv4.py` | 同目录（已同步容器） | ✅ 单 cell 驱动：distinct 前缀 + N×G+1 + 并发 populate + 原生 bench + 正确性重放 + server.log 命中校验 |
| `run_matrix.sh` | 同目录（已同步容器） | ✅ 矩阵驱动 + CSV 断点续跑（`FORCE=1` 重跑） |
| `run_server_probe_pin.sh` / `probe_perf_assumptions.py` | 同目录（已部署/同步） | ✅ 探针工具（`--pin-check` 验 pin） |
