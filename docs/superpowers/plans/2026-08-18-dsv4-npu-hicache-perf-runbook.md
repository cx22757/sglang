# DSV4 HiCache 性能测试 —— 执行 Runbook（交接文档）

> 2026-08-18 初稿；**2026-08-19 更新**：方法论从「W1.5 档 C2(只L1) vs C1(L1+L2) 命中率对比」改为
> **「热组（从 L2 回载命中）vs 冷组（完全 prefill）的 TTFT 收益」**——直击 L2 收益本质：L1 放不下、L2 中存在。
> **4 档正式测试已跑完（数据本机 `/data/dsv4_perf_data_20260819/`），TTFT 收益 92.6% / 96.0% / 97.8% / 92.8%（32K/64K/128K/256K），128K 峰**。
> 设计背景/探针依据见 `2026-08-18-dsv4-npu-hicache-perf-l1l2-vs-l1.md`（只读参照，不需重读即可执行）。

## 0. 目标（一句话）

**L2 收益**：比较「完全 prefill（冷组，C2 无 hicache，0% hit）」vs「从 L2 回载命中（热组，C1，populate 驱逐 L1 后 L2 兜住）」。
收益 = `(TTFT_cold − TTFT_hit) / TTFT_cold`。冷组**必须无 hicache**（有同步 L1→L2 的 write-through 开销，会污染基线）。

> ⚠️ **为什么不做 C2 vs C1 超容对比**：C2 只开 L1 超容即 0%（measure 自驱逐级联，见 CLAUDE.md「L1 超容的 measure 自驱逐级联」），
> 测不出「L2 兜住」之外的 L1 部分命中收益。热/冷组方法直接剥离这一因素。

## 1. 环境

- NPU 远程机：`root@113.46.13.20`（SSH key 免密，`-o BatchMode=yes`），容器 `cx-dsv4`。
- 容器内源码：`/sgl-workspace/sglang`（非 git checkout，靠同步覆盖）。
- 测试脚本在容器：`/sgl-workspace/sglang/test/manual/dsv4_npu_hicache/`（**先确认已含 `--skip-replay` 参数的最新版**）。
- server 启动脚本在远程 host：`/home/cx/hicache_dsv4/`（容器可见）。
- **同步代码**：本机仓库 `test/manual/dsv4_npu_hicache/` 改完跑 `./scripts/sync_to_npu.sh <file...>`。

## 2. 配置与执行顺序（先热组再冷组，服务只重启两次）

| 配置 | 启动脚本（`/home/cx/hicache_dsv4/`） | 差异 |
|---|---|---|
| **C1 热组**（被测） | `run_l1l2_perf_pin.sh` | + `--enable-hierarchical-cache --hicache-io-backend kernel_ascend --hicache-ratio 2.0 --swa-full-tokens-ratio 0.5` |
| **C2 冷组**（基线） | `run_l1only_perf_pin.sh` | 无 hicache，`--swa-full-tokens-ratio 0.5` |

公共：`--max-total-tokens 384000 --context-length 1048576 --mem-fraction-static 0.7 --tp-size 16 --dp-size 16 --enable-dp-attention --prefill-max-requests 1 --max-running-requests 16`。
**两配置唯一差异 = hicache flags**（`swa-full-tokens-ratio 0.5` 两边都要，否则 C1 无法归因）。

**执行顺序（关键）**：起 **C1 → 跑完全部档热组 → 停 C1 → 起 C2 → 跑完全部档冷组 → 停 C2**。
每档测完**立即拉数据回本机**（见 §5），不积攒到最后。

## 3. 档位参数（单请求长度 ×2 ⇒ 请求数 ÷2）

| 档 | input_len | SWA L1 容量 | SWA L2 容量 | 热组 populate | 热组 measure | 冷组 measure |
|---|---|---|---|---|---|---|
| 32K | 32768 | 6 条/rank | 12 条/rank | 192 | 96 | 96 |
| 64K | 65536 | 3 条/rank | 6 条/rank | 96 | 48 | 48 |
| 128K | 131072 | 1 条/rank | 3 条/rank | 48 | 24 | 24 |
| 256K | 262144 | 0 条/rank | 1 条/rank | 24 | 12 | 12 |

**规则**：
- measure 条数 N = 96 × (32768/input_len)（保持 measure 工作集 = 192K SWA token/rank 不变，即 SWA L2 的一半）。
- populate 条数 = 2N（填满 SWA L2=384K/rank，使 L1 驱逐前一半 measure 目标，L2 保留 → measure 纯 L2 回载）。
- 容量账：**SWA L1=192K/rank、SWA L2=384K/rank**（`swa device(192K) × hicache-ratio(2.0)`）是 100% 命中的硬上限；C128 L2=768K 更宽裕非瓶颈。
- seed-base 恒 60000：热组 populate/measure 与冷组 measure 生成 byte-identical 的 input，保证可比。

## 4. 执行步骤（逐档循环）

### 每档热组（C1 上）
```bash
# 1) flush（每档之间必须清缓存）
ssh -o BatchMode=yes root@113.46.13.20 "docker exec cx-dsv4 bash -lc 'curl -s -X POST http://127.0.0.1:30000/flush_cache -d \"{}\" -H \"Content-Type: application/json\"'"
# 2) populate 2N 条（驱逐前 N 条出 L1）+ 自动 replay 2N 条（正确性校验：populate 输出 vs 回载输出）
#    --skip-measure 而非 --populate-only：保留 replay 阶段（replay 对比 populate 的 REPLAY_OUT=8 输出）
ssh -o BatchMode=yes root@113.46.13.20 "docker exec cx-dsv4 bash -lc 'cd /sgl-workspace/sglang/test/manual/dsv4_npu_hicache && python3 -u bench_ids_dsv4.py --input-len <L> --num-prompts <2N> --output-len 1 --concurrency 16 --route roundrobin --pop-conc 8 --skip-measure --tag HOT_<L>_pop_replay --server-log <C1_SERVER_LOG> --seed-base 60000'"
# 3) measure N 条（纯 L2 回载命中，perf；replay 已在步骤 2 验证过，这里跳过）
ssh -o BatchMode=yes root@113.46.13.20 "docker exec cx-dsv4 bash -lc 'cd /sgl-workspace/sglang/test/manual/dsv4_npu_hicache && python3 -u bench_ids_dsv4.py --input-len <L> --num-prompts <N> --output-len 1 --concurrency 16 --route roundrobin --measure-only --skip-replay --tag HOT_<L>_<N> --server-log <C1_SERVER_LOG> --seed-base 60000'"
# 4) **立即拉数据回本机**（见 §5）
```

> **replay 正确性**（2026-08-20 起）：热组 populate 阶段带 replay（`--skip-measure`），脚本在 populate 后并发重放
> 全部 2N 条（REPLAY_OUT=8 token），对比 populate 输出 vs 回载输出，`[replay] N/N identical` 即缓存数据与写入一致。
> 分歧已定位为 decode 数值漂移（DeepEP a2a）非缓存污染，perf-first 模式记录不中断（脚本 NOTE 行）。
> 冷组不带 replay（冷组是基线，无缓存可验）。

### 每档冷组（C2 上，跳过 populate）
```bash
# 1) flush
# 2) measure N 条（完全 prefill，0% hit）
ssh -o BatchMode=yes root@113.46.13.20 "docker exec cx-dsv4 bash -lc 'cd /sgl-workspace/sglang/test/manual/dsv4_npu_hicache && python3 -u bench_ids_dsv4.py --input-len <L> --num-prompts <N> --output-len 1 --concurrency 16 --route roundrobin --skip-populate --skip-replay --tag COLD_<L>_<N> --server-log <C2_SERVER_LOG> --seed-base 60000'"
# 3) 立即拉数据
```

> 热组 measure 预期 `achieved_hit` ≈ 99.6%（96 条实证）~100%；若明显 <90%，检查 populate 驱逐是否生效（L1 未驱逐 → measure 是 L1 命中非 L2）或 flush 是否跨档失效。

## 5. 数据收集（必须：**每档测完立即拉回本机，不积攒、不驻留远程**）

远程机只是测试现场；容器重建或 /data 清理都会丢数据。**任何一档 measure 完成 → 立刻 scp 回本机 `/data/`**。

```bash
# 在**本机**执行（<date> 用当天日期）
mkdir -p /data/dsv4_perf_<date>/
scp -o BatchMode=yes 'root@113.46.13.20:/sgl-workspace/sglang/test/manual/dsv4_npu_hicache/{run_*.log,result_*.jsonl}' /data/dsv4_perf_<date>/
# server.log（命中证据，按配置改名防同名覆盖）
scp -o BatchMode=yes root@113.46.13.20:/data/cx/<C1_SERVER_LOG_DIR>/server.log /data/dsv4_perf_<date>/server_C1_<ts>_hot.log
scp -o BatchMode=yes root@113.46.13.20:/data/cx/<C2_SERVER_LOG_DIR>/server.log /data/dsv4_perf_<date>/server_C2_<ts>_cold.log
# 验证本机文件完整后，清理远程/容器内残留（数据只驻留本机）
ssh -o BatchMode=yes root@113.46.13.20 "rm -f /sgl-workspace/sglang/test/manual/dsv4_npu_hicache/{run_*.log,result_*.jsonl}"
```

> ⚠️ scp 多源同 basename（`server.log`）会互相覆盖，**逐条 scp 并改名**。
> ⚠️ 容器内文件需先 `docker cp` 到远程 host 路径才能 scp（见 §1 说明）；本 runbook 产物在容器 `/sgl-workspace/sglang/...`，可直接 scp（host 可读）。

## 6. 收益计算

每档（W=工作集档位）：
```
L2 TTFT 收益 = (TTFT_cold − TTFT_hit) / TTFT_cold
```

### 6.1 正式测试实测结果（2026-08-19，数据本机 `/data/dsv4_perf_data_20260819/`）

| 档 | 条数 | TTFT_cold (ms) | TTFT_hit (ms) | 热组 hit | TTFT 收益 | 冷 InputTPS | 热 InputTPS | TPS 提升 |
|---|---|---|---|---|---|---|---|---|
| 32K | 96 | 18051.51 | 1330.06 | 99.6% | **92.6%** | 28401 | 382223 | 13.5× |
| 64K | 48 | 35402.24 | 1429.56 | 99.8% | **96.0%** | 28335 | 674550 | 23.8× |
| 128K | 24 | 72876.73 | 1568.90 | 99.9% | **97.8%** | 22337 | 993958 | 44.5× |
| 256K | 12 | 161273.80 | 11575.14 | 100.0% | **92.8%** | 19479 | 267306 | 13.7× |

**趋势解读**：
- **冷组 TTFT 随长度近线性**（18052→35402→72877→161274，×1.96/2.06/2.21）——完全 prefill 成本 ≈ 长度 × 单位 token 计算。
- **热组 TTFT 32K→128K 仅微升**（1330→1429→1569，×1.07/1.10）——L2 回载大部分被 compute 重叠，省下的 prefill 计算随长度放大 → **收益 128K 峰 97.8%**。
- **256K 突跳**：热组 TTFT 1570→11575（×7.4）——**L2→device 回载带宽成为瓶颈**（8 个 32K chunk 逐块回载），收益回落至 92.8%，TPS 仅 13.7×。
- 结论：**L2 收益在 64K~128K 区间最大（96%~97.8%）**；更长请求受 host 回载带宽制约，收益不再提升。

每条记录：TTFT_hit / TTFT_cold、achieved_hit（热组 ~100% / 冷组 0%）、Input TPS。

### 6.2 out=1024 轮实测结果（2026-08-22，数据本机 `/data/dsv4_perf_data_20260821/`，无 abort，热组 hit 99.6~100%）

方法同 §3/§4，仅 `--output-len 1024`（decode 1024 的 E2E），并发 measure `--concurrency 8`。
冷组脚本 `run_l1l2_cold_pin.sh`（参数与热组 `run_l1l2_perf_pin.sh` 完全一致，仅去 hicache flags，日志目录 `perf_C2_`）。

| 档 | 条数 | TTFT_cold (ms) | TTFT_hot (ms) | 热组 hit | **TTFT 收益** | 对比 out=1 收益 |
|---|---|---|---|---|---|---|
| 32K | 96 | 14160.60 | 452.55 | 99.6% | **96.8%** | 92.6% → ↑4.2pp |
| 64K | 48 | 28656.45 | 1320.27 | 99.8% | **95.4%** | 96.0% → ↓0.6pp |
| 128K | 24 | 61402.84 | 1336.07 | 99.9% | **97.8%** | 97.8% → = |
| 256K | 12 | 132565.89 | 7984.82 | 100.0% | **94.0%** | 92.8% → ↑1.2pp |

**结论（out=1024）**：
- **收益与 out=1 档高度一致**（94~98%），L2 收益对 decode 长度稳健；**128K 仍为峰值档 97.8%**。
- **32K 档收益提升 4.2pp**（92.6→96.8%）：out=1024 下 decode 与 L2 回载重叠更充分，短请求的固定回载成本被摊薄（热 TTFT 452ms vs out=1 档 1330ms）。
- **256K 仍受回载带宽瓶颈**（94.0%，热 TTFT 7984ms 中位 11689ms vs 其他档 ~1.3s），与 out=1 档结论一致——长请求受 L2→device 回载带宽制约。
- 无 abort（对比修复 A 前 decode 96 全 abort）；热组 hit 99.6/99.8/99.9/100.0%。

## 7. 已知风险与排查

1. **C2 超容即 0%（measure 自驱逐级联）**：只开 L1 时超容档全 miss 是「顺序 + LRU + 写透驱逐」测试机制下的必然，不是缓存硬行为——**不要用 C2 超容档做任何命中率对照**（热/冷组方法已规避）。
2. **256K 档热组构造受限**：SWA KV 单条 256K > L1=192K → **L1 物理装不下**；且 roundrobin 下 rank0-7 每 rank 2 条 256K 超 L2 单 rank 容量（1 条）→ 部分 measure 目标被整体逐出缓存 → 命中率预期明显偏低。**作为边界档记录现象，不作收益结论**；如 hit<50% 该档标注 INVALID。
3. **flush_cache 跨档有效性**：不同档之间必须 flush；C1（hicache 开）时 flush 是否清 host(L2) 层需现场确认（若只清 device，populate 2N 会与上档残留混合 → 驱逐结果失真）。flush 后看 server.log 确认 cache 清零。
4. **backup 卡死 bug**（历史）：populate 若卡（单请求 >300s），退回 `--pop-conc 1` 串行。
5. **数据丢失**：任何一档跑完立即拉回本机（§5），不要等全部档位。
6. **正确性**：measure 后可选跑一次 `_replay`（去掉 `--skip-replay`）验证回载输出与首轮一致；perf-first 模式记录分歧不中断。

## 8. 关键脚本参数（bench_ids_dsv4.py，已含三阶段独立开关）

```
--input-len N*G   缓存前缀长度（G=2048 的倍数；input = input_len+1 吸收最长命中−1）
--num-prompts     请求数（§3 表）
--output-len      1
--concurrency     16
--route roundrobin（钉 routed_dp_rank=i%16，populate/measure 同 rank）
--pop-conc 8      populate 并发
--seed-base 60000（恒，保证冷热组 input 一致）
阶段开关（任意组合）：
  --populate-only         只 populate
  --measure-only          只 measure（热组 measure 用）
  --skip-populate         跳过 populate（冷组 measure 用）
  --skip-measure          跳过 measure
  --skip-replay           跳过正确性重放（perf-first 用）
```
