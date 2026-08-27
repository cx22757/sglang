# DSV4 replay 分歧根因确认 —— deep_ep 版本差异（环境对比 + 互换验证）

> ⚠️ **适用范围**：§1–§9 根因 = **C2（无 HiCache）** 并发 populate replay 的 **deep_ep 版本差异**（npu-B 新版 bf16 漂移 → 换 npu-D 旧版 int8 修复）。
> **C1（HiCache L2 回载）场景有独立根因**（旧版 deep_ep 下仍分歧，定位到**并发 load_back 数据错误**），见 **§10**。
>
> 2026-08-20。并发 populate → replay 输出分歧问题（见 `dev/main-8.5-debug-con-populate` 分支 `docs/superpowers/debug/replay_divegence.md`）此前已排除缓存污染、定位到"并发下 decode 微小数值漂移（疑 DeepEP a2a）"。本轮通过 **npu-B vs npu-D 的 `cx-dsv4` 容器环境逐项对比 + 互换 deep_ep 验证**，**根因锁定 = 两节点 deep_ep 版本不同**：npu-B 原装新版（a2a 走 bf16 不量化），npu-D 装旧版（走 int8 量化）。把 npu-D 旧版装到 npu-B 后**分歧消失**。

---

## 1. 摘要

**现象回顾**（replay_divegence.md）：C2（只开 L1，无 HiCache flag）perf 矩阵下，并发 populate 预热缓存后 replay 同一输入（temperature=0 贪心）完整命中缓存（`cached=32768`）但输出分歧。串行 populate 不出现。KVSUM digest 6 字段 328 层 diffs=0、冷算 ground truth 证明 replay 正确、logprobs 显示分歧起点 decode #1、logprob 差仅 ~0.14（argmax 边界翻转）。

**本轮新信息**：该分歧**只在 npu-B（113.46.13.20）的 `cx-dsv4` 出现**，npu-D（113.46.46.40）同名容器**没有**。→ 纯"DeepEP 浮点不确定性"解释不了节点相关性，必然存在**环境差异**。

**结论**：两容器逐字段对比后，**唯一运行时相关差异是 `deep_ep` 安装版本**（其余：python/sglang 运行时源码逐字节一致、custom_ops/sgl_kernel_npu 安装后 .so 一致、驱动/CANN/torch 全家桶一致、注册 custom:: 算子一致、启动脚本一致）。把 npu-D 的 deep_ep（`1.0.0+7a396de`）打包目录装到 npu-B（原 `1.0.0+48942e3`）后，**并发 populate replay 分歧消失 → 根因 = deep_ep 版本差异**。

## 2. 环境对比方法论

- 对比对象：npu-B(113.46.13.20) 与 npu-D(113.46.46.40) 的 **`cx-dsv4`** 容器（注意：40 节点清单文档采的是 `mc-l4-test`，是不同容器；本对比针对精度问题涉及的 `cx-dsv4`）。
- 采集方式：只读 `docker inspect` + `docker exec` 内 `sha256sum`/`pip freeze`/import 枚举，不启动模型。
- 采集字段对齐 40 节点清单：容器身份/镜像 digest、env vars、挂载、驱动、CANN、Python 包版本、算子 SHA-256、注册 custom:: 算子、sglang 源码整树哈希。

## 3. 相同项（已排除差异）

| 项目 | 结果 |
|---|---|
| 宿主内核 | `5.10.0-182...hce2.aarch64`，两节点一致 |
| 容器 env vars（ASCEND_*/PATH/LD_LIBRARY_PATH…） | 逐行一致 |
| 容器挂载集合 | 一致（/data、/home、driver、firmware、weights…） |
| 驱动 / ascendhal | `25.5.1` / `7.35.23`，一致 |
| torch / torch_npu / transformers | `2.10.0+cpu` / `2.10.0` / `5.12.1`，一致 |
| `custom_ops` 安装后 `.so` | `41e2c676...`，逐字节一致 |
| `sgl_kernel_npu` 安装后 `.so` | `adbcd5df...`，逐字节一致 |
| **sglang 运行时源码** `python/sglang/`（3231 文件整树哈希） | **逐字节一致** |
| 注册 `custom::` 算子 | 24 个，完全一致 |
| 启动脚本 `run_l1only_perf_pin.sh` 等 | md5 一致，均 `--moe-a2a-backend deepep --deepep-mode auto` |

## 4. 差异项

| # | 差异 | npu-B（原） | npu-D | 相关性 |
|---|---|---|---|---|
| **1** | **`deep_ep` wheel 版本** | `1.0.0+**48942e3**`（sgl-kernel-npu 新 commit） | `1.0.0+**7a396de**`（旧 commit） | ⭐⭐⭐ |
| **2** | **`deep_ep` 安装后 4 个 `.so`** | `deep_ep_cpp.so` + vendor `libcust_opapi.so`/`libcust_opmaster_rt2.0.so`/`libcust_opsproto_rt2.0.so` 全部不同哈希 | — | ⭐⭐⭐ 与 #1 同源 |
| 3 | 基础镜像 | ID `aee0605`，digest `sha256:15f972ff`，build 08-16 | ID `f53d1907`，digest `sha256:40c80254`，build 07-27 | ⭐ tag `main-cann9.0.0-a3` 可变，两节点 pull 到不同内容 |
| 4 | `mooncake-transfer-engine` wheel | `/home/cx/package/` | `/tmp/`，不同 sha | ⭐ C2 无 HiCache，不涉 a2a，低 |
| 5 | sglang pip 元数据 | `dev710+g591cfb088` | `dev241+g8d6549bc4` | 仅安装时刻不同，实际源码一致 |
| 6 | 次要 pip 版本（fastapi/anthropic 等） | 更新 | 旧版 | 无关 |
| 7 | `test/manual/dsv4_npu_hicache/` | 多 9 个 debug/probe 脚本 | 无 | 客户端测试脚本，非运行时 |

## 5. deep_ep 版本 diff（`7a396de..48942e3`，线性，8 个 commit）

区间 commit：`#598` fix tiling bug, a2 expand topk16（7a396de 本体）→ `#599`/#600 FP8 per-token 量化 → `#608` DEEPEP_HCCL_BUFFSIZE → `#614` custom-ops A2-patch → `#618` disable pertoken_fp8_e5m2（动 `deep_ep.cpp`）→ `#619` Precision threshold（只改测试脚本）。

核心改动（a2a dispatch/combine 路径）：
- `csrc/deepep/deep_ep.cpp`（+71）：量化模式字符串重构（`fp8_e4m3`→`mx_fp8_e4m3`、新增 `PER_TOKEN_FP8_SCALES=5`）、`low_latency_dispatch` 签名从 `use_fp8/use_ue8m0/use_mxfp4` bool 改为 `quant_mode_name`、A5-only 量化守卫。
- `strategies/normal_strategy.py`：**删除了 `DEEP_NORMAL_MODE_USE_INT8_QUANT` env 读取**（旧版：`use_quant = os.getenv("DEEP_NORMAL_MODE_USE_INT8_QUANT")=="1"` → int8；新版：改 `quant_mode` 参数，默认 `bf16`，`_intranode_combine` 强制 `use_quant=False`）。
- tiling/kernel：`moe_distribute_dispatch_v2_tiling.cpp`、`cam_moe_dispatch_normal_tiling.cc`、dispatch/combine kernel 等改动。

## 6. 关键开关：`DEEP_NORMAL_MODE_USE_INT8_QUANT=1`

启动脚本 `run_l1only_perf_pin.sh` 里 `export DEEP_NORMAL_MODE_USE_INT8_QUANT=1` **只有旧版 deep_ep 读取**。因此两节点实际跑的 a2a 量化路径本就不同：

| | deep_ep 版本 | 脚本 env | a2a 实际路径 |
|---|---|---|---|
| npu-D | 旧版 `7a396de` | `DEEP_NORMAL_MODE_USE_INT8_QUANT=1` | **int8 量化** dispatch/combine |
| npu-B 原 | 新版 `48942e3` | 同 env（**被忽略**） | **bf16 不量化** |

int8 定点路径并发下数值确定；bf16 浮点 a2a 并发下微小漂移（~0.14 logprob → argmax 翻转）。这解释了两节点行为差异 + 并发相关性。

## 7. 互换验证（根因确认）

**只做单向**：把 npu-D 的 deep_ep 装到 npu-B（不做反向）。

```bash
# 1) npu-D 打包已安装目录（比 wheel 更稳，2.3MB）
ssh root@113.46.46.40 "docker exec cx-dsv4 bash -lc 'cd /usr/local/python3.11.15/lib/python3.11/site-packages && tar czf /tmp/deep_ep_npud.tgz deep_ep' && docker cp cx-dsv4:/tmp/deep_ep_npud.tgz /tmp/deep_ep_npud.tgz"
#    → scp 到本机中转再传 npu-B（或两节点互通则直传）

# 2) npu-B 备份原版（回滚点）
ssh root@113.46.13.20 "docker exec cx-dsv4 bash -lc 'cd .../site-packages && tar czf /tmp/deep_ep_npub_backup.tgz deep_ep' && docker cp cx-dsv4:/tmp/deep_ep_npub_backup.tgz /data/cx/deep_ep_npub_backup.tgz"

# 3) npu-B 覆盖安装 + 校验
ssh root@113.46.13.20 "docker cp /tmp/deep_ep_npud.tgz cx-dsv4:/tmp/ && docker exec cx-dsv4 bash -lc 'cd .../site-packages && mv deep_ep deep_ep.npdbak && tar xzf /tmp/deep_ep_npud.tgz'"
#    校验：5 个 .so 哈希应与 npu-D 一致（deep_ep_cpp.so=12d17935... 等）
#    冒烟：LD_LIBRARY_PATH 加 torch/torch_npu lib 后 `import deep_ep` OK

# 4) 复现（沿用原 C2 脚本，无需改动）
ssh root@113.46.13.20 "docker exec -d cx-dsv4 bash -lc 'nohup bash /home/cx/hicache_dsv4/run_l1only_perf_pin.sh > /home/cx/hicache_dsv4/C2_launch.out 2>&1 &'"
#    等 /health 200，读 C2_launch.out 的 server.log 路径
#   并发 populate + replay（默认 --pop-conc 8）
ssh root@113.46.13.20 "docker exec cx-dsv4 bash -lc 'cd /sgl-workspace/sglang/test/manual/dsv4_npu_hicache && python3 -u bench_ids_dsv4.py --input-len 32768 --num-prompts 32 --output-len 1 --route roundrobin --skip-measure --tag W0.5_32K_deepep_old --server-log <RUN_E>/server.log --seed-base 60000'"
```

**结果：分歧消失**（replay N/N identical）→ 根因锁定 deep_ep 版本。

## 8. 根因链（最终版）

```
新版 deep_ep（48942e3）删掉 DEEP_NORMAL_MODE_USE_INT8_QUANT 处理
  → npu-B 的 a2a dispatch/combine 从 int8 变为 bf16 不量化
  → bf16 浮点通信在并发下产生微小数值漂移（非确定性）
  → 该请求 decode 的 logprob 差 ~0.14，压在 argmax 边界
  → temperature=0 贪心在边界翻转 → first/replay 输出分歧
  → 串行无并发、旧版走 int8 定点 → 均确定，不出现
```

## 9. 回滚与后续建议

- **回滚**：npu-B 若要换回新版 deep_ep：`docker exec cx-dsv4 bash -lc 'cd .../site-packages && rm -rf deep_ep && mv deep_ep.npdbak deep_ep'`（或解压 `/data/cx/deep_ep_npub_backup.tgz`）。
- **版本锁定**：deep_ep 是易变组件，同 flag 不同版本内核/量化路径不同。perf 复现前**锁死 deep_ep 版本**（wheel 名 `+<commit>` 即版本）。
- **升级校验**：升级 deep_ep 前先跑并发 populate replay 校验（`bench_ids_dsv4.py --skip-measure`），确认 bf16 路径并发下仍确定。
- **环境一致性**：`main-cann9.0.0-a3` 是可变 tag，多节点 pull 到不同内容。严格 A/B 前用 `@sha256:` digest 钉镜像，或统一在构建机出包。

---

## 10. 补充：C1（HiCache L2 回载）场景分歧 —— 独立根因（并发 load_back 数据错误）

> **2026-08-20 追加。** §1–§9 的 deep_ep 版本根因**只适用于 C2（无 HiCache）并发 populate replay**。
> 在 **C1（HiCache on，走 L2 回载）** 场景，**旧版 deep_ep 下仍复现分歧** → 这是**独立根因**，不是 deep_ep。

### 10.1 复现（参数与现象）

- 环境：C1（`run_l1l2_perf_pin.sh`，HiCache on，L2=2×L1），旧版 deep_ep（int8 确定性）。
- 复现：`bench_ids_dsv4.py --input-len 65536 --num-prompts 80 --pop-conc 8 --skip-measure`（populate 80 + 自动 replay 80）。
- 结果：**21/80 分歧**（`cached=65536` 全命中，但 replay 输出 ≠ populate 输出）。
- **16 条不复现**：16 条 64K = 1 条/rank，远小于 SWA L1 容量（3 条/rank = 48 条），populate 无驱逐，replay 全 L1 命中，**不走 L2 回载路径**。→ 分歧只在 **L2 回载**（前缀被驱逐出 L1 后从 L2 host 回载）时出现。

### 10.2 排除实验（证据链）

| 实验 | 做法 | 结果 | 排除 |
|---|---|---|---|
| `verify_loadback.py` | 单请求串行 L2 回载 vs flush 后冷算 vs 再回载 | **A==B==C==populate first**（cached=65536） | 串行回载无污染；L2 数据正确 |
| `verify_concurrent.py` | 8 个**相同**请求（同前缀）并发回载 | 全部 == 串行 == first | 纯并发（共享前缀）无影响 |
| `verify_matrix.py` | 满缓存 + 并发 **8 个不同前缀** 回载 vs 串行基准 | **1/8 一致，7/8 分歧** | 复现（需要并发 + 不同前缀 + 满缓存）|
| `verify_serial_repeat.py` | 满缓存下同一请求串行回载 ×4 | 全部稳定 == first | 回载路径对**给定缓存状态**确定 |

### 10.3 KVSUM digest 决定性证据

**方法**：从 `dev/main-8.5-debug-con-populate` 分支移植 KV_DIGEST 插桩（3 文件：`environ.py` 加 2 env、`decode_cuda_graph_runner.py` 加 `rids` 透传、`ascend_dsv4_backend.py` 加 `_kv_digest`），`SGLANG_DSV4_KV_DIGEST=1` 重启 C1 后重跑 populate 80 + replay 80（21/80 分歧复现）。digest 在 populate **extend 写**（phase=write）与 replay **decode 读**（phase=read）时打，覆盖 swa/c4/c128/state/idx/sidecar 6 字段，全压缩层（默认所有 compress_ratio∈{4,128} 层）。

**关键对比（分歧请求 req0，phase=write，ntok=65537，均 cached=65536 命中 + 1 token extend）**：

| 字段 | populate-0 write | replay-0 write | 结论 |
|---|---|---|---|
| layer2 **swa** | 14,331,506 | **11,965,274** | ❌ -17% |
| layer2 **c4** | 5,684,505.5 | 5,684,505.5 | ✅ 逐位一致 |
| layer2 **state** | 1,188,494 | **33,892** | ❌ 差 35× |
| layer15 **swa** | 23,060,146 | **4,464,874** | ❌ -80% |
| layer15 **c128** | 44,033.48 | **42,681.68** | ❌ 微差 |

- req0–4 **全部**同模式：layer2 swa ~14.3M→~11.9M、layer15 swa ~23M→~4.5M（**系统性**偏小，非随机）。
- **analyze_kvsum.py 全量**：requests=80、layers=3280、**diffs=1968（60%）**、suspect=640（全为 `state=inf`）、missing=0。
  - ⚠️ 注意：analyze_kvsum 把 `ntok` 不同直接判 DIFF——populate write（ntok=65537）与 replay read（ntok=65537+decode 步）天然 ntok 不同，**diffs 里含 ntok 假象**；但 **swa/state 的逐值差异（-17%/-80%）与 640 层 state=inf 是真实的**（非 ntok 问题）。

**结论**：并发 replay 时，从 L2 回载写入 device 的 **SWA/compress-state 数据 ≠ populate 时写入的**（C4 一致）。差异 17%–80%，**远超数值漂移量级（~0.14 logprob）**，非 deep_ep。

### 10.4 代码定位（机制层，⚠️ merge 假设已被 §10.6 时序实验推翻）

```
load_back (unified_radix_cache.py:1011)
  → 各 component prepare_load_back (预分配)
  → build_load_back_spec → kv_xfer(主KV) + comp_xfers(SWA/state)
  → cache_controller.load (hybrid_cache_controller.py:506)  ← 队列化
       load_queue.append(CacheOperation)    # 每个请求一个 op
  → start_loading() → CacheOperation.merge_ops(load_queue)   ← 合并多请求
       host_indices = torch.cat([op.host_indices ...])        # 主KV(C4)
       device_indices = torch.cat([op.device_indices ...])
       pool_transfers = merge_pool_transfers(ops)             # SWA/state
  → move_hybrid_indices → load_to_device_per_layer (逐层 H→D)
```

- **`merge_ops` 是并发特有路径**：`len(ops)==1` 直接返回不合并 → 对应"串行回载正确、并发回载错"。
- **C4 与 SWA/state 走不同通道**：主 KV（C4/full）用 `host_indices/device_indices`（一一对应拼接）；SWA/state 用 `pool_transfers`（`merge_pool_transfers` 按 `(name, indices_from_pool)` 分组后 `cat` host/device/keys）。**合并后的 pool_transfers 与主 KV 的对应关系错位 → SWA/state 拷错位置**，C4 不受影响。
- 640 层 `state=inf`：compress-state 位置映射（C4 从 swa_loc、C128 从 req_position）在回载写错后读到未初始化/错位数据。

### 10.5 当前结论与下一步

**结论**：C1 场景分歧根因 = **并发 load_back 时多个请求的 SWA/compress-state 回载（extra_pools）数据被写错**（合并传输/拷贝错位），回载后的 KV ≠ populate 写入 → decode 读到不同数据 → 输出分歧。与 deep_ep 无关（旧版下也错）；串行回载正确（单请求不合并）。

**影响面**：热组 measure 依赖并发 L2 回载 → **回载 decode 输出正确性受损**（TTFT 主要反映回载耗时可能基本有效，但正确性校验失败）。冷组（无 hicache）不受影响。之前 4 档 out=1 收益测试用了 `--skip-replay`，未测正确性。

**下一步（已定位，见 §10.6）**：

> 🚫 **以下 1–3 项随 §10.6 被证伪而失效（见 §10.7）。§10.6 的根因不成立，修复方向 A/B/C 勿采纳；根因重开，当前以 §10.7.3 的取证实验为准。**

1. ~~定位精确 bug 行~~ → **§10.6 完成**：根因是**异步 write-through backup 与 load_back 不同步**（D2H 未完成即标记 backuped，load_back 无等待）。**非 §10.4 的 merge 错位假设**（merge 索引对齐正确，已被时序实验排除）。
2. ~~低并发验证~~ → §10.6 时序对照实验已决定性确认（populate 后 0s 全分歧 / 60s 全对）。
3. 出修复：三选一（§10.6.5，推荐 **A**：load_back 等待 backup 完成）。
4. 修复后回归：80 条 64K populate+replay 应全 identical（`verify_timing.py after` 即是对照）。

**验证脚本**（`test/manual/dsv4_npu_hicache/`）：`verify_loadback.py`、`verify_concurrent.py`、`verify_matrix.py`、`verify_serial_repeat.py`（均本机 + 容器）。KV_DIGEST 插桩 diff：`/tmp/kvdigest.patch`（本地，3 文件）。数据：本机 `/data/dsv4_perf_data_20260820/server_C1_kvsum80.log`（22.5MB，含 [KVSUM] 日志）。

### 10.6 精确 bug 行定位（2026-08-20 补充）—— 异步 write-through backup 与 load_back 不同步

> 🚫 **本节根因已被证伪（2026-08-20 复核，见 §10.7）——请勿据此按方向 A/B/C 修复。**
> §10.6.2 的机制链（"commit_backup 立即标记 backuped → 无检查的 load_back 读到未刷完 host"）
> 在 **write_through 策略（DSV4 C1 默认）下每一环都断**：write_through pending 节点被 `inc_lock_ref`
> 全 component 加锁 → 移出 `evictable_device_leaves` → 不可逐出 → 不可能变 host-only → 其 load_back
> 不可能在 D2H 完成前发生；且 `ack_finish_event` 覆盖 **base + 全部 extra_pools（含 SWA/state）**，
> 锁释放严格 gate 在全池 D2H 完成之后。下面 §10.6.1 的时序实验数据**真实**，但其解释无效，
> 根因重新打开，详见 §10.7。

> ⚠️ **修正 §10.4/§10.5**：后续的**时序对照实验**证明根因**不是 merge（`CacheOperation.merge_ops`）错位**，而是
> **populate 的异步 write-through backup（L1→L2 D2H）未完成时，立即 load_back（H→D）读到未写入的 host 数据**。
> merge 本身索引对齐正确（`host_indices`/`device_indices`/`keys` 均按 op 顺序 `cat`，消费端一一对应）。
> （↑ 此 ⚠️ 结论亦随 §10.6 被证伪而失效：§10.4 的 merge 假设的"推翻"依据本身站不住，见 §10.7。）

#### 10.6.1 决定性实验：同一批 populate，立即 vs 延迟并发

对**同一台 KVSUM C1 服务**（满缓存 80 条），`verify_timing.py` 并发 8 个不同前缀（id0-7，roundrobin），对照**已知正确的串行 ground truth**：

| 时机 | 结果 |
|---|---|
| populate 后 **0s**（立即并发）| **0/8 correct**（全分歧，`cached=65536` 全命中但输出 ≠ ground truth）|
| populate 后 **60s**（延迟并发）| **8/8 correct**（全对）|

唯一变量 = 60s。→ **分歧 = populate 后 backup 未排空时的 load_back**。

辅助对照（同一 KVSUM 服务、不同时间点）：
- `verify_matrix.py`（重新 populate 后立即并发 8 个）→ 1/8 一致（分歧）
- `verify_matrix.py` 在 KVSUM populate **很久后**重跑 → **8/8 一致**（backup 已排空）
- `verify_n2.py`（KVSUM populate 后，2 并发不同前缀）→ 输出全对（延迟），但 digest 仍显"错" → **digest 的"错"是读取时机假象**（回载后 device 位置可能被回收/覆盖），不是输出错

#### 10.6.2 根因机制链

```
populate 每请求写 device KV
  → _execute_kv_backup → cache_controller.write (hybrid_cache_controller.py:374)
       mem_pool_host.alloc() 分配 host 槽
       write_queue.append(op)
       start_writing()  → D2H kernel 提交到 write_stream（异步，GPU stream 排队）
       return host_indices（此时 host 数据【未实际写入】）
  → _execute_and_commit_kv_backup (unified_radix_cache.py:917)
       tree_core.commit_backup(...)  (unified_radix_cache.py:933)
         → 立即标记 node.backuped = True（D2H kernel 还在 write_stream 上飞）

立即 replay（populate 刚返回）
  → needs_host_load_back() 命中（node.backuped == True）
  → load_back (unified_radix_cache.py:1011)  ←【无 backup 完成检查】
       cache_controller.load (load_queue 合并)
       start_loading (hybrid_cache_controller.py:549)
         → D2H 之后 H→D 在 load_stream 执行  ←【未等待 write_stream 的 backup 完成】
       → 读到未写入/部分写入的 host 槽 → 回载的 SWA/state 数据错
       → decode 读到不同 KV → 输出分歧
```

#### 10.6.3 关键代码行

| 文件:行 | 问题 |
|---|---|
| `hybrid_cache_controller.py:374-404` `write` | D2H 只 `write_queue.append` + `start_writing` 提交到 **write_stream（异步）**；返回 `host_indices` 时 host 数据未写 |
| `unified_radix_cache.py:933` `commit_backup` | 紧跟 `write` 之后**立即标记 node.backuped**，D2H kernel 未完成 |
| `hybrid_cache_controller.py:549+` `start_loading` | load 在 **load_stream** 执行，**无与 write_stream 的事件同步**（backup 完成事件未被等待）|
| `unified_radix_cache.py:1011` `load_back` | **无 backup 完成（`ongoing_write_through` / ack / backuped 语义）检查** |

#### 10.6.4 现象全景自洽

| 现象 | 解释 |
|---|---|
| 串行回载 == 冷算（verify_loadback A==B==C）| 单请求 backup 有充分时间完成 |
| 并发 replay 分歧（21/80）| populate 后立即并发 load_back，backup 未排空 |
| 16 条不复现 | 不走 L2 回载（<L1 容量），无 load_back |
| KVSUM：SWA/state 错、C4 对 | SWA/state 走 extra_pools（host 侧），C4 主 KV 回载路径/时机不同 |
| 延迟后 8/8 对 | backup D2H 完成 → host 数据完整 |
| out=1 四档测试"正常" | 用了 `--skip-replay`，未测正确性 |

#### 10.6.5 修复方向（未实施）

- **A（推荐）**：`load_back` 触发前检查对应 node 的 write-through 是否完成（`ongoing_write_through` / ack），未完成则**等待**或**先触发 backup 完成**。
- **B**：`commit_backup` 改为在 D2H ack 后才标记 `backuped`（未完成的 node 不被判定 host hit，`needs_host_load_back` 不命中）。
- **C**：`start_loading` 在 `load_stream` 上**等待 `write_stream` 的完成事件**（跨 stream 同步）。

修复后回归：80 条 64K populate+replay 应全 identical（`verify_timing.py after` 即是对照）。

### 10.7 §10.6 证伪与根因重开（2026-08-20 复核）—— write_through 锁不变量成立，时序矛盾未解

> **本节推翻 §10.6（连带 §10.4/§10.5 的相关结论）。** 对 `dev/main-8.5`
> commit `bd156769763fe25a0f9cffc5318b52cfd864e879` 静态核实整条 write_through 不变量链，
> §10.6 的"backup 未刷完即 load_back"机制在每一环都断，故按 §10.6.5 的 A/B/C 修复大概率修不到点子上。

#### 10.7.1 不变量链核实（每一环都堵死 §10.6）

DSV4 C1 用**默认 write_through 策略**（`run_l1l2_perf_pin.sh` 无 `--hicache-write-policy write_back`）。此策略下：

| 环节 | 代码位置 | 事实 |
|---|---|---|
| backup 对节点全 component 加锁 | `unified_radix_cache.py:936` `_execute_and_commit_kv_backup` → `inc_lock_ref(node_id)` | 锁存入 `ongoing_write_through`，`_finish_write_through_ack` 才释放 |
| 加锁即移出可逐集 | `unified_tree_core.py:548` `inc_lock_ref` 尾部调 `_update_evictable_leaf_sets(node)` | — |
| 被锁节点非 device-leaf | `unified_tree_core.py:1659` `_is_device_leaf`：`any(cd.lock_ref>0)` → `False` | 锁住的节点被移出 `evictable_device_leaves`，`evict_device_leaf`/`_demote` 选不到 |
| **finish_event 覆盖全部池（非仅 base）** | `hybrid_cache_controller.py:437-451` `start_writing`：base + `resolved_pool_transfers`（SWA/state/C4/C128/sidecar）合成**一次** `backup_from_device_all_layer`，`ack_finish_event.record()` 记在其后，同一 `write_stream` 段内 | **SWA/state 与 base 同一事件 gate**，不存在"aux D2H 未被覆盖"的缝 |
| ack 严格 gate 锁释放 | `unified_radix_cache.py:1808` `_count_ready_acks` 遇首个 `finish_event.query()==False` 即停（有序）；`writing_check` 还 `.synchronize()`(:1882/1906)；`_finish_write_through_ack:1004` 才 `dec_lock_ref` | 锁释放 ⇔ 全池 D2H 完成 |

**推论**：任一池 D2H 未完成的 write_through 节点**必被锁 → 非 device-leaf → 不可逐出 → 不可能变 host-only（evicted+backuped）→ 对它的 load_back 不可能在 D2H 完成前发生**。§10.6.2 的机制链结构上无法出现。连"extra_pools 的 aux D2H 没被 Full 的锁/event 覆盖"这个能解释 SWA/state 不对称的退路也不成立（它们本就在同一 finish_event 里）。

#### 10.7.2 由此产生的新硬矛盾（根因重开）

证伪越彻底，§10.6.1 的实测（0s→0/8 全错、60s→8/8 全对，唯一变量=时间）越无法用"节点 backup 没做完"解释：

- `verify_timing.py` 取 id0-7 = **最早** populate 的 8 条前缀，0s（相对 80 条 populate 结束）时它们自身 backup 早已排空。
- 按不变量，这 8 条要么锁在 device（device 命中，读原件→对），要么已逐出且 backup 完成（load_back 读完整 host→对）。**两条路都该对，却在 0s 全错。**

→ 真正触发条件**不是**"某节点 backup 没做完就 load_back"，而是 **并发 load_back 与仍在排空的 write-through（那 80 条里较晚的 backup ack 还在 drain）在 stream 层重叠**；60s 时 write-through 全排空、无重叠 → 恢复正确。

连带结论：
- **§10.4 的 merge 错位假设重新有效**（其"推翻"依据是现已被证伪的 §10.6）；但 merge 单独解释不了"60s 也是并发却全对"——干扰对象是 **write-drain 的重叠**，不是单纯 load 合并。
- 一个可疑面：`load_back` 注明 "the aux build can evict"（`unified_radix_cache.py:1023`），满缓存下并发 load_back 触发的逐出/host 槽 free 与并发 D2H write-through 的 host 槽/index 缓冲，在 stream 层可能存在生命周期或跨 stream 同步缺口。静态阅读到此为止，需上机取证。

#### 10.7.3 下一步（决定性实验，非修复；上机进行中）

按 systematic-debugging，现处 Phase 1 取证，**不提修复**。两个读-only 实验收口（npu-B `cx-dsv4`，旧版 deep_ep + KVSUM 插桩在位）：

1. **路径判别插桩**：8 条 replay 命中时逐请求打印 { device 命中 vs load_back、`write_through_pending_id`、各 component `lock_ref`、实际 host/device indices }。
   - 0s 分歧请求走 **device 命中却仍错** → bug 在**并发 populate 的 device 写**；
   - 走 **load_back 且 pending 已清** → bug 在 **load 与 write-drain 重叠**。
2. **drain 隔离**：0s 场景先 `writing_check(write_back=True)` 强制排空 write-through，再发 8 并发 load_back。若变 8/8 对 → 锁定"write-drain 重叠"为触发器。

#### 10.7.4 上机实验结果（2026-08-20，npu-B `cx-dsv4`，旧版 int8 deep_ep，`DSV4_LB_PROBE` 插桩）

在 `load_back` 入口加单点 `[LBPROBE]` 日志（rid / node / `len(ongoing_write_through)` / node_in_wt / host·swa·mamba_hit），重启 C1（`run_l1l2_perf_pin.sh`，attention=dsv4、write_through、tp16/dp16、旧版 int8 deep_ep）后：

**实验 1 — 0s（populate 80×64K 后立即并发 8 条，roundrobin）：复现 0/8 全分歧。** LBPROBE：

| rid | 路径 | ongoing_wt | node_in_wt | host_hit | swa_hit | rank |
|---|---|---|---|---|---|---|
| timing-now-0..7 | **全部 LOAD_BACK** | **0** | False | 65536 | 128 | 各自 DP0–7 |

- **`ongoing_wt=0`** → load_back 触发时**没有任何 write-through 在排空** → §10.6（backup 未完成即 load_back）与"write-drain 重叠"假设**双双证伪**（实测）。
- **roundrobin → 8 条落 8 个不同 DP rank，每 rank 仅 1 个 load_back op**（`len(ops)==1` 不触发 merge）→ **§10.4 merge 错位假设排除**；且 DP rank 间进程隔离（各自 tree_cache/mem pool），**跨 rank load_back 数据互污结构上不可能**。
- **输出发散点滞后**：id0 前 4 token 与 GROUND 逐位一致、第 5 个才翻（267→1074）；id2 前 6 一致；id1 第 3 个翻。**KV 若被 load_back 写错，token1 即错** → 前几 token 正确说明 **load_back 的主 KV 是对的，发散是 decode 过程累积漂移**。

**实验 2 — device 命中对照（logprob 探测，串行 vs 并发，`return_logprob` top3）：** 由于实验 1 已把 id0-7 load 回 device 驻留，本次 8 条**均 device 命中（LBPROBE 无记录）**，输出 8/8 对且**并发与串行每步 top1−top2 gap 逐位相同**。
→ **对照结论：device 命中下并发 decode 完全确定（旧版 int8 无漂移）→ 并发本身不是分歧来源。**

**证据收敛**：分歧 **同时需要 load_back + 并发 + 紧邻 populate**（`ongoing_wt` 无关；非 merge；非跨 rank；非纯并发）。

#### 10.7.5 当前领先假设（未证实）——LOAD(H2D) 侧 SWA/state sidecar 同步

综合上述 + §10.3 的"SWA/state 错、C4/Full 对" + 发散点滞后 + `swa_hit=128`（SWA 仅回载小尾巴）：

> 领先假设：**load_back 的 SWA/state sidecar（小尾巴 H2D）在并发 + 紧邻 populate（copy/load 引擎繁忙）下，未被正确同步即被 decode 读到（stale/部分）**，主 KV（Full/C4，大块）回载正常；SWA 影响近窗注意力 → 发散在 decode 中累积、滞后翻转。这是 §10.6 的 **LOAD 侧镜像**（write 侧已排除，`ongoing_wt=0`），落点在 H2D load-stream 与 decode 的跨 stream 同步 / sidecar 完成事件。

**下一步取证（未做，需再一个 populate 周期）**：
1. 扩展插桩到 LOAD 侧：load_back / start_loading 记录 load ack、SWA/state sidecar transfer 完成事件、decode 前是否 sync load_stream。
2. 严格 0s vs 60s（各自独立 fresh populate，都强制 load_back）对照，确认"紧邻 populate"这一维度是否经由 load-stream 拥塞/未同步生效。
3. 若确认：修复落在 H2D 侧（decode 前等待 SWA/state sidecar 的 load 完成事件），而非 §10.6.5 的 write 侧 A/B/C。

### 10.8 决定性定位（2026-08-20 复核完成）—— 根因 = load_back 路径本身（内容对、映射错），非时序/并发/同步

> **本节收口。§10.7.3/10.7.5 的"完成同步/时序/并发"方向经上机全部证伪。根因锁定在 HiCache C1 load_back 路径给"触发回载的请求"配置的 decode 读取映射。**

**实验矩阵（npu-B `cx-dsv4`，dev/main-8.5 @ bd15676，旧版 int8 deep_ep，`DSV4_LB_PROBE`/`DSV4_LB_SYNC` 插桩）**：

| 条件 | 结果 | 排除的因素 |
|---|---|---|
| 0s 并发 load_back（overlap 开）| 0/8 | 基线复现 |
| 0s 并发（`--disable-overlap-schedule` + `OVERLAP_PLAN_STREAM=0`）| 1/8 | **overlap 调度** |
| 0s 并发 + `DSV4_LB_SYNC`（start_loading 后强制 `finish_event.synchronize()`）| 0/8 | **H2D 完成同步（layer-overlap 时序）** |
| **0s 串行**（concurrency=1，一次一条）| **0/8** | **并发**（串行也全错）|
| **背靠背 device 命中重读**（同一份 load_back 刚写进 device 的内容，无静置）| **8/8** | **时序/静置**（无需等待即对）|

**扩展探针在 0s 分歧时刻**：`wt=0 lb=0 ackW=0 ackL=0`（所有异步队列空）、`host_hit=65536 swa_hit=128 mamba_hit=0`。→ 无任何 pending 异步状态。

**决定性对照（同一 server、背靠背、同内容）**：
- 第 1 次跑：8 条走 **load_back**（LBPROBE 8 条）→ **0/8 错**；
- 第 2 次跑（立即）：8 条走 **device 命中**（LBPROBE **0 条** = 未 load_back，读的正是上次 load_back 写进 device 的同一份内容）→ **8/8 对**。

**结论**：
1. load_back **把正确的 KV 内容写进了 device**（device 命中重读 8/8 证明内容无误）。
2. 但 **load_back 路径给触发回载的那个请求配置的 decode 读取（KV slot / SWA 位置映射）是错的** → 它自己那次 decode 读错位置 → 分歧。
3. 一旦同一内容作为普通 device 命中被匹配，就正确。

排除清单（全部上机/静态证伪）：deep_ep（旧版 int8 确定性）、backup 未完成（§10.6）、merge 错位（§10.4）、write-drain 重叠、overlap 调度、H2D 完成同步、并发、时序/静置。

**根因域**：HiCache **C1 load_back 对 reused 请求的读取映射**。首要嫌疑 = **SWA 位置/host_hit 回填**（`swa_hit=128`；KVSUM §10.3 "SWA/state 错、Full/C4 对"；输出多在前几 token 对、之后累积翻转，符合近窗 SWA 漂移）。次要嫌疑 = full KV `prefix_indices` 组装（`collect_full_device_indices`）。**这是 HiCache 的锅，非 deep_ep。**

**下一步（定位到具体行 + 修复）**：读 `swa_component.py` 的 load_back 位置回填（`finalize_match_result_in_tree_core` 的 swa_host_hit、host_indices 位置 walk ~1007-1036、swa_uuid/pos）与 `unified_radix_cache.py` `_load_back_transfers`/`init_load_back` 的 `prefix_indices`/`collect_full_device_indices` 组装；配套定点测试（对比 load_back 请求 vs device 命中请求的 `prefix_indices` 与 SWA 读位置）。

### 10.9 根因最终确认与修复（2026-08-20）—— C128 请求 sidecar 在 load_back 后漏刷新

§10.8 将根因域收敛到“内容正确、触发回载请求的读取映射错误”后，静态调用链与已有 KVSUM 日志共同定位到具体缺口：

```text
Req.init_next_round_input
  → UnifiedRadixCache.match_prefix
      → C128SidecarComponent.finalize_match_result_in_cache
          → 只收集当时仍在 device 的 C128 page ids
          → req.c128_prefix_page_ids = 前 31 页
  → UnifiedRadixCache.init_load_back
      → commit_load_back 将最后一个 host-only C128 页恢复到 device/tree
      → 【修复前没有重新刷新当前请求的 c128_prefix_page_ids】
  → req slot alloc
      → req_to_c128_sidecar 安装旧的 31 页列表
      → 第 32 项保持 0 sentinel
  → C128 attention 从 req_to_c128_sidecar 取 page table，最后一组读错物理页
```

**具体错误点**：`c128_sidecar_component.py` 的 `finalize_match_result_in_cache()` 在 load-back **之前**执行；`commit_hicache_transfer(LOAD_BACK)` 恢复 C128 页后，组件没有在 `finalize_load_back()` 中更新触发回载请求的 request-local page table。第二次普通 device-hit 会重新经过 match finalizer，此时 32 页均已在 device，因此自动恢复正确。

**已有日志的页数证据**（64K，`c128_page_size=16`，共 32 个 C128 physical pages）：

- populate：`sidecar=528`，即 `1+2+...+32`；
- 首次 load-back replay：`sidecar=496`，即 `1+2+...+31`；
- 恰好缺最后一个在 match 后才从 host 恢复的 C128 页。对应 replay 的 C128 内容 digest 同时变化，而 C4 保持正确。

**修复**：

1. 在 `C128SidecarComponent` 抽取 `_collect_device_pages(node_id)`，统一按 root 顺序收集路径上的 C128 device pages；
2. `finalize_match_result_in_cache()` 继续在初始 match 时安装已有 device pages；
3. 新增 `finalize_load_back()`：load-back 成功后，从 `req.best_match_node` 重新收集完整 C128 页并调用 `set_c128_prefix_pages()`，覆盖初始 match 阶段的不完整列表；
4. 增加组件测试，覆盖“前缀页已在 device、末页从 host load-back”的混合状态，断言触发回载请求最终拿到 `[device_page, restored_page]`。

**验证结果**（npu-B `cx-dsv4`，C1，fresh server，80×64K populate 后 id0–7 背靠背）：

| 轮次 | cached | 与串行 cold ground truth 对比 |
|---|---:|---:|
| 第一轮 `loadback` | 每条 65536 | **8/8 correct** |
| 第二轮立即 `devicehit` | 每条 65536 | **8/8 correct** |

修复前同一实验第一轮为 **0/8**、第二轮为 **8/8**；修复后首次回载请求与随后 device-hit 均正确。由此确认 C1 replay 分歧根因不是 SWA、Full `prefix_indices`、并发或同步，而是 **C128 request-local sidecar 在 load-back commit 后漏刷新最后一个回载页**。
