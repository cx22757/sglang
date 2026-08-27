# DSV4 main-8.5 A3 Compressor Ring 构建、启动与 16K Smoke Runbook

> 记录日期：2026-08-15
> 适用范围：npu-B 单机 16 die、`cx-dsv4` 容器、DSV4 无 HiCache 启动验证
> 状态：已按本文版本和配置实测通过

## 1. 目的与结论

本文记录以下闭环，供后续在相同环境中复现：

1. 在宿主机准备 `cann-ops-transformer` 的指定提交；
2. 将已提交源码打包并传到 npu-B；
3. 在 `cx-dsv4` 容器内构建包含五组算子的完整 `custom_transformer` 包；
4. 在隔离目录验证完整包，再备份并替换容器内原 vendor；
5. 重启容器，使用 `dev/main-8.5`、不开 HiCache 启动 DSV4；
6. 以稳定 `/health == 200` 判定服务 ready；
7. 执行确定性 16K cold-prefill，并用服务端日志确认真实 prefill 和 decode。

本轮最终结论：

- 默认 decode graph 路径可以完成初始化和图捕获；
- 服务稳定 ready；
- 16K cold-prefill 返回 HTTP 200，并完成 8 token decode；
- 服务端 DP0 日志累计执行 16384 个新 token，累计 cached token 为 0；
- 全程未出现 `aclnnCompressor`、SIGSEGV、OOM、`RuntimeError` 或 Traceback；
- 服务在测试后仍健康运行。

这是一项启动和基础正确性验证，不是性能验收，也不证明 HiCache、L2/L3、并发、长稳或跨节点场景已经通过。

## 2. 冻结版本与环境

### 2.1 SGLang NPU 适配基线

| 项目 | 值 |
| --- | --- |
| 分支 | `dev/main-8.5` |
| commit | `b915d68d81037666f4bd61d42c119ae3aeadd793` |
| 远端源码目录 | `/sgl-workspace/sglang` |
| HiCache | 关闭 |

远端源码是按项目同步协议部署的受版本控制文件快照，不包含 `.git`。因此远端容器内不能用
`git rev-parse` 证明版本；必须在证据目录记录部署 SHA，并由同步记录证明它与本地提交一致。

本轮记录文件：

```text
/data/dsv4_main85_startup/2026-08-15-npu-B-b915d68d-ring-1389e3ac-no-hicache/deployed-source.txt
```

### 2.2 算子源码

| 项目 | 值 |
| --- | --- |
| 本地仓库 | `/root/package/cann-ops-transformer` |
| 分支 | `feature/a3-compressor-ring-bcc6304` |
| commit | `1389e3ac1ae5df7df05f461d4fe243126dfb4d13` |
| merge base | `bcc6304656c9e712b50b1faa22872a158b1e34c5` |
| 源码 tar SHA256 | `c5cb9b0a37fe5c56c79f45300b605af39aefad479e0cf16af2f709a920a5877d` |
| 本轮 `.run` SHA256 | `c5aa72f227f4f8a5eadd5f67bd044b595383c9e664aee6251f0c60d2b4a3fe91` |

`.run` 包可能包含构建时间等元数据。上表的 `.run` SHA 用于识别本轮实际测试的产物；后续从
同一提交重新构建时，应优先核对 API 符号、核心共享库和 kernel 哈希，不应只因整个 `.run`
哈希不同就判定构建错误。

### 2.3 运行环境

| 项目 | 值 |
| --- | --- |
| 节点 | npu-B：`113.46.13.20` |
| 容器 | `cx-dsv4` |
| NPU | Atlas A3，16 die |
| CANN | `9.0.0` |
| CANN inner version | `V100R001C10SPC001B250` |
| Driver | `25.5.1` |
| Driver inner version | `V100R001C23SPC006B220` |
| `torch_npu` | `2.10.0` |
| 模型 | `/mnt/paas/weights/DeepSeek-V4-Flash-w8a8-mtp` |
| 服务端口 | `30000` |

A3 的 `--tp-size` 统计 die，不是物理卡；本文 `--tp-size 16` 需要 16 个空闲 die。

## 3. 为什么必须构建完整五算子包

DSV4 路径不仅使用 Compressor，还依赖以下 API：

```text
aclnnCompressor
aclnnCompressorGetWorkspaceSize
aclnnQuantLightningIndexer
aclnnQuantLightningIndexerGetWorkspaceSize
aclnnQuantLightningIndexerMetadata
aclnnQuantLightningIndexerMetadataGetWorkspaceSize
aclnnSparseAttnSharedkv
aclnnSparseAttnSharedkvGetWorkspaceSize
aclnnSparseAttnSharedkvMetadata
aclnnSparseAttnSharedkvMetadataGetWorkspaceSize
```

`/home/cx/package/compressor-v1.0` 是 Compressor-only vendor。它也提供名为
`libcust_opapi.so`、`libcust_opmaster_rt2.0.so` 和 `libcust_opsproto_rt2.0.so` 的通用库。
把这个 vendor 整体放在原 `custom_transformer` 前面，会遮蔽完整 vendor 中其他算子的同名库：
Compressor 错误虽然消失，但 `aclnnQuantLightningIndexerGetWorkspaceSize` 会发生 SIGSEGV。

只替换两个非 relocatable Compressor kernel `.o` 也不够，因为 Compressor 的 op-api、tiling、
proto 和 kernel 必须来自相互匹配的构建。因此最终方案是从目标 feature commit 一次性构建：

```text
sparse_attn_sharedkv
sparse_attn_sharedkv_metadata
quant_lightning_indexer
quant_lightning_indexer_metadata
compressor
```

这五组算子组成同一个 `custom_transformer` vendor，运行时只有一套 source of truth。

## 4. 在宿主机准备源码

获取 GitHub 源码必须在宿主机进行，构建必须在容器内进行。使用专用 clone，避免切换或污染
其他人的工作区。

```bash
OP_REPO=/root/package/cann-ops-transformer
OP_BRANCH=feature/a3-compressor-ring-bcc6304
OP_SHA=1389e3ac1ae5df7df05f461d4fe243126dfb4d13
SRC_NAME=cann-ops-transformer-ring-1389e3ac
SRC_TAR=/tmp/${SRC_NAME}.tar

git -C "$OP_REPO" fetch origin \
  "refs/heads/${OP_BRANCH}:refs/remotes/origin/${OP_BRANCH}"

# 首次使用该专用 clone 时：
git -C "$OP_REPO" switch --track "origin/${OP_BRANCH}"

# 已存在本地 tracking branch 时：
# git -C "$OP_REPO" switch "$OP_BRANCH"

test "$(git -C "$OP_REPO" rev-parse HEAD)" = "$OP_SHA"
git -C "$OP_REPO" status --short --branch
```

工作区应没有本任务之外的改动。使用 `git archive`，只打包指定 commit 的已提交文件：

```bash
test ! -e "$SRC_TAR"
git -C "$OP_REPO" archive \
  --format=tar \
  --prefix="${SRC_NAME}/" \
  --output="$SRC_TAR" \
  "$OP_SHA"

sha256sum "$SRC_TAR"
```

本轮输出：

```text
c5cb9b0a37fe5c56c79f45300b605af39aefad479e0cf16af2f709a920a5877d
```

## 5. 传输源码到 npu-B 容器

```bash
NPU_B=root@113.46.13.20
CONTAINER=cx-dsv4
SRC_NAME=cann-ops-transformer-ring-1389e3ac
SRC_TAR=/tmp/${SRC_NAME}.tar

scp -o BatchMode=yes "$SRC_TAR" "$NPU_B:$SRC_TAR"

ssh -o BatchMode=yes "$NPU_B" "
  set -e
  docker exec $CONTAINER test ! -e /tmp/$SRC_NAME
  docker cp $SRC_TAR $CONTAINER:$SRC_TAR
  docker exec $CONTAINER bash -lc '
    cd /tmp
    tar -xf $SRC_TAR
    test -f /tmp/$SRC_NAME/build.sh
    find /tmp/$SRC_NAME -type f | wc -l
  '
"
```

本轮解包后有 9278 个普通文件。文件数只是辅助检查，Git commit 和 archive SHA 才是版本依据。

## 6. 在容器内构建完整包

创建构建脚本：

```bash
cat >/tmp/build_ops_transformer_ring_1389e3ac.sh <<'SH'
#!/usr/bin/env bash
set -eo pipefail

exec env -i PATH="${PATH}" bash --login -c '
  set -eo pipefail
  cd /tmp/cann-ops-transformer-ring-1389e3ac
  python3 -c "import packaging; print(\"packaging=\" + packaging.__version__)"
  bash build.sh \
    --pkg \
    --experimental \
    --soc=ascend910_93 \
    --ops="sparse_attn_sharedkv,sparse_attn_sharedkv_metadata,quant_lightning_indexer,quant_lightning_indexer_metadata,compressor"
'
SH
```

上传脚本并在容器中执行。建议把完整输出保存在 `/data` 独立证据目录：

```bash
BUILD_E=/data/dsv4_main85_startup/2026-08-14-npu-B-b915d68d-no-hicache/ops-transformer-ring-1389e3ac-build

ssh -o BatchMode=yes "$NPU_B" "mkdir -p '$BUILD_E'"
scp -o BatchMode=yes /tmp/build_ops_transformer_ring_1389e3ac.sh \
  "$NPU_B:$BUILD_E/build_ops_transformer_ring_1389e3ac.sh"

ssh -o BatchMode=yes "$NPU_B" "
  set +e
  docker exec $CONTAINER bash '$BUILD_E/build_ops_transformer_ring_1389e3ac.sh' \
    >'$BUILD_E/build.log' 2>&1
  rc=\$?
  printf '%s\n' \"\$rc\" >'$BUILD_E/build.status'
  exit \"\$rc\"
"
```

成功产物：

```text
/tmp/cann-ops-transformer-ring-1389e3ac/build_out/
  cann-ops-transformer-custom_linux-aarch64.run
```

构建成功门禁：

```bash
ssh -o BatchMode=yes "$NPU_B" "
  test \"\$(cat '$BUILD_E/build.status')\" = 0
  docker exec $CONTAINER test -f \
    /tmp/cann-ops-transformer-ring-1389e3ac/build_out/cann-ops-transformer-custom_linux-aarch64.run
"
```

## 7. 安装前隔离验证

不要直接覆盖当前 vendor。先把产物复制到持久证据目录，再安装到隔离目录：

```bash
RUN=/tmp/cann-ops-transformer-ring-1389e3ac/build_out/cann-ops-transformer-custom_linux-aarch64.run
VERIFY=/tmp/cann-ops-transformer-ring-1389e3ac-verify

ssh -o BatchMode=yes "$NPU_B" "docker exec $CONTAINER bash -lc '
  set -e
  cp $RUN $BUILD_E/cann-ops-transformer-custom_linux-aarch64.run
  sha256sum $BUILD_E/cann-ops-transformer-custom_linux-aarch64.run \
    >$BUILD_E/package.sha256
  test ! -e $VERIFY
  mkdir -p $VERIFY
  bash $RUN --install-path=$VERIFY \
    >$BUILD_E/verify-install.log 2>&1
'"
```

安装后 vendor 位于：

```text
/tmp/cann-ops-transformer-ring-1389e3ac-verify/vendors/custom_transformer
```

### 7.1 API 符号门禁

```bash
LIB=$VERIFY/vendors/custom_transformer/op_api/lib/libcust_opapi.so

ssh -o BatchMode=yes "$NPU_B" \
  "docker exec $CONTAINER nm -D '$LIB'" \
  | grep ' T aclnn' \
  | sed 's/.* T //' \
  | sort
```

输出必须与第 3 节列出的 10 个符号完全一致，不能只是“至少包含 Compressor”。

### 7.2 包内容门禁

必须存在三个有 kernel 产物的算子目录：

```text
op_impl/ai_core/tbe/kernel/ascend910_93/compressor
op_impl/ai_core/tbe/kernel/ascend910_93/quant_lightning_indexer
op_impl/ai_core/tbe/kernel/ascend910_93/sparse_attn_sharedkv
```

两个 metadata 算子没有独立 kernel 目录；通过对应 API 符号和以下头文件确认它们已打包：

```text
op_api/include/aclnnop/aclnn_quant_lightning_indexer_metadata.h
op_api/include/aclnnop/aclnn_sparse_attn_sharedkv_metadata.h
op_impl/ai_core/tbe/custom_transformer_impl/ascendc/quant_lightning_indexer/quant_lightning_indexer_metadata.h
op_impl/ai_core/tbe/custom_transformer_impl/ascendc/sparse_attn_sharedkv/sparse_attn_sharedkv_metadata.h
```

### 7.3 Compressor kernel 门禁

本轮 feature 包的四个 Compressor kernel 均不同于原已知失败包：

| 文件 | 原包 SHA256 | feature 包 SHA256 |
| --- | --- | --- |
| `Compressor_bef03b9219f6b8f43c9980287f972446.o` | `8adb2e885ef4e54472e7160196bea7306a3eb1eb81e5d6db8185a6afb98fc4e8` | `8a1f903359485fed5b75f957a36a0cab6aade69d53c70ded539dc7e531db4885` |
| `Compressor_bef03b9219f6b8f43c9980287f972446_relocatable.o` | `9a6065689892fc67f935ae3e97d46cc5367d6314b397e238c7292e009381f9af` | `e64d363888979be5fee6bd522341a1a48bc165157e3fed81530b4024f5fdba03` |
| `Compressor_fee6dd135af0359268b769125398b3d2.o` | `cfff68a34518ccf85b680a777cd659e717de72ae21851b3f3e93698df4f960e3` | `7f0e5e2a55e2c2a1e6e9d8022ef1f00e0d2c4423472d9953814877741992e9b0` |
| `Compressor_fee6dd135af0359268b769125398b3d2_relocatable.o` | `de66f3e08f3af55b618f7aaa013f70cc183d1421b5e6ebe1a07af0844a899dd4` | `26f175fc1a9c4ea0d405da2bf2cdec6d491d9db44121b86992a3a3b6314cc9d6` |

若从同一 commit 和相同工具链重新构建，至少应证明生成物不是上表中的旧失败 kernel，并保留
新产物哈希。不能只比较文件名或文件大小。

## 8. 备份并安装完整 vendor

替换前确认没有正在提供服务的 SGLang 进程。若有失败启动残留，按项目规范确认归属后清理；
本项目通常由操作者重启 `cx-dsv4`，不要擅自杀死其他任务。

以下路径定义和替换命令均在 `cx-dsv4` 容器内以 root 执行。建议把它们保存成证据目录中的
安装脚本，再通过 `docker exec cx-dsv4 bash <script>` 运行：

```bash
BUILD_E=/data/dsv4_main85_startup/2026-08-14-npu-B-b915d68d-no-hicache/ops-transformer-ring-1389e3ac-build
VENDORS=/usr/local/Ascend/cann-9.0.0/opp/vendors
CURRENT=$VENDORS/custom_transformer
NEW=/tmp/cann-ops-transformer-ring-1389e3ac-verify/vendors/custom_transformer
STAGE=$VENDORS/.custom_transformer.ring-1389e3ac.new
OLD=/tmp/custom_transformer.pre-ring-1389e3ac
```

备份当前 vendor，并在同一文件系统内准备新目录。若 `STAGE` 或 `OLD` 已存在，不要覆盖；应先
确认它们来自哪一轮，再换用新的明确后缀：

```bash
test -d "$CURRENT"
test -d "$NEW"
test ! -e "$STAGE"
test ! -e "$OLD"

tar -C "$VENDORS" -czf \
  "$BUILD_E/custom_transformer.pre-ring-1389e3ac.tar.gz" \
  custom_transformer
sha256sum "$BUILD_E/custom_transformer.pre-ring-1389e3ac.tar.gz"

# 本轮备份 SHA256：
# 76c4df67c645addf3b32c60a5c67f8708965e53f398e57c50085235e2c147b27

cp -a "$NEW" "$STAGE"
```

切换前至少对比三类核心库：

```bash
cmp "$NEW/op_api/lib/libcust_opapi.so" \
    "$STAGE/op_api/lib/libcust_opapi.so"
cmp "$NEW/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so" \
    "$STAGE/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so"
cmp "$NEW/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so" \
    "$STAGE/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so"
```

用 rename 切换并保留自动回滚：

```bash
rollback() {
  if test ! -e "$CURRENT" && test -e "$OLD"; then
    mv "$OLD" "$CURRENT"
  fi
}
trap rollback EXIT

mv "$CURRENT" "$OLD"
mv "$STAGE" "$CURRENT"
trap - EXIT
```

不要把 standalone `aie_ascendc` vendor 加到最终启动环境，也不要同时 source 两套包含同名
通用库的 vendor。

本轮安装后的核心库 SHA256：

```text
7db68e482387212e12e5b2e1f05a316f08f766f3459d9fe74f599d1907f36dc7  libcust_opapi.so
d0e07b33b3a87b16dcee970b53fef1b0c459aabbfdd1976bc4f5f811737bb694  libcust_opmaster_rt2.0.so
7e5fee1e67b11873b3294c89da03c1eb867e8a3b38d60cfe701d8871ee51b2cc  libcust_opsproto_rt2.0.so
```

## 9. 重启容器并执行启动前检查

替换共享库后必须重启容器，确保没有旧进程继续持有旧 so 或 kernel。重启前确认容器属于本项目、
没有其他任务需要保留：

```bash
ssh -o BatchMode=yes "$NPU_B" 'docker restart cx-dsv4'
```

启动服务前依次检查：

```bash
ssh -o BatchMode=yes "$NPU_B" '
  docker inspect -f "status={{.State.Status}} started={{.State.StartedAt}}" cx-dsv4

  # 不应有旧 SGLang、spawn 或 forkserver。
  docker exec cx-dsv4 bash -lc \
    '\''ps -eo pid,ppid,stat,cmd | grep -E "[s]glang|[m]ultiprocessing.spawn|[f]orkserver" || true'\''

  # 30000 必须空闲。
  ss -lntp | grep -E "[:.]30000[[:space:]]" || true

  # 必须确认 16 个 die 均没有其他进程占用。
  npu-smi info
'
```

同时重新核对已安装的 `libcust_opapi.so` 和四个 Compressor kernel 哈希，确认容器重启没有恢复
旧包。

## 10. 启动 DSV4，无 HiCache

本轮成功启动脚本如下。它只 source `customize` 和完整的 `custom_transformer`，不 source
`/home/cx/package/compressor-v1.0/install/vendors/aie_ascendc`。

```bash
#!/usr/bin/env bash
set -eo pipefail

export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTHONPATH=/sgl-workspace/sglang/python:${PYTHONPATH:-}
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export FORCE_DRAFT_MODEL_NON_QUANT=1
export HCCL_BUFFSIZE=1500
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=lo
export INF_NAN_MODE_FORCE_DISABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_DSV4_FP4_EXPERTS=False
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_SET_CPU_AFFINITY=1
export STREAMS_PER_DEVICE=32

cd /sgl-workspace/sglang
exec python3 -m sglang.launch_server \
  --model-path /mnt/paas/weights/DeepSeek-V4-Flash-w8a8-mtp \
  --page-size 128 \
  --tp-size 16 \
  --trust-remote-code \
  --device npu \
  --attention-backend ascend \
  --watchdog-timeout 9000 \
  --host 0.0.0.0 \
  --port 30000 \
  --mem-fraction-static 0.7 \
  --prefill-max-requests 1 \
  --chunked-prefill-size 32768 \
  --max-running-requests 16 \
  --dp-size 16 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --quantization modelslim \
  --enable-dp-lm-head \
  --kv-cache-dtype auto \
  --random-seed 20260807 \
  --context-length 65536
```

最终成功配置没有使用以下临时规避参数：

```text
--cuda-graph-max-bs-decode 16
--cuda-graph-backend-decode=disabled
```

也就是说，本轮验证覆盖了默认 decode graph 初始化，而不是通过禁用图绕过 Compressor。

后台启动并保存独立日志：

```bash
RUN_E=/data/dsv4_main85_startup/2026-08-15-npu-B-b915d68d-ring-1389e3ac-no-hicache

ssh -o BatchMode=yes "$NPU_B" "
  mkdir -p '$RUN_E'
  nohup docker exec $CONTAINER bash '$RUN_E/launch_no_hicache_ring_1389e3ac.sh' \
    >'$RUN_E/server.log' 2>&1 </dev/null &
  printf '%s\n' \"\$!\" >'$RUN_E/launch-host.pid'
"
```

启动日志：

```text
/data/dsv4_main85_startup/2026-08-15-npu-B-b915d68d-ring-1389e3ac-no-hicache/server.log
```

## 11. Ready 判据

Uvicorn 启动文字不是最终 ready 判据。必须使用 `/health`，并连续检查：

```bash
for i in 1 2 3; do
  code=$(curl --noproxy '*' -sS -o /dev/null -w '%{http_code}' \
    --max-time 5 http://127.0.0.1:30000/health)
  printf 'health_check_%s=%s\n' "$i" "$code"
  test "$code" = 200
  sleep 5
done
```

必须使用 `--noproxy '*'`。npu-B 环境中存在代理变量；不禁用代理时，本地 health 请求曾被转发到
代理端口，产生与服务本身无关的 connection refused。

Ready 后还要确认：

```bash
ss -lntp | grep -E '[:.]30000[[:space:]]'
ps -p "$(cat "$RUN_E/launch-host.pid")" -o pid=,ppid=,stat=,etime=,cmd=
```

## 12. 确定性 16K cold-prefill smoke

### 12.1 Cold 条件

本轮服务是重启后的新实例，测试前没有业务请求。测试使用唯一的 request ID 和输入模式；返回值
必须满足：

```text
cached_tokens == 0
cached_tokens_details == null
```

本文没有调用 `/flush_cache`。如果在同一服务实例上重复完全相同的请求，它不再是 cold test；应
使用新实例或新的唯一输入模式，并仍以服务响应和日志中的 cached token 为 0 作为最终判据。

### 12.2 请求生成方式

使用模型自身 tokenizer，把固定文本模式重复并截断为恰好 16384 个 `input_ids`。先在 npu-B
宿主机记录服务日志字节偏移：

```bash
RUN_E=/data/dsv4_main85_startup/2026-08-15-npu-B-b915d68d-ring-1389e3ac-no-hicache
wc -c <"$RUN_E/server.log" >"$RUN_E/cold16k-server-log-offset.txt"
```

然后把以下脚本先保存为宿主机上的 `/tmp/cold16k-smoke.py`：

```python
import hashlib
import json
import time
import urllib.request

from transformers import AutoTokenizer

base_url = "http://127.0.0.1:30000"
model_path = "/mnt/paas/weights/DeepSeek-V4-Flash-w8a8-mtp"
pattern_text = " dsv4-main85-ring-1389e3ac-cold-prefill"
input_token_count = 16_384
max_new_tokens = 8

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
pattern = tokenizer.encode(pattern_text, add_special_tokens=False)
assert pattern
input_ids = (pattern * ((input_token_count + len(pattern) - 1) // len(pattern)))[
    :input_token_count
]
input_sha256 = hashlib.sha256(
    json.dumps(input_ids, separators=(",", ":")).encode()
).hexdigest()

payload = {
    "rid": "main85-ring-1389e3ac-cold16k",
    "input_ids": input_ids,
    "sampling_params": {
        "temperature": 0,
        "max_new_tokens": max_new_tokens,
    },
    "return_logprob": True,
    "routed_dp_rank": 0,
}
request = urllib.request.Request(
    base_url + "/generate",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
started = time.monotonic()
with opener.open(request, timeout=1800) as response:
    assert response.status == 200
    result = json.loads(response.read())
elapsed = time.monotonic() - started

meta = result["meta_info"]
assert meta["prompt_tokens"] == input_token_count
assert meta["cached_tokens"] == 0
assert meta["cached_tokens_details"] is None
assert meta["completion_tokens"] == max_new_tokens
assert len(result["output_ids"]) == max_new_tokens

print(
    json.dumps(
        {
            "elapsed_seconds": elapsed,
            "input_ids_json_sha256": input_sha256,
            "prompt_tokens": meta["prompt_tokens"],
            "cached_tokens": meta["cached_tokens"],
            "completion_tokens": meta["completion_tokens"],
            "output_ids": result["output_ids"],
            "finish_reason": meta["finish_reason"],
        },
        indent=2,
        sort_keys=True,
    )
)
```

运行并保存客户端输出：

```bash
scp -o BatchMode=yes /tmp/cold16k-smoke.py \
  "$NPU_B:$RUN_E/cold16k-smoke.py"

ssh -o BatchMode=yes "$NPU_B" \
  "docker exec $CONTAINER python3 '$RUN_E/cold16k-smoke.py' \
    | tee '$RUN_E/cold16k-client.log'"
```

上面的代码是请求和断言核心逻辑。本轮实际执行的完整脚本还会把请求摘要、原始响应和结构化
判定分别写入 `cold16k-request-summary.json`、`cold16k-response.json` 和
`cold16k-result.json`；可直接查看第 14.2 节证据目录中的 `cold16k-smoke.py`。

### 12.3 客户端实测结果

```text
HTTP status:             200
input tokens:            16384
input_ids JSON SHA256:   0e0bb0ce2fbf310e5683d79f13096a49b7969aa5ad5074b226943908b2946868
cached tokens:           0
cached token details:    null
completion tokens:       8
elapsed:                 7.1995 s
post-request health:     200
finish reason:           {"type": "length", "length": 8}
output ids:              [1129, 3095, 5044, 475, 283, 12731, 22, 64552]
```

耗时只用于记录本轮现象，不是性能基线。

## 13. 服务端真实执行门禁

不能只看客户端 HTTP 200。对日志偏移之后的 DP0 prefill 行进行汇总：

```python
import pathlib
import re

run_e = pathlib.Path(
    "/data/dsv4_main85_startup/"
    "2026-08-15-npu-B-b915d68d-ring-1389e3ac-no-hicache"
)
offset = int((run_e / "cold16k-server-log-offset.txt").read_text())
text = (run_e / "server.log").read_bytes()[offset:].decode(errors="replace")

new_tokens = []
cached_tokens = []
for line in text.splitlines():
    if "DP0 TP0 EP0" not in line or "Prefill batch" not in line:
        continue
    new_match = re.search(r"#new-token: (\d+)", line)
    cached_match = re.search(r"#cached-token: (\d+)", line)
    if new_match and cached_match:
        new_tokens.append(int(new_match.group(1)))
        cached_tokens.append(int(cached_match.group(1)))

assert sum(new_tokens) == 16_384
assert sum(cached_tokens) == 0
```

本轮 DP0 实际执行 8 个 2048-token prefill chunk：

```text
8 × #new-token: 2048 = 16384
每批 #cached-token: 0
#pending-token: 14336 -> ... -> 0
```

其他 DP rank 在相邻时间可能打印自己的 128-token 日志；本请求明确使用
`routed_dp_rank=0`，因此门禁只统计 `DP0 TP0 EP0`。

同时扫描完整本轮日志，以下模式计数必须为 0：

```text
Traceback (most recent call last)
RuntimeError:
call aclnnCompressor failed
SIGSEGV
Segmentation fault
out of memory
OOM
```

最后再次连续检查三次 `/health == 200`，确认服务进程仍在、端口仍监听，并保存
`npu-smi info` 和容器进程列表。

## 14. 本轮证据索引

### 14.1 构建和安装证据

```text
/data/dsv4_main85_startup/2026-08-14-npu-B-b915d68d-no-hicache/
  ops-transformer-ring-1389e3ac-build/
    build.log
    build.status
    build_ops_transformer_ring_1389e3ac.sh
    cann-ops-transformer-custom_linux-aarch64.run
    package.sha256
    verify-install.log
    verify_ring_package.sh
    opapi-nm.txt
    opapi-symbols.txt
    opapi-symbol-check.txt
    kernel-operator-directories.txt
    compressor-kernel-comparison.txt
    installed-source.txt
    installed-core-libraries.sha256
    installed-compressor-kernels.sha256
    custom_transformer.pre-ring-1389e3ac.tar.gz
    custom_transformer.pre-ring-1389e3ac.tar.gz.sha256
```

### 14.2 启动和 smoke test 证据

```text
/data/dsv4_main85_startup/2026-08-15-npu-B-b915d68d-ring-1389e3ac-no-hicache/
  deployed-source.txt
  operator-source.txt
  launch_no_hicache_ring_1389e3ac.sh
  launch-host.pid
  server.log
  pre-start-ports.txt
  pre-start-processes.txt
  pre-start-npu.txt
  cold16k-smoke.py
  cold16k-request-summary.json
  cold16k-response.json
  cold16k-result.json
  cold16k-client.log
  cold16k-server-log-offset.txt
  cold16k-server-log-extract.txt
  cold16k-server-gate.json
  post-smoke-health.txt
  post-smoke-service-process.txt
  post-smoke-container-processes.txt
  post-smoke-npu.txt
```

## 15. 失败现象与排查顺序

### 15.1 默认启动在 `aclnnCompressor` 失败

原完整 vendor 在 decode graph capture 阶段报错：

```text
Exception: Capture cuda graph failed: call aclnnCompressor failed
```

将 `--cuda-graph-max-bs-decode` 降为 16 没有修复。禁用 decode graph 后，错误只是从图捕获
转移到 eager warmup：

```text
RuntimeError: call aclnnCompressor failed
```

因此根因不是单纯图参数。

### 15.2 standalone Compressor vendor 导致其他算子崩溃

按 Compressor-only README 安装本身可以成功，也能注册 `torch.ops.custom.compressor`，但把其
通用 vendor 库放到完整 `custom_transformer` 前面后，运行时在
`aclnnQuantLightningIndexerGetWorkspaceSize` 发生 SIGSEGV。不要把“安装成功”误判为可以和完整
vendor 同进程叠加。

### 15.3 只覆盖 Compressor kernel 仍失败

只替换两个非 relocatable `.o` 后，`aclnnCompressor` 仍失败。必须使用同一次完整构建产生的
op-api、tiling、proto 和全部 kernel。

### 15.4 仅使用 bcc6304 基线仍是旧 kernel

从 `bcc6304656c9e712b50b1faa22872a158b1e34c5` 构建出的 Compressor kernel 与容器原已知失败
kernel 完全相同，因此没有安装。正确目标是包含 A3 ring 修复的
`feature/a3-compressor-ring-bcc6304`，本轮 SHA 为 `1389e3ac...`。

### 15.5 Health 请求被代理干扰

若 `curl http://127.0.0.1:30000/health` 意外连接其他本地端口，先检查代理环境变量，并使用：

```bash
curl --noproxy '*' http://127.0.0.1:30000/health
```

这类 connection refused 不能直接判定 SGLang 未 ready。

## 16. 回滚

本轮替换前的 vendor 已保留两份：

```text
/tmp/custom_transformer.pre-ring-1389e3ac
/data/dsv4_main85_startup/2026-08-14-npu-B-b915d68d-no-hicache/
  ops-transformer-ring-1389e3ac-build/custom_transformer.pre-ring-1389e3ac.tar.gz
```

回滚前必须停止本项目 SGLang 服务并确认进程归属。不要在进程仍加载算子库时切换目录。回滚后
重启容器，再重新核对 API 和 kernel 哈希。示意：

```bash
VENDORS=/usr/local/Ascend/cann-9.0.0/opp/vendors
CURRENT=$VENDORS/custom_transformer
FAILED=/tmp/custom_transformer.failed-ring-1389e3ac
OLD=/tmp/custom_transformer.pre-ring-1389e3ac

test -d "$CURRENT"
test -d "$OLD"
test ! -e "$FAILED"
mv "$CURRENT" "$FAILED"
mv "$OLD" "$CURRENT"
```

## 17. 复现完成清单

只有以下条件全部满足，才可写“本场景通过”：

- [ ] `dev/main-8.5` 的实际部署 SHA 已记录；
- [ ] 算子 feature branch 和 commit 已记录；
- [ ] 源码 archive 来自指定 commit 的已提交文件；
- [ ] 五组算子在同一次构建中打包；
- [ ] 隔离安装目录含精确的 10 个 API 符号；
- [ ] Compressor kernel 不是旧失败版本；
- [ ] 原 vendor 已备份且可回滚；
- [ ] 替换后已重启容器；
- [ ] 启动前 16 个 die、端口和进程均已检查；
- [ ] 连续三次 `/health == 200`；
- [ ] 16K 请求的 `cached_tokens == 0`；
- [ ] 服务端 DP0 的新 token 累计为 16384；
- [ ] 请求完成 8 token decode；
- [ ] 日志无 Compressor、SIGSEGV、OOM、Traceback 或 RuntimeError；
- [ ] 请求后服务仍健康，进程、端口和 NPU 状态已保存；
- [ ] 未把本轮 smoke 数字描述成性能结论。
