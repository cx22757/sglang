# Remote test environment

- Servers: `113.46.15.88` and `113.46.21.151` (identical environment); the
  test scripts are in `/home/cx/hicache`.
- Run all SGLang service and benchmark commands inside the `cx-dsv4` container,
  for example: `docker exec cx-dsv4 bash -lc 'ls /home/cx'`.

## GLM-5.3-Flash Ascend 适配执行规则

以下规则来自用户在本次会话中的明确要求；与 skill 的默认流程冲突时，以这些规则为准。

### 代码与 Git 工作流

- 所有 SGLang 代码修改都在本地仓库
  `/root/code/ai_infra/sglang-worktrees/sglang` 的 `glm53-adapter` 分支进行。
- 修改完成后，在本地提交并推送到 `cx22757` 远程的 `glm53-adapter` 分支；
  151 的 `/home/cx/sglang` 拉取该分支后再测试，并核对本地、远端提交一致。
- 不在 151 或容器中直接修改 SGLang 源码，不通过临时覆盖源码绕过上述流程。
- 保留已有的无关修改、未跟踪文件和其他分支工作；仅提交本次适配相关文件。
- 用户已授权本次适配的代码修改、提交、推送、远端拉取和容器测试，无需重复确认。

### 测试环境与产物位置

- 测试服务器：`root@113.46.21.151`；运行容器：`cx-dsv4`。
- 模型权重：`/mnt/paas/weights/GLM-5.3-Flash-w4a8`。
- 远端测试目录：`/home/cx/glm53-npu-adapter`。启动服务的脚本、测试文件、
  PID 文件和运行日志可以直接放在该目录，服务和推理测试在容器内执行。
- 分析报告、适配报告、结构化分析结论和最终交付报告只保存在本地，
  当前目录为 `.review/glm53-adapter/`；不要把报告放到远端。
- skill 在本地阅读，不需要将整个 skill 复制到 151；远端只放必要的执行文件。
- 不升级 Transformers，不擅自修改模型权重或模型配置来绕过适配缺口。

### NPU 使用与验证

- 在 151 宿主机执行 `/home/cx/show_npu_usage.sh` 查看设备占用，任选 4 张空闲卡测试。
  先前选择的是物理卡 `4,5,6,7`，启动前重新确认空闲即可。
- 用户已确认设备兼容性，不再核对卡型号。
- 不停止或清理其他用户的服务；只管理本次测试目录中记录的自有进程。
- 参考本地 `sglang-npu-adapter` skill，尽可能复用已有模型、ModelSlim 和 Ascend 后端。
  优先完成正确加载与纯文本推理，再验证其他功能。
- 明确区分静态分析、dummy 测试、真实权重推理和精度验证；服务启动成功或短文本
  冒烟通过，不能单独作为 KPool 长上下文、ACLGraph、MTP 或多模态已支持的证据。

<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root),
reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tools** (when available): `codegraph_explore` answers most code questions in one
  call — the relevant symbols' verbatim source plus the call paths between them.
  `codegraph_node` returns one symbol's source + callers, or reads a whole file with line
  numbers. If the tools are listed but deferred, load them by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` and
  `codegraph node <symbol-or-file>` print the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.
<!-- CODEGRAPH_END -->
