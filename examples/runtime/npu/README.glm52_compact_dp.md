# GLM-5.2 DSpark compact + Attention DP on Ascend

本分支在现有 GLM-5.2 NPU compact 实现上增加 Attention DP 协调。实现覆盖
eager / NPU full graph、开启或关闭 overlap schedule 的路径，保留 QuaRot draft
加载。CPU 合约测试和四进程 Gloo 测试不能替代 NPU 上的 DSA、HCCL/DeepEP、图捕获
及模型输出验证；本机没有 NPU，尚未完成这些硬件验证。

## 启动

双机各 16 卡使用独立的 [双机脚本和说明](README.glm52_compact_dp_2nodes.md)。

在 A3 单机 16 卡上，从仓库根目录运行：

```bash
SPS_TABLE=/home/sps_npu_tp16_adp4.json \
  bash examples/runtime/npu/launch_glm52_dspark_compact_dp.sh
```

默认 TP16、Attention DP4、Attention TP4，开启 compact、图和 overlap schedule。
`--enable-dp-lm-head` 已包含在脚本中。这里使用 `--attn-dp-size`，不设置独立模型
副本的 `--dp-size`。可修改模型、CANN 和 ATB 路径：

```bash
TARGET_MODEL=/home/weights/GLM-5.2-w8a8 \
DRAFT_MODEL=/home/weights/GLM-5.2-DSpark-NPU-0805 \
TP_SIZE=16 ATTN_DP_SIZE=4 MAX_RUNNING_REQUESTS=32 \
SPS_TABLE=/home/sps_npu_tp16_adp4.json \
  bash examples/runtime/npu/launch_glm52_dspark_compact_dp.sh
```

`MAX_RUNNING_REQUESTS` 为服务总上限；默认值对应每个 Attention DP 组约 8 个请求。
图档位按每组请求数配置，运行时会过滤不满足 Attention TP 对齐要求的档位。
单组请求数超过捕获范围时，各组协调回退到原有 uniform eager verify 路径。
权重和 KV 是否可容纳仍取决于实际设备内存；脚本保留原有 69632 token 初始预算。

其他运行方式：

```bash
# 只打印参数；不加载 CANN、不访问权重、不启动服务。
PRINT_COMMAND=1 bash examples/runtime/npu/launch_glm52_dspark_compact_dp.sh

# eager，关闭 overlap，用于与图执行比较。
GRAPH=0 OVERLAP=0 SPS_TABLE=/home/sps_npu_tp16_adp4.json \
  bash examples/runtime/npu/launch_glm52_dspark_compact_dp.sh

# 保留同样的 Attention DP 拓扑，启动 target-only 基线。
MODE=target-only bash examples/runtime/npu/launch_glm52_dspark_compact_dp.sh

# static DSpark 对照。
RAGGED_VERIFY_MODE=static bash examples/runtime/npu/launch_glm52_dspark_compact_dp.sh
```

脚本默认 `ASCEND_LAUNCH_BLOCKING=0`、`SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`、
`SGLANG_NPU_USE_MULTI_STREAM=0`；可通过同名环境变量覆盖。
默认不修改宿主机 sysctl，不注册临时 kernel overlay。原先支持 GLM 192 维 draft
算子的 kernel 依赖保持不变，本次没有引入新的 sgl-kernel-npu API。

## 采集 SPS 成本表

需要在相同硬件、并行拓扑和图配置下重新标定。SPS profiler 的 `--batch-size`
表示**每个 DP 组**的请求数，工具会按 DP 组数放大总请求数。

先用另一个终端启动标定服务：

```bash
SPS_PROFILE=1 TP_SIZE=16 ATTN_DP_SIZE=4 \
  bash examples/runtime/npu/launch_glm52_dspark_compact_dp.sh
```

服务就绪后，在相同的 SGLang Python 环境执行：

```bash
python python/sglang/benchmark/dspark_sps_profiler.py all \
  --base-url http://127.0.0.1:8030 \
  --batch-size 1 2 4 8 \
  --fracs 0.125 0.25 0.5 0.75 1.0 \
  --input-len 256 --temperature 0 --repeats 3 --no-plot \
  --out /home/sps_npu_tp16_adp4.json
```

采集后停止标定服务，使用普通启动命令重新启动。脚本会清除模拟接受相关环境变量，
普通推理不会模拟接受。没有成本表时可设置 `SPS_TABLE=''` 做连通性验证，但预算可能
退化为 verify-all，不能用这种状态判断动态窗口是否带来性能收益。
成本表衡量的是标定负载，实际负载的 DP 不均衡会改变通信与 padding 成本。

## 验证顺序

先在相同 DP 拓扑下运行 target-only、static DSpark、compact eager、compact graph，
每次停止前一个服务再启动下一种配置，普通推理均关闭模拟接受。
对每种配置执行相同请求并保存结果，例如：

```bash
python test/manual/npu/validate_glm52_compact.py capture \
  --concurrency 4 --requests 16 --max-tokens 256 \
  --out /home/compact_dp_graph.json

# 用对应文件名替换 baseline/candidate。
python test/manual/npu/validate_glm52_compact.py compare \
  /home/compact_dp_eager.json /home/compact_dp_graph.json
```

重点单独测试空闲组：将请求固定到一个组，其他组保持空闲。分别对 rank 0/1/2/3
重复运行，随后恢复不指定 rank 的负载；再测试并发 1、3、4、7，并在部分长请求运行时
持续加入短请求，覆盖 busy/idle、prefill/decode 切换。请求相同配置比较时需保持
`--routed-dp-rank` 相同。

```bash
python test/manual/npu/validate_glm52_compact.py capture \
  --routed-dp-rank 0 --concurrency 1 --requests 4 --max-tokens 128 \
  --out /home/compact_dp_rank0_only.json
```

初次验证可使用 `DSPARK_DEBUG=1` 启动。服务日志默认保存在仓库根目录
`server.a3.dspark.compact.adp4.log`，也可通过 `LOG_FILE` 改写。
启动时应出现 `GLM DSpark compact attention DP enabled`，并显示 `attn_dp=4, attn_tp=4`。

- `[DSPARK-CPS]`：本组预算及实际每请求 verify_len。
- `[DSPARK-DP]`：本组请求数、局部 token 档位和各组共同的 target token 宽度。
  同一轮各组的 `target_tokens` 应一致；空闲组应有 `local_reqs=0`。
  `target_tokens=None` 表示共同回退，不表示死锁。
- 图捕获日志只说明已捕获。要确认实际回放，需要结合框架图命中观测或 NPU profiler。
  `[DSPARK-DP]` 本身不能证明发生了图回放。

性能测试关闭 `DSPARK_DEBUG`，避免逐请求日志和张量转 CPU 干扰结果。
除输出一致性外，还应检查长时间混合请求、取消请求和重复请求后的稳定性。
不同并行拓扑或浮点归约顺序可能产生 greedy 分歧，需要结合分歧位置和任务精度排查。

## 图捕获阶段的 HCCL 资源分配失败

`aclnnMoeLowLatencyDispatchV2` 报错代表 auto 模式已进入低延迟通信路径。
如果同时出现 `Failed to allocate ... NPU memory` 和
`stream is not registered with any allocator`，应分别检查剩余内存与通信算子使用的
stream/allocator；捕获前的空闲显存不足以证明错误仅由 OOM 引起。

先保留图、overlap 和原有通信 buffer，仅缩小 KV 预算进行对照：

```bash
MEM_FRACTION_STATIC=0.92 CONTEXT_LENGTH=4096 MAX_TOTAL_TOKENS=8192 \
MAX_RUNNING_REQUESTS=8 CHUNKED_PREFILL_SIZE=2048 MAX_PREFILL_TOKENS=4096 \
GRAPH_BATCH_SIZES="4" ATTN_DP_SIZE=4 GRAPH=1 OVERLAP=1 \
SPS_TABLE=/home/sps_npu_tp16_adp4.json \
  bash examples/runtime/npu/launch_glm52_dspark_compact_dp.sh
```

这一配置是定位用的起点，不是已完成硬件验证的容量配置。
`mem_fraction_static` 必须高于权重预算检查给出的最低值；提高它能放大静态预算，
但不保证给后续通信、图捕获留下足够空间。显式降低 `MAX_TOTAL_TOKENS` 可限制 KV 分配。

脚本允许覆盖 `DEEPEP_MODE`、`HCCL_BUFFSIZE`、`DEEPEP_NORMAL_LONG_SEQ_ROUND`、
`DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS`；`DEEPEP_HCCL_BUFFSIZE` 保持透传，当前框架
对 default/MoE process group 的该配置优先于 `HCCL_BUFFSIZE`。
调小通信 buffer 前需按实际 token 数、hidden size、专家数、通信策略检查容量下界，
不能把小 batch 的设置直接用于长 prefill。

若要比较 normal 路径，可以使用 `DEEPEP_MODE=normal GRAPH=0`。
当前基线会在 normal 模式自动禁用图，因此这种对照不能证明 compact + DP + 图已修复。
如果缩小 KV 后仍报 allocator 未注册，需要对齐服务器 torch、torch_npu、CANN 和
sgl-kernel-npu 的具体版本，并获取首个异常的完整调用栈后再修改流管理。

如果静态预算下界约为 0.911，0.915 仍报同样错误，不再通过反复调整静态比例定位。
可以保留图，以较小的通信 buffer 和 dispatch 容量做一次受限负载验证：

```bash
MEM_FRACTION_STATIC=0.915 CONTEXT_LENGTH=2048 MAX_TOTAL_TOKENS=4096 \
MAX_RUNNING_REQUESTS=8 CHUNKED_PREFILL_SIZE=1024 MAX_PREFILL_TOKENS=2048 \
GRAPH_BATCH_SIZES="4" ATTN_DP_SIZE=4 GRAPH=1 OVERLAP=1 DEEPEP_MODE=auto \
HCCL_BUFFSIZE=512 DEEPEP_HCCL_BUFFSIZE=512 \
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64 \
DEEPEP_NORMAL_LONG_SEQ_ROUND=256 DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=64 \
SGLANG_NPU_DEBUG_GRAPH_CAPTURE=1 \
SPS_TABLE=/home/sps_npu_tp16_adp4.json \
  bash examples/runtime/npu/launch_glm52_dspark_compact_dp.sh
```

这是需要实机验证的定位配置，不保证解决 allocator 注册错误。两个 HCCL 配置同时
设置，避免 MoE process group 的独立覆盖让调整不生效。LL 容量从默认 128 降到 64，
仍覆盖这里的 4 请求 × 9 token = 36 token 图；不能同时换成 8 请求/72 token 图。
正常通信的每轮 token 数也相应减小，`256 × 64 = 16384` 是本配置的 normal 输入
容量上限；不要将这些通信设置直接复用于原来的长 prefill/高并发配置。

`[NPU-GRAPH] phase=warmup-1/warmup-2/capture` 给出每个阶段开始前的空闲、已分配、
已保留显存及当前流/捕获流。`failed phase=...` 标记异常被观察到的阶段，保留原始
异常栈；异步报错也可能来自前序算子。此诊断不增加设备同步或清空 allocator cache，
不调用私有 allocator 注册接口，也不关闭图。关闭该调试变量后不采集这些内存信息。

## 实现与范围

1. `dspark_planner.py`：预算从全 TP 广播改为本请求所在的 Attention TP 组广播。
   GLM 的动态局部档位保留；跨组协调放在 target 验证边界，不在 draft 阶段耦合各组。
2. `hardware_backend/npu/dspark_dp.py`：忙碌组完成本地 dense draft，空闲组跳过 draft，
   所有组恰好参加一次 CPU group all-gather，交换 `(请求数, 局部 token 档位)`。
   取最大值而不求和，避免 Attention TP peers 的重复贡献被累计。
   任一忙碌组没有合法 compact layout 时，全体返回原有 uniform eager 验证。
3. `dspark_worker_v2.py`：将该协调接入 busy 和 idle 两条路径，仅开放 GLM NPU
   的 dense-draft compact DP 图执行，其他模型及 CUDA 的原有限制保持不变。
4. `dspark_verify.py`：发布共同的 packed token 计数；空闲组采用零长度占位行，
   不为占位请求提交 accept 或 draft KV。
5. `forward_batch_info.py`：显式标记这类 NPU batch，MLP padding 保留局部请求行数，
   不再按 `token数 / 9` 反推请求数；输出裁剪保留完整 packed token 区间。
6. `ascend_backend.py`：DP eager 使用与图回放相同的持久元数据和独立 sink 槽，
   padding 不延长任何真实请求的 Q/KV。

本次没有把 CPU 档位协调放进 NPU 图：它交换的是已经存在于 host 的尺寸，必须在
选图前完成，不会为得到这些尺寸额外读取 NPU tensor。会增加每轮一次小型 CPU
collective 的开销；后续性能优化应基于 NPU 实测分析。

支持范围限定为 colocated serving、独立副本 DP=1、CP/DCP/PP=1，要求开启
DP LM head 和 DP MLP 同步。暂不启用 DP speculative prefill coordination。
现有 compact graph 对 two-batch-overlap、LoRA 和禁用图 padding 的限制仍然生效；
这里的 two-batch-overlap 与普通 overlap schedule 是不同功能。
