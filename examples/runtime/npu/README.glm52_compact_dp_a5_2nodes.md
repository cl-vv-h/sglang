# A5 双机 GLM-5.2 DSpark compact + DP Attention

脚本 `launch_glm52_dspark_compact_dp_a5_2nodes.sh` 为独立脚本，两台各放置同一份。
默认每台 8 张可见 NPU，全局 TP16、Attention DP4、Attention TP4，开启图和 overlap。
这沿用上一版双机的初始容量，并未按未经验证的 A5 显存规格扩大预算。

| 配置 | 默认值 |
| --- | --- |
| node 0 | `141.61.45.118` |
| node 1 | `141.61.45.124` |
| HCCL / Gloo 网卡 | `data0.3001` |
| 初始化地址 | `141.61.45.118:50000` |
| HTTP 地址 | `http://141.61.45.118:8030` |
| 总请求数 / 每组图档位 | `32` / `8` |
| context / KV token 上限 | `8192` / `32768` |
| mem_fraction_static | `0.85` |
| SPS 表 | `/home/sps_npu_a5_tp16_adp4.json` |

两台使用相同版本的分支、权重、SPS 表和并行配置，并确保 `data0.3001` 能用于节点间
通信。权重路径、CANN/ATB 路径默认与原双机脚本一致，可用环境变量覆盖。
若脚本未放在仓库 `examples/runtime/npu/` 下，额外设置 `SGLANG_ROOT=/workspace/sglang`。

在两台各自的仓库根目录执行：

```bash
# node 0: 141.61.45.118
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128 \
bash examples/runtime/npu/launch_glm52_dspark_compact_dp_a5_2nodes.sh 0

# node 1: 141.61.45.124
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128 \
bash examples/runtime/npu/launch_glm52_dspark_compact_dp_a5_2nodes.sh 1
```

尚无匹配的 A5 双机 SPS 表时，两台启动命令都加 `SPS_TABLE=''` 可先验证启动和短请求。
无表时预算可能退化为 verify-all，不能据此验证动态窗口性能。
正式采集时两台都设置 `SPS_PROFILE=1`，只在 node 0 运行 profiler：

```bash
python python/sglang/benchmark/dspark_sps_profiler.py all \
  --base-url http://127.0.0.1:8030 \
  --batch-size 1 2 4 8 --fracs 0.125 0.25 0.5 0.75 1.0 \
  --input-len 256 --temperature 0 --repeats 3 --no-plot \
  --out /home/sps_npu_a5_tp16_adp4.json
```

将表复制到 node 1 同一路径，停止两台标定服务，再按普通命令启动。
不要直接复用 A3 或其他 TP/DP/图配置采集的成本表。

默认设置 `ASCEND_LAUNCH_BLOCKING=0`、`SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`、
`SGLANG_NPU_USE_MULTI_STREAM=0`，沿用之前 A5 排查时可用的多流组合。
图档位 8 对应 72 个验证 token，DeepEP dispatch 容量使用 128，不能沿用 64。
系统 sysctl 调优默认关闭。

日志位于各自仓库根目录：

- `server.a5.node0.dspark.compact.tp16.adp4.log`
- `server.a5.node1.dspark.compact.tp16.adp4.log`

可用 `PRINT_COMMAND=1 bash ... 0` 或 `... 1` 只查看命令，不加载设备或启动服务。
内存预算可通过 `MEM_FRACTION_STATIC`、`MAX_TOTAL_TOKENS`、`CONTEXT_LENGTH` 和
`MAX_RUNNING_REQUESTS` 覆盖；两台保持一致。本脚本未在 A5 双机上实测。

## 提前退出时的日志

脚本保留 bash 作为 Python 和 tee 的父进程，避免 Python 的退出清理杀掉 tee 后
吞掉 traceback。Python 使用无缓冲输出，脚本保留原始退出码并记录 `Server exited: status=...`。
这修复的是日志收集，不能据此认定底层服务的启动异常已经解决。

若实际节点是 `141.61.45.113/118`，在两台都设置以下环境，再分别传入 `0`、`1`：

```bash
export NODE0_IP=141.61.45.113
export NODE1_IP=141.61.45.118
export MASTER_ADDR=141.61.45.113
```

默认仍为 `118/124`。两台必须使用一致的 `NODE0_IP`、`NODE1_IP` 和 `MASTER_ADDR`；
node 0 在 `NODE0_IP` 对应主机运行，node 1 在 `NODE1_IP` 对应主机运行。
请查看两台日志中的**第一个 traceback**。只有 `kill_process_tree` 清理日志不能定位根因。
已有成本表需要匹配 A5、TP16、Attention DP4 和实际图配置。

## MXFP8 专家权重与 DeepEP 的量化格式

A5 脚本默认 `DEEP_NORMAL_MODE_USE_INT8_QUANT=0`。本分支的 MXFP8 MoE 使用
BF16 dispatch，再在 GMM1 前通过 `npu_dynamic_mx_quant` 生成 FP8 激活和 E8M0
块 scale。INT8 dispatch 生成的逐 token scale `[M]` 不满足融合 GMM1 的
三维块 scale `[M, ceil(K/64), 2]` 契约，不能靠 unsqueeze 或 reshape 转换。

运行时会在 NPU 的 BF16 dispatch 调用期间屏蔽旧 INT8 环境变量，并在返回或异常时
恢复原值，覆盖 normal prefill 和 low-latency verify/draft。需要 INT8 dispatch
的层保留原行为。已经生成的二维 MX 块 scale 会作为 view 转成配对三维布局；
不符合量化格式的输入会在进入 CANN 前报出明确错误。

使用自定义旧脚本时，请删除其中固定的 `export DEEP_NORMAL_MODE_USE_INT8_QUANT=1`，
并在两台启动前设置 `export DEEP_NORMAL_MODE_USE_INT8_QUANT=0`。脚本后面的固定赋值
会覆盖命令行前缀的设置。此修复保留图、overlap 和 Attention DP，无需新增 kernel。

接口依据：
- [DeepEP 量化模式及旧环境变量回退](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md#quantization-modes-in-normal-dispatch)
- [torch_npu 融合 GMM/SwiGLU/Quant 接口](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_grouped_matmul_swiglu_quant_v2.md)

## A5 动态窗口调度的 Triton 编译错误

若报错位于 `dspark_schedule.py` 的 `_schedule_topk_selected_extra_kernel`，包含
`failed to legalize unresolved materialization` / `ConvertTritonIRToLinalgIR`，
失败发生在动态窗口的候选 token 选择阶段。部分 A5 Triton-Ascend 版本无法转换
该 CUDA kernel 的循环向量归约；强制解除 AutoBlockify 黑名单不能证明计算正确。

本分支在识别 `confidence.device.type == "npu"` 后使用 NPU 原生 PyTorch 调度：

1. FP32 `cumprod` 生成各请求的前缀存活概率。
2. `topk` 只确定预算边界的概率阈值。
3. 高于阈值的候选先入选；等于阈值的候选按“位置优先、请求编号其次”做 `cumsum`，
   仅填满剩余预算。这样不依赖 `topk` 对相同值的返回顺序。
4. 固定形状归约得到每个请求的验证长度，并应用原来的最小/最大长度限制。

调度不读取设备 Tensor 的标量或列表，不使用 FP64 排序、动态布尔索引或原子 scatter。
CUDA 和 CPU 保持原有 dispatch。服务的图、overlap、Attention DP 设置保持不变；
这不是通过关闭 compact 来绕开问题，也无需新增 sgl-kernel-npu 算子。

CPU 对照测试覆盖并列值、零概率、不同预算、截断长度和非连续输入。另附 A5 实机
capture/replay 测试，验证更新 confidence 后图输出也更新：

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/unit/spec/test_dspark_npu_schedule.py
```

无 NPU 的开发环境会跳过两个实机图测试，CPU 对照通过不等于 A5 图实测通过。
更新分支后请重启两台服务；需要重新测量调度耗时与性能后再判断收益。
