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
