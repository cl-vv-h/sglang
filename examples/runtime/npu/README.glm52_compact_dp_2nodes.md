# GLM-5.2 DSpark compact + Attention DP 双机启动

脚本：`launch_glm52_dspark_compact_dp_2nodes.sh`。两台都使用同一版本的本分支、同一份
脚本与参数。默认每台 16 个 NPU rank，全局 TP32 / Attention DP4 / Attention TP8，
开启 compact、full graph 和 overlap schedule。HTTP 请求发给 node 0。

## 默认配置

| 配置 | 默认值 |
| --- | --- |
| node 0 / node 1 | `61.47.19.70` / `61.47.19.71` |
| 网卡 | `enp196s0f0` |
| 分布式初始化地址 | `61.47.19.70:50000` |
| HTTP 端口 | `8030` |
| TP / 每节点 TP | `32` / `16` |
| Attention DP / Attention TP | `4` / `8` |
| 总请求上限 | `32`，每个 Attention DP 组约 `8` |
| 图捕获请求档位 | `8`，target verify 对应 `8 × 9 = 72` 个 token |
| DeepEP 每 rank 最大 dispatch token | 默认 `128` |
| context / KV token 上限 | `8192` / `32768` |
| prefill chunk / max prefill tokens | `2048` / `8192` |
| mem_fraction_static | `0.85`，需按本机实际权重占用调整 |
| SPS 表 | `/home/sps_npu_tp32_adp4.json`，两台内容相同 |

这是一组初始容量配置，未在双机 NPU 上实测。TP32 改变权重切分，不能直接沿用单机
TP16 下测得的静态预算下界。脚本也不代表之前的 allocator/HCCL 故障已经解决。

## 启动

确保两台都具备目标模型、draft 模型、CANN/ATB 和同内容的 SPS 表。默认权重路径与
单机脚本相同。开放主机间必要的 HCCL 通信；`50000` 是初始化端口，不代表全部通信
只使用该端口。保持两台的设备可见性、通信环境和软件版本一致。

在两台各自的仓库根目录运行，两个进程都需保持运行：

```bash
# 61.47.19.70
bash examples/runtime/npu/launch_glm52_dspark_compact_dp_2nodes.sh 0

# 61.47.19.71
bash examples/runtime/npu/launch_glm52_dspark_compact_dp_2nodes.sh 1
```

日志分别写入各自仓库根目录：

```text
server.node0.dspark.compact.tp32.adp4.log
server.node1.dspark.compact.tp32.adp4.log
```

先检查启动参数，无需设备和权重：

```bash
PRINT_COMMAND=1 bash examples/runtime/npu/launch_glm52_dspark_compact_dp_2nodes.sh 0
PRINT_COMMAND=1 bash examples/runtime/npu/launch_glm52_dspark_compact_dp_2nodes.sh 1
```

换 IP、模型路径或内存预算时，两台都使用相同的公共覆盖项。例如：

```bash
# 两台均设置这组环境；启动时分别传 0、1。
export NODE0_IP=61.47.19.70
export NODE1_IP=61.47.19.71
export NETWORK_IFACE=enp196s0f0
export TP_SIZE=32 ATTN_DP_SIZE=4
export TARGET_MODEL=/home/weights/GLM-5.2-w8a8
export DRAFT_MODEL=/home/weights/GLM-5.2-DSpark-NPU-0805
export SPS_TABLE=/home/sps_npu_tp32_adp4.json
export MEM_FRACTION_STATIC=0.85
```

`MASTER_ADDR`、`DIST_PORT` 也可单独覆盖；两台必须使用同一初始化地址。
图、overlap、DeepEP mode、通信 buffer 和图诊断变量与单机脚本保持一致。
若以前设置过 `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64`，本次应清除或改为
至少 `72`（建议恢复默认 `128`）；72-token 图不能使用 64-token dispatch 容量。
脚本会检查这个条件，不会自动关闭图。

## SPS 与启动验证

尚无双机成本表时，可在**两台都设置** `SPS_TABLE=''` 后启动，先验证连通性和短请求。
此时 compact 预算可能退化为 verify-all，不能据此判断动态窗口的性能收益。

正式标定时，两台分别启动：

```bash
# node 0
SPS_PROFILE=1 bash examples/runtime/npu/launch_glm52_dspark_compact_dp_2nodes.sh 0
# node 1
SPS_PROFILE=1 bash examples/runtime/npu/launch_glm52_dspark_compact_dp_2nodes.sh 1
```

服务就绪后，只在 node 0 执行 profiler。`--batch-size` 是每个 Attention DP 组的
请求数；默认 DP4，对应总请求数乘以 4，不能再乘节点数。

```bash
python python/sglang/benchmark/dspark_sps_profiler.py all \
  --base-url http://127.0.0.1:8030 \
  --batch-size 1 2 4 8 --fracs 0.125 0.25 0.5 0.75 1.0 \
  --input-len 256 --temperature 0 --repeats 3 --no-plot \
  --out /home/sps_npu_tp32_adp4.json
```

将表复制到 node 1 的同一路径，停止两台标定服务，按普通启动命令重新启动。
普通启动会清除模拟接受变量；不要将标定服务的模拟输出用于精度验证。
SPS 表应匹配最终硬件、TP/DP、图和通信设置。

普通服务启动后，请求发送到 `http://61.47.19.70:8030`，例如：

```bash
curl -fsS http://61.47.19.70:8030/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"请简要解释张量并行。","sampling_params":{"temperature":0,"max_new_tokens":64},"stream":false}'
```

遇到相同 HCCL/allocator 报错时，两台都设置 `SGLANG_NPU_DEBUG_GRAPH_CAPTURE=1`，收集
两个节点的首个异常和 `[NPU-GRAPH]` 阶段信息。不要只保留进程退出时的同步错误。

A5 双机（`141.61.45.118/124`，网卡 `data0.3001`）见 [A5 启动说明](README.glm52_compact_dp_a5_2nodes.md)。
